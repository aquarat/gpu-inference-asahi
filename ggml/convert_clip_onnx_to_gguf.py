#!/usr/bin/env python3
"""Convert an Immich open_clip *visual* ONNX export (immich-app/<model>/visual/model.onnx)
into a GGUF file for the clip-vit ggml runner (tools/clip_vit.cpp).

The weights are read straight from Immich's ONNX (raw_data or external-data files, memmapped),
so the GGUF is bit-for-bit the model Immich serves. Handles ViT-B-32 / ViT-L-14 / ViT-H-14-378
(any open_clip VisionTransformer export: conv1 patch embed, class+pos embed, ln_pre, N resblocks
with fused in_proj MHA + c_fc/c_proj MLP, ln_post on CLS, visual.proj, L2-normalised output).

usage: convert_clip_onnx_to_gguf.py <model_dir> <out.gguf> [--outtype f16|f32|q8_0]
"""
import argparse, json, os, sys, re
import numpy as np
import onnx
from onnx import numpy_helper
import gguf
from gguf import GGMLQuantizationType as QT

ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("out")
ap.add_argument("--outtype", default="f16", choices=["f16", "f32", "q8_0"])
args = ap.parse_args()

mdir = args.model_dir
onnx_path = os.path.join(mdir, "visual", "model.onnx")
model = onnx.load(onnx_path, load_external_data=False)
g = model.graph
cfg = json.load(open(os.path.join(mdir, "config.json")))
pp = json.load(open(os.path.join(mdir, "visual", "preprocess_cfg.json")))
vcfg = cfg["vision_cfg"]

inits = {t.name: t for t in g.initializer}

def load(name):
    t = inits[name]
    if t.data_location == onnx.TensorProto.EXTERNAL:
        info = {kv.key: kv.value for kv in t.external_data}
        path = os.path.join(os.path.dirname(onnx_path), info["location"])
        off = int(info.get("offset", 0)); length = int(info.get("length", os.path.getsize(path) - off))
        dt = onnx.helper.tensor_dtype_to_np_dtype(t.data_type)
        arr = np.memmap(path, dtype=dt, mode="r", offset=off, shape=(length // np.dtype(dt).itemsize,))
        return np.asarray(arr).reshape(list(t.dims))
    return numpy_helper.to_array(t)

# --- hparams from the graph
width = vcfg["width"]; layers = vcfg["layers"]; patch = vcfg["patch_size"]; image_size = vcfg["image_size"]
head_width = vcfg.get("head_width", 64); heads = width // head_width
embed_dim = cfg["embed_dim"]
n_sig = sum(1 for n in g.node if n.op_type == "Sigmoid"); n_erf = sum(1 for n in g.node if n.op_type == "Erf")
if n_sig == layers and n_erf == 0: gelu = "quick"
elif n_erf == layers and n_sig == 0: gelu = "erf"
else: sys.exit(f"cannot determine GELU type: sigmoid={n_sig} erf={n_erf} layers={layers}")
ln_eps = None
for n in g.node:
    if n.op_type == "LayerNormalization":
        ln_eps = [onnx.helper.get_attribute_value(a) for a in n.attribute if a.name == "epsilon"][0]; break
assert ln_eps is not None

# --- map MatMul weights to modules via node names (/visual/transformer/resblocks.N/attn/MatMul etc.)
mm = {}   # module path -> initializer name
for n in g.node:
    if n.op_type in ("MatMul", "Gemm"):
        w = [i for i in n.input if i in inits]
        if w: mm[n.name] = (n, w[0])
def mm_weight(node_name):
    n, w = mm[node_name]
    arr = load(w).astype(np.float32)
    if n.op_type == "Gemm":
        transB = [onnx.helper.get_attribute_value(a) for a in n.attribute if a.name == "transB"]
        if transB and transB[0]: return arr            # torch layout [out, in]
        return arr.T
    return arr.T                                          # MatMul: [in, out] -> [out, in]

# class embedding and positional embedding: the inputs of /visual/Expand and /visual/Add
cls_name = [i for i in next(n for n in g.node if n.name == "/visual/Expand").input if i in inits][0]
pos_name = [i for i in next(n for n in g.node if n.name == "/visual/Add").input if i in inits][0]
cls = load(cls_name).astype(np.float32).reshape(-1)
pos = load(pos_name).astype(np.float32)
n_pos = (image_size // patch) ** 2 + 1
assert cls.shape == (width,), cls.shape
assert pos.shape == (n_pos, width), (pos.shape, n_pos, width)

# --- write
w = gguf.GGUFWriter(args.out, "clip-vit")
w.add_name(os.path.basename(mdir.rstrip("/")))
w.add_description("open_clip VisionTransformer converted from Immich ONNX for ggml clip_vit")
w.add_file_type({"f16": gguf.LlamaFileType.MOSTLY_F16, "f32": gguf.LlamaFileType.ALL_F32, "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0}[args.outtype])
w.add_uint32("clip-vit.width", width)
w.add_uint32("clip-vit.layers", layers)
w.add_uint32("clip-vit.heads", heads)
w.add_uint32("clip-vit.patch_size", patch)
w.add_uint32("clip-vit.image_size", image_size)
w.add_uint32("clip-vit.embed_dim", embed_dim)
w.add_float32("clip-vit.ln_eps", float(ln_eps))
w.add_string("clip-vit.gelu", gelu)
w.add_array("clip-vit.image_mean", [float(x) for x in pp["mean"]])
w.add_array("clip-vit.image_std", [float(x) for x in pp["std"]])

def add(name, arr, kind):
    """kind: 'mat' (2D GEMM weight, quantisable), 'f16' (f16 unless f32 mode), 'f32'"""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if kind == "f32" or args.outtype == "f32":
        w.add_tensor(name, arr); return
    if kind == "mat" and args.outtype == "q8_0" and arr.shape[-1] % 32 == 0:
        q = gguf.quants.quantize(arr, QT.Q8_0)
        w.add_tensor(name, q, raw_dtype=QT.Q8_0); return
    w.add_tensor(name, arr.astype(np.float16))

add("patch_w", load("visual.conv1.weight"), "f16")          # [width, 3, p, p]
add("class_embd", cls, "f32")
add("pos_embd", pos, "f32")                                  # [n_pos, width]
add("ln_pre_w", load("visual.ln_pre.weight"), "f32"); add("ln_pre_b", load("visual.ln_pre.bias"), "f32")
for i in range(layers):
    p = f"visual.transformer.resblocks.{i}"; q = f"/visual/transformer/resblocks.{i}"
    add(f"blk.{i}.ln1_w", load(f"{p}.ln_1.weight"), "f32"); add(f"blk.{i}.ln1_b", load(f"{p}.ln_1.bias"), "f32")
    add(f"blk.{i}.qkv_w", mm_weight(f"{q}/attn/MatMul"), "mat")            # [3*width, width]
    add(f"blk.{i}.qkv_b", load(f"{p}.attn.in_proj_bias"), "f32")
    add(f"blk.{i}.out_w", load(f"{p}.attn.out_proj.weight"), "mat")        # torch layout [width, width]
    add(f"blk.{i}.out_b", load(f"{p}.attn.out_proj.bias"), "f32")
    add(f"blk.{i}.ln2_w", load(f"{p}.ln_2.weight"), "f32"); add(f"blk.{i}.ln2_b", load(f"{p}.ln_2.bias"), "f32")
    add(f"blk.{i}.fc1_w", mm_weight(f"{q}/mlp/c_fc/MatMul"), "mat")        # [4*width, width]
    add(f"blk.{i}.fc1_b", load(f"{p}.mlp.c_fc.bias"), "f32")
    add(f"blk.{i}.fc2_w", mm_weight(f"{q}/mlp/c_proj/MatMul"), "mat")      # [width, 4*width]
    add(f"blk.{i}.fc2_b", load(f"{p}.mlp.c_proj.bias"), "f32")
add("ln_post_w", load("visual.ln_post.weight"), "f32"); add("ln_post_b", load("visual.ln_post.bias"), "f32")
add("proj_w", load("visual.proj").astype(np.float32).T, "mat")           # [embed_dim, width]

w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"wrote {args.out}: width={width} layers={layers} heads={heads} patch={patch} img={image_size} n_pos={n_pos} embed={embed_dim} gelu={gelu} eps={ln_eps} outtype={args.outtype} size={os.path.getsize(args.out)/1e6:.0f} MB")
