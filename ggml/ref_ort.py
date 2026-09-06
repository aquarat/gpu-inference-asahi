#!/usr/bin/env python3
"""Immich-exact CLIP visual preprocessing + ORT CPU reference embeddings and timing.
usage: ref_ort.py <model_dir> <outdir> <threads> <iters> image1 [image2 ...]
writes <outdir>/<stem>.in.bin (f32 NCHW 1x3xSxS), <outdir>/<stem>.ort.npy (L2-normalised embedding),
prints ORT median latency (warm) over the images.
"""
import sys, json, os, time, statistics
import numpy as np, onnxruntime as ort
from PIL import Image

mdir, outdir, threads, iters = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
images = sys.argv[5:]
os.makedirs(outdir, exist_ok=True)
pp = json.load(open(f"{mdir}/visual/preprocess_cfg.json"))
size = pp["size"][0] if isinstance(pp["size"], list) else pp["size"]
mean, std = np.array(pp["mean"], np.float32), np.array(pp["std"], np.float32)
resample = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR}.get(pp["interpolation"], Image.Resampling.BICUBIC)

def resize_pil(im, s):   # immich_ml.models.transforms.resize_pil: shorter side -> s
    w, h = im.size; scale = s / min(w, h)
    return im.resize((max(s, round(w * scale)), max(s, round(h * scale))), resample)
def crop_pil(im, s):
    w, h = im.size; l, t = (w - s) // 2, (h - s) // 2
    return im.crop((l, t, l + s, t + s))
def preprocess(path):
    img = Image.open(path).convert("RGB")
    x = np.asarray(crop_pil(resize_pil(img, size), size), np.float32) / 255.0
    return ((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32).copy()

so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1; so.enable_cpu_mem_arena = False
s = ort.InferenceSession(f"{mdir}/visual/model.onnx", so, providers=["CPUExecutionProvider"])
iname = s.get_inputs()[0].name
def rss(): return int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0]) // 1024
times = []
for p in images:
    stem = os.path.splitext(os.path.basename(p))[0]
    x = preprocess(p); x.tofile(f"{outdir}/{stem}.in.bin")
    for _ in range(1): s.run(None, {iname: x})
    for _ in range(iters):
        a = time.perf_counter(); emb = s.run(None, {iname: x})[0]; times.append((time.perf_counter() - a) * 1e3)
    emb = emb[0] / np.linalg.norm(emb[0]); np.save(f"{outdir}/{stem}.ort.npy", emb)
    print(f"{stem}: input {x.shape} emb {emb.shape} norm-check {float(np.linalg.norm(emb)):.4f}")
print(f"ORT CPU fp32 thr={threads} size={size}: median {statistics.median(times):.1f} ms min {min(times):.1f} n={len(times)} rss {rss()} MB")
