# CLIP image encoder on ggml (Vulkan)

A converter from Immich's open_clip visual ONNX exports to GGUF and a ~250-line
ggml runner that computes the CLIP image embedding (CLS pooling, `ln_post`,
`visual.proj`, L2-normalised: exactly the ONNX output) on any ggml backend.
Written because no existing ggml tool does this: llama.cpp's `mtmd` returns
projector patch tokens for VLMs and its converters drop `visual.proj` and
`ln_post`; monatis/clip.cpp is pinned to a 2023 ggml without backends.
Details and all measurements: [../reports/PHASE2_GGML.md](../reports/PHASE2_GGML.md).

| file | what |
|---|---|
| `convert_clip_onnx_to_gguf.py` | `<immich model dir>` (with `config.json`, `visual/model.onnx` [+ external data], `visual/preprocess_cfg.json`) to GGUF; `--outtype f16` (default), `f32`, `q8_0`. Reads the weights straight from the ONNX (memmapped), so the GGUF is bit-for-bit what Immich serves. Detects quickGELU vs erf-GELU from the graph. Works unchanged for ViT-B-32, ViT-L-14 and ViT-H-14-378. Needs only `onnx`, `numpy` and `gguf` (from `llama.cpp/gguf-py`). |
| `clip_vit.cpp` | The runner: loads the GGUF, builds the tower with `ggml_conv_2d` + N transformer blocks (classic attention or `ggml_flash_attn_ext`), runs it through `ggml_backend_sched` on `Vulkan0` or `CPU`, times build/compute/download per call, reports unsupported ops (none on Vulkan for the three models), writes the embedding. |
| `ref_ort.py` | Immich-exact preprocessing (shorter side to S, centre crop, mean/std, NCHW f32) and the onnxruntime CPU fp32 reference embedding + timing. Produces `<refdir>/<img>.in.bin` and `<refdir>/<img>.ort.npy`. |
| `cmp.py` | Cosine of the runner's embedding against the reference. |
| `run_clip.sh` | One configuration over the four test images: per-image median and the median of medians. |
| `bench_chain{1,2,3}.sh` | The benchmark sequences the report quotes (llama-bench, ViT-B-32 variants, ViT-L/ViT-H, precision options, `test-backend-ops`). |

## Build

llama.cpp at commit `9e0e220` (2026-09-06, `git clone --depth 1
https://github.com/ggml-org/llama.cpp`). Fedora Asahi 44 has `glslc` (shaderc
2026.1), `vulkan-headers` and `glslang` but not `spirv-headers-devel`, which
`ggml-vulkan` needs (`find_package(SPIRV-Headers CONFIG)` plus
`spirv/unified1/spirv.hpp`); without root, install it into a prefix:

```sh
git clone https://github.com/KhronosGroup/SPIRV-Headers
cmake -S SPIRV-Headers -B SPIRV-Headers/build -DCMAKE_INSTALL_PREFIX=$PWD/prefix && cmake --install SPIRV-Headers/build
cmake -S llama.cpp -B llama.cpp/build-vk -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=$PWD/prefix -DCMAKE_CXX_FLAGS=-I$PWD/prefix/include
cmake --build llama.cpp/build-vk -j4 --target llama-bench llama-quantize test-backend-ops   # ~25 min on an M1
g++ -O2 -std=c++17 clip_vit.cpp -o clip_vit -Illama.cpp/ggml/include -Lllama.cpp/build-vk/bin \
    -Wl,-rpath,$PWD/llama.cpp/build-vk/bin -lggml -lggml-base -lggml-cpu -lggml-vulkan
```

(The `CMAKE_CXX_FLAGS` include is needed because the SPIRV-Headers config
file alone does not add its include directory to the `ggml-vulkan.cpp`
compile.) Python side: `uv venv && uv pip install onnx onnxruntime numpy
pillow -e llama.cpp/gguf-py`.

## Use

```sh
python3 convert_clip_onnx_to_gguf.py $IMMICH_CACHE/clip/ViT-H-14-378-quickgelu__dfn5b vith14_q8_0.gguf --outtype q8_0
python3 ref_ort.py $IMMICH_CACHE/clip/ViT-H-14-378-quickgelu__dfn5b ref/vith14 4 2 img1.png img2.jpg ...
./clip_vit vith14_q8_0.gguf Vulkan0 ref/vith14/img1.in.bin out.bin 1 3 4 1     # warmup iters threads flash-attn
python3 cmp.py ref/vith14 out/
GGML_VK_DISABLE_F16=1 ./clip_vit ...                                          # fp32 shader math: cosine 0.99999, ~1.75x slower
```

`run_clip.sh <tag> <model.gguf> <refdir> <Vulkan0|CPU> <warmup> <iters> <threads> <fa 0|1> [MemoryMax]`
expects the four test images named `shapes`, `noise`, `checker`, `t1` in
`<refdir>` (synthetic PIL images plus one public sample were used; none are
included) and wraps each run in `systemd-run --user --scope -p MemoryMax=`.

## Results (Apple M1, Mesa 26.1.8 Honeykrisp, next to a live NVR; median of 3-5 per image, median over 4 images)

| model | ggml-Vulkan | cosine vs ORT fp32 | ORT CPU fp32, 4 threads | notes |
|---|---:|---|---:|---|
| ViT-B-32 (224 px, 12x768, 50 tokens) | 39 ms f16 / 38 ms +FA / 40 ms Q8_0 | 0.9991-0.9996 | 33 ms | too small to fill the GPU (~0.1 ms per node); ggml CPU Q8_0 32 ms |
| ViT-L-14 (224 px, 24x1024, 257 tokens) | 348 ms f16 / 341 ms Q8_0 | 0.9997-0.9998 | 563 ms | 466 GFLOPS, 19% of peak |
| ViT-H-14-378 (378 px, 32x1280, 730 tokens, 1007 GFLOP) | 1874 ms f16 / **1776 ms f16+FA / 1757 ms Q8_0+FA** | 0.9985-0.9991 | 3678 ms (3218 idle) | 566 GFLOPS, 23% of peak; host RSS 75 MB, UMA weights 1268 MB (f16) or 677 MB (Q8_0) vs ORT's 2.5 GB; fp32 math 3119 ms at cosine 0.99999 |

For comparison on the same driver: MNN-Vulkan fp16 ViT-H 5.8-7.7 s (2% of
peak, 2.4 GB GPU + 2.8 s CPU per call), MNN CPU fp16 1.9 s. ggml's own CPU
path is poor for ViTs on this chip (ViT-H 4.0-5.4 s), so "ggml" here means
ggml on the GPU; the CPU fallback should stay ORT.

`llama-bench` on Qwen2.5-0.5B: Vulkan Q8_0 pp512 783 t/s (~0.77 TFLOPS,
~30% of the 2.5 TFLOPS clpeak fp16 figure), F16 767 t/s; CPU 4 threads Q8_0
743 t/s, F16 303 t/s. Batch-1 token generation is slower on the GPU (42 vs
64 t/s): the per-dispatch overhead still shows for tiny memory-bound ops.

Vulkan device facts that shape this: `fp16: 1 | bf16: 0 | int dot: 0 | warp
size: 32 | shared memory: 32768 | matrix cores: none`, so ggml uses its
scalar fp16 GEMM and scalar flash-attention shaders. If Honeykrisp gains
`VK_KHR_cooperative_matrix`, ggml picks its coopmat shaders up automatically.

## Towards an Immich backend

Section 5 of the report sketches it: a `libclipvit.so` with a four-function
C API, a `GgmlSession(ModelSession)` in Immich's `sessions/` calling it over
ctypes, convert-on-first-load from the ONNX Immich already downloads, visual
tower on ggml-Vulkan, text/faces/OCR on ORT-CPU. That work is happening in a
separate Immich fork.
