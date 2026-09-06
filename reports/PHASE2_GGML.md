# Phase 2: ggml / llama.cpp Vulkan backend for Immich's CLIP ViT-H-14-378 on the M1 Mac mini

Date: 2026-09-06. Host: Apple M1 Mac mini (T8103, G13G GPU, 8 cores, 16 GB shared with the live Frigate NVR + `frigate-detector.service`), Fedora Asahi 44, Mesa 26.1.8 Honeykrisp (Vulkan 1.4, fp16 yes, **no cooperative matrix**), llvmpipe also present.
Working dir: `gpu-opt/ggml/` (a local work tree; `tools/` is `ggml/` in this repository) (`llama.cpp/` clone, `build-vk/`, `tools/` converter + runner, `models/` GGUFs, `ref/` ORT reference outputs, `out/` ggml outputs, `venv/`). This report is written incrementally; background numbers come from `IMMICH_ML_GPU.md` (the Immich-side write-up, published with the Immich fork) and `VULKAN_DISPATCH_EXPERIMENT.md`.

## 0. TL;DR

* **Yes -- ggml's Vulkan backend runs Immich's ViT-H-14-378 efficiently on Honeykrisp.** `clip_vit` (a ~250-line ggml runner fed by a converter that reads Immich's own ONNX) encodes one 378-px image in **1.76-1.87 s on the GPU** (Q8_0/f16 weights, fp16 shader math, cosine 0.9985-0.9994 vs ORT fp32), or **3.1 s in fp32-math mode with cosine 0.99999**. That is **2x faster than Immich's ORT-CPU (3.7 s today, 3.2 s idle), 4x faster than MNN-Vulkan (7.0-7.7 s) on the same driver**, and level with the best CPU path (MNN CPU fp16 1.9 s) -- while the process holds **75 MB of host RAM plus 0.68-1.27 GB of GPU/UMA weights** instead of ORT's 2.5 GB. Effective 566 GFLOPS = 23 % of the GPU's fp16 peak vs 2 % for MNN.
* The earlier "Vulkan is hopeless on this driver" conclusion was a kernel/runtime finding, not a driver one: ggml's llama-bench reaches **783 t/s pp512 (0.77 TFLOPS) on a Q8_0 0.5B LLM**, 15x MNN's efficiency, with one command buffer per graph and tiled fp16 shaders. Caveat: ggml's *CPU* Q8_0 GEMM is nearly as fast (743 t/s) on this chip, so the GPU is only a clear win for fp16/large-M work like ViT-H (1.6x on ViT-L, ~1x on ViT-B-32; ggml-CPU itself is poor for ViTs: 4.0-5.4 s for ViT-H).
* No existing ggml CLIP tool does this out of the box: llama.cpp `mtmd` has all the ops (incl. quickGELU) but outputs VLM patch tokens, its converters drop `visual.proj`/`ln_post`; monatis/clip.cpp is CPU-only on a 2023 ggml. Converting from Immich's ONNX (`tools/convert_clip_onnx_to_gguf.py`, no torch, no 4 GB HF download) + the small runner was the direct route and works for all three Immich CLIP ViTs unchanged.
* Deployment shape: `libclipvit.so` + ctypes `GgmlSession` in Immich's `sessions/` (or a host-side service like the Frigate detector), visual tower on ggml-Vulkan, text/faces/OCR stay on ORT-CPU. ~3 days to a PR-shaped fork. Measure `frigate-detector` p95 under a sustained embedding burst first. `immich-ml` container was **not** stopped (peak host usage stayed inside 5 GB caps); `frigate-detector` logged 0 fallbacks.

## 1. Survey: ggml-based CLIP image encoders and the Vulkan backend

| implementation | state (2026-09-06) | fit for Immich's ViT-H-14-378 DFN5B |
|---|---|---|
| **llama.cpp `tools/mtmd` (`clip.cpp`, `clip-graph.h`, `models/llava.cpp`, ...)** | actively maintained; ~60 projector types (LLaVA, SigLIP, Qwen-VL, Gemma, InternVL, ...); runs on every ggml backend incl. Vulkan; graph code has `build_vit()`, `build_attn()` (flash-attn or classic), `FFN_GELU / FFN_GELU_ERF / FFN_GELU_QUICK` (`clip.cpp:662-679`, quickGELU already exists, selected by the `clip.use_gelu`/`use_silu` KV pair -> defaults to gelu_quick when neither is set, `clip.cpp:1369-1384`) | Built for **VLM projectors**: `clip_image_encode()` returns the *patch-token sequence after the projector* (for llava: layer N-1 features -> MLP), not CLIP's pooled CLS -> `ln_post` -> `visual.proj` embedding. `convert_hf_to_gguf.py`'s `MmprojModel` subclasses (`conversion/llava.py: LlavaVisionModel`, `hidden_act` only `gelu`/`silu`, `conversion/llava.py:86-92`) and the legacy `tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py` (HF `CLIPModel`/`CLIPVisionModel`; explicitly **drops** `visual_projection.weight` and `post_layernorm` when building an mmproj, `legacy-models/convert_image_encoder_to_gguf.py:27`) both expect the HF-transformers checkpoint layout. `apple/DFN5B-CLIP-ViT-H-14-378` on HF ships only `pytorch_model.bin` / `open_clip_pytorch_model.bin` (3.95 GB each, pickle, no safetensors). So: the *kernels and graph pieces* are all there, but the CLIP-embedding graph (CLS pooling + post-LN + projection + L2 norm) and a converter that keeps those tensors need to be added; the 378-px positional embedding (730 positions) is just a tensor size, and quickGELU is already an op. |
| **monatis/clip.cpp** (standalone, 568 stars) | README updated 2026-08-24 but the code is stale: ggml submodule pinned at `c3ae31e5` = **2023-09-16**, i.e. pre-`ggml-backend` -> **CPU only, no Vulkan** (the Vulkan backend landed in ggml in 2024). Its `convert_hf_to_gguf.py` reads HF `CLIPModel` (vision + text + projections, `clip-cpp-gguf` HF tag, Q4/Q5/Q8/f16). Author now points to `monatis/ggmlc` (a PyTorch/JAX -> GGML compiler, new, unreviewed here). | Would need a port to modern ggml to get Vulkan; the model-file format ideas (vision-only GGUF with `v.` tensors + projection) are reusable but the graph is the same ~200 lines we wrote ourselves. |
| **staghado/vit.cpp**, wwwsctvcom/ggml-vit | ViT classification (timm) ports, last pushed 2024, CPU only | not CLIP; nothing to reuse. |
| **this work: `gpu-opt/ggml/tools/clip_vit.cpp` + `convert_clip_onnx_to_gguf.py`** | ~250 lines of C++ on the public ggml backend API (`ggml_backend_sched`, `gguf`), ~130 lines of Python | converts **Immich's own ONNX** (`cache/clip/<model>/visual/model.onnx`, external-data files memmapped) -> GGUF (f32/f16/Q8_0), so weights are byte-identical to what Immich serves, no HF download (the graph's node names `/visual/transformer/resblocks.N/attn/MatMul` etc. identify the transposed `onnx::MatMul_*` weights; GELU type is detected from the graph: ViT-H = 32 `Sigmoid` = quickGELU, ViT-B-32__openai = 12 `Erf`; LN eps 1e-5). Graph = conv patch embed (im2col + GEMM), CLS+pos, `ln_pre`, N x (LN, fused QKV GEMM, attention (classic soft-max path or `ggml_flash_attn_ext`), out-proj, LN, fc1, GELU, fc2), `ln_post` on CLS only, `visual.proj`, L2-normalise -> exactly the ONNX output. Runs unmodified on `Vulkan0` or `CPU`. |

**Conversion route decision**: converting from Immich's ONNX beats the HF route on every axis (no 4 GB pickle download, identical weights, preprocessing config already next to it, works for every Immich CLIP model with the open_clip layout). The HF `pytorch_model.bin` route would need `torch` + the legacy converter patched to keep `post_layernorm`/`visual_projection` and to emit `use_gelu=false` (quick); not pursued.

Vulkan backend facts relevant to this GPU (from `ggml_vulkan` device init and `ggml-vulkan.cpp`): `Apple M1 (G13G B1) (Honeykrisp) | uma: 1 | fp16: 1 | bf16: 0 | int dot: 0 | warp size: 32 | shared memory: 32768 | matrix cores: none`. So ggml uses its **scalar fp16 GEMM shaders** (`FLOAT_TYPE = float16_t`, `ggml-vulkan.cpp:4076`), the **scalar flash-attention path** (`FA_SCALAR`, `ggml-vulkan.cpp:3922`), no integer-dot MMQ, no coopmat; fusion (`GGML_VK_DISABLE_FUSION`) and graph optimisation are on by default. All shader variants (coopmat/coopmat2/int-dot/bf16) are compiled in but selected at runtime.

## 2. Build notes

* `git clone --depth 1 https://github.com/ggml-org/llama.cpp` @ `9e0e220` (2026-09-06) -> `gpu-opt/ggml/llama.cpp`, `build-vk/`.
* `glslc` is already installed (`/usr/bin/glslc`, shaderc 2026.1); `vulkan-headers 1.4.341`, `glslang 16.2`. **Missing: `spirv-headers-devel`** (ggml-vulkan needs `find_package(SPIRV-Headers CONFIG)` + `spirv/unified1/spirv.hpp`). Not installed system-wide (no sudo use): cloned KhronosGroup/SPIRV-Headers, `cmake --install` into `gpu-opt/ggml/prefix`, then `-DCMAKE_PREFIX_PATH=.../prefix` and `-DCMAKE_CXX_FLAGS=-I.../prefix/include` (the config file alone does not add the include dir for the `ggml-vulkan.cpp` compile).
* `cmake -B build-vk -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release` under `systemd-run --user --scope -p MemoryMax=6G`, `cmake --build -j4` (targets `llama-bench llama-cli llama-mtmd-cli llama-quantize test-backend-ops`); ~25 min wall, peak well under the cap; CPU backend variant `-mcpu=apple-m1+crc+aes+sha3+fp16+dotprod` (no i8mm/SVE/SME on the M1). Shared libs in `build-vk/bin/` (`libggml*.so`, `libllama.so`).
* Runner: `g++ -O2 -std=c++17 tools/clip_vit.cpp -Illama.cpp/ggml/include -Lllama.cpp/build-vk/bin -lggml -lggml-base -lggml-cpu -lggml-vulkan` (rpath set). Python venv `gpu-opt/ggml/venv` (uv, 3.12): onnx 1.22, onnxruntime 1.29, numpy, pillow, `gguf` editable from `llama.cpp/gguf-py`.
* Test images (non-personal): `images/shapes.png`, `noise.png`, `checker.png` (synthetic, PIL) and `t1.jpg` (InsightFace's public sample used by the earlier research). Preprocessing = Immich's `OpenClipVisualEncoder.transform` (shorter side -> S, centre crop, mean/std, NCHW f32) in `tools/ref_ort.py`, which also produces the ORT-CPU reference embeddings (4 threads).

## 3. llama-bench: ggml GEMM throughput on Honeykrisp vs the M1 CPU

Qwen2.5-0.5B-Instruct (`Qwen/Qwen2.5-0.5B-Instruct-GGUF`, 630 M params incl. the 136 M tied embedding/lm_head), pp512 = prompt processing (GEMM-bound), tg128 = token generation (bandwidth/latency-bound), 3 repetitions, `-t 4`. Frigate's NVR was live during all runs. ~0.99 GFLOP/token excluding attention, so **783 t/s ≈ 0.77 TFLOPS**.

| model | device | pp512 t/s | tg128 t/s | notes |
|---|---|---:|---:|---|
| Q8_0 | **Vulkan0 (Honeykrisp)** | **783 ± 7** (fa off) / 768 (fa on) | 42 / 45 | pp128 546, pp1024 713 |
| F16 | Vulkan0 | 767 / 758 | 38 / 22 | fp16 scalar GEMM shaders (`matmul_f16`), no coopmat |
| Q8_0 | CPU, 4 threads (`-dev none`) | **743 ± 4** | 64 | ggml's int8 `dotprod` GEMM (`-mcpu=apple-m1+dotprod`) |
| Q8_0 | CPU, 8 threads | 635 | 45 | E-cores slow it down, as with ORT |
| F16 | CPU, 4 threads | 303 | 42 | NEON fp16 GEMM |
| F16 | CPU, 8 threads | 338 | 8 | |

Reading:
* **ggml's Vulkan GEMM reaches ~0.77 TFLOPS on this GPU, i.e. ~30 % of the 2.5 TFLOPS `clpeak` fp16 figure -- 15x the ~2 % MNN-Vulkan achieved** (47 GFLOPS on ViT-H). The per-op dispatch cost that crippled MNN (one command buffer per op) is absent: ggml records the whole graph into one command buffer with the minimal barriers, fuses adds/activations, and its tiled fp16 shaders use the 32 KB shared memory well. So "Vulkan on Honeykrisp is hopeless for transformers" was an MNN/ncnn kernel finding, not a driver finding.
* **But the same chip's CPU is nearly as fast for int8 GEMMs** (743 vs 783 t/s with 4 threads): ggml's Q8_0 CPU path uses the NEON `sdot` instruction, and the M1's 4 P-cores deliver ~0.73 TFLOPS int8-effective. For fp16 GEMMs the GPU is 2.5x the CPU (767 vs 303).
* tg (memory-bound, batch 1) is *slower* on the GPU (42 vs 64 t/s) -- irrelevant for CLIP (730 tokens per image behaves like prompt processing) but it shows the per-dispatch overhead is still there for tiny ops.

`test-backend-ops perf -o MUL_MAT` (its built-in shape list; the 5-min cap only reached the mat-vec cases n=1..5, k=14336, m=4096 -- i.e. bandwidth-bound decode shapes, not the ViT GEMMs): Vulkan f16 13-97 GFLOPS, Q8_0 42-102 GFLOPS; CPU (4 thr) f16 41-167, Q8_0 75-186 GFLOPS. Same picture as tg128: for small-N, memory-bound ops the CPU beats this GPU, so the GPU advantage is specifically the large-N GEMMs of a 730-token ViT (566 GFLOPS effective) and LLM prompt processing (770 GFLOPS). (`MESA: error: Opening /dev/dri/card2 failed: Permission denied` printed by test-backend-ops is harmless: the render node is what ggml uses.)

## 4. CLIP results

Methodology: `tools/run_clip.sh` runs `clip_vit` on each of the 4 test images (1 warm-up + N timed iterations, each iteration = rebuild graph + upload input + compute + download embedding, i.e. what a service call costs), reports the median per image and the median of those; cosine = dot product of the L2-normalised ggml embedding and the ORT-CPU fp32 embedding (Immich's reference). All under `systemd-run --scope -p MemoryMax=4-5G`, 4 threads for CPU paths, Frigate live.

### 4.1 ViT-B-32__openai (224 px, 12 x 768, 50 tokens, 8.8 GFLOP/image, erf-GELU)

| runtime / precision | latency (median, ms) | weights | cosine vs ORT fp32 (min / mean over 4 images) |
|---|---:|---:|---|
| ORT CPU fp32 (Immich, 4 thr) | 33.1 | 351 MB (RSS 624) | 1 (reference) |
| MNN CPU fp16 (earlier research) | 19.0 | | 0.9998 |
| MNN-Vulkan fp16 (earlier research) | 133-216 | | 0.99998 |
| ncnn-Vulkan fp16 (earlier research) | 66 | | 0.9995 |
| **ggml Vulkan f16** | **39.4** | 176 MB | 0.99912 / 0.99955 |
| ggml Vulkan f16 + flash-attn (scalar FA) | 38.1 | 176 MB | 0.99926 / 0.99960 |
| ggml Vulkan Q8_0 | 39.5 | 96 MB | 0.99901 / 0.99944 |
| ggml Vulkan f32 weights (activations still f16 in shaders) | 43.7 | 351 MB | 0.99853 / 0.99951 |
| ggml Vulkan f16, `GGML_VK_DISABLE_F16=1` (fp32 shader math) | 49.8 | 176 MB | **0.999999 / 1.000000** |
| ggml CPU f16 | 43.1 | 176 MB (RSS 189) | 0.99995 / 0.99997 |
| ggml CPU Q8_0 | **31.8** | 96 MB (RSS 113) | 0.99939 / 0.99958 |
| ggml CPU f32 | 72.7 | 351 MB | 0.99995 / 0.99998 |

* ggml-Vulkan is **3.4-5.5x faster than MNN-Vulkan and 1.7x faster than ncnn-Vulkan** on the same model and driver, and now within 20 % of ORT-CPU -- but this model is too small to fill the GPU (8.8 GFLOP in 39 ms = 225 GFLOPS; 370 graph nodes -> ~0.1 ms per node is dispatch/latency-bound, consistent with the ~80-100 us submit round trip measured earlier).
* Precision: the ~1e-3 cosine loss on Vulkan comes from **fp16 shader arithmetic** (the Vulkan backend computes the GEMMs with `float16_t` when the device reports fp16), not from the f16 weights: forcing fp32 math (`GGML_VK_DISABLE_F16=1`) gives cosine 0.999999 at +27 % time, and the CPU f16 path (f16 weights, f32 accumulation) gives 0.99995. 0.999 is on par with ncnn fp16 (0.9995) and well above the ORT-int8 drift (0.989) flagged as risky in the earlier research; fine for retrieval.

### 4.2 ViT-L-14__openai (224 px, 24 x 1024, 257 tokens, 162 GFLOP/image, erf-GELU)

| runtime / precision | latency (median, ms) | weights | cosine vs ORT fp32 (min / mean) |
|---|---:|---:|---|
| ORT CPU fp32 (4 thr) | 563 | 1.6 GB (RSS 2.05 GB) | 1 |
| **ggml Vulkan f16** | **348** | 609 MB | 0.99977 / 0.99981 |
| ggml Vulkan f16 + FA | 355 | 609 MB | 0.99977 / 0.99982 |
| ggml Vulkan Q8_0 | 341 | 325 MB | 0.99968 / 0.99973 |
| ggml CPU f16 | 766 | 609 MB | 0.99997 / 0.99998 |
| ggml CPU Q8_0 | 603 | 325 MB | 0.99975 / 0.99981 |

466 GFLOPS effective on the GPU (19 % of peak); GPU 1.6x faster than ORT-CPU, 1.7x faster than ggml's own CPU path.

### 4.3 ViT-H-14-378-quickgelu__dfn5b (378 px, 32 x 1280, 730 tokens, 1007 GFLOP/image incl. 87 GFLOP attention, quickGELU) -- Immich's production model

| runtime / precision | latency (median of 3 per image, median over 4 images) | memory | cosine vs ORT fp32 (min / mean) |
|---|---:|---:|---|
| ORT CPU fp32 (Immich, 4 thr) | 3678 ms (3218 ms in the earlier research; today's box is busier) | RSS 2.5 GB | 1 |
| MNN CPU fp16 (earlier research) | 1893 | RSS 1.3-1.7 GB | 0.9997 |
| MNN-Vulkan fp16 (earlier research) | 6958-7697 (5791 on the patched driver) | 2.4 GB GPU + 2.8 s CPU per inference | 0.99999 |
| **ggml Vulkan f16** | **1874** (1851-1907) | **1268 MB Vulkan buffer + 57 MB compute buffer; host RSS 75 MB** | 0.99854 / 0.99903 |
| **ggml Vulkan f16 + flash-attn (scalar FA)** | **1776** (1774-1799) | same | 0.99852 / 0.99903 |
| ggml Vulkan Q8_0 | 1810 (1804-1827) | **677 MB** Vulkan buffer | 0.99858 / 0.99907 |
| ggml CPU f16 (4 thr) | 5368 | RSS 1.29 GB | 0.99986 / 0.99993 |
| ggml CPU Q8_0 (4 thr) | 3987 | RSS 726 MB | 0.99979 / 0.99986 |

Per-call breakdown (Vulkan f16): graph build + allocation + input upload 5 ms, GPU compute 1871 ms, download <0.1 ms; model load (1.27 GB file -> Vulkan buffer) 0.4-1.2 s; first-call shader compile/warm-up adds ~10 ms only (pipelines are compiled at backend init, ~1 s).

Reading:
* **ggml-Vulkan runs Immich's ViT-H-14-378 in 1.8 s: 2.0x faster than Immich's ORT-CPU (3.7 s today / 3.2 s idle), 4x faster than MNN-Vulkan on the same driver, and on par with the best CPU result (MNN CPU fp16 1.9 s).** Effective throughput 566 GFLOPS = 23 % of the fp16 peak, vs 2 % for MNN-Vulkan; the GEMMs are no longer 88 us tiles but ggml's tiled fp16 shaders, and the graph is one command buffer (950 nodes, ~2 ms of CPU per call instead of MNN's 2.8 s).
* **Memory is the bigger win**: the host process holds 75 MB, the GPU (UMA) holds 1.27 GB (f16) or 0.68 GB (Q8_0) of weights + 57 MB of activations. That is 2-3.5x less than ORT's 2.5 GB resident and well inside the 16 GB box's budget next to Frigate and the rest of the Immich set.
* Q8_0 is no faster than f16 here (the dequant-on-the-fly GEMM shader is not cheaper without integer dot product on this GPU) but halves the weight memory at no measurable accuracy cost (0.9986 vs 0.9985).
* Accuracy: cosine 0.9985-0.9994 against ORT fp32. As with ViT-B-32 the loss is the fp16 shader arithmetic (ggml computes GEMMs in `float16_t` on fp16-capable devices, and the 730-token quickGELU tower accumulates it); the CPU f16 path with f32 accumulation gives 0.9999. It is the same tolerance class as MNN-CPU-fp16 (0.9997) and ncnn fp16 (0.9995) that the Phase-1 plan accepted, and far from the int8 drift (0.989). Section 4.4 quantifies the fp32-math option.
* ggml's **CPU** path is not competitive for this model (5.4 s f16, 4.0 s Q8_0 vs MNN CPU fp16 1.9 s and ORT 3.7 s): ggml's ARM fp16 GEMM without i8mm/SME is tuned for LLM shapes, not 730-row activations. So "ggml" here means "ggml on the GPU"; a CPU fallback should stay ORT/MNN.
* The live `frigate-detector` logged 0 fallbacks during all runs; the GPU was shared with Frigate's ffmpeg scalers and detector throughout, so these numbers include normal NVR contention.

### 4.4 ViT-H precision/speed options on Vulkan (all + flash-attn)

| variant | latency (ms) | GPU weights | cosine vs ORT fp32 (min / mean) |
|---|---:|---:|---|
| f16 weights, fp16 shader math (default) | 1776 | 1268 MB | 0.99852 / 0.99903 |
| **Q8_0 weights, fp16 shader math** | **1757** | **677 MB** | 0.99857 / 0.99907 |
| f16 weights, **fp32 shader math** (`GGML_VK_DISABLE_F16=1`) | 3119 | 1268 MB | **0.99999 / 1.00000** |

fp32 math on the GPU is bit-for-bit as accurate as ORT (cos 0.99999) at 3.1 s -- still 1.2x faster than ORT-CPU (3.7 s) with 33x less host memory, so it is a legitimate "exact" mode; fp16 math is 1.75x faster at cosine 0.9985. The Q8_0 + FA combination is the best default: 1.76 s, 0.68 GB.

## 5. What an Immich session backend on ggml would need

Immich's `machine-learning` already has the seam: `schemas.ModelSession` (`run(output_names, input_feed) -> list[ndarray]`, `get_inputs()/get_outputs()`), `ModelFormat`, per-backend `sessions/<name>/` packages (`ort`, `ann`, `rknn`), `models/base.py::_make_session()` dispatching on the file suffix, and the `-<device>` Dockerfile stages. The Phase-1 MNN design in `IMMICH_ML_GPU.md` section 5 maps 1:1; the differences for ggml:

1. **Encoder library.** `tools/clip_vit.cpp` is ~250 lines on the public ggml API (`gguf`, `ggml_backend_sched`, `ggml_backend_alloc_ctx_tensors`). Turn it into `libclipvit.so` with a 4-function C API -- `clip_vit_init(gguf_path, backend_name, n_threads, flags)`, `clip_vit_encode(ctx, const float * nchw, int n, float * out)` (batch of preprocessed images, returns L2-normalised embeddings), `clip_vit_embed_dim(ctx)`, `clip_vit_free(ctx)` -- and call it from Python with `ctypes` (numpy arrays in/out, no copies). No pybind, no torch. A `GgmlSession(ModelSession)` in `sessions/ggml/` is then ~60 lines: `get_inputs()` reports `image [1,3,S,S]`, `get_outputs()` `embedding [1,D]`, `run()` calls `clip_vit_encode`. Immich's existing `OpenClipVisualEncoder.transform` produces exactly the NCHW f32 tensor the runner consumes (verified: cosine 0.999999 in fp32-math mode).
2. **Model files.** Nobody publishes CLIP GGUFs in this layout, so convert on first load from the ONNX Immich already downloads: `convert_clip_onnx_to_gguf.py` needs only `onnx` + `gguf` (pure Python, memmaps the external data; ~10 s and <2 GB for ViT-H; f16 default, `--outtype q8_0` to halve GPU memory). Mirrors `FaceRecognizer._add_batch_axis` rewriting a model once on load. Keep the ONNX for the fallback path.
3. **Text tower.** Keep it on ORT CPU: 157 ms per query, 1.4 GB fp32 (or an fp16 ONNX/MNN variant), exact. A ggml port is possible (a 24-layer causal transformer with EOT-token pooling + `text_projection` is llama.cpp's home turf and would be ~150 lines) but it buys nothing: text is 2 % of the visual cost, and MNN-Vulkan's wrong text embeddings (cos 0.67) are a warning that every GPU text path must be validated separately. Hybrid = visual on ggml-Vulkan, text/faces/OCR on ORT-CPU.
4. **Precision policy.** Default fp16 shader math (cos 0.9985-0.9994 vs fp32; 1.8 s). Expose `MACHINE_LEARNING_GGML_FP32_MATH=true` -> `GGML_VK_DISABLE_F16=1` for exact results at a latency cost (section 4.4). Q8_0 weights are an option for memory, not speed.
5. **Concurrency and batching.** One `ggml_backend_sched` per session, guarded by a lock (Immich's request pool is 8 threads; the GPU graph is serial anyway). `ggml_conv_2d` and every op in the graph take a batch dimension, so `n>1` images per call is a small change to the runner (the ONNX is fixed at batch 1 today too).
6. **Packaging / GPU access.** The `-cpu` image is Debian bookworm; Honeykrisp lives in Fedora Asahi's Mesa. Two viable shapes: (a) run immich-ml on the host or in a Fedora-based image with `mesa-vulkan-drivers` + `--device /dev/dri` (the pattern `frigate-detector.service` uses); (b) keep the stock container and put the encoder behind a tiny host-side service (ZMQ/HTTP), exactly like the Frigate detector, with the `GgmlSession` as its client. Building ggml-vulkan needs `glslc` + SPIRV-Headers in the builder stage (~25 min on this box, or cross-build); the runtime needs only `libggml*.so` (`libggml-vulkan.so` links the Vulkan loader; the ICD comes from the host Mesa).
7. **Memory budget on this box**: ViT-H visual f16 1.27 GB (Q8_0 0.68 GB) in GPU/UMA + 75 MB host, text tower 1.4 GB ORT, antelopev2 0.3 GB, OCR small -> ~3.1 GB total vs 4.5 GB today, and the visual weights can be evicted/reloaded in ~1 s (`MODEL_TTL`).
8. **Upstream paths.** (i) Immich: same shape as the MNN discussion #30449 -- a `GgmlSession` + `ModelFormat.GGUF` PR; the maintainer's stated preference for ORT-side optimisation applies, so plan for a rebased fork. (ii) llama.cpp: `tools/mtmd` could gain a "CLIP embedding" mode (CLS pooling + `ln_post` + `visual.proj` behind a new projector type, plus an mmproj converter that keeps those tensors); then Immich could link `libmtmd` instead of a custom runner, and quantisation/backends come for free. That is a modest PR (the graph pieces exist; `PROJECTOR_TYPE_*` enum + `clip-model.h` tensors + converter), and it would make monatis/clip.cpp (ggml from 2023, CPU only) obsolete.

Effort estimate: C API + ctypes session + convert-on-load + host-side packaging: ~1.5 days; soak test and detector-contention measurement: 0.5 day; tests/lint/PR: 1 day.

## 6. Risks and open questions

* **GPU contention with the NVR**: a ViT-H call saturates the GPU for 1.8 s. During these benchmarks `frigate-detector` reported 0 fallbacks, but its p95 under a sustained burst of embeddings (an Immich re-index) has not been measured -- do that before deployment (`journalctl -u frigate-detector` stats while looping `clip_vit` for a few minutes). At the normal ~130 requests/day it is a non-issue.
* Accuracy is fp16-class (0.9985+). Retrieval quality should be spot-checked on a few known queries, as the earlier report asked for int8; fp32-math mode is the fallback.
* Driver dependence: Mesa updates change Honeykrisp; ggml's Vulkan backend is exercised by many users on Mesa (RADV/ANV/Turnip) so regressions are likely to be caught upstream, but re-run `tools/bench_chain2.sh` after each Mesa update. If Honeykrisp ever exposes `VK_KHR_cooperative_matrix`, ggml switches to its coopmat shaders automatically (expected 2-3x on the GEMMs).
* `ggml_backend_sched` silently falls back to CPU for unsupported ops; the runner prints the unsupported-op list at start (empty on Vulkan for all three models with this ggml) -- keep that check in the session's health log.

## 7. Files

`gpu-opt/ggml/` (a local work tree; `tools/` is `ggml/` in this repository): `llama.cpp/` (@ `9e0e220`, `build-vk/bin/{llama-bench,llama-cli,llama-mtmd-cli,llama-quantize,test-backend-ops,libggml*.so}`), `prefix/` (SPIRV-Headers), `SPIRV-Headers/`, `venv/`, `tools/convert_clip_onnx_to_gguf.py`, `tools/clip_vit.cpp` + binary `tools/clip_vit`, `tools/ref_ort.py` (Immich preprocessing + ORT reference), `tools/run_clip.sh`, `tools/cmp.py`, `tools/bench_chain{1,2,3}.sh`, `models/` (`vitb32_{f16,f32,q8_0}.gguf`, `vitl14_{f16,q8_0}.gguf`, `vith14_{f16,q8_0}.gguf`, `qwen2.5-0.5b-instruct-{q8_0,fp16}.gguf`), `images/`, `ref/<model>/<img>.{in.bin,ort.npy}`, `out/<run>/{log.txt,<img>.bin}`, raw logs `results_chain1.txt` (llama-bench, ViT-B-32, ORT refs), `results_chain2.txt` (ViT-L, ViT-H), `results_chain3.txt` (ViT-H variants, GEMM micro-benchmarks).

Re-run: `tools/run_clip.sh <tag> models/vith14_q8_0.gguf ref/vith14 Vulkan0 1 3 4 1 5G` (needs `ref/vith14` from `tools/ref_ort.py`, ~30 s of GPU).
