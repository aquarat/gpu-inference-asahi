# GPU inference on Apple Silicon under Fedora Asahi (Honeykrisp Vulkan)

The record of getting neural-network inference onto the M1 GPU through Mesa's
Honeykrisp Vulkan driver, on a headless M1 Mac mini (T8103, 8-core G13G GPU,
16 GB unified memory, Fedora Asahi Remix 44, Mesa 26.1.8) that also runs a
Frigate NVR. Two workloads drove it: a small YOLOv9-t 320x320 object detector
(the NVR) and CLIP vision transformers up to ViT-H-14-378 (Immich's
smart-search model).

What is here: the patches to MNN and ncnn that came out of it, a ggml/llama.cpp
CLIP image encoder (converter + runner), the benchmark sources, the three
write-ups, and the raw profiler/benchmark logs they quote. The deployment
itself and the first round of runtime evaluation live in
[aquarat/frigate-asahi](https://github.com/aquarat/frigate-asahi); the
patched Honeykrisp driver and its harness in
[aquarat/got-bringup](https://github.com/aquarat/got-bringup).

All numbers below were measured on that one machine, next to the live NVR
(its ffmpeg Vulkan scalers and Vulkan detector share the GPU; light but
present contention, roughly 1 inference/s), as medians of 100 iterations
after warm-up unless a report says otherwise. Each report states its own
conditions; read them before reusing a figure.

## The thread, in order

### 1. Detector on Vulkan (2026-09-05)

Write-up: `research/GPU_INFERENCE.md` and `research/GPU_RESCALE.md` in
[frigate-asahi](https://github.com/aquarat/frigate-asahi) (not duplicated here).

YOLOv9-t 320, batch 1, idle GPU: ONNX Runtime CPU 12 ms per inference when
idle but 30-130 ms under NVR load; ncnn-Vulkan fp16 31 ms average, p95 38 ms,
flat under load; MNN-Vulkan the fastest GPU option at ~19 ms; IREE
(vulkan-spirv) slowest and its default target produced NaNs. OpenCL through
rusticl was rejected: numerically wrong results and kernel compile failures
for these workloads. The GPU is not faster than the M1's NEON cores for a
network this small; its value is CPU offload (detector process ~40% of a core
to ~5%). The other GPU user is zero-copy Vulkan rescaling of AVD-decoded
frames (`GPU_RESCALE.md`). MNN needed a one-line CMake fix to build with GCC 16
([mnn/0001](mnn/)).

### 2. Does a lighter GPU barrier help inference? (2026-09-06)

Write-up: [reports/VULKAN_DISPATCH_EXPERIMENT.md](reports/VULKAN_DISPATCH_EXPERIMENT.md);
code: [vk-dispatch/](vk-dispatch/).

The got-bringup Honeykrisp fork (weak CDM barrier between compute dispatches,
5.5x on Ghost of Tsushima's heaviest scene) was loaded per process through a
private ICD, never installed. A chain test of 1000 back-to-back dispatches
puts numbers on it:

| chain shape | stock Mesa 26.1.8 | patched | note |
|---|---:|---:|---|
| dependent (barrier between every dispatch), empty shader | 3.47 us/dispatch | 3.17 us | unchanged: the driver must drain either way |
| dependent, 1000-iteration shader loop | 33.6 us | 31.9 us | shader latency paid serially on both |
| independent (no barrier), empty shader | 3.30 us | **0.32 us** | 10x |
| independent, shader loop | 31.6 us | **0.71 us** | 45x, all 1000 overlap |

Inference graphs are dependent chains: ncnn puts a barrier after every layer,
MNN (as shipped) ends the command stream after every op. Result: YOLOv9-t
MNN-Vulkan fp16 19.0 vs 19.0 ms, ncnn-Vulkan 36.3 vs 36.6 ms in a detector
A/B at 25 requests/s, outputs bit-identical. CLIP on MNN-Vulkan did get
1.33-1.45x faster (ViT-B-32 ~190 to ~135 ms, ViT-H 7.70 to 5.79 s), but the
`HK_PERFTEST=nooverlap,noconstdata` ablation runs exactly as fast and plain
upstream 26.2.2 is as slow as 26.1.8, so the gain is in the fork's other
compiler-side commits, not the barrier. The driver was not deployed: nothing
the system driver serves benefits.

### 3. Runtime fixes: submission structure and fusion (2026-09-06)

Write-up: [reports/PHASE1_RUNTIME.md](reports/PHASE1_RUNTIME.md);
patches: [mnn/](mnn/), [ncnn/](ncnn/); benchmarks: [bench/](bench/).

Profiling with the fork's firmware profiler (`HK_GPUTIME=1`), `strace` and
`perf` showed where the 19-28 ms actually went:

* **MNN** recorded one `VkCommandBuffer` per op (~512 per inference).
  Honeykrisp turns each into its own GPU control stream and DRM submit ioctl
  (~610 ioctls, 6.4 ms of CPU per inference), and the firmware leaves ~23 us
  between streams: ~11 ms of the 18.7 ms was GPU idle. MNN already has a
  whole-graph-in-one-command-buffer mode (`MNN_GPU_RECORD_BATCH`) that nobody
  reaches, because `ScheduleConfig::mode` is a union with `numThread`, so
  every caller that sets a thread count silently selects per-op mode.
  [mnn/0002](mnn/) makes batch recording the Vulkan default:
  **fp16 18.7 to 11.6 ms, fp32 21.2 to 15.7 ms, CPU per inference 6.4 to
  0.7 ms, 474 to ~3 control streams, GPU busy 50% to 78%, output md5
  identical.**
* **ncnn** records one command buffer but its graph kept SiLU as a separate
  layer after every convolution (and a quarter of them as Split + Sigmoid +
  Mul), because neither `onnx2ncnn` nor `ncnnoptimize` knows the pattern.
  [ncnn/0001](ncnn/) adds Swish as Convolution `activation_type 7` on every
  backend and teaches `ncnnoptimize` to fuse it: **655 to 388 layers, 978 to
  ~740 dispatches per inference**, fp16 ~29.0 to 25.2 ms and fp32 33.8 to
  30.2 ms in the same (NVR-loaded) pass; fp32 max abs diff vs onnxruntime
  0.0016 (0.0021 before). [ncnn/0002](ncnn/) coalesces the 2-3
  `vkCmdPipelineBarrier` calls per dispatch into one (~1680 to ~580 barrier
  calls per inference), bit-identical output, latency within the noise of the
  shared GPU: Honeykrisp already makes back-to-back barriers nearly free.
* What is left is kernel time: ~700 dispatches x 14 us (MNN) or ~740 x
  27 us (ncnn) against a 3.5 us dependent-dispatch floor. Staging is already
  zero-copy on this unified-memory GPU; fp16/subgroup options change nothing.
  The best GPU path (MNN image fp16, batched, 11.6 ms at 0.75 ms CPU) is still
  2.4x the latency of MNN/ncnn NEON fp16 (4.8-4.9 ms at ~20 thread-ms) on a
  quiet machine, which is the whole reason to use the GPU on this box.

### 4. ggml for transformers (2026-09-06)

Write-up: [reports/PHASE2_GGML.md](reports/PHASE2_GGML.md); code: [ggml/](ggml/).

## Phase 3: fuse SiLU into MNN's Vulkan convolution epilogue

YOLO-style graphs follow almost every convolution with SiLU, and MNN ran each
one as its own dispatch (179 of 783 per inference for YOLOv9-t 320). The
fusion is made at schedule time, inside MNN, when the session is created, so
it applies to models converted by any converter -- including a detector that
converts ONNX to `.mnn` at load time with the pip wheel. Dispatches per
inference 783 -> 604, fp16 median 12.1 -> 11.15 ms, fp32 14.7 -> 14.0 ms;
fp32 output bit-identical, fp16 slightly closer to the fp32 reference, no
detection changes on the 45-frame set.

Write-up: [reports/PHASE3_MNN_FUSION.md](reports/PHASE3_MNN_FUSION.md);
patch: [mnn/0003](mnn/).

The "Vulkan is hopeless for transformers on this driver" conclusion from the
MNN runs (ViT-H 7 s, 2% of peak) was a runtime finding, not a driver one.
ggml's Vulkan backend records the graph into one command buffer with minimal
barriers and uses tiled fp16 GEMM shaders (no cooperative matrix on
Honeykrisp, so scalar fp16 paths):

| workload | ggml-Vulkan | others on the same machine |
|---|---:|---|
| Qwen2.5-0.5B Q8_0, llama-bench pp512 | **783 t/s = ~0.77 TFLOPS, ~30% of the 2.5 TFLOPS clpeak fp16 peak** | ggml CPU Q8_0 4 threads 743 t/s (NEON `sdot`); ggml CPU F16 303 t/s |
| CLIP ViT-B-32 (224 px) | 39 ms, cos 0.9991 | ORT CPU fp32 33 ms; MNN-Vulkan 133-216 ms; ncnn-Vulkan 66 ms |
| CLIP ViT-L-14 (224 px) | 348 ms | ORT CPU 563 ms |
| CLIP ViT-H-14-378 (Immich's model, 1007 GFLOP) | **1.76-1.87 s, 566 GFLOPS = 23% of peak; 75 MB host RSS + 0.68 (Q8_0) or 1.27 GB (f16) of UMA weights** | ORT CPU fp32 3.7 s (3.2 s idle), 2.5 GB RSS; MNN CPU fp16 1.9 s; MNN-Vulkan 5.8-7.7 s at 2% of peak |

Accuracy: cosine 0.9985-0.9994 vs ORT fp32 with fp16 shader math (the loss
is the arithmetic, not the f16 weights); `GGML_VK_DISABLE_F16=1` gives cosine
0.99999 at 3.1 s, still faster than ORT-CPU. The GPU only wins for
fp16/large-M work: for int8 GEMMs the M1's four P-cores are nearly as fast,
and for memory-bound batch-1 decode they are faster. No existing ggml CLIP
tool produced a CLIP embedding (llama.cpp's `mtmd` outputs projector patch
tokens and its converters drop `visual.proj`/`ln_post`; monatis/clip.cpp is
CPU-only on a 2023 ggml), so [ggml/](ggml/) converts Immich's own ONNX to
GGUF and runs the tower in ~250 lines on the public ggml backend API.

### 5. Cooperative matrix on Honeykrisp (2026-09-06)

Write-up: [reports/PHASE4_COOPMAT.md](reports/PHASE4_COOPMAT.md);
patches: [mesa/](mesa/) (branch `coopmat` on aquarat/mesa).

The G13 has an 8x8x8 SIMD-group matrix FMA (`simd_matrix_fmadd16/32`, the
instruction behind Metal's `simdgroup_matrix`; encoding from dougallj's
`applegpu`, lane layout from metal-flash-attention). Three patches on the
Honeykrisp fork teach the `agx` compiler the opcode and give `hk` a
panvk-style lowering of `VK_KHR_cooperative_matrix` onto it (16x16x16 and
8x8x8, f16/f32, mixed f16 x f16 + f32 natively). Bit-exact on a standalone
test, ggml `test-backend-ops` MUL_MAT 1106/1106; one register-allocator bug
found by ggml's flash-attention shader (the destination must not overlap A or
B). Loaded per process through `VK_DRIVER_FILES`, never installed:

| workload | stock 26.1.8 | coopmat build |
|---|---:|---:|
| llama-bench Qwen2.5-0.5B Q8_0 pp512 | 783 t/s | **1055 t/s** (~1.04 TFLOPS, 40 % of FMA peak) |
| CLIP ViT-H-14-378, one image, fa=0 / fa=1 | 1874 / 1776 ms | **1505** / 1578 ms |

The M1 has no separate matrix unit (the op runs on the FMA ALUs), so this is
utilisation, not extra peak; Apple's Metal GEMMs on the same instruction reach
~80 %, so ~1.5-2x more is on the table from vectorised tile loads and
ggml-side tile tuning. Not CTS-tested. Deployed for Immich's ML service only,
through two `Environment=VK_*` lines in its unit: ViT-H image embedding
1.80 -> 1.53 s in the service (cosine 0.9999 vs the stock driver), with the
shared detector's p95 unchanged at ~36 ms during an embedding burst.

## What was learned

* On Honeykrisp the per-dispatch floor for a dependent chain is ~3.2-3.5 us
  and a submit/fence round trip is ~80-160 us. Neither is the reason a
  runtime is slow; submission structure (command buffers per graph) and
  kernel efficiency (tile sizes, fp16 accumulation, fusion) are.
* One command buffer per op is catastrophic on this driver: every
  `VkCommandBuffer` becomes a GPU control stream with a ~23 us gap. Check
  what your runtime's "default" mode really records before profiling
  kernels.
* Consecutive `vkCmdPipelineBarrier` calls with no work between them cost
  nothing extra here; "2000 barriers" in a profile are not 2000 drains.
* For a 320x320 CNN the M1's NEON fp16 path beats the GPU on latency (4.8 vs
  11.6 ms); the GPU buys CPU time (0.75 vs ~20 thread-ms per inference).
  For ViT-H-class transformers the GPU is 2x ORT-CPU and holds the weights
  in UMA instead of 2.5 GB of host RSS.
* The got-bringup barrier change transfers to inference only where an
  application issues no barrier between dispatches; ML graphs never do that.
  Cooperative-matrix support and a cheaper submit path would transfer.

## Layout

| path | what |
|---|---|
| [mnn/](mnn/) | Two `git am` patches on alibaba/MNN `bef71b9`: the GCC 16 build fix and the Vulkan batch-recording default. |
| [ncnn/](ncnn/) | Two patches on Tencent/ncnn tag `20260526`: SiLU fusion (`activation_type 7`) + optimizer pass, and barrier coalescing. |
| [mesa/](mesa/) | Three patches on aquarat/mesa `local-deploy` @ `d105715` (= branch `coopmat`): `VK_KHR_cooperative_matrix` for Honeykrisp on the G13 SIMD-group matrix FMA, the native mixed-precision form, and the register-allocation fix. |
| [ggml/](ggml/) | `convert_clip_onnx_to_gguf.py` (Immich ONNX to GGUF, f16/f32/Q8_0), `clip_vit.cpp` (ggml runner, Vulkan or CPU), the ORT reference/preprocessing script, comparison and benchmark scripts, build notes for llama.cpp `9e0e220`. |
| [bench/](bench/) | The MNN/ncnn YOLO benchmark sources and the interleaved final A/B script behind PHASE1. |
| [vk-dispatch/](vk-dispatch/) | The chain test, the per-process ICD loading of a private driver build, and the run scripts of the dispatch experiment. |
| [reports/](reports/) | `VULKAN_DISPATCH_EXPERIMENT.md`, `PHASE1_RUNTIME.md`, `PHASE2_GGML.md`, `PHASE3_MNN_FUSION.md`, `PHASE4_COOPMAT.md`, sanitised. |
| [results/](results/) | Raw logs the reports quote: firmware-profiler timelines, strace/perf summaries, chain-test output, llama-bench and CLIP runs. |

Not included: models (`.onnx`, `.mnn`, `.param/.bin`, `.gguf`), test images,
binaries, the extracted driver RPMs (see got-bringup's dnf repository), and
the ORT reference embeddings. Every script that needs them takes their
location as an argument or an environment variable.

## Related repositories

| repository | relation |
|---|---|
| [aquarat/frigate-asahi](https://github.com/aquarat/frigate-asahi) | The NVR deployment: host detector service with its Vulkan backends (ncnn, MNN shim), `research/GPU_INFERENCE.md`, `research/GPU_RESCALE.md`, the earlier `bench_*.py` scripts, the AVD decode work. |
| [aquarat/got-bringup](https://github.com/aquarat/got-bringup) | The patched Honeykrisp driver, its measurement harness and the `cstest`/`coherence` tests reused here. |
| [aquarat/mesa](https://github.com/aquarat/mesa) `local-deploy`, `coopmat` | The driver source the RPM was built from; `coopmat` = `local-deploy` + the three cooperative-matrix patches in [mesa/](mesa/). |
| [aquarat/MNN](https://github.com/aquarat/MNN) | Intended mirror for the MNN branches; empty until a `workflow`-scoped token can push MNN's history (its `.github/workflows` are rejected otherwise). The patches here are the same commits. |
| [aquarat/fedora-asahi-remix-notes](https://github.com/aquarat/fedora-asahi-remix-notes) | The index of all the Asahi work, with the project page for this thread. |
