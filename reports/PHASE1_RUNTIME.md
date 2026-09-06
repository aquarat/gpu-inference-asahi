# Phase 1: GPU inference runtime latency on the M1 Mac mini (Honeykrisp Vulkan)

Date: 2026-09-06. Machine: Mac mini M1 (T8103, G13G B1 8-core GPU), Fedora Asahi 44, kernel 7.1.6-400.asahi, system Mesa 26.1.8 (Honeykrisp), 16 GB unified memory, headless. Patched-driver profiler = got-bringup Honeykrisp fork loaded per process via `vk-dispatch/with-patched.sh` with `HK_GPUTIME=1` (never installed system-wide).
Model: `fox-person-cat-320.onnx` (YOLOv9-t, 320x320, 1x8x2100 output, DFL decode in-graph), batch 1, fixed input `input_320.bin` (seed-0 random), reference `ref_ort_output.bin` (onnxruntime fp32 CPU).
Background load during all measurements: the live Frigate NVR (ffmpeg Vulkan scalers + `frigate-detector.service` ncnn-Vulkan fp16 on the system driver) -- untouched; ~0.5-1 inference/s of GPU contention. Numbers are medians of 100 iterations after 10 warm-ups unless stated; p95 is noisy because of the NVR and the parallel builds noted below.
Working dir: `gpu-opt/` (a local work tree; the sources, patches and logs it refers to are in this repository) -- `MNN/` (git branch `gpu-opt` on top of MNN `bef71b9`), `ncnn/` (git branch `gpu-opt` on tag `20260526` = the pip wheel `ncnn==1.0.20260526`, outputs bit-identical to the wheel), `bench/` (benchmarks, models, outputs), `results/` (raw profiler/perf/strace logs), `patches/` (`git format-patch` output), `scripts/`.
Memory discipline: every build and benchmark ran inside `systemd-run --user --scope -p MemoryMax=5G`, `ninja -j4`, `free -h` checked before each heavy step (never below 4.5 GB available).

## TL;DR

* **MNN-Vulkan: 18.7 ms -> 11.6 ms fp16 (21.2 -> 15.7 ms fp32), bit-identical output, CPU per inference 6.4 -> 0.7 ms.** The ~474 command streams were one VkCommandBuffer per op ("direct" mode); MNN already has a whole-graph-in-one-command-buffer mode (`MNN_GPU_RECORD_BATCH`) that nobody reaches because `ScheduleConfig::mode` is a union with `numThread`. Patch `patches/mnn/0001` makes it the Vulkan default (opt-out `MNN_GPU_RECORD_OP`). No new barrier code was needed: the per-op image barriers already describe the dependencies; 2-3 control streams and 78% GPU busy per inference.
* **ncnn-Vulkan: 978 -> ~740 dispatches, 655 -> 388 layers, ~2000 -> ~580 barrier calls; fp16 ~22-25 -> ~20-25 ms (best 19.4 vs 20.5-22.4), fp32 28.5 -> 25.8 ms best-case; fp32 exact vs onnxruntime (0.0016).** Patch `patches/ncnn/0001` adds SiLU as Convolution `activation_type 7` (all backends) and teaches `ncnnoptimize` to fuse Conv+Swish and the Split+Sigmoid+Mul remnants; `0002` issues one `vkCmdPipelineBarrier` per dispatch. Staging was already zero-copy on this unified-memory GPU (ncnn detects the mappable device-local type, `rebar=1`); option sweeps (fp16 modes, subgroup ops) change nothing.
* **GPU vs CPU**: the best GPU path (MNN image fp16, batch) is 11.6 ms vs 4.8-4.9 ms for MNN/ncnn NEON fp16 on a quiet machine. The remaining GPU cost is kernel time: ~700 dispatches x ~14 us (MNN) / ~740 x 27 us (ncnn) with a 3.5 us dependent-dispatch floor -> Phase 2 = fused SiLU epilogue in MNN's conv shaders (~-2 ms), fewer packing dispatches and better conv kernels/local sizes in ncnn, not submission structure.
* **Contention caveat**: all numbers were taken next to the live NVR, which was detecting at ~4 fps during the last hour (its own average rose from 35 to 61 ms while my benchmarks ran; back to normal afterwards, 0 fallbacks, 0 restarts, no GPU faults). Under that contention a single 10 ms control stream gets time-sliced against the NVR's 20 ms stream, so the batched MNN run shows higher tail latency in the interleaved A/B (medians 11.7 / 21.6 / 50.3 ms across passes vs direct 18.3-18.5 ms; mins 11.1 vs 17.6). On a GPU it does not share, batch mode wins everywhere (11.1-11.6 ms quiet-machine medians).

## 1. Profile before any change (Task 1)

### 1.1 Per-inference structure (patched-driver `HK_GPUTIME` firmware profiler + `strace -c` + `perf`)

Raw: `results/gputime_mnn_fp16_mode0x4_stockbuild.txt`, `results/gputime_mnn_fp16_mode0x204_stockbuild.txt`, `results/ncnn_pip_option_sweep_gputime.txt`, `results/strace_mnn_stockbuild.txt`, `results/perf_mnn_summary.txt`, `results/perf_mnn_m0x4.data`, `results/perf_mnn_m0x204.data`; prior session's `research/vk-dispatch/results/gputime_*.txt`.

| runtime (YOLOv9-t 320 fp16) | vkQueueSubmit / inference | VkCommandBuffers / inference | GPU control streams / inference | dispatches / inference | Vulkan barriers / inference | GPU time per dispatch | GPU busy | median latency | CPU time in runtime / inference |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MNN-Vulkan (image), default `gpuMode` 0x4 = "direct" (one VkCommandBuffer per op) | 2 (`_finish` after input-copy+graph, and after output-copy) | ~512 (1 per op + 2 converters) | ~474 (driver merges some) | ~700-790 | image layout barriers inside each cmdbuf only (profiler counts 0 pipeline barriers that end work) | 14.4 us | 49.5% | 18.7-19.0 ms | 6.4-7.2 ms (`perf`: `queue_submit` 18%, `memcpy` 30% (driver cmdbuf copies), `VulkanBasicExecutionDirect::onExecute` 15%, `vk_drm_syncobj_init` 3.6%); ~610 DRM ioctls / inference |
| ncnn-Vulkan (pip wheel, detector options: fp16 packed+storage+arithmetic, blob/staging allocators) | 1 | 1 | 1 (978 dispatches / stream) | ~978 | ~1900-2000 buffer barriers (`VkCompute::record_pipeline` inserts one `vkCmdPipelineBarrier` per bottom blob whose last access differs; all "compute-only, ending real work" = each one drains the GPU) | 19.9 us | 66-78% (rest = CPU side between inferences) | 22.3-25 ms (`-opt` model); 28 ms (raw model) | 1.9-3.0 ms |

Notes:
* MNN direct mode: `VulkanBackend::_finish()` does issue only one `vkQueueSubmit` with `commandBufferCount = N` (N ~ 500), but Honeykrisp turns every VkCommandBuffer into its own control stream + DRM submit ioctl (strace: ~610 ioctls per inference, `queue_submit` dominates the CPU profile), and the firmware inserts a ~23 us gap between control streams -> ~11 ms of GPU idle per inference. That is the ~474-command-stream finding of the previous experiment.
* Dependent dispatch floor on this driver is 3.2-3.5 us; both runtimes' kernels are 14-20 us each, so ~25% of GPU time is drain/launch overhead even with perfect submission.

### 1.2 Layer types in the converted graphs and dispatches each needs

MNN graph (`mnnconvert` default, 510 ops; `bench/fox-person-cat-320.mnn`):

| MNN op | count | Vulkan dispatches each (image backend) |
|---|---:|---|
| Convolution | 186 | 1 (3x3 s1 with >= 4 output ch may use winograd: 3 dispatches; 1x1: 1) |
| SILU | 179 | 1 (`VulkanUnary`) -- **not fused into Convolution** (MNN's Vulkan conv fuses only relu/relu6 via `Convolution2DCommon.relu/relu6`) |
| ADD (residual) | 44 | 1 (`VulkanBinary`) |
| Concat | 31 | 1 per input slice via `VulkanRaster` (region copies) |
| StridedSlice | 18 | 1 (raster) |
| Const | 18 | 0 |
| Reshape / Permute / Flatten | 9 / 2 / 1 | 0-1 (raster if layout changes) |
| Pooling | 8 | 1 |
| ConvertTensor | 4 | 1 |
| Interp | 2 | 1 |
| MUL / SUB / SIGMOID / Softmax / Shape | 2 / 2 / 1 / 1 / 1 | 1 each (Shape 0) |
| Input | 1 | 1 converter dispatch (NCHW buffer -> NC4HW4 image) + 1 for the output |

Measured: ~700 dispatches per inference in batch mode (profiler: 259.8 dispatches per control stream x 2.7 streams).

ncnn graph (`onnx2ncnn` + `ncnnoptimize ... 65536`, 655 layers / 794 blobs, `bench/fox-person-cat-320-opt.param`; the optimizer changed nothing: raw = 655 layers too):

| ncnn layer | count | Vulkan dispatches each |
|---|---:|---|
| Convolution | 186 | 1 (+ packing/cast dispatches when the neighbour's elempack differs) |
| Swish (x*sigmoid(x) recognised by onnx2ncnn) | 135 | 1 -- **not fused**: `ncnnoptimize` fuses only ReLU/Clip/Sigmoid/Mish/HardSwish into Convolution (`activation_type` 1-6) |
| Split | 123 | 0 (reference copy) |
| BinaryOp | 92 | 1 (45 are the `Mul` half of an unrecognised SiLU: `Conv -> Split -> {Sigmoid, Mul}`; 44 residual adds; 3 decode) |
| Sigmoid | 45 | 1 (the other half of those 45 SiLUs) |
| Concat | 31 | 1 (+ unpacking when channel counts are not pack-aligned) |
| Crop | 18 | 1 |
| Reshape / Permute / Interp / Pooling / Softmax / MemoryData | 8 / 2 / 2 / 8 / 1 / 3 | 1 each (MemoryData 0) |

978 dispatches measured vs ~530 "real" layers: the remaining ~450 are ncnn's implicit `Packing` / cast layers (pack1<->pack4<->pack8, fp32<->fp16) inserted around Concat/Crop/Split boundaries.

### 1.3 Baselines (this session, same binaries as the previous documents)

| config | median ms | p95 | CPU ms/iter | correctness vs ORT fp32 (max abs diff / mean) |
|---|---:|---:|---:|---|
| MNN-Vulkan fp16 image, direct (stock build) | 18.73 | 20.9 | 6.4 | 3.61 / 0.056 (md5 `6d1d0a17`) |
| MNN-Vulkan fp32 image, direct | 21.19 | 23.5 | 7.2 | 0.0017 / 0.00002 (md5 `301ed9a3`) |
| ncnn-Vulkan fp16 (pip wheel, opt model, detector options) | 22.33 | 27.6 | 2.3 | 22.6 / 0.19 (fp16 arithmetic; box coords in px) |
| ncnn-Vulkan fp32 (pip wheel) | 28.34 | 29.1 | 1.5 | 2.08 / 0.021 |
| ncnn CPU fp16 4 thr | 11.2 (NVR load) | 14.4 | 45 | 11.4 / 0.22 |
| MNN CPU fp16 4 thr | 16.9 (measured while a build ran; 4.7 idle per GPU_INFERENCE.md) | | 68 | 8.9 / 0.12 |

## 2. MNN Vulkan backend (Task 2)

### 2.1 Root cause of the ~474 command streams

`source/backend/vulkan/image/backend/VulkanBackend.cpp`: `mDirect = (gpuMode & MNN_GPU_RECORD_BATCH) == 0`. In direct mode every op gets its own `VulkanCommandPool::Buffer` (`VulkanBasicExecutionDirect`, recorded once at resize), `onExecute` only pushes the handle, and `_finish()` (called from `onExecuteEnd` and from every `onCopyBuffer`) issues one `vkQueueSubmit` with `commandBufferCount = ~512`, then `vkWaitForFences`. Honeykrisp submits each VkCommandBuffer as its own control stream (one DRM submit ioctl each: strace shows ~610 ioctls per inference; `perf`: `queue_submit` + driver `memcpy` + `vk_drm_syncobj_init` = most of the 6.4 ms CPU per inference), and the firmware leaves ~23 us between control streams -> ~11 ms of the 18.7 ms is GPU idle.

MNN already has the batching mode: `MNN_GPU_RECORD_BATCH` (0x200; documented "All ops share one commandBuffer (Vulkan)"). In that mode `VulkanBasicExecutionInDirect::onResize` encodes every op into the backend's single `mCmdBuffer` (opened in `onResizeBegin`, closed in `onResizeEnd`) with the same per-op `VulkanImage::barrierRead/barrierWrite` image-memory barriers (layout tracked per image at record time, so the dependencies between consecutive ops are already correct), and one inference is: host->device converter cmdbuf + graph cmdbuf in one submit (+ fence wait), then the device->host converter cmdbuf in a second submit. Nobody uses it because `ScheduleConfig::mode` is a **union with `numThread`**: every caller that sets `numThread = 4` (the benchmark, the detector shim, MNN's own tools) silently selects gpuMode 0x4 = TUNING_WIDE + direct mode, and any mode outside the six whitelisted values (e.g. the documented default 0x84 with the OpenCL memory bit) is rejected back to 0x4.

### 2.2 Result of batch mode (no source change needed; `bench/bench_mnn2 ... <mode>`)

| MNN-Vulkan image backend, stock research build | median | p95 | min | CPU ms/iter | control streams / inference | GPU busy | dispatches / inference | output |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fp16, mode 0x4 (direct) | 18.73 (18.8-23 under load) | 20.9 | 18.0 | 6.4 | ~474 | 49.5% | ~700-790 | md5 `6d1d0a17` |
| **fp16, mode 0x204 (RECORD_BATCH)** | **11.10-12.26** (11.6 on the gpu-opt build, quiet machine) | 11.9 (quiet) / 22 (loaded) | 11.2 | **0.6-0.75** | **~2.7** | **78.1%** | ~700 | **md5 `6d1d0a17` identical** |
| fp32, mode 0x4 | 21.19 | 23.5 | 20.1 | 7.2 | ~474 | | | md5 `301ed9a3` |
| **fp32, mode 0x204** | **15.73** | 24.7 (loaded) | 14.5 | 0.68 | ~2.7 | | | **md5 `301ed9a3` identical** |

Phase breakdown (fp16, batch, `bench/bench_mnn_phases`): copyFromHost 0.08 ms, runSession 10.84 ms, copyToHost 0.18 ms; GPU busy per inference ~10.1 ms -> the remaining ~1 ms is two submit/fence round trips plus the converter dispatches. GPU time is now the floor: ~700 dispatches x 14 us.

Patch (`patches/mnn/0001-Vulkan-record-the-whole-graph-into-one-command-buffer-by-default.patch`, commit `330e565` on `MNN/gpu-opt`): `VulkanRuntime` now validates only the tuning bits, ignores the OpenCL-only memory bits, and defaults to `MNN_GPU_RECORD_BATCH` unless the caller asks for `MNN_GPU_RECORD_OP` (0x100) explicitly. Verified: mode 0x4 on the patched build = batch behaviour (CPU 0.77 ms/iter, md5 identical to stock), mode 0x104 = old direct behaviour (8.2 ms CPU, 19.3 ms), mode 0x204 unchanged. Both the image and the buffer backend read the same `mGpuMode`, so both are covered; the CPU fallback path is untouched (the flag lives in the Vulkan runtime only).

### 2.3 Buffer memory mode (`-DMNN_VULKAN_IMAGE=OFF`, `MNN/build-buffer`)

| MNN-Vulkan buffer backend | direct 0x4 | batch 0x204 | correctness vs ORT (max abs / mean) |
|---|---:|---:|---|
| fp32 | 21.56 ms | 17.53 ms | 0.0044 / 0.00003 |
| fp16 | 20.36 ms | 15.02 ms | **20.05 / 0.159** (much worse than image mode's 3.6 / 0.056) |

Batch mode helps the buffer backend too (also md5-identical between direct and batch), but it is ~30% slower than the image backend on this GPU and its fp16 path loses accuracy, so the image backend stays the choice.

### 2.4 Not done / measured and rejected for MNN

* SILU is a separate `VulkanUnary` dispatch after each of 179 convolutions (MNN's `Convolution2DCommon` only carries `relu`/`relu6` post-ops); fusing it needs a schema field + converter pass + CPU/GPU epilogues in every backend, i.e. a multi-backend change outside this time box. Estimated saving on the GPU-time floor: ~179 x (3.5 us drain + ~6 us kernel) ~ 1.5-2 ms of the 11.6 ms.
* The two submits per inference could become one by deferring the graph submit to the output copy; gain <= one submit/fence round trip (~0.2 ms, <2%), not attempted.
* Auto-tuning (`MNN_GPU_TUNING_WIDE`, default) only affects private pipelines; no cache file was used in these runs and it made no measurable difference between passes.

## 3. ncnn (Task 3) -- source build of tag `20260526` (= pip wheel `ncnn==1.0.20260526`; outputs bit-identical to the wheel, md5 `4e2a4db2`)

Build: `ncnn/build` (`-DNCNN_VULKAN=ON -DNCNN_SHARED_LIB=ON -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_BENCHMARK=ON`, glslang submodule, ~12 min at `-j4` inside the 5 GB scope). The live detector keeps using the pip wheel in `detector/venv` (untouched).

### 3.1 What was checked and found already optimal (no change)

* **Unified memory / staging.** `vulkaninfo`: one heap, both memory types are `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` (type 0 also `HOST_CACHED`); ncnn prints `rebar=1`. `VkBlobAllocator` therefore ends up on a mappable type (`allocator.cpp` prefers a non-host-visible device-local type and falls back to the host-visible one), and every path already uses that: `VkTransfer::record_upload` (weights) memcpys straight into the mappable destination (no staging, no copy); `VkCompute::record_upload` (input) memcpys into the staging buffer and runs **one** `convert_packing` dispatch that does the fp32->fp16 cast + pack directly into the blob (that dispatch is needed anyway on an integrated GPU); `record_download` casts/unpacks with one dispatch into a mappable blob and memcpys out. Nothing to skip: 2 dispatches and 2 memcpys per inference (input 1.2 MB, output 67 KB).
* **Options.** `use_fp16_packed/storage/arithmetic` on vs `arithmetic` off vs `use_subgroup_ops` off: 25.0 / 28.0 / 27.5 ms, no dispatch-count change (`results/ncnn_pip_option_sweep_gputime.txt`). `use_shader_pack8`, `use_image_storage` no longer exist in this ncnn version (pack8 and image storage were removed upstream in 2025). `use_bf16_storage`: ncnn reports `bf16-p/s=1/0` (no bf16 storage support on Honeykrisp) -> not applicable. int8: no int8 model; `int8-p/s/u/a=1/1/1/1` is available but would need calibration (out of scope).
* **`ncnnoptimize` had been applied** (detector: `... 0`, research: `... 65536`), but on this graph it fuses nothing: Ultralytics already folds BN into the convs, and SiLU is not in its activation list -> 655 layers in and out.

### 3.2 Change 1: fuse SiLU into Convolution (patch `patches/ncnn/0001-...`, commit `1b00e9b`)

* `ncnnoptimize`: new pass `fuse_split_sigmoid_mul_swish()` (x -> Split -> {a,b}; Sigmoid(b); Mul(a, s) => Swish(x)) recovers the 44 SiLUs that `onnx2ncnn` left as Split+Sigmoid+BinaryOp, then the Convolution / ConvolutionDepthWise / InnerProduct activation fusions accept `Swish` as `activation_type 7`.
* Runtime: `activation_type 7 = x*sigmoid(x)` in `fused_activation.h` (scalar + `create_activation_layer` -> Swish layer for the generic fallback), `arm_activation.h` (f32 NEON, fp16 scalar/f16x4/f16x8), `x86_activation.h` (sse/avx/avx512, existing `swish_*` helpers), loongarch/mips/riscv helpers, and `vulkan_activation.comp` (both `afp` and `afpvec4`). x86/loongarch/mips/riscv are compile-untested here (aarch64 host) but mirror the existing type-6 code one line each.
* Graph: 655 -> **388 layers** (`bench/fox-person-cat-320-fused.param`: 186 Convolution of which 179 carry `9=7`, Split 123 -> 79, BinaryOp 92 -> 48, Sigmoid 45 -> 1, Swish 135 -> 0).

| ncnn-Vulkan (src build, same pass, NVR busy) | dispatches / inference | barrier calls / inference | GPU time / inference | median | p95 | correctness |
|---|---:|---:|---:|---:|---:|---|
| `-opt` graph (655 layers), fp16 | ~1000 | ~2020 | 21.2 ms | 28.99 (22.3 in a quiet pass) | 41.8 | fp32: 0.0021 vs ORT |
| **fused graph (388 layers), fp16** | **~740** | ~1680 | 19.8 ms | **25.20** | 26.4 | vs ORT 18.6 / 0.18 (fp16; unfused 22.6 / 0.19); vs unfused fp16 output 7.1 / 0.115 (fp16 rounding of the fused epilogue) |
| fused graph, fp32 | | | | 30.16 (unfused 33.80 same pass) | 31.7 | **0.0016 / 0.00002 vs ORT** (bit-level exactness not expected: different op order; better than unfused 0.0021) |
| fused graph, CPU fp32 NEON 4 thr | | | | 25.3 (NVR load) | | 0.0015 / 0.00002 vs ORT -> the NEON `activation_type 7` path is right |
| fused graph, CPU fp16 NEON | | | | 15.3 | | 26.8 / 0.21 vs ORT (unfused CPU fp16: 11.4 / 0.22; fp16 arithmetic noise) |

Raw: `results/gputime_ncnn_opt_fp16.txt`, `results/gputime_ncnn_fused_fp16.txt`.

Why only ~13% faster for 26% fewer dispatches: the removed dispatches were the cheap element-wise ones; ncnn's remaining kernels average 26.7 us each (MNN's 14 us) -- ncnn's Vulkan convolution shaders / local sizes are simply less efficient than MNN's image kernels on this GPU (GPU time per inference 19.8 ms vs MNN's 10.1 ms for the same network). That is a kernel-efficiency problem (Phase 2), not a feeding problem.

### 3.3 Change 2: one `vkCmdPipelineBarrier` per dispatch (patch `patches/ncnn/0002-...`, commit `4049775`)

`VkCompute::record_pipeline` called `barrier_readwrite()` per buffer binding -> 2-3 separate barrier calls before every dispatch. The patch gathers them into one call (srcStageMask = OR of the bindings' stages; delayed-record path kept for non-push-descriptor devices; image bindings unchanged). Barrier calls per inference ~1680 -> **~580** (`results/gputime_ncnn_fused_coalesced_fp16.txt`), output bit-identical (fp16 md5 `42c8108e`, fp32 md5 `3cd2317f`). Latency: within noise on the shared GPU (interleaved passes: 24.2 vs 23.9 / 24.6 vs 29.6 / 56.8 vs 69.2 ms while the NVR was detecting at ~4 fps; best-case 19.4 vs 20.5 ms). Honeykrisp evidently makes consecutive barriers with no work between them nearly free, so the "2000 barriers" were never 2000 drains -- ~740 dependent dispatches at 3.5 us drain each is the real floor (~2.6 ms).

### 3.4 Not done for ncnn

* The ~450 implicit packing/cast dispatches (978 measured vs ~530 layers) come from pack4/pack1 changes around Concat/Crop with channel counts that are not multiples of 4 on every branch; eliminating them needs shader-level "pack-agnostic" concat/crop -- Phase 2.
* Kernel local-size tuning (`Pipeline::set_optimal_local_size_xyz`) for the AGX (subgroup 32, 1024 invocations) -- Phase 2.
* The pip wheel cannot load the fused graph (it does not know `activation_type 7`: the conv would silently run without activation). Deploying needs a wheel built from `gpu-opt/ncnn` with `-DNCNN_PYTHON=ON` (not done, the live wheel was not to be touched).

## 4. Graph-level fusion at conversion (Task 4)

* **MNN**: `mnnconvert --optimizeLevel 0/1/2` and `--fp16` were run on the ONNX (`results/mnnconvert_optlevels.txt`): see the op counts there; the converter already folds BN and produces the 510-op graph with SILU as a separate op at every level (MNN has no conv+SiLU fusion), so no dispatch reduction is available from the converter. `onnx-simplifier` is not installed in the research venvs and the ONNX (Ultralytics export, opset 12) is already constant-folded (663 nodes, 186 Conv), so it was not pursued.
* **ncnn**: dispatches per inference 978 -> ~740 with the patched `ncnnoptimize` (section 3.2). `ncnnoptimize`'s other passes did nothing on this graph.
* Dispatch counts before/after: MNN 700-790 -> ~700 (no graph change; the win was submission), ncnn 978 -> ~740.

## 5. Combined result vs the baselines (Task 5)

Quiet-machine medians (NVR idle, nothing building) plus the range across the interleaved final A/B (`results/final_ab.txt`, 3 passes, NVR detecting at ~4 fps: report min / medians):

| configuration | quiet median | A/B min | A/B medians (3 passes) | correctness | CPU ms/iter |
|---|---:|---:|---|---|---:|
| **MNN-Vulkan fp16 image, batch (patched default)** | **11.6 ms** | **11.1** | 11.7 / 21.6 / 50.3 | md5 identical to stock (`6d1d0a17`), 3.6 max / 0.056 mean vs ORT | 0.75 |
| MNN-Vulkan fp16 image, direct (baseline 19.0) | 18.7 | 17.6 | 18.3 / 18.5 / 18.4 | same | 6.4-8.2 |
| MNN-Vulkan fp32 image, batch | 15.7 | 14.0 | 14.7 / 18.5 / 21.9 | md5 identical (`301ed9a3`), 0.0017 vs ORT | 0.8 |
| MNN-Vulkan fp32 image, direct | 21.2 | 19.5 | 20.0 / 20.2 / 20.1 | same | 8-9 |
| ncnn-Vulkan fp16 fused+coalesced (src) | ~24 (25.2 in a loaded pass) | 20.8 | 24.7 / 31.2 / 41.1 | 18.6 / 0.18 vs ORT (fp16) | 4.3-5.5 |
| ncnn-Vulkan fp16 pip `-opt` (baseline 28 raw / 22.3 opt) | 22.3 | 22.4 | 27.2 / 30.6 / 30.9 | 22.6 / 0.19 | 5.5-6 |
| ncnn-Vulkan fp32 fused+coalesced | 30.2 | 25.8 | 29.3 / 29.5 / 31.3 | 0.0016 vs ORT | 4-4.8 |
| ncnn-Vulkan fp32 pip `-opt` | 28.3 | 28.5 | 32.6 / 32.9 / 37.2 | 0.0021 | 5.7-5.9 |
| MNN CPU fp16 4 thr (reference) | 4.9 | 4.6 | 4.9 / 14.7 / 14.3 (CPU contention from the NVR) | 8.9 / 0.12 | 19-52 |
| ncnn CPU fp16 4 thr (pip) | 4.8-5.4 | 4.8 | 5.4 / 5.4 / 4.8 | 11.4 / 0.22 | 19-21 |

Best GPU configuration: **MNN, image backend, fp16, `MNN_GPU_RECORD_BATCH`** = 11.6 ms / 86 fps at 0.75 ms CPU per inference (vs 19.0 ms baseline, -39%; vs ncnn-Vulkan 28 ms baseline, -59%). It is still 2.4x the CPU fp16 path in latency, but costs 0.75 ms of CPU instead of ~20 thread-ms, which is the reason to use the GPU at all on this box.

For the detector: `detector/mnn_shim/mnn_shim.cpp` sets `cfg.numThread = threads` (i.e. mode 0x4 = direct); with the patched `libMNN.so` from `gpu-opt/MNN/build` it gets batch mode automatically, or with the stock library one line (`cfg.mode = MNN_GPU_TUNING_WIDE | MNN_GPU_RECORD_BATCH` for the Vulkan case) does the same. Neither was applied to the live service.

## 6. Deliverables and files

* `PHASE1_RUNTIME.md` (this file).
* `patches/mnn/0001-Vulkan-record-the-whole-graph-into-one-command-buffe.patch` (MNN `gpu-opt` branch, commit `330e565`, on `bef71b9`).
* `patches/ncnn/0001-Fuse-Swish-SiLU-into-Convolution-activation_type-7-a.patch`, `patches/ncnn/0002-Vulkan-issue-one-vkCmdPipelineBarrier-per-dispatch-i.patch` (ncnn `gpu-opt` branch, commits `1b00e9b`, `4049775`, on tag `20260526`).
* `MNN/` (source + `build/` image backend patched, `build-buffer/` buffer backend), `ncnn/` (source + `build/` with `tools/ncnnoptimize`), `bench/` (`bench_mnn2.cpp` [+ `_stock`, `_buffer` binaries], `bench_mnn_phases.cpp`, `bench_ncnn2.py` (pip wheel), `bench_ncnn3.cpp` (source lib), `cmp.py`, models: `fox-person-cat-320.mnn`, `fox-person-cat-320{,-opt,-fused}.param/.bin`, `fox_opt{0,1,2}.mnn`, `fox_fp16.mnn`, `lib-swish/` (ncnn lib before barrier coalescing), all `out_*.bin` outputs), `scripts/final_ab.sh`, `results/` (`gputime_*.txt` raw profiler timelines, `perf_mnn_*.data` + `perf_mnn_summary.txt`, `strace_*.txt`, `ncnn_pip_option_sweep_gputime.txt`, `mnnconvert_optlevels.txt`, `final_ab.txt`).

## 7. What remains (Phase 2 candidates, in order of expected payoff)

1. MNN: fuse SiLU (and the residual ADD) into the convolution epilogue -- needs a `Convolution2DCommon` post-op field + converter pass + CPU/Vulkan epilogues; removes ~180-220 of ~700 dispatches (~2 ms of 11.6).
2. ncnn: kernel efficiency -- its conv dispatches take 27 us vs MNN's 14 us for the same layers (local-size selection for the AGX, winograd/gemm path choice); and the ~450 implicit packing/cast dispatches around Concat/Crop.
3. ncnn wheel rebuild from `gpu-opt/ncnn` (`-DNCNN_PYTHON=ON`) if the fused graph is to reach the detector; the stock wheel silently drops `activation_type 7`.
4. Submission-level: MNN's two submits per inference could be one (~0.2 ms); ncnn already does one.
5. Re-measure on an idle GPU (stop or pause the NVR briefly) to get clean p95s; every number above carries the NVR's contention.
