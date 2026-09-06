# Phase 3: Conv + SiLU epilogue fusion in MNN's Vulkan backend

Date: 2026-09-06. Same machine, driver, model, input and reference as `PHASE1_RUNTIME.md` (Mac mini M1, Fedora Asahi 44, kernel 7.1.6, Mesa 26.1.8 Honeykrisp; YOLOv9-t `fox-person-cat-320.onnx` converted by `mnnconvert` to the 510-op `bench/fox-person-cat-320.mnn`; `input_320.bin`, `ref_ort_output.bin`). Baseline = the batch-recording library from Phase 1 (`MNN` commit `330e565`, = `detector/mnn-batch/lib/libMNN.so`, kept as `bench/lib-batch/libMNN.so`). Live NVR (ncnn-Vulkan detector, ffmpeg Vulkan scalers) and another agent's detector benchmark on port 5556 shared the GPU throughout; passes with medians above 20 ms are contention and are listed but excluded from the summary.
Work: branch `conv-silu-fusion` of `the MNN tree`, commit `d9f1652` on top of `330e565`; patch `patches/mnn/0002-Vulkan-fuse-SiLU-into-the-convolution-epilogue-at-schedule-time.patch`. Builds inside `systemd-run --user --scope -p MemoryMax=5G`, `ninja -j4`.

## TL;DR

* The 179 SiLU ops that follow convolutions in the YOLOv9-t graph are now applied inside the convolution shaders. Scheduled ops 469 -> 290, **GPU dispatches per inference 783 -> 604 (-179, -23%)**, measured exactly (gdb count of `vkCmdDispatch`: -179 with tuning disabled) and with the firmware profiler (261.0 -> 201.3 dispatches per command stream, 3 streams per inference).
* **fp16 median 12.08 -> 11.15 ms (-0.9 ms, -8%), fp32 14.73 -> 14.00 ms (-5%)** in an 8-pass interleaved A/B (100 iterations each, medians of the clean passes); best-case minimums 10.94 -> 10.33 ms and 13.90 -> 12.83 ms. GPU time per inference from the profiler: 783 x 13.7 us = 10.7 ms -> 604 x 16.1 us = 9.7 ms; the removed dispatches were the cheap ones, so the remaining average kernel is longer.
* Correctness: fp32 output **bit-identical** to the unfused library (md5 `301ed9a3`). fp16 differs because SiLU now sees the fp32 accumulator instead of the fp16-rounded tensor: max abs diff 2.0 (mean 0.030) vs the unfused fp16 output, and it is slightly *closer* to the onnxruntime fp32 reference (mean abs 0.0546 vs 0.0562). On the 45-frame detector set the post-processed detections are identical (36/36, no class/count change, max score delta 0.003, max box-corner delta 0.37 px vs the batch lib).
* The fusion is a **runtime** decision made when the session is scheduled, so it applies to the detector as it is: the detector converts the ONNX at load time with the pip `mnn` wheel's `_tools.mnnconvert`, which a converter-side pass would never touch. Model file, conversion cache and shim are unchanged; only `DETECTOR_MNN_LIB` moves to `detector/mnn-batch/lib-v2/libMNN.so`.

## 1. How MNN handles activations today (Task 1)

* **Converter** (`tools/converter/source/optimizer/`): BN folding, `x*sigmoid(x)` -> `UnaryOp(SILU)` (`FuseTemplateOp.cpp`), relu/relu6 folded into `Convolution2DCommon.relu/relu6` by the ONNX front-end. There is no Conv+SiLU (or Conv+any-unary) merge; `Convolution2DCommon` (`schema/default/CaffeOp.fbs`) has only the two booleans and no activation enum. Confirmed in Phase 1: every `--optimizeLevel` yields the same 510-op graph with 179 separate `SILU` ops.
* **Vulkan image backend**: `VulkanConvolutionCommon::getPostTreatMacro()` maps relu/relu6 to `RELU_`/`RELU6_` and picks a precompiled SPIR-V variant of the epilogue shader: `convolution1x1{,_w4,_c8w4}.comp` (1x1 path, private auto-tuned pipelines), `convolutionDepthwise{,_s1d1_w2}.comp`, `col2Im.comp` (im2col + `gemm16x16` path), `winogradTransformDest2_3_1.comp` (3x3 s1 winograd). Variants are enumerated in `glsl/macro.json` and baked into `compiler/AllShader.cpp` + `VulkanShaderMap.cpp` by `compiler/makeshader.py` (glslangValidator, not run by CMake). On this GPU the `FP32_` variants are used even in fp16 mode (the `FP16_` half-float kernels are enabled only for Adreno/Mali), i.e. the convolution accumulates in fp32 and stores into an fp16 image.
* **SiLU today**: `VulkanUnary` with `unaryImage.comp` `SILU` (`value / (1 + exp(-value))`), **one dispatch per SiLU** (the converter already merged sigmoid+mul; only the head's 1 `Sigmoid` and 2 `Mul` remain separate). So the target was exactly 179 dispatches.
* **Dispatch accounting**: Phase 1 estimated "~700" from 259.8 dispatches per stream x 2.7 streams. The per-inference structure in batch mode is 3 command streams (host->device converter: 1 dispatch; the recorded graph: 781; device->host converter: 1) = **783**, which the profiler now shows directly (261.0 per command over a steady window with 3 commands per inference: `results/gputime_mnn_fp16_batchlib_phase3.txt`). The gdb count of `vkCmdDispatch` at session creation (batch mode records the graph once) is 1165 with tuning disabled: 783 + ~380 one-time weight-upload/reorder dispatches.

## 2. Design (Task 2)

Runtime fusion, because the production conversion path cannot be changed (see TL;DR). Five parts, all in patch `0002` (20 files, of which `AllShader.cpp` is generated):

1. **Schema**: `Convolution2DCommon.fusedActivation:int = 0`, appended at the end of the table (flatbuffers field id 18, so old models and old readers are unaffected). Non-zero = a `UnaryOpOperation` id applied after bias/relu/relu6. The converter never writes it; the generated `schema/current/CaffeOp_generated.h` was regenerated with MNN's bundled flatc 1.10 (regenerating the unmodified schema gave a zero diff, so the header diff is just the field).
2. **Core hook**: `Backend::onSupportFusedActivation(inputs, outputs, op, activation)` (default false).
3. **Pipeline pass** `Pipeline::_fuseConvolutionActivation()`, called at the end of `Pipeline::encode()` after geometry compute and quant-cast insertion. It walks the command list of the pipeline once; for each `UnaryOp` whose input is produced by a `Convolution`/`ConvolutionDepthwise` command with a single consumer, no relu/relu6, no quant attributes, no raster regions, same shape/format/type on both tensors and a `NORMAL` (not session-output) intermediate, it asks the main backend; on yes it unpacks the conv op, sets `fusedActivation`, repacks it into a `BufferStorage` cached per original op (`mFusedOps`, so the op pointer and MNN's execution cache stay stable across resizes), points the conv at the unary's output and erases the unary command. The pass is idempotent: a retained command list already carries `fusedActivation != 0`; a rebuilt one is re-fused. Cost: one copy of each fused conv's weights (a few MB for this model) for the session's lifetime.
4. **Vulkan**: `VulkanBackend::onSupportFusedActivation` accepts `SILU` for group-1 `Convolution` and for `ConvolutionDepthwise` when the same `_supportImageSize` check as `onCreate` passes. `getPostTreatMacro` returns `SILU_` for it, and all seven epilogue shaders gained a `SILU`/`SILU_FP32`/`SILU_FP16` block (`x / (1 + exp(-x))` in fp32, converting from/to `f16vec4` in the FP16 variants, the same expression as `unaryImage.comp`). Twelve new SPIR-V variants were compiled with the same recipe as `makeshader.py` (first line, `#define MACRO`, `glslangValidator -V -Os`) and spliced after their siblings in `AllShader.cpp`/`AllShader.h`/`VulkanShaderMap.cpp` (`scripts/mnn_gen_silu_shader_variants.py`); the existing blobs are untouched, and `macro.json` lists the variants so a full `makeshader.py` regeneration reproduces them.
5. **CPU fallback**: if a convolution op carrying `fusedActivation` reaches the CPU backend (the pipeline falls back per op when the GPU declines one), `CPUBackend::onCreate` wraps the execution in `CPUConvolutionFusedActivation`, which runs the convolution and then applies the core's unary function (`MNNSelectUnaryFunctionForFloat`, fp32 or fp16 core) over the packed output. The unary kernels are fed from a stack scratch copy because `MNNSiLu` is not in-place safe (it writes `exp(-x)` into `dst` before reading `src` again) -- found by the test below, where the first version silently returned the plain convolution in fp32.

Not done on purpose: a converter-side pass. It would only help models converted by a patched `mnnconvert`, and any *other* backend (OpenCL, Metal, CoreML, buffer-Vulkan) would silently ignore the field and drop the activation; keeping the decision at runtime, gated on `onSupportFusedActivation`, means a model can never be wrong on a backend that does not implement it. The buffer-mode Vulkan backend (`MNN_VULKAN_IMAGE=OFF`) keeps the default (no fusion).

## 3. Verification (Task 3)

Tools: `bench/phase3/count_ops.cpp` (ops executed per session via `runSessionWithCallBackInfo`), `scripts/count_dispatch.sh` (gdb breakpoint count on the loader's `vkCmdDispatch`), `scripts/ab.sh` (interleaved A/B, 100 iterations per config per pass), the got-bringup profiler via `research/vk-dispatch/with-patched.sh` + `HK_GPUTIME=1`, `bench/phase3/test_fused.cpp` (CPU fallback + small-net Vulkan), `scripts/acc_dump.py`/`acc_cmp.py` (45-frame detector set through `detector/backends.py`'s `mnn-vulkan` backend with `DETECTOR_MNN_LIB`). Raw logs: `results/phase3_chain1.txt`, `results/phase3_chain2.txt`, `results/phase3_ab.txt`, `results/phase3_ab2.txt`, `results/gputime_mnn_fp16_{batchlib,fused}_phase3.txt`.

### 3.1 Graph and dispatches

| | batch lib (`330e565`) | fused lib (`d9f1652`) |
|---|---:|---:|
| ops executed per inference, Vulkan fp16 (`count_ops`) | 469 (186 Convolution, 179 SILU, ...) | **290** (186 Convolution, 1 UnaryOp = the head Sigmoid) |
| ops executed, CPU fp32 session (fusion must not apply) | 469 | 469 |
| `vkCmdDispatch` calls at session creation, tuning off (`mode 0x1`) | 1165 | **986 (-179)** fp16 and fp32 alike |
| dispatches per inference, profiler steady state (3 streams x per-stream count) | 3 x 261.0 = **783** | 3 x 201.3 = **604** |
| average GPU time per dispatch (profiler) | 13.5-13.9 us | 16.0-16.2 us |
| GPU time per inference (product) | ~10.7 ms | ~9.7 ms |

(With tuning on, the creation-time count also includes the auto-tuning trial dispatches, which differ between the two libraries because the fused variants are new pipeline keys; that is why the tuning-off count is the exact one.)

### 3.2 Latency (interleaved A/B, `scripts/ab.sh`, 100 iterations per run after 10 warm-ups; `results/phase3_ab.txt` 5 passes + `phase3_ab2.txt` 3 passes)

| config | medians per pass (ms) | median of clean passes | range (clean) | median p95 (clean) | best min |
|---|---|---:|---:|---:|---:|
| batch fp16 | 12.21 12.08 11.99 12.31 (35.09) 11.32 11.72 12.12 | **12.08** | 11.32-12.31 | 14.30 | 10.94 |
| **fused fp16** | 10.97 11.33 11.06 11.15 (25.86) 11.65 11.10 11.19 | **11.15** | 10.97-11.65 | 13.81 | 10.33 |
| batch fp32 | 14.80 14.70 14.86 14.65 (35.74) 17.24 14.72 14.73 | **14.73** | 14.65-17.24 | 17.12 | 13.90 |
| **fused fp32** | 14.12 13.62 13.77 (36.33) (37.13) 14.71 13.88 14.16 | **14.00** | 13.62-14.71 | 16.89 | 12.83 |

Parenthesised passes coincided with the other agent's detector benchmark / NVR bursts (both configs of that pass jump together). CPU time per inference is unchanged (0.6-0.8 ms). A 30-iteration run of the delivered `lib-v2` library on a quiet moment gave 10.86 ms fp16 / 13.31 ms fp32.

### 3.3 Outputs

| comparison | fp32 | fp16 |
|---|---|---|
| fused vs batch lib, Phase 1 input | **identical** (md5 `301ed9a3` both) | max abs 2.00, mean 0.030 (md5 `1841c2cc` vs `6d1d0a17`) |
| fused vs onnxruntime fp32 reference | 0.00174 / 0.00002 (unchanged) | 3.856 / 0.0546 (batch lib: 3.606 / 0.0562) |
| `lib-v2/libMNN.so` vs the research build | identical outputs, library md5-identical (`7b03f0c3`) | identical |

45 real detector frames (`scripts/acc_dump.py` through `backends.py` `mnn-vulkan`, fp16, score > 0.4, NMS), raw `output0` (1x7x2100) and post-processed detections:

| | fused vs batch lib | fused vs `cpu` (ORT fp32) | batch lib vs `cpu` (Phase 1 result) |
|---|---|---|---|
| raw max / mean abs diff | 2.63 / 0.0249 | **3.51 / 0.0467** | 3.76 / 0.0481 |
| detections A / B | 36 / 36 | 36 / 36 | 36 / 36 |
| frames with class or count mismatch | 0 | 0 | 0 |
| max score delta / max box-corner delta | 0.0029 / 0.37 px | 0.0043 / 0.25 px | 0.0042 / 0.25 px |

The fp16 shift is the SiLU input no longer being rounded to fp16 between the convolution and the activation; it moves the result towards the fp32 reference, and no detection changes.

### 3.4 CPU fallback and resize (`bench/phase3/test_fused.cpp`)

A hand-built net (Input -> 3x3 Conv with `fusedActivation = SILU`, set directly in the flatbuffer) vs Input -> Conv -> SILU on the CPU backend: max abs diff **0.0** in fp32 and in fp16 (Arm82 core), after the in-place fix described in section 2. The same Conv+SILU net on Vulkan fuses at runtime (4 commands instead of 5) and matches the CPU fp32 result to 3.2e-5 (fp32) / 0.010 (fp16).

Resize: `bench/phase3/resize_check.cpp` (batch-1 session, release, new session resized to batch 2 with the input duplicated) segfaults inside `VulkanImageConverter` -> `hk_UpdateDescriptorSets` at `copyFromHostTensor` for the real model with **both** the batch lib and the fused lib, identically; with the small Express net the resized session runs but its second half is wrong in both. This is a pre-existing behaviour of the Vulkan backend's `resizeTensor`+`resizeSession` path in this harness, not introduced here and not investigated further; noted because the detector shim (v2) resizes sessions for batch > 1 (the accuracy and soak runs so far are batch 1).

## 4. Deliverables (Task 4)

* `patches/mnn/0002-Vulkan-fuse-SiLU-into-the-convolution-epilogue-at-schedule-time.patch` (commit `d9f1652`, branch `conv-silu-fusion` on `330e565`); applies on top of `0001`.
* `detector/mnn-batch/lib-v2/`: `libMNN.so` (md5 `7b03f0c3c843b4c65e95487cd66ec757`, also in `lib/`), `PROVENANCE.txt`, `build.sh` (same flags as `mnn-batch/build.sh`, sources in `lib-v2/src` = `git archive d9f1652`), `build.log`. Nothing else under `detector/` was touched; the live service is unchanged. To switch: `DETECTOR_MNN_LIB=detector/mnn-batch/lib-v2/libMNN.so`; rollback = the old path.
* `PHASE3_MNN_FUSION.md` (this file), `bench/phase3/`, `scripts/{ab.sh,chain.sh,chain2.sh,count_dispatch.sh,mnn_gen_silu_shader_variants.py,acc_dump.py,acc_cmp.py}`, `results/phase3_*`, `results/gputime_mnn_fp16_*_phase3.txt`, `bench/lib-batch/libMNN.so` (the Phase 1 baseline library), `bench/out_fused_p{1,2}.bin`, `bench/out_libv2_p{1,2}.bin`.
* Preservation repo `gpu-inference-asahi`: `mnn/0003-...patch` and `reports/PHASE3_MNN_FUSION.md` (paths sanitised).

Note for the detector: at the time of the final check, `detector/detector/backends.py` had been changed by the other agent to require `mnn_shim_create_multi` from the shim, which the on-disk `libmnn_shim.so` does not export, so `mnn-vulkan` currently falls back to `cpu` regardless of the library. The lib-v2 accuracy run therefore used the saved pre-change `backends.py` with `DETECTOR_MNN_SHIM` pointing at the on-disk shim: it loaded lib-v2 (`mnn-vulkan(Vulkan, fp16)`, 11.6 ms average in-process on the contended GPU) and its raw outputs are bit-identical to the research build on all 45 frames.

## 5. What remains

1. The 44 residual `BinaryOp(ADD)` after convolutions (`Conv -> Add(skip)` in the RepNCSP blocks): same mechanism, a second input to the conv epilogue; ~44 dispatches (~0.4 ms).
2. The `Concat`/`StridedSlice` raster copies (31 + 18 ops, one dispatch per input slice) -- writing convolution outputs straight into the concat destination would remove most of them.
3. ncnn's kernel efficiency and packing dispatches (Phase 1 section 7) are unchanged.
4. GPU time is now ~9.7 ms of the 11.15 ms; the ~1.4 ms remainder is the two submit/fence round trips and the converters.
