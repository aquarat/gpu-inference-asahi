# MNN patches

Base: [alibaba/MNN](https://github.com/alibaba/MNN) commit `bef71b9`
("[Vulkan:Perf] Optimize INT4 cooperative matrix path", master as of
2026-09-05). The patches are `git format-patch` output and apply in order with `git am`:

```sh
git clone https://github.com/alibaba/MNN && cd MNN && git checkout bef71b9
git am ../gpu-inference-asahi/mnn/*.patch
```

| patch | what | evidence |
|---|---|---|
| `0001-cmake-drop-D__STRICT_ANSI__-on-Linux-to-fix-the-build-with-GCC-16.patch` | With GCC 16's libstdc++ the Linux build fails when `__STRICT_ANSI__` is defined (`__int128` redefinition). Drops the define from `CMAKE_CXX_FLAGS` on Linux. One line. | `research/GPU_INFERENCE.md` in frigate-asahi |
| `0003-Vulkan-fuse-SiLU-into-the-convolution-epilogue-at-schedule-time.patch` | YOLO-style graphs follow almost every convolution with SiLU, which MNN ran as a separate `VulkanUnary` dispatch (`Convolution2DCommon` only knows relu/relu6). Adds `Convolution2DCommon.fusedActivation` (appended field, converter never writes it), a `Backend::onSupportFusedActivation` hook and a pass at the end of `Pipeline::encode` that fuses Convolution -> UnaryOp chains at schedule time when the backend accepts (so it works for models converted by any converter, including a detector that converts at load time), SiLU variants of all Vulkan image-backend convolution epilogue shaders (1x1, depthwise, col2Im, winograd; SiLU evaluated in fp32), and a CPU fallback wrapper so a fused op is still right if the GPU declines it. | `../reports/PHASE3_MNN_FUSION.md` |
| `0002-Vulkan-record-the-whole-graph-into-one-command-buffer-by-default.patch` | `VulkanRuntime` accepted only six exact `gpuMode` values and defaulted to one `VkCommandBuffer` per op. Because `ScheduleConfig::mode` shares storage with `numThread`, nearly every caller lands on mode `0x4` = per-op recording. The patch validates only the tuning bits, ignores the OpenCL-only memory bits, and defaults to `MNN_GPU_RECORD_BATCH` unless the caller asks for `MNN_GPU_RECORD_OP` explicitly. Touches `source/backend/vulkan/runtime/VulkanRuntime.cpp` only; both the image and the buffer Vulkan backends read the same flag; the CPU path is untouched. | `../reports/PHASE1_RUNTIME.md` section 2 |

Measured effect of 0002 on YOLOv9-t 320, batch 1, Apple M1, Mesa 26.1.8
Honeykrisp, image backend, next to a live NVR (quiet-machine medians of 100):

| | per-op (before) | batch (after) |
|---|---:|---:|
| fp16 median latency | 18.7 ms | 11.6 ms |
| fp32 median latency | 21.2 ms | 15.7 ms |
| CPU time per inference | 6.4 ms | 0.7 ms |
| GPU control streams per inference | ~474 | ~3 |
| DRM ioctls per inference | ~610 | a handful |
| GPU busy | 50% | 78% |
| output | md5 `6d1d0a17` (fp16) / `301ed9a3` (fp32) | identical |

Verified on the patched build: mode `0x4` now behaves as batch, mode `0x104`
(`RECORD_OP`) reproduces the old per-op numbers (8.2 ms CPU, 19.3 ms), mode
`0x204` is unchanged. On the buffer backend (`-DMNN_VULKAN_IMAGE=OFF`) batch
mode helps too (fp32 21.6 to 17.5 ms) but that backend is ~30% slower than
the image backend here and its fp16 path loses accuracy (max abs diff 20 vs
3.6), so the image backend remains the choice.

Measured effect of 0003 on top of 0002 (same setup, 8-pass interleaved A/B of
100 iterations, medians of the uncontended passes):

| | 0002 only | 0002 + 0003 |
|---|---:|---:|
| ops scheduled per inference | 469 | 290 |
| GPU dispatches per inference | 783 | 604 |
| fp16 median latency | 12.08 ms | 11.15 ms |
| fp32 median latency | 14.73 ms | 14.00 ms |
| fp32 output | md5 `301ed9a3` | identical |
| fp16 output vs onnxruntime fp32 (max / mean abs) | 3.61 / 0.0562 | 3.86 / 0.0546 |
| 45-frame detector set vs the 0002 library | | 36/36 detections, 0 mismatches, max score delta 0.003, max box delta 0.37 px |

Callers that cannot rebuild MNN get the same effect with one line:
`cfg.mode = MNN_GPU_TUNING_WIDE | MNN_GPU_RECORD_BATCH` for the Vulkan case
(instead of, or in addition to, `cfg.numThread`).

Build used for the measurements (aarch64, GCC 16, `ninja -j4` inside a 5 GB
memory scope):

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMNN_VULKAN=ON -DMNN_VULKAN_IMAGE=ON -DMNN_ARM82=ON -DMNN_BUILD_SHARED_LIBS=ON -DMNN_BUILD_TOOLS=ON -DMNN_SEP_BUILD=OFF -DMNN_OPENCL=OFF
cmake --build build
```

Under the git history that produced these files, 0001 is commit `94555fa`
(on a full clone of MNN), 0002 is `330e565` on a local `gpu-opt` branch and
0003 is `d9f1652` on a `conv-silu-fusion` branch on top of it,
whose baseline commit carries the same CMake change. Pushing those branches to
[aquarat/MNN](https://github.com/aquarat/MNN) is blocked until the GitHub
token has the `workflow` scope, because MNN's history contains
`.github/workflows/*` files.
