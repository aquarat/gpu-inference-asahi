# ncnn patches

Base: [Tencent/ncnn](https://github.com/Tencent/ncnn) tag `20260526` (commit
`e54f7b1`), which is the source of the pip wheel `ncnn==1.0.20260526`; a
source build of the tag produced outputs bit-identical to the wheel. Both
patches are `git format-patch` output and apply with `git am`:

```sh
git clone https://github.com/Tencent/ncnn && cd ncnn && git checkout 20260526
git submodule update --init glslang
git am ../gpu-inference-asahi/ncnn/*.patch
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_SHARED_LIB=ON -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_BENCHMARK=ON
cmake --build build      # ~12 min at -j4 on an M1
```

## 0001 Fuse Swish/SiLU into Convolution (`activation_type 7`) and recover x*sigmoid(x) in ncnnoptimize

YOLO-family models use SiLU after every convolution. `ncnnoptimize` fuses
ReLU/Clip/Sigmoid/Mish/HardSwish into `Convolution` (`activation_type` 1-6)
but not Swish, and `onnx2ncnn` only recognises `x*sigmoid(x)` when the
convolution output has a single consumer, so a quarter of the SiLUs survive
as `Split + Sigmoid + BinaryOp(MUL)`: three dispatches and two barriers on
Vulkan for one activation.

* `activation_type 7 = x * sigmoid(x)` in `src/layer/fused_activation.h`
  (scalar and the `create_activation_layer` fallback, which yields a Swish
  layer), the arm (f32 NEON, fp16 scalar/f16x4/f16x8), x86 (sse/avx/avx512),
  loongarch, mips and riscv activation helpers (each already had swish or
  sigmoid helpers; one line each), and `vulkan_activation.comp` (`afp` and
  `afpvec4`).
* `tools/ncnnoptimize.cpp`: new pass `fuse_split_sigmoid_mul_swish()` that
  turns `x -> Split -> {a, b}; Sigmoid(b); Mul(a, sigmoid)` into `Swish(x)`,
  then the `Convolution` / `ConvolutionDepthWise` / `InnerProduct` activation
  fusions accept `Swish` as type 7.

x86, loongarch, mips and riscv are compile-untested (aarch64 host); they
mirror the existing type-6 code.

YOLOv9-t 320 (655 layers from `onnx2ncnn`): the optimised graph has 388
layers (179 convolutions carry `9=7`; Split 123 to 79, BinaryOp 92 to 48,
Sigmoid 45 to 1, Swish 135 to 0). On an Apple M1 (Mesa 26.1.8 Honeykrisp)
that is 978 to ~740 dispatches and ~2020 to ~1680 barrier calls per
inference; Vulkan fp16 29.0 to 25.2 ms and fp32 33.8 to 30.2 ms in the same
(NVR-loaded) pass. fp32 Vulkan max abs diff vs onnxruntime 0.0016 (0.0021
before), fp32 NEON 0.0015; fp16 differs from the unfused fp16 graph only by
the rounding of the fused epilogue. The removed dispatches were the cheap
element-wise ones, which is why 26% fewer dispatches gave ~13% less time:
ncnn's convolution kernels average 27 us on this GPU against MNN's 14 us for
the same layers, a kernel-efficiency gap this patch does not address.

**Deployment note**: a stock ncnn (including the pip wheel) does not know
`activation_type 7` and will run the fused graph *without* the activation,
silently. Only feed fused `.param` files to a build that carries this patch.

## 0002 Vulkan: one `vkCmdPipelineBarrier` per dispatch instead of one per buffer binding

`VkCompute::record_pipeline` called `barrier_readwrite()` per buffer binding,
so each dispatch was preceded by 2-3 separate barrier calls (~1700-2000 per
YOLOv9-t inference). The patch gathers the `VkBufferMemoryBarrier`s of all
bindings and records them with one call whose `srcStageMask` is the OR of the
bindings' stage flags; the delayed-record path for devices without
`VK_KHR_push_descriptor` is kept; image bindings are unchanged.

Effect on the M1: ~1680 to ~580 barrier calls per inference for ~740
dispatches, output bit-identical (fp16 and fp32), latency within the noise
of the shared GPU (interleaved passes 24.2 vs 23.9-29.6 ms, best 19.4 vs
20.5 ms). Honeykrisp makes consecutive barriers with no work between them
nearly free, so the real floor is the ~740 dependent dispatches at ~3.5 us
drain each. Kept because it is correct and cheaper on drivers that do charge
per barrier.

## Things checked and left alone

Staging is already zero-copy on this unified-memory GPU (ncnn detects the
mappable device-local memory type, `rebar=1`; weights are memcpy'd straight
into the destination, the input needs one packing/cast dispatch either way).
`use_fp16_*` and `use_subgroup_ops` sweeps change nothing; `use_shader_pack8`
and `use_image_storage` no longer exist in this version; bf16 storage is not
supported by Honeykrisp. The ~450 implicit packing/cast dispatches around
`Concat`/`Crop` with channel counts that are not multiples of 4, and local-size
tuning for the AGX (subgroup 32, 1024 invocations), are the remaining
ncnn-side work. See `../reports/PHASE1_RUNTIME.md` section 3.
