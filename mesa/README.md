# Honeykrisp: VK_KHR_cooperative_matrix on the M1's SIMD-group matrix FMA

Three `git am` patches on [aquarat/mesa](https://github.com/aquarat/mesa) branch `local-deploy` @ `d105715`
(Mesa 26.3.0-devel; the fork's Honeykrisp with the got-bringup compiler work). The same commits are pushed as branch
[`coopmat`](https://github.com/aquarat/mesa/tree/coopmat) (`77e6f99`, `7dd673b`, `71969e3`). Nothing in them depends
on the fork's other commits, so they should apply to upstream `main` with at most path drift.
Write-up: [../reports/PHASE4_COOPMAT.md](../reports/PHASE4_COOPMAT.md).

| patch | what |
|---|---|
| `0001-asahi-hk-prototype-VK_KHR_cooperative_matrix-on-the-.patch` | The extension. `agx`: one new opcode `simd_matrix_fmadd` (encoding `0x800000000800006f`, the G13's `simd_matrix_fmadd16/32` from dougallj's `applegpu`, 8x8x8 on a 32-lane SIMD-group, two elements per lane in Metal's `simdgroup_matrix` layout), packed by the generic ALU packer; a `cmat_muladd_agx` NIR intrinsic; emitter, validator, read/write-register cases. `hk`: a ~370-line lowering pass modelled on panvk's (`nir_lower_cooperative_matrix_flexible_dimensions` splits every shape into 8x8 tiles, then `cmat_*` intrinsics become per-lane `vec2` ops and per-element loads/stores with the lane's `(row, col)` from the layout formula); `KHR_cooperative_matrix` advertised for compute with six shapes: 16x16x16 and 8x8x8 in f16/f16/f32/f32, f16/f16/f16/f16, f32/f32/f32/f32. No integer shapes (the ISA has none on G13), no bf16. |
| `0002-asahi-hk-use-the-native-mixed-f16-x-f16-f32-matrix-F.patch` | The hardware honours per-operand size flags: f16 register pairs for A and B with an f32 pair for C/D (instruction bit 26 set from the accumulator size) is bit-exact and accumulates in true f32 (a `4096 + 8 x 1/64` probe returns 4096.125 exactly). Makes that the default; `HK_CMAT_WIDEN=1` keeps the widen-to-f32 form. |
| `0003-agx-never-allocate-a-simd_matrix_fmadd-destination-o.patch` | The bug ggml's flash-attention shader found: the register allocator "early-kills" a source whose last use is the current instruction and may put D over A or B; the multi-cycle SIMD-group op keeps reading its inputs while writing outputs, so that corrupts one output block. `can_kill_early()` refuses it for sources 0 and 1 (D over C, i.e. accumulate in place, is fine). |

## Numbers (M1 Mac mini, G13G, NVR live on the same GPU)

* Correctness: bit-exact on a standalone test in every advertised shape/type/layout (incl. column-major B and a padded-stride shared-memory staging variant); ggml `test-backend-ops -o MUL_MAT` **1106/1106** with `matrix cores: KHR_coopmat`; the FLASH_ATTN_EXT subsets that exercise the coopmat path pass (the only remaining failures are q5_0/q5_1 K/V cases that fail identically with coopmat disabled).
* `llama-bench` Qwen2.5-0.5B pp512: Q8_0 **783 -> 1055 t/s** (+35 %, ~1.04 TFLOPS = 40 % of the 2.6 TFLOPS FMA peak), F16 757 -> 1016 t/s; token generation unchanged (memory-bound).
* CLIP ViT-H-14-378 (1007 GFLOP): **1874 -> 1505 ms** (1.25x) without flash-attention, 1776 -> 1578 ms with it (ggml's coopmat1 FA shader is slower than its non-FA path once the GEMMs use the instruction); cosine vs ORT fp32 unchanged, 0.9987-0.9991.
* On the M1 the instruction runs on the FMA ALUs (no separate matrix unit), so the gain is utilisation, not peak; Apple's own Metal GEMMs on it reach ~80 % of peak, ggml is at ~40 %.

## Build and load per process (never installed)

```
git clone -b local-deploy https://github.com/aquarat/mesa && cd mesa && git am ../mesa/000*.patch   # or: git checkout coopmat
meson setup build --prefix=$PWD/install --libdir=lib64 --buildtype=release -Dvulkan-drivers=asahi -Dgallium-drivers= \
  -Dopengl=false -Dgles1=disabled -Dgles2=disabled -Dglx=disabled -Degl=disabled -Dgbm=disabled -Dglvnd=disabled \
  -Dvideo-codecs= -Dtools= -Dvulkan-layers= -Dbuild-tests=false -Dllvm=enabled -Dshared-llvm=enabled -Dxmlconfig=enabled
ninja -C build install            # ~10 min in a 3-CPU / 6 GB container (the host lacked llvm/clang headers)
export VK_DRIVER_FILES=$PWD/install/share/vulkan/icd.d/asahi_icd.aarch64.json   # + VK_ICD_FILENAMES for old loaders
vulkaninfo --summary | grep driverInfo                                           # Mesa 26.3.0-devel, KHR_cooperative_matrix
```

The Vulkan loader honours `VK_DRIVER_FILES` per process, so one service (here Immich's ML service, through two
`Environment=` lines in its unit) can run on this build while everything else, including a live detector on the same
GPU, keeps the system Mesa. ggml/llama.cpp selects its coopmat shaders automatically when the extension is advertised;
`GGML_VK_DISABLE_COOPMAT=1` isolates the extension from the fork's other compiler work for A/B runs.

## What remains

1. **Vulkan CTS**: `dEQP-VK.compute.cooperative_matrix.*` has not been run (no deqp build fitted next to the NVR); the standalone test and ggml's 1106 MUL_MAT cases cover the layouts, both element types, mixed precision and tile composition, not `cmat_extract/insert` element order, per-element ops or robustness.
2. **Vectorised tile loads/stores**: the lowering emits two scalar loads per lane per tile; row-major pairs are adjacent (one 32-bit load for f16, 64-bit for f32) and `nir_opt_load_store_vectorize` did not merge them. Halving the shared-memory instruction count per matrix op is the first thing to try, and a transposed-register form (`(AB)^T = B^T A^T`) would do the same for column-major operands.
3. **ggml-side tile tuning**: ggml picks warptile sizes by vendor and Honeykrisp lands in the generic bucket; a ~20-cycle 8x8x8 op wants more accumulator tiles per warp than NVIDIA-tuned settings, and the FA coopmat1 path should be benchmarked against `fa=0` per model.
4. Add the D-over-A/B constraint to `agx_validate` (post-RA) so it cannot regress silently; audit `agx_pressure_schedule` latency for the op; state the reverse-engineered encoding provenance (no CTS run) in any upstream MR.
