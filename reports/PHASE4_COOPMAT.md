# Phase 4: VK_KHR_cooperative_matrix for Honeykrisp on the M1 (G13G) -- feasibility, design, prototype

Date: 2026-09-06. Host: a Mac mini M1 (T8103, GPU G13G B1, 8 cores, 32-wide SIMD), Fedora Asahi 44, kernel 7.1.6-400.asahi, system Mesa 26.1.8 (untouched), live Frigate NVR + `frigate-detector.service` on the system driver throughout.
Working dir: `~/gpu-opt/coopmat/` (`mesa/` = shallow clone of `aquarat/mesa` `local-deploy` @ `d105715` on a new local branch `coopmat`; `applegpu/` = dougallj's disassembler; `mfa/` = philipturner/metal-flash-attention; `build-in-container.sh` + `build-logs/`; `install/` = private driver prefix; `tests/`). This file is written incrementally; sections are appended as gates are passed.

Constraints honoured: userspace only, nothing installed under `/usr`, the private driver is loaded per process via `VK_DRIVER_FILES` (pattern from `frigate/research/vk-dispatch/with-patched.sh`), builds under `--memory 6g --cpus 3` + `nice`, GPU runs kept short, `dmesg` and the detector's fallback count checked after each.

## 0. TL;DR

* **Gate 1 (ISA): pass.** The G13 has `simd_matrix_fmadd16/32`, an 8x8x8 SIMD-group multiply-accumulate; its encoding is in dougallj's `applegpu` (derived on an M1) and the lane layout is the one Metal's `simdgroup_matrix` uses (metal-flash-attention `morton_order`, Apple patent US11256518B2). Two elements per lane, register pairs for every operand, f16 or f32; **mixed f16 x f16 + f32 is supported natively** (found here: per-operand size flags are honoured, the result accumulates in true f32). No integer variant. It is not a separate matrix unit: it runs on the FMA ALUs, so on the M1 the win is utilisation, not extra FLOPS.
* **Gate 2 (design): pass.** panvk's cooperative-matrix lowering is a direct template; the AGX operand encoding coincides with the generic ALU layout so the packer needed no special case; the pass is ~370 lines, the compiler side ~40.
* **Gate 3 (prototype): done and correct.** `VK_KHR_cooperative_matrix` is advertised by a private Honeykrisp build (fork `local-deploy` @ `d105715` + 3 commits on branch `coopmat`, `patches/mesa/000[1-3]-*.patch`), 16x16x16 and 8x8x8 in f16/f16/f32, f16/f16/f16 and f32/f32/f32. Bit-exact on a standalone test in every shape/type/layout; ggml's `test-backend-ops` passes **1106/1106 MUL_MAT** and the FLASH_ATTN_EXT subsets that exercise the coopmat path (the only failures left are q5_0/q5_1 K/V cases that fail identically without coopmat). One real bug was found and fixed on the way: the register allocator must not place D over A or B of the multi-cycle instruction (D over C is fine).
* **Numbers (this M1, NVR live):** ggml `llama-bench` pp512 Q8_0 **783 -> 1055 t/s (+35 %, ~1.04 TFLOPS)**, F16 757 -> 1016 t/s; ViT-H-14-378 CLIP **1874 -> 1505 ms (1.25x)** without flash-attention, 1776 -> 1578 ms (1.13x) with it; cosine vs ORT unchanged (0.9987-0.9991). Token generation unchanged (memory-bound). This is a first, unoptimised lowering (scalar per-element tile loads); the ceiling analysis (Apple's own GEMMs reach ~80 % of FMA peak on this instruction, ggml is now at ~40 %) says another ~1.5-2x is available from vectorised tile loads and ggml-side tile tuning.
* Nothing system-wide changed; the live detector logged 0 fallbacks and no GPU fault appeared in `dmesg` during any run. The private driver is used only through `coopmat/with-coopmat.sh`.

## 1. Gate 1 -- does the G13 ISA have SIMD-group matrix instructions, and are the encodings known?

**Yes on both counts.** The instruction exists on the M1 (Metal `simdgroup_matrix` / `simdgroup_multiply_accumulate` is available from Apple GPU family 7 = A14/M1), and its encoding and register layout are public reverse-engineering results; nothing needs to be recovered from the Mac.

### 1.1 Sources

| source | what it gives | notes |
|---|---|---|
| dougallj/applegpu `applegpu.py` (`SimdMatrixFMadd32InstructionDesc`, `SimdMatrixFMadd16InstructionDesc`, commit `70738db` "Add simdgroup matrix operations", 2022-10-30, PR #17) | **instruction encoding**, both variants, operand fields, "paired" register semantics | Reverse-engineered on Dougall Johnson's M1 (the whole disassembler targets G13 first). No emulator (`exec`) for the op -- semantics are documented only through Metal. Local copy: `coopmat/applegpu/applegpu.py:4296-4335`. |
| philipturner/metal-flash-attention `GEMMHeaders.swift` `simdgroup_matrix_storage::morton_order()` (cites Apple patent US11256518B2) | **register/lane layout** of an 8x8 tile in a 32-lane SIMD-group: which two elements each lane holds | Used in production Metal kernels (MFA, also MLX's `steel` GEMM and llama.cpp's Metal backend load/store tiles through the same `thread_elements()` layout). Local copy: `coopmat/mfa/Sources/FlashAttention/GEMM/GEMMHeaders.swift:553-567`. |
| philipturner/metal-benchmarks README ("Apple's 'tensor core' is the `simdgroup_matrix` instruction, which decreases register pressure and improves ALU utilization in existing FP32 pipelines") | **throughput model**: the instruction is *not* a separate matrix unit on M1; it runs on the FP32/FP16 FMA ALUs | Table "Matrix FFMA32 per core-cycle": A14 ~56.9, A15+/M1+ 101.7 (vs 128 FMA/core-cycle scalar peak, i.e. ~80 % of peak achieved in a GEMM kernel); Matrix FFMA16 102.5 (M1 has no fp16 rate doubling). |
| Metal Shading Language spec (`simdgroup_half8x8`, `simdgroup_float8x8`, `simdgroup_multiply_accumulate(d,a,b,c)`, `thread_elements()` = `vec<T,2>`) | **shape and types**: 8x8 tiles, f16 or f32, all four operands same type in the public API; 2 elements per thread (64 elements / 32 lanes) | int8 / bf16 simdgroup matrices do not exist on family 7 (bf16 arrived with M3/family 9). |
| Mesa `src/asahi/compiler/` (fork @ `d105715`) | **nothing yet**: no opcode, no TODO, no `cmat` handling anywhere under `src/asahi/` (`grep -ri "simdgroup\|coopmat\|cmat\|matrix"` finds only unrelated hits) | The SIMD family the op belongs to *is* modelled: `agx_opcodes.py` has `quad_/simd_ shuffle/shuffle_xor/up/down`, `quad_/simd_ reduce/prefix` (opcode byte `0b01101111`), packed by the generic `agx_pack_alu`. |
| Honeykrisp extension list (`hk_physical_device.c`) | `KHR_cooperative_matrix` absent; `KHR_shader_float16_int8`, `KHR_16bit_storage`, `KHR_shader_subgroup_extended_types`, `subgroupSize = 32`, `computeFullSubgroups` present (all prerequisites for ggml's `coopmat` path) | Upstream: only a user wish-list issue is referenced in the Mesa 25.2.0 release notes ("Request for coop_matrix and bfloat16 (ML/AI related) VK exts ... for Asahi HoneyKrisp driver (if/when possible/supported in HW)"); no MR, no branch. `dougallj/applegpu` issue #44 (SIMD futures / async copies on A16+) is about later chips. |

Mac route: not needed and in any case not available in this session (no SSH access to the Mac from here). If a Metal disassembly is ever wanted for cross-checking, the recipe is: `xcrun -sdk macosx metal -c t.metal -o t.air && xcrun metallib t.air -o t.metallib`, then `applegpu/compiler_explorer.py`/`disassemble.py` on the extracted binary here.

### 1.2 The instructions

Two 8-byte instructions in the SIMD-op family (opcode low byte `0x6f`, the same family as `simd_shuffle`/`simd_reduce`), one per element size; `applegpu.py`:

```
simd_matrix_fmadd32   bits[6:0]=0x6f, bit15=0, bit26=1, bit27=1, bit63=1, size 8
simd_matrix_fmadd16   bits[6:0]=0x6f, bit15=0, bit26=0, bit27=1, bit63=1, size 8
   D = PairedALUDst   : Dt bits[8:7] (bit7 cache hint, bit8 = 32-bit), D bits[14:9], Dx bits[61:60]
   A = PairedFloatSrc : A bits[21:16], At bits[25:22] (hint/size flags), Ax bits[59:58], modifier Am bits[53:52]
   B = PairedFloatSrc : B bits[33:28], Bt bits[37:34], Bx bits[57:56], Bm bits[39:38]
   C = PairedFloatSrc : C bits[45:40], Ct bits[49:46], Cx bits[55:54], Cm bits[51:50]
   semantics (Metal): D = A * B + C on 8x8 tiles held collectively by the 32 lanes
```

* **"Paired" operands**: every operand is *two* consecutive registers per lane -- the two matrix elements a lane owns. The size flag in the operand field selects the element type, and the register named is the first of the pair: a 32-bit-flagged `r4` means the pair `r4_r5` (2 x f32), a 16-bit-flagged `r4l` means `r4l_r4h`, i.e. the whole 32-bit `r4` holding 2 x f16 (dougallj: "R0L -> R0, or R0 -> R0_R1"). This is exactly Metal's `thread_elements()` = `vec<T,2>`.
* The operand fields sit where the generic AGX ALU encoding puts them (D at 7/9 + ext 60, src s at 16+12s + ext 58-2s), so Mesa's `agx_pack_alu` produces the right bits without a special case as long as no float modifiers are set (src0's modifier slot at bits 26-27 is repurposed as the size-select/constant bits here; `Am` moved to 52-53).
* Element types: f16 (`fmadd16`) and f32 (`fmadd32`). Whether the size flags may be *mixed* (f16 A/B with f32 C/D -- the shape ggml prefers) is not documented anywhere; dougallj's comment says each operand "accept[s] either 16-bit or 32-bit", which suggests the hardware reads the per-operand flags. This is the one unknown, and it is cheaply testable on this machine once the opcode exists in the compiler (section 3). No integer variant exists on G13.
* Cost model: ~20 cycles per instruction has been quoted in the community for the 8x8x8 MAC (512 FMAs = 16 cycles of a 32-lane FMA pipe, plus overhead), consistent with metal-benchmarks' finding that a Metal GEMM built on it reaches ~80 % of the chip's FMA peak. So on **M1 the upside is utilisation, not extra FLOPS**: the ceiling is the same 2.6 TFLOPS (8 cores x 128 FMA x 2 x 1.278 GHz) that scalar FMAs have; ggml's scalar path is at 0.77 TFLOPS (30 %), and Apple's own GEMMs on this instruction reach ~2.0-2.1 TFLOPS (80 %). Realistic target for ggml's `coopmat` shader on this hardware: 1.5-2.5x on large GEMMs, with the instruction removing the register-blocking/shuffle overhead that keeps the scalar shader at 30 %.

### 1.3 The register layout (which lane holds what)

From `morton_order()` (MFA, patent US11256518B2), for lane `l` (0..31) of the SIMD-group, the lane holds elements `(row, col)` and `(row, col+1)`:

```
quadrant  q  = l >> 3          (0: rows 0-3/cols 0-3, 1: rows 0-3/cols 4-7, 2: rows 4-7/cols 0-3, 3: rows 4-7/cols 4-7)
row          = (q >> 1) * 4 + ((l >> 1) & 3)
col          = (q & 1)  * 4 + (l & 1) * 2       (+0 and +1 for the two elements)
```

i.e. the 8x8 tile is split into four 4x4 quadrants of 8 lanes each; inside a quadrant lane pairs walk the rows, and each lane owns two horizontally adjacent elements. All four operands (A, B, C, D) use the *same* layout -- Metal's `simdgroup_load`/`simdgroup_store` do the row-/column-major addressing and the transpose in the address computation, not in the instruction. That makes the Vulkan `cmat_load`/`cmat_store` lowering pure address arithmetic (row-major: `base + row*stride + col`, column-major: `base + col*stride + row`, per element), with the two elements of a lane adjacent in memory for row-major A/C and strided for column-major.

### 1.4 Gate 1 verdict

**Pass.** Encoding, operand semantics and lane layout are known from public, M1-derived sources; the only unverified detail (mixed f16/f32 operand flags) is testable here in minutes once the compiler can emit the instruction. Upstream Mesa has no code for it, so anything built here is new.

## 2. Gate 2 -- design: mapping `VK_KHR_cooperative_matrix` onto `simd_matrix_fmadd`

### 2.1 How the three existing Mesa implementations do it (what was reusable)

| driver | lowering | notes |
|---|---|---|
| radv (`src/amd/vulkan/nir/radv_nir_lower_cooperative_matrix.c`, 1510 lines) | cmat types -> per-lane vectors, `cmat_muladd_amd` -> WMMA/SWMMAC | Handles many shapes, int8/fp8, conversions between operand layouts; far more than needed here. |
| nvk (`src/nouveau/compiler/nak_nir_lower_cmat.c`, 1118 lines) | NV-specific: `cmat_muladd_nv`, `cmat_load_shared_nv`, `cmat_mov_transpose_nv` | Register layouts differ per use (A/B/C), so it needs explicit transposes. |
| **panvk** (`src/panfrost/vulkan/panvk_nir_lower_cooperative_matrix.c`, 540 lines) | generic `nir_lower_cooperative_matrix_flexible_dimensions` splits every matrix into hardware-sized tiles, then a ~400-line pass maps `cmat_*` intrinsics onto per-lane vectors and one hardware intrinsic (`cmat_muladd_pan`) | Closest match: a single small hardware tile (4x4 there, 8x8 here), every use in the same per-lane layout, memory accesses expressed as derefs so the existing explicit-IO lowering handles SSBO/shared/global. **Used as the template.** |

Common infrastructure that already exists upstream and is used unchanged: SPIR-V `CooperativeMatrixKHR` -> NIR `cmat_*` intrinsics + `glsl_cmat` types (`vtn_cmat.c`), `nir_lower_cooperative_matrix_flexible_dimensions` (`nir_lower_cooperative_matrix.c`, Red Hat 2025) for splitting arbitrary shapes at a granularity, `vk_features`/`vk_properties` fields for the extension, and the SPIR-V capability derived automatically from the `cooperativeMatrix` feature (`vk_nir.c`).

### 2.2 The AGX mapping

* **Hardware tile** = 8x8x8, subgroup scope, 32 lanes, 2 elements per lane, one layout for all uses (section 1.3). So `coopmat<T, gl_ScopeSubgroup, 8, 8, Use>` becomes a per-lane `vec2<T>`; `cmat_length` = 2.
* **Shapes advertised** (`hk_GetPhysicalDeviceCooperativeMatrixPropertiesKHR`): 16x16x16 and 8x8x8, each in f16/f16/f32/f32 (A/B/C/Result), f16/f16/f16/f16, f32/f32/f32/f32. The 16x16x16 entries exist because that is what applications (ggml, CTS lists, most GEMM libraries) probe first; the pass splits them into 8x8 tiles with `m_gran = n_gran = k_gran = 8`, so a 16x16x16 muladd is 8 hardware instructions on 4+4+4 tiles.
* **Types**: no integer shapes (the ISA has none on G13; ggml's MMQ/int-dot path stays off, as it is today), no bf16.
* **Mixed precision** f16 x f16 + f32: the operand size flags are per operand in the encoding; bit 26 (fmadd16 vs fmadd32) is set from the accumulator size. Verified natively on the hardware (section 3.2); `HK_CMAT_WIDEN=1` selects the alternative (widen A and B to f32 elementwise, then the all-f32 form) for experiments.
* **Loads/stores**: per element, `base + outer*stride + inner` with the lane's `(row, col+p)` from the layout formula; row-major A/C reads are two adjacent elements per lane (NIR's load/store vectoriser can merge them into one 32-bit (f16) or 64-bit (f32) access when alignment allows -- left to the existing passes, not hand-vectorised in this prototype). Column-major is the same formula with the roles swapped, so `gl_CooperativeMatrixLayoutColumnMajor` B operands (ggml's choice) cost nothing extra.
* **Other cmat ops**: construct = replicate, extract/insert = vector ops on the 2-vector, convert = elementwise (same layout for every type -- simpler than panvk, which has to repack fp16), unary/binary/scalar ops = elementwise ALU.
* **Compiler (`agx`)**: one new opcode `simd_matrix_fmadd` (`agx_opcodes.py`), packed by the *generic* ALU packer because the operand fields coincide with the ALU layout (section 1.2): dest and sources are `agx_index` vectors of 2 channels (f16: two 16-bit halves = one 32-bit register; f32: two 32-bit registers), which is exactly the "paired" register semantics. Additions: bit 26 from the accumulator size in `agx_pack_alu`; `agx_read_registers`/`agx_write_registers` cases so liveness/RA see both halves of every pair; the emitter (`cmat_muladd_agx` -> instruction + cached split of the 2-channel result); `is_float = false` so the optimiser never folds `fneg/fabs` into an operand (src0's modifier bits are the instruction's constant bits). No new register class, no scheduling constraints found: the instruction behaves as a normal (long-latency) ALU op, the register allocator's natural alignment (vectors aligned to their size) satisfies it, and the existing `wait` insertion for the loads feeding it is enough.
* **Driver (`hk`)**: extension + feature + property + the properties query; the pass runs first thing in `hk_preprocess_nir_internal` for compute shaders with `has_cooperative_matrix`, before `nir_lower_vars_to_ssa` meets the cmat-typed variables and before explicit-IO lowering consumes the derefs it emits.

### 2.3 Effort estimate vs actual

Estimated at the gate: ~6-8 hours (build environment 1-2 h, compiler 1 h, pass 2 h, plumbing 0.5 h, debugging unknown). Actual: the build environment took the longest (no llvm/clang headers on the host; solved with a Fedora 44 container, `build-in-container.sh`, ~10 min per full build, ~1 min incremental), the code was ~480 lines, and the first hardware run was bit-exact. **Gate 2: pass -- proceeded to the prototype.**

## 3. Prototype

### 3.1 What was built (branch `coopmat` in `coopmat/mesa`, one commit on top of `local-deploy` @ `d105715`)

| file | change |
|---|---|
| `src/compiler/nir/nir_intrinsics.py` | `cmat_muladd_agx(vec2 a, vec2 b, vec2 c) -> vec2` (result bit size = c's; subgroup semantics) |
| `src/asahi/compiler/agx_opcodes.py` | `simd_matrix_fmadd`: encoding `0x800000000800006f` (opcode `0x6f`, bit 27, bit 63), 8 bytes, 3 sources |
| `src/asahi/compiler/agx_pack.c` | bit 26 from the accumulator size; asserts dest size == C size |
| `src/asahi/compiler/agx_validate.c` | register-pair read/write counts for the new op |
| `src/asahi/compiler/agx_compile.c` | emit `cmat_muladd_agx` |
| `src/asahi/vulkan/hk_nir_lower_cooperative_matrix.c` (new, 373 lines) | the lowering described in 2.2 |
| `src/asahi/vulkan/hk_shader.c/.h`, `meson.build` | run the pass |
| `src/asahi/vulkan/hk_physical_device.c` | `KHR_cooperative_matrix`, `cooperativeMatrix` feature, `cooperativeMatrixSupportedStages = COMPUTE`, `hk_GetPhysicalDeviceCooperativeMatrixPropertiesKHR` (6 shapes) |

Build: `coopmat/build-in-container.sh setup|compile|build` (Fedora 44 container `hk-coopmat-build`, `--memory 6g --cpus 3`, `nice`, `ninja -j3`, source bind-mounted at its host path; deps = the fork's own spec's BuildRequires). Output: `coopmat/install/lib64/libvulkan_asahi.so` + `install/share/vulkan/icd.d/asahi_icd.aarch64.json`; `coopmat/with-coopmat.sh CMD` runs CMD against it via `VK_DRIVER_FILES`. Nothing installed on the host; `ldd` resolves entirely against system libraries. Identity: `driverInfo = Mesa 26.3.0-devel (git-d105715f01)`, driverVersion 26.2.99 (vs 26.1.8 for the system driver).

### 3.2 Correctness

Standalone test `coopmat/tests/cmat_test.c` + `cmat.comp` (GLSL `GL_KHR_cooperative_matrix`, one 32-lane workgroup computes `C = A*B + C0` for one tile, specialisation constants M/N/K, integer-valued inputs exactly representable in f16 so the reference is exact; `CMAT_PREC=1` switches to an accumulate-precision probe). Every run under `with-coopmat.sh`, shader cache disabled where an env var changes code generation, `dmesg` and the detector's fallback count checked after each batch (nothing: the only kernel line in the window is a pre-existing `avd ... Frame processing timed out` from the NVR's video decoder at 69654 s, before any GPU work of this phase).

| shape | A/B | C/D | B layout | result |
|---|---|---|---|---|
| 8x8x8 | f32 | f32 | row-major | **OK, max abs err 0** (first hardware run of the encoding) |
| 8x8x8 | f16 | f16 | row-major | OK, err 0 |
| 8x8x8 | f16 | f32 | row-major | OK, err 0 -- native mixed encoding (`simd_matrix_fmadd ^r3l...r3h, ^r1l...r1h, ^r4...r5`: 16-bit-flagged pairs for A/B, 32-bit pair for C/D, bit 26 set) |
| 8x8x8 | f16 | f32 | row-major, `HK_CMAT_WIDEN=1` (A/B widened, all-f32 instruction) | OK, err 0 |
| 8x8x8 | f16 | f32 | column-major (B stored transposed) | OK, err 0 |
| 16x16x16 | f32 | f32 | row-major | OK, err 0 (8 instructions on 4+4+4 tiles) |
| 16x16x16 | f16 | f32 | row-major | OK, err 0 |
| 16x16x16 | f16 | f16 | row-major | OK, err 0 |
| 16x16x16 | f16 | f32 | column-major | OK, err 0 |
| 8x8x8, `CMAT_PREC=1` (A=B=0.125, C0=4096, expect 4096.125) | f16 | f32 | | **4096.125 exactly** -> the mixed instruction accumulates in f32 |
| 8x8x8, `CMAT_PREC=1` | f16 | f16 | | 4096 (err 0.125): f16 accumulator rounds, as it must |

Disassembly (`AGX_MESA_DEBUG=shaders`, `tests/*_shaders.txt`): the f32 tile product is `r4...r5 = simd_matrix_fmadd ^r4...r5, ^r6...r7, ^r8...r9` after two `wait`s for the loads; the register allocator kept every pair contiguous and aligned without any change.

ggml oracle: `test-backend-ops -b Vulkan0 -o MUL_MAT` on the private driver reports `matrix cores: KHR_coopmat` and passes **1106/1106** (all f32/f16/bf16/quantised MUL_MAT shapes, compared against the CPU backend), both with the widened path and with the native mixed path (`tests/tbo_mul_mat_coopmat*.txt`). ggml picked the first f16/f16/f32/f32 entry, so its `mul_mm` coopmat shader runs with TM = TN = TK = 16, which the pass turns into 8 hardware instructions per 16x16x16 step.

Not run: the Vulkan CTS `dEQP-VK.compute.cooperative_matrix.*` group (no deqp build on this machine; building one is a multi-GB job that the memory budget here does not allow alongside the NVR). The standalone test plus ggml's 1106 MUL_MAT cases cover the load/store layouts, both element types, mixed precision, both memory layouts and tile composition, but not e.g. `cmat_extract/insert` element order, per-element ops, or robustness edge cases.

### 3.3 The bug the oracle found (and the padded-stride red herring)

The first full ViT-H run *with* flash-attention came back with cosine 0.66-0.89 although every MUL_MAT case passed, and `test-backend-ops -o FLASH_ATTN_EXT` failed 1254/5173 cases with ~0.02 errors. ggml's `flash_attn_cm1.comp` differs from the GEMM shader in staging K/V/P through **shared memory with a padded row stride** and loading A-use tiles column-major. A padded-stride shared-memory variant of the standalone test (`tests/cmat2.comp -DSTAGE`, `cmat3.comp`) reproduced it in 30 lines: one 8x8 output block wrong, all input tiles individually right (the other blocks that use them were correct). The NIR addressing was right; the difference was in the *register assignment*: the failing block's instruction was `r6...r7 = simd_matrix_fmadd r6l...r6h, r10l...r10h, r14...r15` -- the destination allocated over the A operand because the allocator "early-kills" a source whose last use is the current instruction. Accumulating in place (D over C, `r4...r5 = simd_matrix_fmadd ^r3l...r3h, ^r1l...r1h, ^r4...r5`) had always worked; D over A/B corrupts, as expected for a multi-cycle SIMD-group op that keeps reading its inputs while writing outputs. Fix: `can_kill_early()` in `agx_register_allocate.c` refuses early-kill for sources 0 and 1 of `simd_matrix_fmadd` (commit `2395e4e`, patch 0003). After it: every standalone variant passes, the ViT-H flash-attention path is back to cosine 0.9987-0.9991, and the FLASH_ATTN_EXT subsets `hsk=64/kv=1024` (3/3), `hsk=80` (112/112, ViT-H's head size) pass; `hsk=72,kv=113` still shows 12 failures, all `type_K/V = q5_0/q5_1`, and exactly the same 12 fail with `GGML_VK_DISABLE_COOPMAT=1` on the same driver -- pre-existing, not from this work. (One variant, `cmat2_stagea`, kept failing after the fix; it was a broken test -- a leftover `c4[i] = sc[i]` copy-out loop overwriting the result -- not a driver problem. Noted so nobody chases it again.)

The full FLASH_ATTN_EXT list (5173 cases, ~10 min of GPU) was not re-run after the fix: the coordinator asked for <= 30 s single-instance GPU runs because the live detector was being starved by concurrent benchmarking; the subsets above were chosen to cover the previously failing shapes and ViT-H's.

### 3.4 Numbers

All on this M1 with the NVR live, `nice`, `systemd-run --scope -p MemoryMax=4-5G`, detector `inference_speed` checked < 60 ms before every GPU run (`coopmat/gpu_ok.sh`); raw output in `coopmat/results_bench.txt`, `coopmat/bench.log`, `ggml/out/p4_*`. Three arms: **coopmat** = private driver; **nocoopmat** = same private driver with `GGML_VK_DISABLE_COOPMAT=1` (isolates the extension from the fork's other compiler work); **stock** = system Mesa 26.1.8.

`llama-bench`, Qwen2.5-0.5B, 3 repetitions (t/s; ~0.99 GFLOP/token):

| test | stock 26.1.8 | private, nocoopmat | **private, coopmat** | gain vs nocoopmat |
|---|---:|---:|---:|---:|
| Q8_0 pp512, fa=0 | 778.6 | 783.6 | **991.1** | 1.26x |
| Q8_0 pp512, fa=1 | 782.5 | 631.1 (scalar FA path, noisy) | **1046.6** / 1055.0 (final build) | 1.35x vs stock (~**1.04 TFLOPS**) |
| F16 pp512, fa=0 | 782.8 | 763.0 | **952.1** | 1.25x |
| F16 pp512, fa=1 | 756.9 | 774.2 | **1046.1** / 1016.2 (final build) | 1.34x |
| Q8_0 pp128 / pp1024, fa=1 | 620 / 711 | 641 / 709 | **878 / 993** | 1.37x / 1.40x |
| Q8_0 tg128 | 43.7-52.3 | 40-47 | 54 | memory-bound, unchanged within noise |
| F16 tg128 | 38.7-40.8 | 39.9-40.3 | 38-43 | unchanged |

ViT-H-14-378 CLIP (`tools/run_clip.sh`, 4 images, 1 warm-up + 3 iterations unless noted, ms median-of-medians, cosine vs ORT fp32 min/mean):

| configuration | latency | cosine | vs Phase 2 stock |
|---|---:|---|---:|
| Phase 2, stock, f16, fa=0 / fa=1 | 1873.8 / 1775.8 | 0.9985 / 0.9990 | 1 |
| private nocoopmat, f16, fa=1 | 1836.6 | 0.99852 / 0.99903 | 0.97x |
| private nocoopmat, Q8_0, fa=1 | 1782.5 | 0.99857 / 0.99907 | 1.0x |
| **private coopmat, f16, fa=0** (1+1 it) | **1505.0** | 0.99867 / 0.99907 | **1.25x** (669 GFLOPS effective, 26 % of peak) |
| private coopmat, f16, fa=1 (final build, 1+1 it) | 1578.2 | 0.99866 / 0.99905 | 1.13x |
| private coopmat, Q8_0, fa=1 (pre-fix build; GEMM path identical) | 1590.6 | (FA path was wrong then; 0.66) | 1.12x |

Reading: the GEMMs got ~1.3x, the whole ViT-H graph 1.13-1.25x (attention, LayerNorm, GELU and the 730-token softmax are unchanged), and ggml's coopmat1 flash-attention shader is *slower* than its non-FA path once the GEMMs use the matrix instruction, so for ViT-H `fa=0` is now the better setting. The 0.77 -> 1.04 TFLOPS on pp512 is 40 % of the chip's 2.6 TFLOPS FMA peak; Apple's Metal GEMMs on the same instruction reach ~80 %.

## 4. What remains

1. **Vectorised tile loads/stores.** The lowering emits two scalar loads per lane per tile; for row-major A/C the pair is adjacent (one 32-bit load for f16, one 64-bit for f32) and ggml's shaders read A row-major and B column-major from shared memory. Emitting `vec2` accesses directly (or checking why `nir_opt_load_store_vectorize` did not merge them) halves the shared-memory instruction count per matrix op (16 loads per 8 `simd_matrix_fmadd` today). This is the first thing to try; expected to be worth a good part of the remaining gap.
2. **A transposed-register alternative for column-major operands**: since `(AB)^T = B^T A^T`, keeping every tile transposed in registers turns column-major loads into adjacent pairs -- a per-shader choice the pass could make when most loads are column-major.
3. **ggml-side tuning**: ggml picks warptile sizes by vendor; Apple/Honeykrisp falls into the generic bucket. A 20-cycle 8x8x8 op wants more accumulator tiles per warp than NVIDIA-tuned settings; also the FA coopmat1 path should be benchmarked against `fa=0` per model (it lost here).
4. **Verification**: run `dEQP-VK.compute.cooperative_matrix.*` (needs a deqp build, ~several GB; not done here) and the full FLASH_ATTN_EXT list once the GPU is not shared with a live service; add the D-over-A/B constraint to `agx_validate` (post-RA check) so it can never regress silently; audit `agx_pressure_schedule` latency for the op (treated as a plain ALU op now).
5. **Shapes**: 16x8/8x16 style shapes, `cmat_extract/insert` element order (implementation-defined, untested), robustness (`cooperativeMatrixRobustBufferAccess = false` as advertised), 32x32 via splitting (free with the generic splitter, just not advertised).
6. **Upstreaming**: the three patches are self-contained on the fork; for upstream Mesa the same code applies to `main` (nothing depends on the fork's other commits), but the encoding provenance (reverse-engineered, no CTS run) should be stated in the MR, and the M1-only `is_float = false`/no-modifier rule documented in `agx_opcodes.py`.

## 5. Artefacts

* Report: this file. Branch: `coopmat/mesa` `coopmat` (3 commits over `local-deploy` @ `d105715`, not pushed). Patches: `patches/mesa/0001-asahi-hk-prototype-VK_KHR_cooperative_matrix-on-the-.patch`, `0002-asahi-hk-use-the-native-mixed-f16-x-f16-f32-matrix-F.patch`, `0003-agx-never-allocate-a-simd_matrix_fmadd-destination-o.patch`.
* Driver: `coopmat/install/lib64/libvulkan_asahi.so` (+ ICD json), `coopmat/with-coopmat.sh`, `coopmat/gpu_ok.sh`, `coopmat/build-in-container.sh` (`setup|compile|build`; the Fedora 44 container `hk-coopmat-build` is left in place for incremental builds -- `docker rm -f hk-coopmat-build` removes it), `coopmat/bench.sh`.
* Tests: `coopmat/tests/cmat_test.c` + `cmat*.comp`/`.spv` (standalone), `tbo_mul_mat_coopmat*.txt`, `tbo_fa_*.txt` (ggml oracles), `*_shaders.txt` (disassemblies).
* Sources consulted: `coopmat/applegpu` (dougallj), `coopmat/mfa` (philipturner), metal-benchmarks README, Mesa `src/panfrost/vulkan/panvk_nir_lower_cooperative_matrix.c`, `src/compiler/nir/nir_lower_cooperative_matrix.c`, radv/nak equivalents.
