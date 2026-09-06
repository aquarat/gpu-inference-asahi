# Phase 5: feeding the matrix instruction -- vectorised tile loads (driver) and coopmat shader tuning (ggml)

Date: 2026-09-06. Host: a Mac mini M1 (T8103, GPU G13G B1, 8 cores, 32-wide SIMD), Fedora Asahi 44,
kernel 7.1.6-400.asahi, system Mesa 26.1.8 (untouched), live Frigate NVR + `frigate-detector.service`
and `immich-ml-ggml.service` on the GPU throughout. Follows `PHASE4_COOPMAT.md`.

Working dirs: `gpu-opt/coopmat/mesa` (branch `coopmat-vecload` = `coopmat` @ `71969e3` + 1 commit),
`gpu-opt/ggml/llama.cpp` (2 commits), `gpu-opt/coopmat/deploy-v2/` (new build, not deployed).

## 0. TL;DR

* **Part A (driver, vectorised tile load/store): implemented, correct, and *not* a win on this GPU.**
  A lane's two elements of a tile are adjacent in memory for the row-major orientation, so the pair can
  be one access instead of two. That is now `HK_CMAT_VECLOAD=1` (two-component access) or `=2`
  (single 32-bit access). Both are bit-exact everywhere and cut ggml's `mul_mm` inner-loop threadgroup
  loads from 24 to 20 per k step (A tiles only; B is column-major and stays scalar), producing a
  *smaller* shader (2813 vs 2990 instructions, 147 vs 155 GPRs, same 640-thread occupancy). On the ggml
  shader as shipped it is worth **+5.6 %** (1002 -> 1058 t/s). But once the shader's redundant tile
  loads are removed (Part B) it is **-12 %** (1134 -> 997 t/s) and -15 % on ViT-H. Cause not identified;
  mode 2 behaves identically, so it is not the access width. **Shipped off by default.**
* **Part B (ggml): the real win was a loop-nest bug, not tile sizes.** ggml's coopmat1 `mul_mm` inner
  loop re-issues the same `coopMatLoad` `cms_per_row` times per k step (16 tile loads where 8 suffice
  for the 128x128x32 warptile). AGX does not eliminate those (its `nir_opt_load_store_vectorize` does
  not cover `nir_var_mem_shared`, and enabling it there costs 10 % elsewhere -- tried and reverted).
  Hoisting the A tiles into a register array: **llama-bench pp512 Q8_0 1002 -> 1134 t/s (+13 %),
  F16 1017 -> 1110, ViT-H-14-378 CLIP 1505 -> 1375 ms.**
* **Warptile shape barely matters here**: splitting the 128x128 workgroup tile 64x64 / 32x64 / 64x32 /
  32x32 between warps moves pp512 by < 3 %. Added `GGML_VK_WT_*` env overrides so the sweep is
  repeatable.
* **Targets not reached.** pp512 Q8_0 **1134 t/s** (target ~1300), ViT-H CLIP **1375 ms** (target ~1200).
  ~1.13 TFLOPS, 43 % of the chip's 2.6 TFLOPS FMA peak. Section 5 says where the rest goes: for a
  Q8_0 GEMM the on-the-fly dequantisation + threadgroup staging is now comparable in instruction count
  to the matrix work itself, and that is not something tile tuning can remove.
* **Correctness unchanged**: standalone test bit-exact in all three driver modes, `test-backend-ops -o
  MUL_MAT` **1106/1106** with the ggml change (and 1106/1106 with `HK_CMAT_VECLOAD=1`),
  `FLASH_ATTN_EXT` `hsk=64,kv=1024` 6/6 and `hsk=80` 112/112, `hsk=72,kv=113` 168/216 with exactly the
  same 48 q5_0/q5_1 failures as `GGML_VK_DISABLE_COOPMAT=1` on the same driver (pre-existing).
  ViT-H cosine vs the ORT fp32 reference unchanged: min 0.99867, mean 0.99907.
* **Recommendation: do not redeploy the driver** (`deploy-v2/` is behaviourally identical to `deploy/`
  with the default settings, so there is nothing to gain). **Do rebuild Immich's ggml with
  `patches/ggml/0001`** if its CLIP latency matters -- that is where the ~9 % is.
* Health: 0 detector fallbacks and no GPU fault in `dmesg` across every run of this phase.

## 1. What the lowering emitted per `coopMatLoad`, before

`hk_nir_lower_cooperative_matrix.c` maps a `coopmat<T, subgroup, 8, 8, Use>` to a per-lane `vec2<T>`
(the Morton layout: lane `l` owns `(row, col)` and `(row, col+1)`, see Phase 4 §1.3). Larger shapes are
split into 8x8 tiles first, so a 16x16 tile is 4 hardware tiles.

`lower_cmat_load_store()` walked the two elements and emitted one access each:

```
for p in 0..1:
    col   = col0 + p
    outer = row_major ? row : col
    inner = row_major ? col : row
    idx   = outer * stride + inner          # stride scaled from pointee units to scalars
    load/store deref[idx]                   # one scalar element
```

So, per `coopMatLoad`/`coopMatStore` of one hardware 8x8 tile, per lane:

| use / layout | element type | accesses emitted | width | contiguous? |
|---|---|---|---|---|
| A (M x K), row-major | f16 | 2 | 16 bit | yes, `col0` and `col0+1` adjacent |
| A (M x K), row-major | f32 | 2 | 32 bit | yes |
| A, column-major | f16/f32 | 2 | 16/32 bit | no, one row stride apart |
| B (K x N), column-major (what ggml uses) | f16 | 2 | 16 bit | **no**, one row stride apart |
| B (K x N), row-major | f16 | 2 | 16 bit | yes |
| C/D accumulator, row-major | f32 | 2 | 32 bit | yes |
| C/D accumulator, column-major (what ggml uses) | f32 | 2 | 32 bit | **no** |

Packing: none. The two 16-bit values are `collect`ed into the 32-bit register the instruction wants as
its "paired" operand, and AGX's register allocator already placed them as `rNl` / `rNh` of the same
register, so there were no extra moves -- only the extra load instruction and its address arithmetic.
Nothing merged them later: `agx_optimize_nir` calls `nir_opt_load_store_vectorize` with
`modes = nir_var_mem_global | nir_var_mem_constant | nir_var_shader_temp`, and ggml stages its tiles in
`nir_var_mem_shared`.

Measured on ggml's `mul_mm` coopmat inner loop (post-RA AGX IR, f16 128x128x16 warptile, `cms_per_row =
cms_per_col = 2` per warp in that variant), per k step: **8 threadgroup loads for A, 16 for B, 16
`simd_matrix_fmadd`** -- 24 loads per 16 matrix ops.

## 2. Part A -- the vectorised path

`HK_CMAT_VECLOAD` (patch `patches/mesa/0004-*`):

* `0` (default): the scalar path above.
* `1`: for the row-major orientation only, one access of a 2-component vector of the element type at
  `row * stride + col0`.
* `2`: same, but a 4-byte-aligned 16-bit pair moves as one 32-bit scalar (`pack/unpack_32_2x16`).

Alignment. The address is computed in scalar elements because the row stride is arbitrary, so the
vector access is not naturally aligned in general. `col0` is always even, so the pair is 2*elem-aligned
whenever the pointee is a vector (`ptr_vec % 2 == 0`) -- which is ggml's case, `buf_a`/`buf_b` are
`f16vec2[]`. That is all the pass claims (`align_mul = elem_size` otherwise). It is enough: agx's
`nir_lower_mem_access_bit_sizes` turns a 4-byte f16 pair into a **two-component 16-bit** access
(`lload rNl_rNh, ..., i16, xy`) rather than back into two accesses, so even the unaligned case is a
single instruction. Verified with a deliberately odd-strided (21 f16) scalar threadgroup staging test.

Column-major keeps the scalar path -- the two elements are a row stride apart. There is no cheap fix
in the driver: making B's pair contiguous would need the staging buffer transposed (Part B, §4.3), and
the transpose identity `(AB)^T = B^T A^T` only swaps which operand is strided, it does not vectorise
both. A shuffle-based gather (one contiguous vec4 load per lane plus a cross-lane exchange) was
considered and rejected: it needs 4 lanes' worth of loads to feed 8 lanes plus two shuffles and a
select per lane, against the two scalar loads it would replace.

RA constraints: none new. The destination of a vector load is a register pair, which is exactly what
`simd_matrix_fmadd` wants for its operand; the Phase 4 `can_kill_early()` fix (D must not land on A or
B) still covers the hazard, and no new overlap is introduced (the load's destination may overlap its
own address register, which is what the scalar path already did).

### 2.1 Correctness

Standalone test (`coopmat/tests/cmat_test.c`), every case run at `HK_CMAT_VECLOAD` 0, 1 and 2, all
**bit-exact (max abs err 0)**:

| shape | A/B | C/D | staging | 0 | 1 | 2 |
|---|---|---|---|---|---|---|
| 16x16x16 / 8x8x8 | f16 | f32 | global, row-major | OK | OK | OK |
| 16x16x16 | f16 | f16 | global | OK | OK | OK |
| 16x16x16 | f32 | f32 | global | OK | OK | OK |
| 16x16x16 | f16 | f32 | global, B column-major | OK | OK | OK |
| 16x16x16 | f16 | f32 | shared `f16vec4`, padded stride | OK | OK | OK |
| 16x16x16 | f16 | f32 | shared scalar `f16`, stride 20 | OK | OK | OK |
| 16x16x16 | f16 | f32 | shared scalar `f16`, **stride 21 (odd)** (new test `cmat4_oddstride`) | OK | OK | OK |
| 16x16x16 | f16 | f32 | shared C staging | OK | OK | OK |

Two standalone cases fail identically in all modes and are test artefacts, not driver bugs, as in
Phase 4: `cmat2_gpad` needs `CMAT_APAD=20` (it passes with it), and `cmat2_stagea` has a leftover
copy-out loop that overwrites its result.

ggml: `test-backend-ops -b Vulkan0 -o MUL_MAT` = **1106/1106** with `HK_CMAT_VECLOAD=1`
(`tests/p5_tbo_mul_mat_vecload1.txt`) and 1106/1106 with the default 0 and the Part B shader change
(`tests/p5_tbo_mul_mat_ahoist.txt`).

### 2.2 Loads per k step, before and after

ggml `mul_mm`, f16 variant, `cms_per_row = cms_per_col = 2`, post-RA AGX IR of the inner loop
(`tests/p5_loop_{vec,novec}.txt`, extracted from `tests/p5_dis_*.txt`):

| | A loads | B loads | matrix ops | inner-loop IR lines |
|---|---:|---:|---:|---:|
| Phase 4 lowering, ggml as shipped | 8 (scalar) | 16 x cms_per_row (scalar) | 16 | -- |
| `VECLOAD=0`, ggml with §4.1 | 8 (scalar) | 16 (scalar) | 16 | 146 |
| `VECLOAD=1`, ggml with §4.1 | **4** (`i16, xy`) | 16 (scalar) | 16 | **124** |

Whole-shader `shaderdb` for the Q8_0 `mul_mm` pipeline: `VECLOAD=1` 2813 instructions / 147 GPRs /
640 threads / 0 spills; `VECLOAD=0` 2990 / 155 / 640 / 0. The vectorised shader is strictly smaller
on every counter the compiler reports.

### 2.3 ... and it is still slower

`llama-bench`, Qwen2.5-0.5B, pp512, `-r 3`, `MESA_SHADER_CACHE_DISABLE=1` in both arms (so the
env-dependent code generation is not cached), medians of the reported means:

| ggml `mul_mm` | `VECLOAD=0` | `VECLOAD=1` | `VECLOAD=2` |
|---|---:|---:|---:|
| as shipped (B reloaded per cm_row), Q8_0 fa=1 | 1001.6 | **1058.4** (+5.6 %) | -- |
| A hoisted (§4.1), Q8_0 fa=1 | **1134.3** | 997.1 (-12 %) | -- |
| B hoisted (§4.2), Q8_0 fa=1 | 1103.5 | 993.9 (-10 %) | 986.9 |
| B hoisted, Q8_0 fa=0 | 1035.2 | 930.7 | -- |
| B hoisted, F16 fa=1 | 1110.3 | 1007.3 | -- |
| A hoisted, ViT-H-14-378 CLIP f16 fa=0 (ms, lower better) | **1375.0** | 1583.6 | -- |

Reproducible to +-2 t/s across repeats, and the sign is the same for f16 and Q8_0 and with and without
flash attention, so it is the GEMM shader, not the FA shader. No explanation was found in the time
box: the vectorised shader has fewer instructions, fewer registers, the same occupancy, no spills, and
the same number of `wait`s; mode 2 (a genuinely different encoding, `i32, x` instead of `i16, xy`)
behaves the same, so it is not the access width or a threadgroup-bank effect (the bank pattern of the
A tile is conflict-free either way -- 32 distinct banks across the 32 lanes for the padded strides ggml
uses). The most likely remaining candidate is instruction scheduling / memory-level parallelism around
the ~20-cycle matrix op, which would need a cycle-level profile to confirm.

**Decision: keep the code, default it off, record the numbers.** Reverting it entirely would only
guarantee the experiment gets repeated.

### 2.4 A driver variant that was tried and reverted

Adding `nir_var_mem_shared` to `agx_optimize_nir`'s `nir_opt_load_store_vectorize` modes -- which would
have made the redundant-tile-load elimination of §4.1 happen in the driver, for every application --
is correct (all standalone tests pass) but costs **10 %**: pp512 Q8_0 1058 -> 949 t/s. Reverted, not
shipped.

## 3. Part B -- ggml tuning

### 3.1 Where the knobs are

* `ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp`, `#ifdef COOPMAT` blocks. Spec constants
  0..10 = `BLOCK_SIZE, BM, BN, BK, WM, WN, WMITER, TM, TN, TK, WARP`; 11 = `ALIGNED`;
  12 = `SHMEM_STRIDE_PAD` (default 4); 13 = `APPLY_SLM_A_RESHAPE`. `SHMEM_STRIDE = BK/2 +
  SHMEM_STRIDE_PAD`, and `buf_a` / `buf_b` are `shared FLOAT_TYPEV2[]`, i.e. A is staged M x K
  row-major and B is staged **N x K** (so B is loaded `ColumnMajor`).
* `ggml-vulkan.cpp` ~4330-4390: `l_/m_/s_warptile[_mmq]`. `TM = TN = TK = device->coopmat_m/n/k`, which
  for this driver is 16 (the first advertised shape). With `subgroup_size = 32`:
  `l_warptile_mmq = {128, 128, 128, 32, 64, 64, 2, 16, 16, 16, 32}` -> 4 warps, each owning a 64x64
  output tile = **16 accumulator tiles of 16x16 f32 = 128 32-bit values per lane**.
  Selection (`ggml_vk_guess_matmul_pipeline`, non-coopmat2): `m<=32||n<=32` -> s, `m<=64||n<=64` -> m,
  else l. pp512 uses l.
* `bank_conflict_offset = coopmat ? 8 : 1` in `ggml_vk_matmul_shmem_support` is only the shared-memory
  budget estimate; the actual pad is spec constant 12.

### 3.2 The loop-nest fix (the win)

```
for (i = 0; i < BK; i += TK)
  for (cm_row ...) {
      coopMatLoad(cache_a, ...);                 // once per (k, cm_row)
      for (cm_col ...) {
          coopMatLoad(cache_b, ...);             // once per (k, cm_row, cm_col)  <-- cms_per_row times
          sums[..] = coopMatMulAdd(cache_a, cache_b, sums[..]);
      }
  }
```

`cache_b` is re-loaded for every `cm_row`. On a driver that CSEs shared loads that is free; AGX does
not (§2.4 shows why it should not). For `l_warptile_mmq` that is `cms_per_row * cms_per_col = 16` B
tile loads per k step where 4 would do -- 8x more work than the A loads.

`patches/ggml/0001` loads all `cms_per_row` A tiles into a register array once per k step and moves
`cm_col` outside, so both operands are fetched exactly once:

```
for (i = 0; i < BK; i += TK) {
    for (cm_row ...) coopMatLoad(cache_a[cm_row], ...);
    for (cm_col ...) {
        coopMatLoad(cache_b, ...);
        for (cm_row ...) sums[..] = coopMatMulAdd(cache_a[cm_row], cache_b, sums[..]);
    }
}
```

Extra registers: `cms_per_row` A tiles (4 x 4 = 16 32-bit values for the large warptile), against the
128 already held by the accumulators. No spills appeared (`shaderdb`: 0:0 spills:fills, 640 threads,
unchanged).

Hoisting **B** instead (`cache_b[cms_per_col]`, `cm_col` inner) gives the same load counts but measured
lower -- 1103.5 vs 1134.3 t/s -- so A-hoisting is what shipped.

### 3.3 Warptile sweep

`patches/ggml/0002` adds `GGML_VK_WT_{L,M,S}[_MMQ]` so the eleven spec constants can be set from the
environment (the workgroup denominators are not overridable, so `BM`/`BN` must stay 128x128 for the
large tile). pp512 Q8_0 fa=1, `VECLOAD=0`, A-hoisted shader:

| `BLOCK_SIZE,BM,BN,BK,WM,WN,WMITER,TM,TN,TK,WARP` | warps | acc tiles/warp | t/s |
|---|---:|---:|---:|
| `128,128,128,32,64,64,2,16,16,16,32` (default) | 4 | 16 | 1098.0 |
| `256,128,128,32,32,64,2,16,16,16,32` | 8 | 8 | 1102.2 |
| `256,128,128,32,64,32,2,16,16,16,32` | 8 | 8 | 1100.8 |
| `512,128,128,32,32,32,2,16,16,16,32` | 16 | 4 | 1072.8 |

Under 3 % across a 4x change in per-warp accumulator area, so **the generic table is fine for this
device** and no Honeykrisp branch was added to `ggml-vulkan.cpp`. (Measured before the A-hoist landed
in the binary used for the final numbers; the ordering is what matters.)

`SHMEM_STRIDE_PAD` was left at 4: the A-tile access pattern with `stride = BK/2 + 4` words already maps
the 32 lanes onto 32 distinct banks (`row * 20 mod 32` is a permutation for rows 0..7), while pad 0 or
8 would fold rows onto each other.

### 3.4 What was not tried

* Staging B transposed (`buf_b` as K x N) so its `coopMatLoad` becomes row-major and vectorisable. This
  is the only way to get both operands contiguous -- the transpose identity does not help, it just swaps
  which one is strided. It means rewriting `load_b_to_shmem` for every quantised type in
  `mul_mm_funcs.glsl` and re-checking bank conflicts on the (now strided) staging writes; too large for
  the time box, and given §2.3 the vectorised loads it would enable may not even be a win here.
* Advertising 8x8x8 first so ggml picks `TM=TN=TK=8`. The hardware instruction count is identical
  (a 16x16x16 `coopMatMulAdd` is 8 of them either way) and the load counts work out the same, so there
  is nothing to gain.
* `dEQP-VK.compute.cooperative_matrix.*` (still no deqp build on this machine).

## 4. Numbers

All with the NVR live, `systemd-run --user --scope -p MemoryMax=5G`, `nice -n 10`, detector
`inference_speed < 60 ms` checked before every run (`coopmat/gpu_ok.sh`), `dmesg` and the detector's
fallback count checked after. "Phase 4" = the numbers in `PHASE4_COOPMAT.md`.

**Caveat on absolute numbers.** The GPU is shared with a live NVR, and the background load moved over
the afternoon: the shipped configuration measured 1134 t/s (+-2) in the quietest window and 1066-1077
t/s (+-9 to +-24) two hours later with the detector busier. Every A/B pair below was measured
back-to-back in one window, so the *ratios* are solid; treat single absolute figures as +-6 %.

### 4.1 llama-bench, Qwen2.5-0.5B, pp512, `-r 3` (t/s)

| configuration | Q8_0 fa=1 | Q8_0 fa=0 | F16 fa=1 |
|---|---:|---:|---:|
| Phase 2, stock Mesa 26.1.8 | 782.5 | 778.6 | 756.9 |
| Phase 4, private driver, coopmat (shader cache on) | 1055.0 | 991.1 | 1016.2 |
| Phase 5 baseline (same, cache off) | 1001.6 | -- | -- |
| + driver `VECLOAD=1` only | 1058.4 | -- | -- |
| + ggml A-hoist only (**shipped**) | **1134.3** (1066-1077 later, busier) | -- | not re-measured |
| + ggml A-hoist + `VECLOAD=1` | 997.1 | -- | -- |
| + ggml B-hoist only | 1103.5 | 1035.2 | 1110.3 |
| + ggml B-hoist + `VECLOAD=1` | 993.9 | 930.7 | 1007.3 |

F16 was measured on the B-hoist build only; the A-hoist build was re-measured for Q8_0 and CLIP.

**1134 t/s = ~1.13 TFLOPS = 43 % of the 2.6 TFLOPS FMA peak** (Phase 4: 1.04 TFLOPS / 40 %; Phase 2
scalar: 0.77 TFLOPS / 30 %).

### 4.2 ViT-H-14-378 CLIP (`tools/run_clip.sh`, 4 images, 1 warm-up + 3 iterations, f16, fa=0)

| configuration | median-of-medians | cosine vs ORT fp32 (min / mean) |
|---|---:|---|
| Phase 2, stock, fa=0 | 1873.8 ms | 0.9985 / -- |
| Phase 4, private driver, coopmat | 1505.0 ms | 0.99867 / 0.99907 |
| Phase 5, ggml B-hoist | 1463.6 ms | 0.998665 / 0.999072 |
| **Phase 5, ggml A-hoist (shipped)** | **1375.0 ms** | 0.998665 / 0.999072 |
| Phase 5, ggml A-hoist + `VECLOAD=1` | 1583.6 ms | 0.998665 / 0.999072 |

1.36x over Phase 2 stock, 1.09x over Phase 4. Target was ~1200 ms; not reached.

### 4.3 Correctness

| check | result |
|---|---|
| standalone `cmat_test`, 10 cases x 3 `VECLOAD` modes | bit-exact, max abs err 0 |
| `test-backend-ops -o MUL_MAT`, shipped config | **1106/1106** |
| `test-backend-ops -o MUL_MAT`, `VECLOAD=1` | **1106/1106** |
| `FLASH_ATTN_EXT -p hsk=64,...,kv=1024` | 6/6 |
| `FLASH_ATTN_EXT -p hsk=80` (ViT-H's head size) | 112/112 |
| `FLASH_ATTN_EXT -p hsk=72,...,kv=113` | 168/216 -- the 48 failures are all `type_K/V = q5_0/q5_1` |
| same, `GGML_VK_DISABLE_COOPMAT=1` | 168/216, **identical failures** -> pre-existing |
| ViT-H cosine vs ORT | min 0.998665, mean 0.999072 (unchanged from Phase 4) |
| `dmesg` GPU faults | none |
| `journalctl -u frigate-detector | grep -ci fallback` | 0 |

## 5. Why 1134 t/s and not 1300 -- where the time goes

Per warp, per k block of the large warptile (`BM=BN=128`, `BK=32`, 4 warps, 64x64 output per warp):

* matrix work: `cms_per_row * cms_per_col * (BK/TK) * 8` = 4*4*2*8 = **256 `simd_matrix_fmadd`**
* tile loads after §3.2: `cms_per_row * 4 * (BK/TK)` A + `cms_per_col * 4 * 2 * (BK/TK)` B
  = 32 + 64 = **96** (`VECLOAD=1` would make it 16 + 64 = 80, and is slower anyway, see §2.3)
* staging: `(BM + BN) * BK = 8192` elements per workgroup / 128 threads = **64 elements per thread**,
  each needing an address, a dequantisation step (Q8_0) and a threadgroup store -- on the order of
  150 instructions.

So roughly 256 matrix instructions against ~250 non-matrix ones. That is the 43 %-of-peak number, and
it is dominated by **staging and dequantisation, not by the tile loads**. Halving the tile loads (which
is all Part A can do, and only for A) moves at most ~6 % of the instruction stream even when it works.
Getting materially closer to Apple's ~80 % of peak on this instruction would need the staging cost
amortised over a larger tile -- `BM`/`BN` of 256 -- which the 32 KiB threadgroup limit and the register
file (16 accumulator tiles per warp is already 128 32-bit values per lane) do not allow at 16x16
fragments on this chip. That is a structural ceiling, not a tuning one.

## 6. Deliverables

* Patches: `patches/mesa/0004-asahi-hk-add-an-opt-in-vectorised-cooperative-matrix.patch`;
  `patches/ggml/0001-vulkan-hoist-the-A-tile-load-out-of-the-coopmat-mul_.patch`,
  `patches/ggml/0002-vulkan-allow-the-matmul-warptiles-to-be-overridden-f.patch`.
* Branch `coopmat-vecload` on `https://github.com/aquarat/mesa` (`coopmat` + 1 commit).
* Driver build `coopmat/deploy-v2/` (+ `PROVENANCE.txt`). **Behaviourally identical to `deploy/`** with
  default settings -- the new code is opt-in and off.
* Tests/evidence: `coopmat/tests/cmat4_oddstride.{comp,spv}` (new), `p5_tbo_mul_mat_*.txt`,
  `p5_fa72_nocoopmat.txt`, `p5_dis_{vec,novec}.txt` (full AGX dumps), `p5_loop_{vec,novec}.txt`
  (the extracted inner loops), `p5_sdb_{vec,novec}.txt` (shaderdb). Scripts: `coopmat/bench5.sh`,
  `coopmat/sweep5.sh`.

## 7. Recommendation

* **Driver: do not redeploy.** `deploy-v2/` differs from the deployed `deploy/` only by an opt-in code
  path that is off by default and slower when on. `immich-ml-ggml.service` was not touched. If you want
  the newer build in place anyway (for the extra standalone-test coverage and the odd-stride fix in the
  test suite, not for speed), point the unit's `VK_DRIVER_FILES` at
  `coopmat/deploy-v2/asahi_icd.aarch64.json` -- but there is no measured reason to.
* **ggml: worth rebuilding.** `patches/ggml/0001` is a ~9-13 % GEMM win with no driver change and no
  correctness change, and it is a plain upstream bug fix (it will help any coopmat1 device whose
  compiler does not CSE shared loads). Immich's CLIP tower runs its own ggml build at
  `immich-ml/immich-fork/machine-learning/ggml/build`; applying the patch and rebuilding there would
  give it the ViT-H 1505 -> 1375 ms. Not done here (out of scope: it is not this repo's tree).
* **Next**, in decreasing order of expected value: (1) stage `buf_b` transposed so both operands are
  row-major, then re-test `VECLOAD=1` -- it is the only untried change that could make Part A pay;
  (2) find out why the vectorised load is slower (needs a cycle-level profile or a microbenchmark
  isolating one `lload i16, xy` against two `lload i16, x`); (3) attack the staging/dequantisation
  cost, which §5 says is now the bigger half of the shader.
