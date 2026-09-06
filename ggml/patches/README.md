# llama.cpp / ggml Vulkan patches

`git am` patches on llama.cpp (based on `9e0e220`). Measured on an Apple M1 (G13G, 32-wide
subgroups) against the Honeykrisp `VK_KHR_cooperative_matrix` implementation in
[../../mesa/](../../mesa/); write-up in
[../../reports/PHASE5_COOPMAT_TUNING.md](../../reports/PHASE5_COOPMAT_TUNING.md).

| patch | what |
|---|---|
| `0001-vulkan-hoist-the-A-tile-load-out-of-the-coopmat-mul_.patch` | The coopmat1 `mul_mm` inner loop fetches the same `cache_b` tile `cms_per_row` times per k step (16 tile loads where 8 suffice for the 128x128x32 warptile), relying on the backend to CSE shared-memory loads. AGX does not. Load the `cms_per_row` A tiles into a register array once per k step and iterate `cm_col` outside, so each operand is fetched exactly once. **pp512 Q8_0 1002 -> 1134 t/s (+13 %), F16 1017 -> 1110, ViT-H-14-378 CLIP 1505 -> 1375 ms**, `test-backend-ops -o MUL_MAT` 1106/1106 unchanged. Not device-specific -- it should help any coopmat1 device whose compiler leaves the redundant loads in. |
| `0002-vulkan-allow-the-matmul-warptiles-to-be-overridden-f.patch` | `GGML_VK_WT_{L,M,S}[_MMQ]="BLOCK_SIZE,BM,BN,BK,WM,WN,WMITER,TM,TN,TK,WARP"` so tile shapes can be swept on a new device without a rebuild (the workgroup denominators are not overridable, so `BM`/`BN` must be left alone). Used to establish that on this GPU the warp split of the 128x128 tile is worth under 3 %, so no device-specific table was added. |
