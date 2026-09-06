# Does the patched Honeykrisp driver (got-bringup weak CDM barrier) cut per-dispatch overhead for ML inference on this M1?

Date: 2026-09-06. Machine: Mac mini M1 (T8103, GPU G13G B1, 8 cores), Fedora Asahi Remix 44, kernel 7.1.6-400.asahi, system Mesa 26.1.8, 16 GB unified, headless.
Background load during every measurement: the live Frigate NVR (ffmpeg Vulkan scalers, `frigate-detector.service` = ncnn-Vulkan fp16) on the SYSTEM driver. Its journal shows ~1000 inferences per ~20 min (about 0.8/s at this time of day, not 25/s), so GPU contention was light but present; numbers here are ~10-15% slower than the idle-GPU figures in `GPU_INFERENCE.md` for that reason.
Nothing system-wide was changed: no package installed, `/usr/lib64`, `/usr/share/vulkan` and the live service untouched (verified at the end: `rpm -q honeykrisp-got` = not installed, `rpm -V mesa-vulkan-drivers` clean). The `immich-ml` container was NOT stopped (permission was given, but the ViT-H run fitted inside the 5 GB cap with 6.2 GB available).
Working dir: `vk-dispatch/` (scripts `run-*.sh`, raw outputs in `results/`, tests in `tests/`).

## TL;DR

* **The barrier change does nothing for inference on this machine.** Dependent dispatch chains cost the same on both drivers (3.2-3.7 us/dispatch for empty shaders, work-bound otherwise); only chains with NO Vulkan barrier between dispatches get faster (10-45x for tiny dispatches), and inference runtimes either put a barrier between every layer (ncnn) or a submission boundary between every op (MNN), so there is nothing to overlap.
* YOLOv9-t 320: ncnn-Vulkan fp16 36.3 ms vs 36.6 ms (detector A/B at 25 req/s), MNN-Vulkan fp16 19.0 vs 19.0 ms. No change. Outputs bit-identical.
* CLIP transformers on MNN-Vulkan DID get 1.33-1.45x faster (ViT-B-32 fp16 ~190 -> ~135 ms; ViT-H-14-378 fp16 7.70 -> 5.79 s), bit-identical outputs -- **but the ablation shows this is not the barrier change**: with `HK_PERFTEST=nooverlap,noconstdata` the fork is exactly as fast, and plain upstream Mesa 26.2.2 is as slow as 26.1.8. The gain comes from the fork's other (compiler-side) commits. It is still 7x slower than MNN CPU fp16 (18 ms) on the same chip, so it changes no decision.
* Live service unaffected (0 restarts, no errors, no GPU faults in dmesg); its inference average rose 34 -> 37 ms while my benchmarks shared the GPU.
* **Recommendation: do not deploy system-wide.** No workload here benefits enough to justify replacing the driver under the ffmpeg scalers and the live detector.

## 1. Driver provenance

| item | value |
|---|---|
| harness repo | `aquarat/got-bringup` main @ `6b25a95` (cloned to `vk-dispatch/got-bringup`); README, INSTALL, `deploy-system-driver.sh`, `mesa-source.env`, `data/barrier-bit-cost.md`, `data/per-fix-results.md`, `tests/` read |
| package | `honeykrisp-got-26.3.0.devel-1.20260905git0a5367a9c506.fc44.aarch64.rpm` (1,984,749 B, sha256 `43cf185c921326b3ebf8f5ed601e55b6e1636ffaf6a685e616dd89f9026d56c8`) from the dnf repo `https://aquarat.github.io/got-bringup/fedora/44/aarch64/` (same file as GitHub release `driver-0a5367a9c506`). Downloaded with curl, NOT `dnf install`. |
| signature | OK against `RPM-GPG-KEY-honeykrisp-got` (fingerprint `22ffa844134808c480e5108633473d8ec8883cd3`), checked with `rpmkeys` in a throw-away rpmdb (`vk-dispatch/rpmdb`). Built 2026-09-05 19:55 BST. |
| Mesa commit | RPM built from `aquarat/mesa` `local-deploy` @ `0a5367a9c506c9d7e97e5467556365c3ef54442b` (Mesa 26.3.0-devel). **`mesa-source.env` pins `d105715f01c3` (branch HEAD), one commit later** ("fix the findings from an external review", described as correctness-only: profiler use-after-free, subqueue-overlap holes, an alloc error path, cache-key hash). The published RPM lags the pin by one commit; this experiment used the RPM. |
| fork contents | upstream Mesa main (26.3.0-devel) + 36 commits: dispatch overlap / weak CDM barrier `0x80` (on by default; `HK_PERFTEST=nooverlap` restores the old full barrier), constant tables out of scratch (`HK_PERFTEST=noconstdata` disables), `HK_GPUTIME` firmware profiler, `iadd(amul)` bounds-check lowering, `load_agx` divergence fix, AGX compiler cleanups, tessellation, fragment-shader-interlock, subqueue overlap (default off). |
| extraction | `rpm2cpio | cpio` into `vk-dispatch/patched-mesa/`; driver at `patched-mesa/usr/lib64/honeykrisp-got/libvulkan_asahi.so` (13.7 MB). `ldd`: every dependency resolves to the system libs (glibc 2.43, libstdc++ 16.1.1, libdrm 2.4.134, SPIRV-Tools, xcb, wayland) -- no `LD_LIBRARY_PATH` needed. |
| loading | `vk-dispatch/patched_icd.json` (copy of the system ICD with `library_path` -> private .so, api 1.4.359) and `vk-dispatch/with-patched.sh` (exports `VK_DRIVER_FILES` + `VK_ICD_FILENAMES`, execs its arguments). |
| verification | `with-patched.sh vulkaninfo --summary`: apiVersion 1.4.359, driverVersion **26.2.99**, driverInfo **Mesa 26.3.0-devel**, Apple M1 (G13G B1). Plain `vulkaninfo --summary`: driverVersion 26.1.8, Mesa 26.1.8. The detector logs agree (stock instance `driver 0x6801008`, patched instance `driver 0x6802063`). The fork prints `[hk] HK_PERFTEST active: 0x80 ()` at device creation, i.e. the weak barrier is on. |
| G13G | got-bringup was tested only on an M1 Max (G13C). Same driver code; it enumerates the G13G, passes the chain test's data check, and every inference output is bit-identical to the stock driver's. No warnings. No CTS run here. |
| control arm | plain Fedora rawhide `mesa-vulkan-drivers-26.2.2-1.fc46` (no fork patches) extracted the same way into `vk-dispatch/upstream-mesa/`, `with-upstream.sh` -- used only to attribute gains to upstream progress vs the fork. |

Confound to keep in mind: "stock vs patched" is 26.1.8 vs 26.3-devel+fork, so it includes two upstream cycles of Mesa. The `HK_PERFTEST` ablation and the 26.2.2 control arm separate the pieces (section 5).

## 2. Per-dispatch microbenchmarks

### 2.1 Chain test (`tests/chaintest.c`, the key number)

1000 back-to-back `vkCmdDispatch` of a trivial shader (local_size 64) in one command buffer, one submit. **dependent** = every dispatch reads+writes the same buffer with a `vkCmdPipelineBarrier` (compute->compute, shader write -> shader read buffer barrier) between consecutive dispatches -- the shape of an inference graph; result verified (each element must equal N). **independent** = each dispatch has its own slice and there is no barrier at all. Metric: CPU wall from `vkQueueSubmit` to `vkQueueWaitIdle` divided by N (median of 3 runs, each run a fresh recording+submit). Optional 1000-iteration serial loop per invocation to give each dispatch real latency.

| chain | workgroups | shader | stock 26.1.8 | patched (0x80) | patched `nooverlap` | upstream 26.2.2 |
|---|---:|---|---:|---:|---:|---:|
| dependent | 1 | empty | 3.47 us | 3.17 us | 3.15 us | |
| dependent | 64 | empty | 3.70 | 3.62 | 3.63 | |
| dependent | 512 | empty | 6.38 | 5.70 | 6.38 | |
| dependent | 1 | 1000-iter loop | 33.6 | 31.9 | 32.0 | 31.9 |
| dependent | 64 | loop | 33.2 | 33.3 | 32.8 | |
| dependent | 512 | loop | 174.4 | 174.4 | 174.3 | |
| independent | 1 | empty | 3.30 | **0.32** | 3.30 | |
| independent | 64 | empty | 4.62 | **1.33** | 4.70 | |
| independent | 512 | empty | 10.04 | **6.04** | 10.36 | |
| independent | 1 | loop | 31.6 | **0.71** | 31.5 | 31.9 |
| independent | 64 | loop | 34.0 | **20.0** | 35.0 | |
| independent | 512 | loop | 172.3 | **158.3** | 172.0 | |

Reading:
* **Dependent chains: no change** (differences are within the run-to-run spread). With a Vulkan barrier between dispatches the driver must drain the previous dispatch on both drivers; the floor is ~3.2 us of fixed per-dispatch/barrier cost, and once the shader has real latency (the loop) it is the shader latency (~31 us) that is paid serially, on every driver.
* **Independent chains: 10-45x faster for small dispatches** on the patched driver (0.32 us vs 3.30 us empty; 0.71 vs 31.6 us with the loop, i.e. all 1000 dispatches overlap) and 1.1-1.7x once each dispatch already fills the GPU (512 wg). `HK_PERFTEST=nooverlap` reverts exactly to the stock behaviour, so this is the barrier change and nothing else. Plain 26.2.2 still serialises.
* Data check: "ok" on every empty-shader run on all drivers -- with an explicit Vulkan barrier the weak CDM barrier is coherent (the got-bringup coherence data says bit 7 alone is not; the driver still emits the full barrier where the application asks for one).

### 2.2 got-bringup `tests/cstest.c` (GPU timestamps, 3 runs)

Only usable on the patched driver here: on 26.1.8 the bottom-of-pipe timestamp is written without waiting for un-barriered dispatches, so stock reports 0.6 ms for 64 dispatches of 100000 iterations, which is physically impossible (one such dispatch takes ~3 ms). Patched: 64x1x100000 = 3.00 ms (all 64 overlap; one dispatch's latency, cf. 2.89 ms floor on the M1 Max in `data/barrier-bit-cost.md`), 64x128x1000 = 1.31 ms, 256x1x100000 = 9.34 ms (~64-70 dispatches in flight). This is why the chain test above uses CPU wall time, not timestamps.

### 2.3 clpeak kernel-launch latency (`--latency --vulkan`, 3 runs, median)

| | dispatch | roundtrip |
|---|---:|---:|
| stock 26.1.8 | 81.9 us (79.1-92.3) | 158.9 us (158.7-180.5) |
| patched | 79.6 us (77.6-82.2) | 158.3 us (153.9-163.7) |

Unchanged: this is the cost of one submit -> fence round trip (ioctl + firmware), which the barrier change does not touch.

## 3. Inference benchmarks (same scripts, models and inputs as `GPU_INFERENCE.md` / `IMMICH_ML_GPU.md`)

| model / runtime | stock 26.1.8 | patched | correctness on patched |
|---|---:|---:|---|
| YOLOv9-t 320, ncnn-Vulkan fp16 (`bench_ncnn.py`, 100 it, median, 2 passes) | 28.4 / 28.0 ms | 34.1 / 28.6 ms | vs ORT fp32: max diff 13.8 / 8.2 (fp16 rounding; stock 10.0 / 14.7 on random inputs) |
| YOLOv9-t 320, ncnn-Vulkan fp32 | 34.0 / 35.1 | 40.2 / 34.5 | max diff vs ORT 0.0011 / 0.0010 (stock 0.0029 / 0.0034) |
| YOLOv9-t 320, MNN-Vulkan fp32 (`bench_mnn_cpp`, fixed `input_320.bin`, 100 it) | 20.38 / 20.51 | 20.20 / 20.13 | output **md5 identical** to stock (`301ed9a3…`) |
| YOLOv9-t 320, MNN-Vulkan fp16 | 18.99 / 18.90 | 19.00 / 19.01 | **md5 identical** (`6d1d0a17…`); upstream 26.2.2: 19.1 ms, same md5 |
| CLIP ViT-B-32 visual, MNN-Vulkan fp32 (`mnnrun`, 20 it) | 216.4 / 213.7 | **155.9 / 155.8** | embedding **md5 identical** (`a64b3d9f…`), cos vs ORT 0.999986 |
| CLIP ViT-B-32 visual, MNN-Vulkan fp16 (20-30 it, 4 passes) | 190.7 / 167.1 / 194.6 / 220.3 | **127.3 / 135.9 / 133.7 / 156.7** | **md5 identical** (`c4007997…`), cos 0.999983 |
| CLIP ViT-H-14-378 visual, MNN-Vulkan fp16 weights (1 warm-up + 3 it; 2.4 GB session, under a 5 GB cap) | 7697 ms | **5791 ms** | **md5 identical** (`139a3759…`), cos vs ORT 0.999990 |
| detector A/B: private `zmq_onnx_client.py --backend ncnn-vulkan` on :5556, `test_client.py --duration 60 --rate 25` (1501 req) | avg 36.34, p50 36.02, **p95 43.27**, p99 46.11, max 58.57 ms, 0 timeouts | avg 36.64, p50 36.37, **p95 43.04**, p99 45.77, max 57.09 ms, 0 timeouts | detector outputs valid (20x6 rows) |

No correctness mismatch anywhere: every MNN output (YOLO fp32/fp16, ViT-B-32 fp32/fp16, ViT-H fp16) is byte-identical between the two drivers, and ncnn's error vs ORT is the usual fp16 rounding, in the same range as stock.

### 3.1 Where the time goes (fork's `HK_GPUTIME=1` firmware profiler, patched driver, steady state)

| workload | command streams / inference | dispatches / inference | GPU time per dispatch | Vulkan pipeline barriers / inference | GPU busy | GPU idle | with `nooverlap` |
|---|---:|---:|---:|---:|---:|---:|---|
| MNN-Vulkan YOLO fp16 (46.7 fps) | ~474 | ~788 | 13.9 us | **0** | 50.0% (10.9 ms) | 50% in ~474 gaps (~23 us each) | identical (50.1%) |
| ncnn-Vulkan YOLO fp16 (35 fps, `bench_ncnn_cpp`) | 1 (978 dispatches per stream) | ~978 | 19.9 us | **~2000, all "compute-only, ending real work"** | 66% (18.9 ms) | 34% between inferences (CPU side) | identical (67%) |
| MNN-Vulkan ViT-B-32 fp16 (7 fps) | ~130 | ~680 | 87.9 us | ~212 | 41.8% | 58% in ~400 gaps | 40.2%, same latency |

* MNN's Vulkan backend submits (roughly) one command buffer per op and uses **no pipeline barriers**; the per-submission gap (~23 us x ~474 = 10.9 ms of a 21 ms inference) is the overhead, and a barrier change cannot touch it.
* ncnn records the whole graph in one command buffer with a full barrier after every layer; every layer depends on the previous one, so every barrier legitimately drains the GPU. The remaining 34% idle is the Python/ZMQ/CPU side between inferences.
* ViT-B-32's 88 us dispatches are unfused GEMM/attention tiles with low efficiency (no cooperative matrix on Honeykrisp), and the GPU is still idle 58% of the time even on the patched driver (MNN burns 45-50 ms of CPU per iteration).

## 4. Effect on the live service

* `frigate-detector.service`: `NRestarts=0`, active since 2026-09-05 17:00, no errors or warnings in `journalctl -u frigate-detector --since -4h`. Its own stats went avg 34.3 ms -> 35.8 -> 37.4 ms (max 88 -> 100 ms) across the experiment window: about +3 ms from sharing the GPU with my benchmark processes, nothing else. Frigate's API reported `host_zmq` inference_speed 48.6 ms (its end-to-end number) and 0 detection fps on all cameras at the time.
* `sudo dmesg`: no AGX/GPU fault, timeout or reset message during the window (the only 08:xx kernel line is an unrelated fs-verity entry). The earlier OOM record in dmesg is from the previous session (2026-09-05 23:05), not from this one; `free` was >= 6.0 GB available before every heavy run and all runs were inside `systemd-run --user --scope -p MemoryMax=5G`.
* The private :5556 detector instance was stopped after each arm (port free, no `zmq_onnx_client` on :5556 at the end). The `frigate` and `immich-ml` containers were not touched.

## 5. Interpretation

1. **Does the barrier change help dependent inference chains? No -- measurably zero.** The mechanism (letting consecutive CDM dispatches overlap by dropping the full cache barrier to bit 7) only applies where the application emits no barrier between dispatches. Inference graphs are chains of dependent layers: ncnn puts a full Vulkan barrier after each (2000 per YOLO inference), MNN ends the command stream after each op. In both cases the driver must still drain, and the profiler confirms `nooverlap` changes nothing. The chain test puts a number on the best case that would have been available: even for *independent* tiny dispatches the saving is ~3 us each, i.e. ~2 ms over YOLO's ~650 dispatches -- against a 20-36 ms inference. The 5.5x from Ghost of Tsushima came from ~414 *independent* single-workgroup dispatches per frame; that shape does not occur here.
2. **The per-dispatch overhead that limits inference here is not the GPU-side barrier.** It is ~3 us; what costs 20-36 ms is (a) MNN's one-submission-per-op structure (~11 ms of GPU idle gaps per YOLO inference), (b) ncnn's per-layer drains multiplied by the *latency* of each small dispatch (20 us GPU time each for tiny convolutions that cannot fill 8 cores), and (c) CPU-side runtime work (5-9 ms for YOLO, 45-50 ms for ViT-B-32, 2.7 s for ViT-H). clpeak's 80-100 us "dispatch latency" is the single-submit round trip and is likewise untouched.
3. **The CLIP speed-up (1.33-1.45x) is real, bit-exact, and not from the barrier or the constant-table change.** Ablation on ViT-B-32 fp16 (30 it, two passes): stock 194.6/220.3 ms; patched 133.7/156.7; `nooverlap` 140.2/157.4; `noconstdata` 135.1/136.0; both off 132.4/137.5. ViT-H: default 5791 ms, `nooverlap` 5794 ms. Plain upstream 26.2.2: 194.1/214.4 ms (= stock). So the gain sits in the fork's non-gated commits (candidates: the `iadd(amul)` bounds-check lowering that halves robustness code per load, the `load_agx` divergence fix, the pre-preamble/post-phi cleanups) and/or upstream main between 26.2.2 and 26.3-devel; separating those would need a build of upstream main, which is out of scope. Either way MNN-Vulkan ViT-B-32 at 135 ms remains 7x slower than MNN CPU fp16 (18.2 ms measured today) and ViT-H at 5.8 s is 3x slower than CPU fp16 (1.9 s), so the Immich conclusion ("run it on the CPU") stands.
4. **What remains for GPU inference on this stack**: cooperative-matrix / efficient GEMM (the 88 us ViT dispatches are the ceiling on transformers), fused kernels (fewer, larger dispatches -- ncnn's pnnx graph or Immich's fused ONNX graphs), a runtime that batches ops into one command buffer with the minimum barriers (MNN's per-op submissions are the biggest single loss for the CNN), and CPU-side runtime cost. None of these is a driver barrier problem.

## 6. Recommendation

**Do not deploy the patched driver system-wide for the NVR detector.**

* Benefit: none measurable for the workload the system driver serves -- the ncnn-Vulkan detector is 36.3 vs 36.6 ms (p95 43.3 vs 43.0), MNN-Vulkan YOLO 19.0 vs 19.0 ms.
* Risk: the swap replaces `/usr/lib64/libvulkan_asahi.so`, which the ffmpeg Vulkan scalers and the live detector load continuously; it is a 26.3-devel snapshot with 36 non-upstream commits, tested by its author only on a G13C, with no CTS run on this G13G; a `dnf` mesa update would fight with it (the package's file trigger re-applies the swap); the published RPM is one commit behind its own pin. On this headless box the compositor risk does not apply, but the recovery path (`honeykrisp-got disable`) still means a stopped detector until someone intervenes.
* If any process ever wants it (e.g. an MNN-Vulkan transformer job), load it per process through `vk-dispatch/with-patched.sh` exactly as done here; it coexists with the system driver with no side effects. Outputs were bit-identical in every test, so correctness is not the concern -- there is just nothing to gain for the detector.
* Re-check when Fedora ships Mesa 26.3 (to see whether the CLIP-side compiler gain landed upstream) or when Honeykrisp gains cooperative matrix; `run-clip.sh`, `run-clpeak-yolo.sh` and `tests/chaintest` re-run in minutes.

## 7. Files

`vk-dispatch/`: `rpms/` (RPM + key), `patched-mesa/` (extracted), `patched_icd.json`, `with-patched.sh`, `upstream-mesa/` + `upstream_icd.json` + `with-upstream.sh` (26.2.2 control), `tests/` (`chaintest.c/.comp`, built `cstest`, `coherence`), `run-micro.sh`, `run-clpeak-yolo.sh`, `run-clip.sh`, `run-ablate.sh`, `run-upstream.sh`, `run-detector-arm.sh`, `detector-models/` (private copy for the :5556 instance), `results/` (`chaintest.txt`, `cstest.txt`, `clpeak.txt`, `yolo_ncnn.txt`, `yolo_mnn.txt`, `mnn_out_*.bin`, `clip_vitb32.txt`, `clip_vith.txt`, `gputime_*.txt`, `ablation.txt`, `upstream_ab.txt`, `client_*.log`, `detector_*.log`).
