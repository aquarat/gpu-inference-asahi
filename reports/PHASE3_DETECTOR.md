# Phase 3 (detector): batching several cameras into one GPU inference, and the request round trip

Date: 2026-09-06. Machine and stack as in `PHASE1_RUNTIME.md` (Mac mini M1, Fedora Asahi 44, Mesa 26.1.8
Honeykrisp, 16 GB). Subject: the host-side Frigate detector service (`~/frigate/detector`, systemd
unit `frigate-detector.service`, ZMQ protocol of Frigate's `zmq_ipc` plugin), which since 10:55 today runs MNN on
Vulkan with the Phase 1 batch-recording library (`mnn-batch/lib/libMNN.so`): engine ~12-15 ms, ~18 ms round trip.
Everything was measured next to the live NVR (4 cameras, motion-gated detection at 2-13 req/s) and, for part of the
afternoon, next to another agent's ggml/MNN GPU benchmarks; those windows are marked. Load generation against the
live service is not repeatable (see 1.1), so every load test ran against a second instance on port 5556.

## TL;DR

* **Frigate never has two requests in flight on one detector**: its detector process is a strict
  dequeue -> preprocess -> send -> wait -> publish loop, and between two back-to-back requests it spends **13.5 ms
  (p50)** of its own CPU time. Batching in the service is therefore only possible with several Frigate detector
  processes on the same endpoint (config-only: N `zmq` entries in `detectors:`; Frigate feeds them from one shared
  queue) and a service that can hold several requests at once. That is what was built: a **ROUTER**-socket service
  (v2 of `zmq_onnx_client.py`, protocol unchanged) with an optional batching queue, zero-copy input, an idempotent
  model handshake and per-stage timing.
* **Batching itself does not pay on this GPU/runtime.** MNN-Vulkan's convolutions scale almost linearly with the
  batch (engine 12.0 / 25.3 / 37.3 / 74.5 ms for batch 1/2/4/8 = 12.0 -> 9.3 ms per frame at best): the ~500
  dispatches are not the cost, their kernel time is, and batch>1 also loses the winograd 3x3 path. Under every load
  pattern tried (4 or 8 clients, in phase or staggered, windows 0/2/4 ms, max batch 2/4/8) the batched service was
  10-25 ms *worse* on average latency than the same service serialising requests, and only trimmed the p95 in the
  artificial "8 clients firing in lock-step" case. Timeout risk goes up (one 4-batch took 61 ms). **Deployed with
  `--max-batch 1`**; the batching path stays as a knob.
* Getting batch>1 to compute correctly on MNN-Vulkan needed two fixes that are useful on their own: sessions must be
  created at their batch size (a resized session computes garbage / segfaults), and the YOLO DFL head's last-axis
  softmax is flattened by MNN to `[N*8400, 16]` rows, over the driver's 16384-row image limit from N=2 (the
  fixed-batch model variants rewrite it as a middle-axis softmax + weighted sum). Batched outputs then match batch-1
  on all 45 frames (36/36 detections, score <= 0.003, box <= 0.37 px).
* **Round trip**: the service's own Python cost is ~1.1 ms per request; the engine is 11.2-12 ms quiet and 14-15 ms
  next to the NVR; the remaining ~3 ms is ZMQ/TCP transport of the 1.23 MB float32 tensor plus the client. The old
  and new service measure the same 18 ms at 25 req/s. Shared memory or sending uint8 would need the Frigate plugin
  patched and was not done.
* **What changed in production** (switch at the time in section 6): v2 service with `--max-batch 1`, the other
  agent's conv+SiLU fused library (`mnn-batch/lib-v2`, engine 11.2 vs 12.0 ms, -6 %) and a second Frigate detector
  process (`host_zmq_b`) so that Frigate's 13.5 ms per-request overhead overlaps the GPU. With two processes the
  service also stops reloading the model on every handshake (the old REP service did, 0.4 s each, which under a
  handshake storm took the live detector to 190-250 ms and 0 detections/s -- section 1.1).

## 1. Where batching could help: the live arrival pattern

Captured with `tcpdump -i any 'tcp port 5555'` for 89 s on the live service (2026-09-06 12:08, ncnn → mnn-vulkan batch-lib
service, Frigate on 4 cameras, motion-gated detection), reconstructed per request from the TCP segments
(`scratchpad/arrival.py`; the `any` capture sees each packet on the veth and the bridge, deduplicated by interface):

| metric (529 tensor requests, 1.23 MB each, 6.1 req/s) | avg | p50 | p95 | max |
|---|---:|---:|---:|---:|
| request transfer, first -> last TCP segment | 0.92 ms | 0.77 | 1.93 | 4.13 |
| service time, last request byte -> reply on the wire | 16.8 ms | 16.8 | 20.2 | 23.9 |
| client "think", reply -> next request's first byte | 147 ms | **13.5** | 194 | 13392 |
| inter-arrival, request start -> next request start | 165 ms | 33.7 | 214 | 13412 |

Histogram of inter-arrival gaps (ms): `<5: 0, 5-10: 32, 10-20: 250, 20-50: 6, 50-100: 188, 100-200: 44, >200: 8`.
**No request ever arrived while another was in flight** (0 % within 3 ms of the previous one), and even
back-to-back requests are separated by >= 5 ms (p50 13.5 ms) of Frigate-side work: with one detector process
Frigate is a strict send / wait / publish / dequeue / preprocess loop (`frigate/object_detection/base.py`
`DetectorRunner.run`: one `detection_queue.get`, one `detect_raw`, one shm write + publish per request; the
zmq plugin adds `tensor_input.tobytes()` of 1.2 MB). So on the single REQ socket the service can never see two
requests at once; batching on the service side needs Frigate to run more than one detector process.
Frigate's app.py creates one `ObjectDetectProcess` per entry in `detectors:` and all of them consume the *same*
`detection_queue` (cameras are not pinned to a detector), so N `zmq` entries on the same endpoint = N requests in
flight, without any plugin change. That is design (a); it also hides the 13.5 ms of Frigate-side per-request work
behind the GPU time of the other process. Design (b) (a pipelining plugin) was not pursued: it would need a plugin
change for the same effect.

Side finding: the live service reloads the model on *every* `model_request` handshake (2-7 s incl. warm-up), and
Frigate re-handshakes after each 200 ms timeout. With N detector processes every one of them handshakes, so the
v2 service answers a handshake for the already-loaded file (same name, same mtime) without reloading.

## 2. Round-trip breakdown (task 2)

Same client (`test_client.py`, one REQ socket, real cam2 frame, 25 req/s, 40 s, GPU shared with the live NVR and
with the other agent's ggml benchmarks that were running intermittently):

| | old service (REP, `recv_multipart` copy, `np.frombuffer`, per-call `np.empty`, shim memcpy in/out) | v2 service (ROUTER, `copy=False` frames, host tensors wrapping the buffers, no shim memcpy) |
|---|---|---|
| round trip avg / p50 / p95 / p99 / max | 17.9 / 16.9 / 24.0 / 32.9 / 102.6 ms (1 timeout) | 18.3 / 17.7 / 25.2 / 33.2 / 45.3 ms (0 timeouts) |
| service "Inference stats" (engine + post) | 14.9 ms avg, max 23.6 | engine 14.2-14.8 ms avg, post 0.27-0.40 ms |

The two are equal within the GPU noise of that hour (the engine itself measured 11.6 ms on a quiet GPU and 14-15 ms
here). Where the rest goes, from `cProfile` of the v2 service under the same load (36 s, 751 requests):

| stage | per request | notes |
|---|---:|---|
| `zmq.Poller.poll` | (idle) 17.2 s total | waiting for the next request: the service is ~50 % idle at 25 req/s |
| shim call `mnn_shim_infer_batch` (input copy to the GPU image, `runSession`, output copy) | 19.5 ms (contended; 14.2 in the stats line, 11.6 quiet) | everything else below is the whole Python cost |
| `recv_multipart(copy=False)` + header json + `np.frombuffer` | 0.11 ms | |
| YOLO decode + `cv2.dnn.NMSBoxes` | 0.40 ms | |
| `_build_response` (json + `tobytes` of 480 B) + `send_multipart` | 0.45 ms | |
| batch bookkeeping (`_flush`, `_run_batch`) | 0.15 ms | |

Python overhead on the service side is now ~1.1 ms per request; the old service was within ~0.5 ms of that (its
extra 1.2 MB copies cost ~0.1-0.2 ms each on this memory system). The remaining ~3 ms between engine time and the
client's round trip is transport: the 1.23 MB float32 tensor takes 0.9 ms on the wire (pcap, above), the ZMQ I/O
thread -> application thread hand-off on both sides, and the client's own `tobytes()`. The shared-memory transport
option (noted in the task) would remove the 0.9 ms transfer and the copies; a cheaper protocol change would be for the
plugin to send uint8 NCHW (Frigate converts to float32 *before* sending, 4x the bytes) and let the service divide by
255 -- both need the plugin patched, neither was implemented because the ZMQ path is already within ~3 ms of the
engine.

### 1.1 The old service under several clients (do not repeat)

A 2-client check against the *live* REP service (loadgen at 2 x 3 req/s for 15 s, 12:46) produced 34 timeouts and
0 successes: every `model_request` handshake reloaded the model (0.4 s with the GPU busy), the other client's
in-flight request hit Frigate's 200 ms timeout, reset its socket and handshaked again, and so did Frigate's own
detector process -- a storm of reloads that held the live detector at 190-250 ms `inference_speed` and
`detection_fps` 0 for about 2 minutes after the load stopped. The v2 service answers a handshake for the already
loaded file (same name and mtime) without touching the backend, which is a prerequisite for design (a).

## 3. Batched inference on MNN-Vulkan: making it correct

`mnn_shim` v2 (`detector/mnn_shim/mnn_shim.cpp`) holds one MNN session per batch size and exposes
`mnn_shim_create_multi(paths[], batches[])` / `mnn_shim_infer_batch(h, b, ...)`; host tensors wrap the caller's
buffers (`MNN::Tensor::create(shape, data)`), so the per-call memcpys of v1 are gone. The v1 entry points are kept.
`backends.MnnBackend.load()` builds sessions for 1 and the powers of two up to `DETECTOR_MAX_BATCH`.

What went wrong before it computed the right thing (all on the real model, image backend, fp16, `lib/libMNN.so`):

| attempt | result | diagnosis |
|---|---|---|
| ONNX export has batch 1 hard-coded (input `[1,3,320,320]`, every Reshape constant starts with 1) | MNN `resizeSession` to batch 2 fails: `Reshape error: 268800 -> 134400` | rewrite the Reshape constants (dim 0 -> -1 / 0, other dims from shape inference) -- `detector/detector/onnx_batch.py` |
| one session, `resizeTensor` + `resizeSession` to batch N | outputs garbage (max abs diff 1635 vs batch 1; on the CPU backend the same session is right to 0.1) | per-tensor bisect (`saveTensors`, CPU fp32 reference for images A and B, `scratchpad/bisect2.cpp`): image B wrong from the first stride-1 1x1 conv, image A's last row wrong from the first 3x3 conv -- images sized for batch 1 survive the resize. The MNN-fusion agent saw the same operation segfault in `VulkanImageConverter`. Not chased further: **create each session at its batch size** from a model whose batch dim is fixed at conversion (`<name>.b{N}.onnx` -> `<name>.b{N}.mnn`, cached next to the model). |
| fixed-batch model | correct through the whole backbone and heads for both images, **wrong from `/model.22/dfl/Softmax`** for image B (513 vs <= 1) | the softmax in isolation is right at batch 2 (5 synthetic shapes), the DFL chain in isolation is right, so it is context: with a debug print in `VulkanSoftmax.cpp` (build/ only, source restored) MNN's geometry pass had flattened the input to `[16800, 16]`; `VulkanTensor` stores that as an image of `N*H` = 16800 rows, splits it into two `VkImage` blocks at `maxImageDimension2D` = 16384, and the softmax kernel only reads block 0. Batch 1 (8400 rows) never hits it. |
| DFL rewritten in the batch variants: `Reshape -> Transpose -> Softmax(axis 3) -> Transpose -> Conv(1x1, w=0..15)` becomes `Softmax(bin axis) -> Mul(0..15) -> ReduceSum` (opset 12 -> 13 for the axis-wise softmax; verified in isolation at batch 2/8 and with ONNX Runtime: 0.0005 max diff vs the original) | **correct** | the middle-axis form is `[N*4, 16, 2100]`, 32 rows at batch 8 |

Correctness of the final variants on the 45 real frames (post-processed detections, score > 0.4, NMS), Vulkan fp16:

| | batch 1 (unchanged model) | batch 2 | batch 4 | batch 8 |
|---|---|---|---|---|
| raw `output0` max / mean abs diff vs batch 1 | - | 2.5 / 0.025 | same | same |
| raw max / mean abs diff vs ORT fp32 | 3.76 / 0.048 | 5.12 / 0.048 | same | same |
| detections (ORT: 36) | 36 | 36 | 36 | 36 |
| frames with class/count mismatch vs batch 1 / vs ORT | - / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| max score delta / box-corner delta vs batch 1 | - | 0.003 / 0.37 px | 0.003 / 0.37 px | 0.003 / 0.37 px |
| padded slot influence on image 0 (full batch vs zero-padded) | - | 0.0 | 0.0 | 0.0 |

Batch>1 is **not bit-identical** to batch 1 (different convolution kernels: MNN's winograd path is batch-1 only; the
DFL is computed in a different but equivalent order), and the fp16 raw error vs ORT grows from 3.8 to 5.1 in the box
coordinates (pixels, 0..320) with no effect on the post-processed detections. The batch-1 path is the untouched
`fox-person-cat-320.mnn`, so production stays bit-identical to the 10:55 service.

### 3.1 Engine time per batch size

Quiet-ish GPU (live NVR only, other benchmarks gated off), `bench_backend.py`, medians of 20-60 calls through the
real backend:

| library | batch 1 | batch 2 | batch 4 | batch 8 | best per-frame |
|---|---:|---:|---:|---:|---:|
| `mnn-batch/lib` (Phase 1 batch recording) | 11.95 ms (min 11.2) | 25.3 (12.6/frame) | 37.3 (9.3/frame) | 74.5 (9.3/frame) | 9.3 ms |
| `mnn-batch/lib-v2` (+ conv+SiLU fusion, `PHASE3_MNN_FUSION.md`) | 11.22 ms (min 10.5) | 24.1 (12.1/frame) | 35.9 (9.0/frame) | 67.0 (8.4/frame) | 8.4 ms |
| MNN CPU fp32, 4 threads (for scale; wrong tool under ffmpeg load) | 6.7 | 11.4 | 20.8 | 42.0 | 5.2 |

So a batch of 2 costs 2.1x a single frame, a batch of 4 3.1x, 8 6.2x: on this GPU the per-frame kernel time, not
the dispatch count, dominates once the graph is recorded in one command buffer. Phase 1's "~510 dispatches
regardless of batch" is true and irrelevant here. The backbone alone (model cut at the head concats) scales the same
way (12.1 -> 25.3 -> 36.2 -> 63.5 ms min), the DFL tail is 1-2 ms in every form.

## 4. Load sweep on the v2 service (port 5556)

`loadgen.py`: N independent REQ clients (one thread + socket each, exactly like N Frigate detector processes), each
pacing real frames from the 45-frame set at a fixed rate, every reply compared against the batch-1 reference
(`--check`, `--tol 0.01` for batch>1). "In phase" = all clients fire at the same instants (the worst case for the
server: the whole burst queues); "staggered" = phases spread over the period (what N Frigate processes actually look
like). 20-30 s per point, live detector gated to < 60 ms before each run, no other GPU users during any run below.
Service CPU 6-8 % of a core at 20 req/s. `mnn-batch/lib` unless stated.

| point | clients x rate | max batch / window | req/s | timeouts | avg | p50 | p95 | p99 | max | batches (sizes) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mb1 | 4 x 5, in phase | 1 | 20.1 | 0 | **43.8** | 45.0 | **66.6** | 72.0 | 86.5 | 600 x 1 |
| mb2-w0 | 4 x 5, in phase | 2 / 0 ms | 20.1 | 0 | 51.6 | 52.9 | 70.6 | 73.9 | 77.2 | |
| mb2-w2 | | 2 / 2 ms | 20.1 | 0 | 50.2 | 51.1 | 67.8 | 70.0 | 74.8 | |
| mb2-w4 | | 2 / 4 ms | 20.2 | 0 | 59.8 | 59.1 | 104.3 | 114.9 | 120.6 | |
| mb4-w0 | | 4 / 0 ms | 20.2 | 0 | 70.3 | 72.5 | 109.5 | 119.2 | 126.5 | |
| mb4-w2 | | 4 / 2 ms | 20.2 | 0 | 54.7 | 53.5 | 71.7 | 75.5 | 80.5 | |
| mb4-w4 | | 4 / 4 ms | 20.2 | 0 | 54.2 | 52.1 | 73.0 | 76.5 | 78.0 | |
| mb8-w0 | | 8 / 0 ms | 20.2 | 0 | 54.2 | 61.7 | 73.9 | 77.8 | 82.3 | |
| mb8-w4 | | 8 / 4 ms | 20.2 | 0 | 53.2 | 52.2 | 70.9 | 75.3 | 78.4 | |
| mb1-stag | 4 x 5, staggered | 1 | 20.2 | 0 | **18.9** | 17.5 | **28.3** | 31.7 | 33.7 | 400 x 1 |
| mb4-w0-stag | 4 x 5, staggered | 4 / 0 ms | 20.0 | **3** | 21.6 | 17.3 | 31.6 | 100.3 | 247.4 | 391 x 1, 1 x 2, 1 x 3, 1 x 4 (the 4-batch took 61 ms) |
| mb1-8c | 8 x 3, in phase | 1 | 24.4 | 0 | 77.1 | 76.7 | 131.4 | 150.9 | 169.4 | |
| mb4-w0-8c | 8 x 3, in phase | 4 / 0 ms | 24.4 | 0 | 76.6 | 74.0 | 115.7 | 122.5 | 129.2 | |
| mb8-w0-8c | 8 x 3, in phase | 8 / 0 ms | 24.4 | 0 | 94.6 | 101.4 | 121.6 | 136.1 | 138.5 | |
| mb1-2c | 2 x 5, in phase | 1 | 10.1 | 0 | 28.1 | 27.9 | 38.7 | 40.7 | 41.9 | |
| mb2-w0-2c | 2 x 5, in phase | 2 / 0 ms | 10.1 | 0 | 31.6 | 35.2 | 41.2 | 44.0 | 56.0 | |
| v2lib-mb1 | 4 x 5, in phase, **lib-v2** | 1 | 20.2 | 0 | 41.8 | 42.2 | 68.5 | 76.8 | 87.1 | |
| v2lib-mb4-w0 | 4 x 5, in phase, lib-v2 | 4 / 0 ms | 20.2 | 0 | 67.5 | 62.0 | 107.6 | 113.6 | 114.6 | |
| v2lib-mb1-8c | 8 x 3, in phase, lib-v2 | 1 | 24.4 | 0 | **68.5** | 69.0 | **114.0** | 122.6 | 126.2 | |

Reading it: with clients in phase, the service sees a burst of N requests and the average latency is the queueing of
N x 14 ms whatever the batch setting; batching only converts "the last request waits N x 14" into "everyone waits
the batch time", which is longer than the average wait because a batch of N costs ~0.8 N x 14. It helps the tail
only when N is large (8 in phase: p95 131 -> 116 ms) and hurts everywhere else; a wait window makes it worse
(mb2-w4, mb4-w0 vs w2 differ mostly by which requests happened to be grouped). With staggered clients (the realistic
case) the sequential service delivers 18.9 ms average at 20 req/s -- the single-client round trip -- because the GPU
is only ~30 % busy; the batched service reached the same average but produced 3 timeouts. The fused library saves
5-11 % across the board (v2lib rows). `--max-batch 1` it is.

Frigate's real arrival rate is 6-13 req/s and never in phase, so the production numbers in section 6 are the
staggered rows, not the in-phase ones.

## 5. Soak of the final design (port 5556, 13:41-13:51)

v2 service, `--max-batch 1 --batch-window-ms 0`, `mnn-batch/lib-v2`, 4 staggered clients x 3 fps = 12 req/s for
600 s (rate-limited to leave the live detector alone; a guard sampled Frigate's `inference_speed` every 10 s and
would have aborted above 60 ms -- it never fired, max seen 39.5 ms), every reply checked against the batch-1
reference at tolerance 0.01:

| requests | timeouts / errors / mismatches | avg | p50 | p95 | p99 | max | engine (`Inference stats`) | service Python per request | RSS | service CPU | dmesg |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| 7204 (12.0/s, 600 s) | 0 / 0 / 0 | 21.4 ms | 20.3 | 32.2 | 40.8 | 56.4 | 16.0-16.1 ms avg, max 28-32 (GPU shared with the live detector) | recv 0.09 + decode 0.16 + build 0.01 + post 0.50 + send 0.36 = 1.1 ms | 158 MB flat | 28 s / 600 s = 4.7 % of a core | 27 -> 27 AGX/GPU lines |

## 6. Production switch (13:52 service, 14:08 Frigate)

Before: the 10:55 service (REP, `mnn-batch/lib`), one Frigate detector process. Live numbers that morning:
`inference_speed` 19.7 ms EMA at 6 req/s, service `Inference stats` 14.0-15.9 ms avg / 20-24 ms max per 1000
requests, detector process `cpu` 12.5 % in Frigate's stats.

Steps taken (backups `research/frigate-detector.service.before-v2`, `config/config.yaml.bak.20260906-135200-v2`):

1. `detector/frigate-detector.service` (adds `--max-batch 1 --batch-window-ms 0`, `DETECTOR_MNN_LIB` ->
   `mnn-batch/lib-v2/libMNN.so`) copied to `/etc/systemd/system/`, `daemon-reload`, `restart`. Journal: shim
   `libmnn_shim.so`, `libMNN .../lib-v2/libMNN.so`, `Using inference backend: mnn-vulkan(Vulkan, fp16)`,
   `max_batch=1`, no FALLBACK; Frigate re-handshaked 4 s later (`Model ready`).
2. `config/config.yaml`: second `zmq` detector `host_zmq_b` on the same endpoint, `docker compose restart frigate`.
   (The first attempt at 13:53 put the entry under `go2rtc:` -- a generator bug, Frigate ignored it silently and ran
   one detector for 15 minutes on the new service; corrected and restarted again at 14:08.)

After (14:10, both detector processes handshaked at 14:08:42, both answered "already loaded; handshake answered
without reload"): `host_zmq` 24.0 ms / `host_zmq_b` 26.5 ms `inference_speed`, cameras 5.0-5.1 fps, detections
following motion, 21 recording segments written in 2 minutes, no ffmpeg restarts except the dead `cam1`
camera (pre-existing, no route to host), dmesg AGX/GPU lines 27 -> 27. The two detector processes cost 1.9-2.0 %
CPU each in Frigate's stats.

15-minute verification (14:09-14:24, motion-gated real load, no synthetic traffic):

| | value |
|---|---|
| Frigate `inference_speed` (EMA incl. container round trip) | `host_zmq` 24-26 ms, `host_zmq_b` 26-27 ms (single process before: 19.7 ms at 6 req/s; the EMA now also averages the moments when both processes' requests overlap in the service) |
| service `Inference stats` (3 x 1000 requests) | engine 13.4 / 13.6 / 14.5 ms avg, max 20.5-22.7; per-request queue-wait 0.1-0.2 ms avg, **max 1.0-1.8 ms** -- the two Frigate processes rarely collide at this rate |
| service Python per request | recv 0.06 + decode 0.11 + post 0.28 + send 0.24 = 0.7 ms |
| cameras | 4 live cameras at 5.0-5.1 fps, `skipped_fps` 0; `cam1` is the pre-existing dead camera (RTSP 404, ffmpeg restarts every second as before) |
| handshakes / fallbacks / errors in the service journal | 2 (one per detector process, at 14:08:42) / 0 / 0; no `Initializing model` after that in Frigate's log = no 200 ms timeouts |
| recordings | 125 segments written in 15 minutes; 0 ffmpeg restarts on the live cameras |
| host | 88 % idle, load 1.17; service RSS 138 MB, 25.7 s CPU in 32 minutes (1.3 % of a core); Frigate's detector processes 0.3 % each idle, 2 % busy |
| dmesg | 27 -> 27 AGX/GPU lines |

Rollback: `research/frigate-detector.service.before-v2` back to `/etc/systemd/system/` + `daemon-reload` +
`restart` (old REP service, batch lib; the on-disk code is v2 but the old unit runs it with the defaults, which is
the same sequential behaviour), and/or `config/config.yaml.bak.20260906-135200-v2` back + `docker compose restart
frigate` (one detector process). Either half can be reverted alone.

## 7. Files

* Service: `detector/detector/zmq_onnx_client.py` (v2), `backends.py` (batch API, `MnnBackend` multi-session),
  `onnx_batch.py` (fixed-batch ONNX variants + DFL rewrite), `mnn_shim/mnn_shim.cpp` + `libmnn_shim.so`
  (`libmnn_shim.so.v1-20260905` is the old binary), `Makefile` (`make mnn-shim` against `mnn-batch/`),
  `requirements-linux.txt` (+ `onnx`), `frigate-detector.service`, `loadgen.py`; docs `detector/README-linux.md`
  ("v2 service" section), `frigate/README.md` (Object detection row).
* Frigate: `config/config.yaml` (`host_zmq_b`); nothing else in the container changed.
* Scratch (this session's `scratchpad/`): `arrival.py` + `arrival.pcap`, `bisect.cpp` / `bisect2.cpp` (per-tensor
  MNN comparison), `sm_test.py` / `dfl_test.py` / `dfl2_test.py` (isolated softmax / DFL checks), `bench_batch.py`,
  `bench_backend.py`, `time_parts.py`, `sweep.sh` + `sweep/results.jsonl` + per-run service logs, `profile.log` +
  `v2.prof`, `soak2/`, `verify15.log`, `gen_config.py`.
* Sanitised export: `push-staging/gpu-inference-asahi/detector/detector-v2.patch` + `reports/PHASE3_DETECTOR.md`.

## 8. Not done / open

* Shared-memory or uint8 transport (needs the Frigate plugin patched): would remove ~1-2 ms of the ~18 ms round
  trip; not worth a plugin fork at this rate.
* MNN-Vulkan multi-block images (tensors over 16384 rows) are silently wrong in at least `VulkanSoftmax`; the DFL
  rewrite sidesteps it for this model. Worth an upstream issue; not patched here (the other agent owns the MNN tree).
* `resizeSession` to batch>1 on the Vulkan image backend: wrong results here, segfault in the fusion agent's tree.
  Not investigated beyond the bisect.
* If a bigger model or more cameras ever make the GPU the bottleneck (queue-wait max in the stats line climbing past
  a few ms), `--max-batch 2..8` is a one-line change and the fixed-batch variants build themselves at the next load.
