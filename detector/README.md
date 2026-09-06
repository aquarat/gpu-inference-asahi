# Frigate host detector service: v2 (ROUTER socket, batching knobs, zero-copy)

`detector-v2.patch` is a plain `diff -ruN` against the `detector/` tree of
[aquarat/frigate-asahi](https://github.com/aquarat/frigate-asahi) (the 2026-09-05 export) and
brings the host-side ZMQ detector service to the state measured in
`../reports/PHASE3_DETECTOR.md`:

| file | change |
|---|---|
| `detector/zmq_onnx_client.py` | REP -> ROUTER socket (Frigate's REQ clients unchanged), so N Frigate detector processes can have requests in flight; drain-then-run batching queue (`--max-batch`, `--batch-window-ms`); tensors used straight from the ZMQ frame; a `model_request` for the already-loaded file is answered without reloading; per-stage timing in the stats line |
| `detector/backends.py` | `batch_sizes` / `infer_batch()` on the backend interface; `MnnBackend` builds one MNN session per batch size (1 and powers of two up to `DETECTOR_MAX_BATCH`) from fixed-batch model files |
| `detector/onnx_batch.py` | derives a fixed-batch-N ONNX from the batch-1 export (Reshape constants, input/output dims) and rewrites the YOLO DFL head as a middle-axis softmax + weighted sum, which keeps MNN's flattened softmax under the 16384-row Vulkan image limit |
| `mnn_shim/mnn_shim.cpp` | `mnn_shim_create_multi()` (one `.mnn` per batch size, sessions created at their batch -- a *resized* session is wrong on the Vulkan image backend), `mnn_shim_infer_batch()`, host tensors wrapping the caller's buffers; v1 entry points kept |
| `loadgen.py`, `gen_frigate_detectors_config.py` | N-client load generator with reply checking; helper that adds `host_zmq_b` .. to Frigate's `detectors:` |
| `frigate-detector.service`, `Makefile`, `requirements-linux.txt`, `README-linux.md` | unit with `--max-batch 1` and the conv+SiLU fused library, shim build against `mnn-batch/`, `onnx` wheel, docs |

Outcome (details and every number in the report): batching several cameras' frames into one MNN-Vulkan
inference is correct but does not pay on the M1 (a batch of 2/4/8 costs 2.1/3.1/6.2x one frame), so production
runs `--max-batch 1` with two Frigate detector processes on the ROUTER service.
