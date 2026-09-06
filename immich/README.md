# Immich machine-learning: ggml-Vulkan CLIP session backend

Three commits on top of immich-app/immich `main` @ 4c7b30c (2026-09-06), as `git am`-able patches:

1. `libclipvit` — a C library (ggml + ggml-vulkan statically linked) that runs CLIP ViT image encoders from GGUF, with a converter from Immich's own ONNX files (f16 / q8_0).
2. `GgmlSession` — a session backend selected only for CLIP *visual* models when `MACHINE_LEARNING_GGML=true`; text tower, faces and OCR stay on ONNX Runtime. Knobs: `MACHINE_LEARNING_GGML_PRECISION` (fp16 shader math, or fp32 for exact results), `_WEIGHT_TYPE` (f16/q8_0), `_FLASH_ATTN`, `_THREADS`, `DEVICE_ID`.
3. Docker `-ggml` image stages, `hwaccel.ml.yml` entry and docs.

Measured on an Apple M1 (Honeykrisp, Mesa 26.1.8): ViT-H-14-378 image embedding 1.8–1.9 s vs 3.2 s on ORT CPU, cosine 0.9985–0.9994 (fp16 math) / 0.99999 (fp32), ~1.1–1.7 GB less resident memory. When the GPU is shared with a latency-sensitive client (an object detector with a 200 ms budget), run the service with `GGML_VK_MAX_NODES_PER_SUBMIT=1`: it costs ~1 % CLIP throughput and keeps the detector's p95 under 50 ms during embedding bursts; the default submission size pushed 88 % of detector requests past 200 ms.

Not submitted upstream: Immich's CONTRIBUTING asks for a Discord discussion before large changes and declines LLM-generated PRs; see the deployment write-up in the companion `frigate-asahi`-style repository for the full evaluation.
