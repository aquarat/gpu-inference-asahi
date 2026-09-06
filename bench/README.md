# YOLOv9-t benchmarks behind PHASE1_RUNTIME.md

All take a 320x320 NCHW f32 input file (`input_320.bin`, seed-0 random in the
report) and the converted models (`.mnn` from `mnnconvert`, `.param/.bin` from
`onnx2ncnn` + `ncnnoptimize`); neither is included.

| file | what |
|---|---|
| `bench_mnn2.cpp` | MNN: `bench_mnn2 model.mnn <forwardType 0=CPU/7=Vulkan> <precision 1=fp32/2=fp16> input.bin output.bin [iters] [gpuMode hex, default 0x4] [warmup]`. Reports median/p95/min/max, CPU time per iteration (`getrusage`), and writes the output for `cmp.py`. Build: `g++ -O2 -std=c++17 bench_mnn2.cpp -I<MNN>/include -L<MNN>/build -lMNN`. |
| `bench_mnn_phases.cpp` | Same session, but times `copyFromHostTensor` / `runSession` / `copyToHostTensor` separately. |
| `bench_ncnn3.cpp` | ncnn against a given `libncnn` with the detector's Vulkan options (fp16 packed/storage/arithmetic, blob + staging allocators): `bench_ncnn3 <param_base> <gpu 0/1> <fp16 0/1> <iters> <out.bin|->`. Reads `input_320.bin` from the current directory. Build: `g++ -O2 -std=c++17 bench_ncnn3.cpp -I<ncnn>/build/src -I<ncnn>/src -L<ncnn>/build/src -lncnn -fopenmp`. |
| `bench_ncnn2.py` | The same options through the pip wheel (`ncnn==1.0.20260526`). |
| `cmp.py` | Max/mean abs diff and md5 of two output files (vs the onnxruntime fp32 reference). |
| `final_ab.sh` | The interleaved 3-pass A/B of section 5 of the report (MNN stock vs patched, ncnn pip vs fused+coalesced source build, CPU references). Needs `DETECTOR_PY` (a python with the ncnn wheel) and the built binaries next to it. |

Everything ran under `systemd-run --user --scope -p MemoryMax=5G` on a 16 GB
machine shared with a live NVR; the report gives the contention caveats.
