# The dispatch experiment

Does the got-bringup Honeykrisp fork (weak CDM barrier between compute
dispatches) reduce per-dispatch overhead for ML inference? Write-up:
[../reports/VULKAN_DISPATCH_EXPERIMENT.md](../reports/VULKAN_DISPATCH_EXPERIMENT.md).
Short answer: 10-45x on chains with no barrier between dispatches, nothing on
dependent chains, so nothing for inference; the CLIP gain it did show comes
from other commits in the fork.

## Loading a private driver build per process

The fork's RPM (`honeykrisp-got-*.rpm` from the got-bringup dnf repository)
was downloaded, signature-checked in a throw-away rpmdb, and extracted with
`rpm2cpio | cpio` into `patched-mesa/`, never installed. `patched_icd.json`
is a copy of the system ICD with `library_path` pointing at the extracted
`libvulkan_asahi.so` (fill in the absolute path); `with-patched.sh` exports
`VK_DRIVER_FILES`/`VK_ICD_FILENAMES` at it and execs its arguments. Only that
process and its children see the fork; the system driver, the compositor (if
any) and the live services are untouched. `ldd` on the extracted library
resolved every dependency against the system libraries, so no
`LD_LIBRARY_PATH` was needed. `with-upstream.sh`/`upstream_icd.json` do the
same for a plain Fedora rawhide Mesa 26.2.2 package, used as the control arm
that separates upstream progress from the fork's changes.

## Tests

`tests/chaintest.c` + `chaintest.comp` (the key number): 1000 back-to-back
`vkCmdDispatch` of a trivial shader in one command buffer, one submit, in two
shapes. *dependent*: every dispatch reads and writes the same buffer with a
compute-to-compute buffer memory barrier between consecutive dispatches (the
shape of an inference graph; the result is verified). *independent*: each
dispatch has its own slice and there is no barrier. Metric: CPU wall from
`vkQueueSubmit` to `vkQueueWaitIdle` divided by N, median of 3, with an
optional serial loop per invocation to give each dispatch real latency.

```sh
glslangValidator -V tests/chaintest.comp -o tests/chaintest.spv
cc -O2 -o tests/chaintest tests/chaintest.c -lvulkan
tests/chaintest dep 1000 1 0 3        # dep|indep  N  workgroups  loops  reps
```

GPU timestamps were not used for the metric because on Mesa 26.1.8 the
bottom-of-pipe timestamp is written without waiting for un-barriered
dispatches (it reported 0.6 ms for work that takes 3 ms). got-bringup's
`tests/cstest.c` and `tests/coherence.c` were also run unchanged; they are in
that repository.

## Run scripts

They read `VK` (this directory), `FRIGATE_HOME` (the frigate-asahi tree, for
`research/bench_*.py`, `bench_mnn_cpp`, the model cache and the detector) and
`IMMICH_WORK` (the Immich-side MNN runner `mnnrun`, published with the Immich
fork) from the environment.

| script | what |
|---|---|
| `run-micro.sh` | chain test x {stock, patched, patched `HK_PERFTEST=nooverlap`} x 6 shapes; `cstest`; clpeak kernel latency |
| `run-clpeak-yolo.sh` | clpeak latency and YOLOv9-t on ncnn-Vulkan and MNN-Vulkan, stock vs patched, outputs saved for md5 comparison |
| `run-clip.sh` | CLIP ViT-B-32 and ViT-H-14-378 on MNN-Vulkan stock vs patched, then the `HK_GPUTIME=1` firmware-profiler attribution of MNN's YOLO run |
| `run-ablate.sh` | which fork change is responsible for the CLIP gain: `nooverlap`, `noconstdata`, both |
| `run-upstream.sh` | the plain Mesa 26.2.2 control arm |
| `run-detector-ab.sh`, `run-detector-arm.sh` | a second, private instance of the NVR's ZMQ detector on another port, 60 s at 25 requests/s per arm, plus dmesg/journal checks of the live service |

Raw outputs are in `../results/vk-dispatch/`.
