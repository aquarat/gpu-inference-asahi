# Raw logs

The files the reports quote, as produced (home-directory paths shortened to
`~`, journal hostnames removed). Binary outputs (`perf.data`, model outputs)
are not included.

| directory | contents | report |
|---|---|---|
| `vk-dispatch/` | `chaintest.txt`, `cstest.txt`, `clpeak.txt` (micro), `yolo_{mnn,ncnn}.txt`, `clip_{vitb32,vith}.txt`, `gputime_*.txt` (`HK_GPUTIME` firmware profiler timelines), `ablation.txt`, `upstream_ab.txt`, `detector_*.log` / `client_*.log` (the detector A/B) | `VULKAN_DISPATCH_EXPERIMENT.md` |
| `phase1/` | `gputime_mnn_*` and `gputime_ncnn_*` profiler runs before/after each change, `strace_*.txt` (`strace -c` ioctl counts), `perf_mnn_summary.txt`, `ncnn_pip_option_sweep_gputime.txt`, `mnnconvert_optlevels.txt`, `final_ab.txt` (the interleaved 3-pass A/B) | `PHASE1_RUNTIME.md` |
| `ggml/` | `results_chain1.txt` (llama-bench, ViT-B-32, ORT references), `results_chain2.txt` (ViT-L, ViT-H), `results_chain3.txt` (ViT-H precision variants, `test-backend-ops` GEMM figures) | `PHASE2_GGML.md` |
