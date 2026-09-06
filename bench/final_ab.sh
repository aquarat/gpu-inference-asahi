#!/bin/bash
# Interleaved final A/B: 3 passes, each config 100 iters (CPU configs 50). Output: results/final_ab.txt
cd "$(dirname "$0")"
PY=${DETECTOR_PY:?path to a python with the ncnn pip wheel}
R=${RESULTS:-.}/final_ab.txt
run() { local label=$1; shift; echo "### $label"; systemd-run --user --scope -p MemoryMax=5G -q "$@" 2>&1 | grep -E "^MNN|^ncnn"; }
: > $R
for pass in 1 2 3; do
  echo "===== pass $pass $(date +%T) load $(cut -d' ' -f1 /proc/loadavg)" | tee -a $R
  {
  run "MNN stock direct fp16"         ./bench_mnn2_stock fox-person-cat-320.mnn 7 2 input_320.bin /dev/null 100 0x4
  run "MNN patched default fp16"      ./bench_mnn2 fox-person-cat-320.mnn 7 2 input_320.bin /dev/null 100 0x4
  run "MNN stock direct fp32"         ./bench_mnn2_stock fox-person-cat-320.mnn 7 1 input_320.bin /dev/null 100 0x4
  run "MNN patched default fp32"      ./bench_mnn2 fox-person-cat-320.mnn 7 1 input_320.bin /dev/null 100 0x4
  run "ncnn pip opt fp16"             $PY bench_ncnn2.py fox-person-cat-320-opt 1 1 100
  run "ncnn src fused+coalesced fp16" ./bench_ncnn3 fox-person-cat-320-fused 1 1 100 -
  run "ncnn pip opt fp32"             $PY bench_ncnn2.py fox-person-cat-320-opt 1 0 100
  run "ncnn src fused+coalesced fp32" ./bench_ncnn3 fox-person-cat-320-fused 1 0 100 -
  run "MNN CPU fp16 4thr"             ./bench_mnn2 fox-person-cat-320.mnn 0 2 input_320.bin /dev/null 50 0x4
  run "ncnn pip CPU fp16 4thr"        $PY bench_ncnn2.py fox-person-cat-320-opt 0 1 50
  } | tee -a $R
done
echo DONE | tee -a $R
