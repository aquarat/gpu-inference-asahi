#!/bin/bash
VK=${VK:-$(dirname "$(readlink -f "$0")")}; FRIGATE_HOME=${FRIGATE_HOME:-$HOME/frigate}; IMMICH_WORK=${IMMICH_WORK:-$HOME/immich-ml/work}
# Microbenchmarks stock vs patched. Sequential so runs never overlap on the GPU.
cd $VK/tests
WP=$VK/with-patched.sh
R=$VK/results
run() { # label cmd...
  local label=$1; shift
  echo "### $label :: $*"
  "$@" 2>&1 | grep -v 'MESA: error: Opening'
}
{
for cfg in "1 0" "64 0" "512 0" "1 1000" "64 1000" "512 1000"; do
  set -- $cfg
  for mode in dep indep; do
    run "stock"            ./chaintest $mode 1000 $1 $2 3
    run "patched"          $WP ./chaintest $mode 1000 $1 $2 3
    run "patched-nooverlap" env HK_PERFTEST=nooverlap $WP ./chaintest $mode 1000 $1 $2 3
  done
done
} > $R/chaintest.txt
{
for i in 1 2 3; do
  run "stock run$i"   ./cstest cstest.spv
  run "patched run$i" $WP ./cstest cstest.spv
done
} > $R/cstest.txt
{
for i in 1 2 3; do
  run "stock run$i"   clpeak --kernel-latency --vulkan
  run "patched run$i" $WP clpeak --kernel-latency --vulkan
done
} > $R/clpeak.txt
echo MICRO_DONE
