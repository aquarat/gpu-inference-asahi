#!/bin/bash
VK=${VK:-$(dirname "$(readlink -f "$0")")}; FRIGATE_HOME=${FRIGATE_HOME:-$HOME/frigate}; IMMICH_WORK=${IMMICH_WORK:-$HOME/immich-ml/work}
# Sequential: clpeak latency (3x each) then YOLOv9-t inference on ncnn and MNN, stock vs patched.
WP=$VK/with-patched.sh
R=$VK/results
RS=$FRIGATE_HOME/research
M=$FRIGATE_HOME/config/model_cache
run() { local label=$1; shift; echo "### $label :: $*"; "$@" 2>&1 | grep -v 'MESA: error: Opening'; }
{
for i in 1 2 3; do
  run "stock run$i"   clpeak --latency --vulkan --vk-device 0
  run "patched run$i" $WP clpeak --latency --vulkan --vk-device 0
done
} > $R/clpeak.txt
cd $RS
source venv314/bin/activate
{
for pass in 1 2; do
  run "stock pass$pass"   python bench_ncnn.py ncnn_models/fox-person-cat-320 $M/fox-person-cat-320.onnx 100
  run "patched pass$pass" $WP python bench_ncnn.py ncnn_models/fox-person-cat-320 $M/fox-person-cat-320.onnx 100
done
} > $R/yolo_ncnn.txt
{
for pass in 1 2; do
  for prec in 1 2; do
    run "stock pass$pass prec$prec"   ./bench_mnn_cpp mnn_models/fox-person-cat-320.mnn 7 $prec input_320.bin $R/mnn_out_stock_p${prec}_pass$pass.bin 100
    run "patched pass$pass prec$prec" $WP ./bench_mnn_cpp mnn_models/fox-person-cat-320.mnn 7 $prec input_320.bin $R/mnn_out_patched_p${prec}_pass$pass.bin 100
  done
done
} > $R/yolo_mnn.txt
echo YOLO_DONE
