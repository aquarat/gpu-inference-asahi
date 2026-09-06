#!/bin/bash
VK=${VK:-$(dirname "$(readlink -f "$0")")}; FRIGATE_HOME=${FRIGATE_HOME:-$HOME/frigate}; IMMICH_WORK=${IMMICH_WORK:-$HOME/immich-ml/work}
# CLIP ViT-B-32 visual and ViT-H-14-378 visual fp16 on MNN-Vulkan, stock vs patched; then HK_GPUTIME attribution of MNN YOLO.
WP=$VK/with-patched.sh
R=$VK/results
W=$IMMICH_WORK
MM=$IMMICH_WORK/../models/mnn
run() { local label=$1; shift; echo "### $label :: $*"; "$@" 2>&1 | grep -vE 'MESA: error: Opening|CPU Group|device supports'; }
cd $W; source $IMMICH_WORK/../venv/bin/activate
{
for pass in 1 2; do
 for prec in 1 2; do
  for drv in stock patched; do
    o=$R/clip_vitb32_${drv}_p${prec}_pass$pass; mkdir -p $o
    if [ $drv = patched ]; then pre=$WP; else pre=env; fi
    run "$drv pass$pass prec$prec" $pre ./mnnrun/mnnrun $MM/clip_ViT-B-32__openai_visual.mnn 7 $prec 4 3 20 $o image=ref/vitb32_buffalo/clip_visual_input.bin:f32:1x3x224x224
    python compare.py $o ref/vitb32_buffalo/clip_visual_ref.npy
    md5sum $o/*.bin
  done
 done
done
} > $R/clip_vitb32.txt
echo "free before ViT-H:"; free -h | head -2
{
for drv in stock patched; do
  o=$R/clip_vith_${drv}_p2; mkdir -p $o
  if [ $drv = patched ]; then pre=$WP; else pre=env; fi
  free -h | head -2
  run "$drv fp16 weights, prec2" $pre timeout 900 ./mnnrun/mnnrun $MM/clip_ViT-H-14-378_visual_fp16.mnn 7 2 4 1 3 $o image=ref/vith/clip_visual_input.bin:f32:1x3x378x378
  python compare.py $o ref/vith/clip_visual_ref.npy
  md5sum $o/*.bin
done
} > $R/clip_vith.txt
cd $FRIGATE_HOME/research
{
run "patched HK_GPUTIME default"   env HK_GPUTIME=1 $WP ./bench_mnn_cpp mnn_models/fox-person-cat-320.mnn 7 2 input_320.bin /dev/null 200
run "patched HK_GPUTIME nooverlap" env HK_GPUTIME=1 HK_PERFTEST=nooverlap $WP ./bench_mnn_cpp mnn_models/fox-person-cat-320.mnn 7 2 input_320.bin /dev/null 200
} > $R/gputime_mnn_yolo.txt
echo CLIP_DONE
