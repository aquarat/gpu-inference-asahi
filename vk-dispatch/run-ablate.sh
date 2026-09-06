#!/bin/bash
VK=${VK:-$(dirname "$(readlink -f "$0")")}; FRIGATE_HOME=${FRIGATE_HOME:-$HOME/frigate}; IMMICH_WORK=${IMMICH_WORK:-$HOME/immich-ml/work}
# Ablation of the CLIP gain: which driver change is responsible? Alternating arms, 2 passes.
WP=$VK/with-patched.sh
R=$VK/results
MM=$IMMICH_WORK/../models/mnn
cd $IMMICH_WORK; source $IMMICH_WORK/../venv/bin/activate
arm() { # name env-prefix...
  local name=$1; shift
  printf "%-28s " "$name"; "$@" ./mnnrun/mnnrun $MM/clip_ViT-B-32__openai_visual.mnn 7 2 4 3 30 $R/abl_$name image=ref/vitb32_buffalo/clip_visual_input.bin:f32:1x3x224x224 2>&1 | grep -E 'RESULT|HK_PERFTEST active' | sed -E 's/RESULT type=7\(resolved 7\) prec=2 thr=4: //; s/\| mnn session.*//' | tr '\n' ' '; echo
}
for pass in 1 2; do
  echo "--- pass $pass"
  arm stock                     env
  arm patched                   $WP
  arm patched-nooverlap         env HK_PERFTEST=nooverlap $WP
  arm patched-noconstdata       env HK_PERFTEST=noconstdata $WP
  arm patched-nooverlap-noconst env HK_PERFTEST=nooverlap,noconstdata $WP
done
echo "--- ViT-H fp16, patched nooverlap (1 warmup + 3 iters)"
free -h | head -2
o=$R/clip_vith_patched_nooverlap_p2; mkdir -p $o
HK_PERFTEST=nooverlap $WP timeout 900 ./mnnrun/mnnrun $MM/clip_ViT-H-14-378_visual_fp16.mnn 7 2 4 1 3 $o image=ref/vith/clip_visual_input.bin:f32:1x3x378x378 2>&1 | grep -E 'RESULT|HK_PERFTEST active' | sort -u
md5sum $o/*.bin
echo ABL_DONE
