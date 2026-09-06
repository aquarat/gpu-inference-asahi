#!/bin/bash
VK=${VK:-$(dirname "$(readlink -f "$0")")}; FRIGATE_HOME=${FRIGATE_HOME:-$HOME/frigate}; IMMICH_WORK=${IMMICH_WORK:-$HOME/immich-ml/work}
V=$VK; WP=$V/with-patched.sh; WU=$V/with-upstream.sh; R=$V/results; MM=$IMMICH_WORK/../models/mnn
cd $IMMICH_WORK; source $IMMICH_WORK/../venv/bin/activate
arm() { local name=$1; shift; printf "%-20s " "$name"; "$@" ./mnnrun/mnnrun $MM/clip_ViT-B-32__openai_visual.mnn 7 2 4 3 30 $R/up_$name image=ref/vitb32_buffalo/clip_visual_input.bin:f32:1x3x224x224 2>&1 | grep -E 'RESULT' | sed -E 's/RESULT type=7\(resolved 7\) prec=2 thr=4: //; s/\| mnn session.*//'; md5sum $R/up_$name/*.bin | cut -c1-12; }
echo "--- ViT-B-32 fp16 MNN-Vulkan, 30 iters"
for pass in 1 2; do echo "pass $pass"; arm stock env; arm upstream-26.2.2 $WU; arm patched $WP; done
cd $FRIGATE_HOME/research
echo "--- YOLO MNN fp16"; for pass in 1 2; do printf "upstream-26.2.2  "; $WU ./bench_mnn_cpp mnn_models/fox-person-cat-320.mnn 7 2 input_320.bin $R/mnn_out_upstream_p2_pass$pass.bin 100 2>&1 | grep MNN; done; md5sum $R/mnn_out_upstream_p2_pass1.bin
echo "--- YOLO ncnn fp16 (bench_ncnn_cpp)"; printf "upstream-26.2.2  "; $WU ./bench_ncnn_cpp ncnn_models/fox-person-cat-320 1 1 4 100 2>&1 | grep median
cd $V/tests
echo "--- chaintest upstream-26.2.2"; for m in dep indep; do $WU ./chaintest $m 1000 1 1000 3 2>&1 | grep -E '^(dep|indep)'; done
echo "--- ViT-B-32 fp16 HK-free CPU baseline for context (MNN CPU fp16)"; cd $IMMICH_WORK; ./mnnrun/mnnrun $MM/clip_ViT-B-32__openai_visual.mnn 0 2 4 3 30 $R/up_cpu image=ref/vitb32_buffalo/clip_visual_input.bin:f32:1x3x224x224 2>&1 | grep RESULT | sed -E 's/\| mnn session.*//'
echo UP_DONE
