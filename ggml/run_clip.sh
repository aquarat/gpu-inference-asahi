#!/bin/bash
# usage: run_clip.sh <tag> <model.gguf> <refdir> <backend> <warmup> <iters> <threads> <fa> [MemoryMax]
# runs all 4 test images, prints per-image RESULT + cosine vs ORT, then the overall median.
set -u
tag=$1; model=$2; ref=$3; backend=$4; warm=$5; iters=$6; thr=$7; fa=$8; mem=${9:-5G}
cd "$(dirname "$0")"; mkdir -p out/$tag
: > out/$tag/log.txt
for img in shapes noise checker t1; do
  systemd-run --user --scope -p MemoryMax=$mem --quiet ./clip_vit $model $backend $ref/$img.in.bin out/$tag/$img.bin $warm $iters $thr $fa >> out/$tag/log.txt 2>&1
  tail -1 out/$tag/log.txt | sed "s/^/$img: /"
done
python3 cmp.py $ref out/$tag
grep -h "^RESULT" out/$tag/log.txt | sed 's/.*median_ms=\([0-9.]*\).*/\1/' | sort -n | awk '{a[NR]=$1} END{print "OVERALL '"$tag"' median-of-medians:", a[int((NR+1)/2)], "ms over", NR, "images"}'
grep -h "backend:\|graph:\|weights:" out/$tag/log.txt | sort -u | head -3
