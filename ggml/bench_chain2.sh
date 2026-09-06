#!/bin/bash
# ViT-L-14 and ViT-H-14-378: Vulkan (f16, q8_0, fa on/off) and ggml CPU (f16, q8_0), cosine vs ORT
cd "$(dirname "$0")"
chk() { echo "--- $(date +%T) free: $(free -h | awk '/Mem/{print $7}') avail; other GPU jobs: $(pgrep -af 'bench_|test_client|mnnrun|ncnn_' | grep -v pgrep | grep -v 'bash -c' | wc -l)"; }
echo "### ViT-L-14 (224 px, 24 x 1024, 257 tokens) Vulkan"; chk
./run_clip.sh vitl14_f16_vk models/vitl14_f16.gguf ref/vitl14 Vulkan0 1 5 4 0 4G
./run_clip.sh vitl14_f16_vk_fa models/vitl14_f16.gguf ref/vitl14 Vulkan0 1 5 4 1 4G
./run_clip.sh vitl14_q8_vk models/vitl14_q8_0.gguf ref/vitl14 Vulkan0 1 5 4 0 4G
echo "### ViT-L-14 ggml CPU 4 threads"; chk
./run_clip.sh vitl14_f16_cpu models/vitl14_f16.gguf ref/vitl14 CPU 1 3 4 0 4G
./run_clip.sh vitl14_q8_cpu models/vitl14_q8_0.gguf ref/vitl14 CPU 1 3 4 0 4G
echo "### ViT-H-14-378 (378 px, 32 x 1280, 730 tokens) Vulkan"; chk
./run_clip.sh vith14_f16_vk models/vith14_f16.gguf ref/vith14 Vulkan0 1 3 4 0 5G
chk
./run_clip.sh vith14_f16_vk_fa models/vith14_f16.gguf ref/vith14 Vulkan0 1 3 4 1 5G
chk
./run_clip.sh vith14_q8_vk models/vith14_q8_0.gguf ref/vith14 Vulkan0 1 3 4 0 5G
echo "### ViT-H-14-378 ggml CPU 4 threads"; chk
./run_clip.sh vith14_f16_cpu models/vith14_f16.gguf ref/vith14 CPU 1 2 4 0 5G
./run_clip.sh vith14_q8_cpu models/vith14_q8_0.gguf ref/vith14 CPU 1 2 4 0 5G
echo "### frigate-detector fallback check"; journalctl -u frigate-detector --since -60min 2>/dev/null | grep -ci fallback
echo CHAIN2_DONE
