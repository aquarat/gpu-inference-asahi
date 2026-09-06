#!/bin/bash
cd "$(dirname "$0")"; B=${LLAMA_BUILD:-llama.cpp/build-vk/bin}
echo "### ViT-H Vulkan Q8_0 + flash-attn"
./run_clip.sh vith14_q8_vk_fa models/vith14_q8_0.gguf ref/vith14 Vulkan0 1 3 4 1 5G
echo "### ViT-H Vulkan f16 weights, fp32 shader math (GGML_VK_DISABLE_F16=1), fa=1"
GGML_VK_DISABLE_F16=1 ./run_clip.sh vith14_f16_vk_fa_nof16 models/vith14_f16.gguf ref/vith14 Vulkan0 1 3 4 1 5G
echo "### test-backend-ops perf MUL_MAT on Vulkan0 (raw GEMM ceiling)"
timeout 300 systemd-run --user --scope -p MemoryMax=4G --quiet $B/test-backend-ops perf -o MUL_MAT -b Vulkan0 2>&1 | grep -E "MUL_MAT\(type_a=(f16|f32|q8_0),type_b=f32.*m=.*GFLOPS|Backend|error" | head -40
echo "### test-backend-ops perf MUL_MAT on CPU"
timeout 300 systemd-run --user --scope -p MemoryMax=4G --quiet $B/test-backend-ops perf -o MUL_MAT -b CPU 2>&1 | grep -E "MUL_MAT\(type_a=(f16|f32|q8_0),type_b=f32.*m=.*GFLOPS" | head -40
echo "### frigate fallback / kernel GPU messages"; journalctl -u frigate-detector --since -90min 2>/dev/null | grep -ci fallback; journalctl -k --since -90min 2>/dev/null | grep -iE "agx|gpu|fault|timeout" | tail -5
echo CHAIN3_DONE
