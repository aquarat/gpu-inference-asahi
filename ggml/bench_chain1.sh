#!/bin/bash
cd "$(dirname "$0")"; B=${LLAMA_BUILD:-llama.cpp/build-vk/bin}; R=results; mkdir -p $R
echo "### llama-bench GPU (Vulkan0), Q8_0 + fp16, pp512/tg128, fa on/off, 3 reps"
for m in q8_0 fp16; do systemd-run --user --scope -p MemoryMax=5G --quiet $B/llama-bench -m models/qwen2.5-0.5b-instruct-$m.gguf -dev Vulkan0 -ngl 99 -fa 0,1 -p 512 -n 128 -t 4 -r 3 -o md 2>&1 | grep -E "^\||ggml_vulkan|device"; done
echo "### llama-bench CPU only (-dev none), 4 threads and 8 threads"
for m in q8_0 fp16; do systemd-run --user --scope -p MemoryMax=5G --quiet $B/llama-bench -m models/qwen2.5-0.5b-instruct-$m.gguf -dev none -ngl 0 -fa 0 -p 512 -n 128 -t 4,8 -r 3 -o md 2>&1 | grep -E "^\|"; done
echo "### llama-bench GPU pp at 128/512/1024 (GEMM scaling), q8_0 fa=1"
systemd-run --user --scope -p MemoryMax=5G --quiet $B/llama-bench -m models/qwen2.5-0.5b-instruct-q8_0.gguf -dev Vulkan0 -ngl 99 -fa 1 -p 128,1024 -n 0 -t 4 -r 3 -o md 2>&1 | grep -E "^\|"
echo "### ViT-B-32 variants on Vulkan"
./run_clip.sh vitb32_f32_vk models/vitb32_f32.gguf ref/vitb32 Vulkan0 1 5 4 0 4G
./run_clip.sh vitb32_q8_vk models/vitb32_q8_0.gguf ref/vitb32 Vulkan0 1 5 4 0 4G
./run_clip.sh vitb32_f16_vk_fa models/vitb32_f16.gguf ref/vitb32 Vulkan0 1 5 4 1 4G
GGML_VK_DISABLE_F16=1 ./run_clip.sh vitb32_f16_vk_nof16 models/vitb32_f16.gguf ref/vitb32 Vulkan0 1 5 4 0 4G
echo "### ViT-B-32 on ggml CPU (f16, q8_0, f32), 4 threads"
./run_clip.sh vitb32_f16_cpu models/vitb32_f16.gguf ref/vitb32 CPU 1 5 4 0 4G
./run_clip.sh vitb32_q8_cpu models/vitb32_q8_0.gguf ref/vitb32 CPU 1 5 4 0 4G
./run_clip.sh vitb32_f32_cpu models/vitb32_f32.gguf ref/vitb32 CPU 1 5 4 0 4G
echo "### ORT references (quiet CPU): ViT-B-32 and ViT-L-14 timing, ViT-H embeddings + timing"
free -h | head -2
python3 ref_ort.py ${IMMICH_CACHE:?set to the Immich model cache}/clip/ViT-B-32__openai ref/vitb32 4 10 images/shapes.png images/noise.png images/checker.png images/t1.jpg 2>&1 | grep -v cpuid | tail -1
systemd-run --user --scope -p MemoryMax=4G --quiet python3 ref_ort.py ${IMMICH_CACHE:?set to the Immich model cache}/clip/ViT-L-14__openai ref/vitl14 4 3 images/shapes.png images/noise.png images/checker.png images/t1.jpg 2>&1 | grep -v cpuid | tail -1
systemd-run --user --scope -p MemoryMax=5G --quiet python3 ref_ort.py ${IMMICH_CACHE:?set to the Immich model cache}/clip/ViT-H-14-378-quickgelu__dfn5b ref/vith14 4 2 images/shapes.png images/noise.png images/checker.png images/t1.jpg 2>&1 | grep -v cpuid
echo CHAIN1_DONE
