// ncnn C++ bench against a given libncnn (detector-equivalent Vulkan options) + output dump for correctness.
// usage: bench_ncnn3 <param_base> <gpu 0|1> <fp16 0|1> <iters> <out.bin|-> [pack8 0|1] [image_storage 0|1] [threads]
#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sys/resource.h>
static double cpu_ms() { struct rusage r; getrusage(RUSAGE_SELF, &r); return (r.ru_utime.tv_sec + r.ru_stime.tv_sec) * 1000.0 + (r.ru_utime.tv_usec + r.ru_stime.tv_usec) / 1000.0; }
int main(int argc, char** argv) {
    std::string base = argv[1]; int gpu = atoi(argv[2]); int fp16 = atoi(argv[3]); int iters = atoi(argv[4]); std::string outf = argv[5];
    int pack8 = argc > 6 ? atoi(argv[6]) : 1; int imgst = argc > 7 ? atoi(argv[7]) : 0; int threads = argc > 8 ? atoi(argv[8]) : (gpu ? 1 : 4);
    if (gpu) ncnn::create_gpu_instance();
    ncnn::VulkanDevice* vkdev = gpu ? ncnn::get_gpu_device(0) : nullptr;
    ncnn::VkAllocator* blob = gpu ? new ncnn::VkBlobAllocator(vkdev) : nullptr;
    ncnn::VkAllocator* staging = gpu ? new ncnn::VkStagingAllocator(vkdev) : nullptr;
    {
    ncnn::Net net;
    net.opt.use_vulkan_compute = gpu; net.opt.num_threads = threads; net.opt.openmp_blocktime = 0;
    net.opt.use_fp16_packed = fp16; net.opt.use_fp16_storage = fp16; net.opt.use_fp16_arithmetic = fp16; net.opt.use_bf16_storage = false;
    (void)pack8; (void)imgst;
    if (gpu) { net.set_vulkan_device(vkdev); net.opt.blob_vkallocator = blob; net.opt.workspace_vkallocator = blob; net.opt.staging_vkallocator = staging; }
    if (net.load_param((base + ".param").c_str()) || net.load_model((base + ".bin").c_str())) { fprintf(stderr, "load failed\n"); return 1; }
    std::vector<float> x(3 * 320 * 320); std::ifstream("input_320.bin", std::ios::binary).read((char*)x.data(), x.size() * 4);
    ncnn::Mat in(320, 320, 3, x.data());
    ncnn::Mat out;
    auto infer = [&]() { ncnn::Extractor ex = net.create_extractor(); ex.input("images", in); ex.extract("output0", out); };
    for (int i = 0; i < 10; i++) infer();
    std::vector<double> t; double c0 = cpu_ms(); auto w0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) { auto a = std::chrono::steady_clock::now(); infer(); auto b = std::chrono::steady_clock::now(); t.push_back(std::chrono::duration<double, std::milli>(b - a).count()); }
    double c1 = cpu_ms(); double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    std::sort(t.begin(), t.end());
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("ncnn %s gpu=%d fp16=%d pack8=%d img=%d thr=%d: median %.2f ms p95 %.2f min %.2f max %.2f -> %.1f fps | cpu %.2f ms/iter | wall %.1f ms/iter (out %dx%dx%d)\n", base.c_str(), gpu, fp16, pack8, imgst, threads, t[t.size() / 2], t[t.size() * 95 / 100], t[0], t.back(), 1000.0 / t[t.size() / 2], (c1 - c0) / iters, wall / iters, out.w, out.h, out.c);
    if (outf != "-") { std::ofstream f(outf, std::ios::binary); for (int c = 0; c < out.c; c++) for (int hh = 0; hh < out.h; hh++) f.write((const char*)out.channel(c).row(hh), out.w * 4); }
    }
    delete blob; delete staging;
    if (gpu) ncnn::destroy_gpu_instance();
    return 0;
}
