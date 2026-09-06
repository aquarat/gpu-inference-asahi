// MNN bench with gpu mode + CPU-time accounting.
// usage: bench_mnn2 model.mnn <forwardType 0/7> <precision 1=fp32 2=fp16> input.bin output.bin [iters] [mode hex, default 0x4] [warmup]
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sys/resource.h>
static double cpu_ms() { struct rusage r; getrusage(RUSAGE_SELF, &r); return (r.ru_utime.tv_sec + r.ru_stime.tv_sec) * 1000.0 + (r.ru_utime.tv_usec + r.ru_stime.tv_usec) / 1000.0; }
int main(int argc, char** argv) {
    auto net = MNN::Interpreter::createFromFile(argv[1]);
    int ft = atoi(argv[2]), prec = atoi(argv[3]); int iters = argc > 6 ? atoi(argv[6]) : 100;
    int mode = argc > 7 ? (int)strtol(argv[7], nullptr, 0) : 0x4; int warm = argc > 8 ? atoi(argv[8]) : 10;
    MNN::ScheduleConfig cfg; cfg.type = (MNNForwardType)ft; cfg.numThread = 4; cfg.mode = mode;
    MNN::BackendConfig bc; bc.precision = (MNN::BackendConfig::PrecisionMode)prec; cfg.backendConfig = &bc;
    auto sess = net->createSession(cfg);
    auto in = net->getSessionInput(sess, nullptr); auto out = net->getSessionOutput(sess, nullptr);
    std::vector<float> x(3 * 320 * 320); std::ifstream(argv[4], std::ios::binary).read((char*)x.data(), x.size() * 4);
    MNN::Tensor hin(in, MNN::Tensor::CAFFE); std::copy(x.begin(), x.end(), hin.host<float>());
    MNN::Tensor hout(out, MNN::Tensor::CAFFE);
    auto infer = [&]() { in->copyFromHostTensor(&hin); net->runSession(sess); out->copyToHostTensor(&hout); };
    for (int i = 0; i < warm; i++) infer();
    std::vector<double> t; double c0 = cpu_ms(); auto w0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) { auto a = std::chrono::steady_clock::now(); infer(); auto b = std::chrono::steady_clock::now(); t.push_back(std::chrono::duration<double, std::milli>(b - a).count()); }
    double c1 = cpu_ms(); double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    std::sort(t.begin(), t.end());
    printf("MNN type=%d prec=%d mode=0x%x: median %.2f ms p95 %.2f min %.2f max %.2f -> %.1f fps | cpu %.2f ms/iter | wall %.1f ms/iter (out %d elems)\n", ft, prec, mode, t[t.size()/2], t[t.size()*95/100], t[0], t.back(), 1000.0/t[t.size()/2], (c1-c0)/iters, wall/iters, hout.elementSize()); fflush(stdout);
    std::ofstream(argv[5], std::ios::binary).write((char*)hout.host<float>(), hout.elementSize() * 4);
    return 0;
}
