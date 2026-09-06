// phase breakdown: copyFromHost / runSession / copyToHost per inference
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <fstream>
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size()/2]; }
int main(int argc, char** argv) {
    auto net = MNN::Interpreter::createFromFile(argv[1]);
    int ft = atoi(argv[2]), prec = atoi(argv[3]); int iters = argc > 5 ? atoi(argv[5]) : 100; int mode = argc > 6 ? (int)strtol(argv[6], nullptr, 0) : 0x4;
    MNN::ScheduleConfig cfg; cfg.type = (MNNForwardType)ft; cfg.numThread = 4; cfg.mode = mode;
    MNN::BackendConfig bc; bc.precision = (MNN::BackendConfig::PrecisionMode)prec; cfg.backendConfig = &bc;
    auto sess = net->createSession(cfg);
    auto in = net->getSessionInput(sess, nullptr); auto out = net->getSessionOutput(sess, nullptr);
    std::vector<float> x(3 * 320 * 320); std::ifstream(argv[4], std::ios::binary).read((char*)x.data(), x.size() * 4);
    MNN::Tensor hin(in, MNN::Tensor::CAFFE); std::copy(x.begin(), x.end(), hin.host<float>());
    MNN::Tensor hout(out, MNN::Tensor::CAFFE);
    std::vector<double> t1, t2, t3, tt;
    for (int i = 0; i < iters + 10; i++) {
        auto a = clk::now(); in->copyFromHostTensor(&hin); auto b = clk::now(); net->runSession(sess); auto c = clk::now(); out->copyToHostTensor(&hout); auto d = clk::now();
        if (i >= 10) { t1.push_back(ms(a,b)); t2.push_back(ms(b,c)); t3.push_back(ms(c,d)); tt.push_back(ms(a,d)); }
    }
    printf("MNN type=%d prec=%d mode=0x%x: total median %.2f ms | copyFromHost %.3f | runSession %.3f | copyToHost %.3f (medians)\n", ft, prec, mode, med(tt), med(t1), med(t2), med(t3));
    return 0;
}
