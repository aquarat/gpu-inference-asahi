# ncnn pip-wheel bench mirroring the detector's Vulkan options. usage: bench_ncnn2.py <param_base> <gpu 0|1> <fp16 0|1> [iters] [out.bin] [extra opt flags k=v ...]
import sys, time, numpy as np, resource
import ncnn
base, gpu, fp16 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 100
outf = sys.argv[5] if len(sys.argv) > 5 else None
extra = dict(kv.split("=") for kv in sys.argv[6:])
x = np.fromfile("input_320.bin", dtype=np.float32).reshape(3, 320, 320)
if gpu: ncnn.create_gpu_instance()
net = ncnn.Net(); opt = net.opt
opt.use_vulkan_compute = bool(gpu); opt.use_fp16_packed = bool(fp16); opt.use_fp16_storage = bool(fp16); opt.use_fp16_arithmetic = bool(fp16); opt.use_bf16_storage = False
for k, v in extra.items(): setattr(opt, k, bool(int(v)) if v in ("0", "1") else int(v))
if gpu:
    opt.num_threads = 1; opt.openmp_blocktime = 0
    net.set_vulkan_device(0); vkdev = ncnn.get_gpu_device(0)
    blob = ncnn.VkBlobAllocator(vkdev); staging = ncnn.VkStagingAllocator(vkdev)
    opt.blob_vkallocator = blob; opt.workspace_vkallocator = blob; opt.staging_vkallocator = staging
else:
    opt.num_threads = 4; opt.openmp_blocktime = 0
net.opt = opt
assert net.load_param(base + ".param") == 0 and net.load_model(base + ".bin") == 0
def infer():
    ex = net.create_extractor(); ex.input("images", ncnn.Mat(x)); r, out = ex.extract("output0"); return out
for _ in range(10): out = infer()
t = []; c0 = resource.getrusage(resource.RUSAGE_SELF); w0 = time.perf_counter()
for _ in range(iters):
    a = time.perf_counter(); out = infer(); t.append((time.perf_counter() - a) * 1000)
c1 = resource.getrusage(resource.RUSAGE_SELF); wall = (time.perf_counter() - w0) * 1000 / iters
cpu = ((c1.ru_utime - c0.ru_utime) + (c1.ru_stime - c0.ru_stime)) * 1000 / iters
t.sort(); o = np.array(out)
print(f"ncnn {base.split('/')[-1]} gpu={gpu} fp16={fp16} {extra}: median {t[len(t)//2]:.2f} ms p95 {t[len(t)*95//100]:.2f} min {t[0]:.2f} max {t[-1]:.2f} -> {1000/t[len(t)//2]:.1f} fps | cpu {cpu:.2f} ms/iter | wall {wall:.1f} ms/iter (out {o.shape})", flush=True)
if outf: o.astype(np.float32).T.copy().tofile(outf) if o.shape == (2100, 8) else o.astype(np.float32).tofile(outf)
del net
if gpu: ncnn.destroy_gpu_instance()
