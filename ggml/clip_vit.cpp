// clip_vit: minimal ggml runner for an open_clip VisionTransformer image encoder (CLIP image embedding)
// converted with convert_clip_onnx_to_gguf.py. Runs the whole tower on one ggml backend (Vulkan0 / CPU)
// through ggml_backend_sched (CPU fallback for unsupported ops), times it, and writes the L2-normalised embedding.
//
// usage: clip_vit <model.gguf> <backend: CPU|Vulkan0> <input.bin f32 NCHW 1x3xSxS> <out.bin> [warmup] [iters] [threads] [fa 0|1]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <map>

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static long rss_mb(const char * key) {
    std::ifstream f("/proc/self/status"); std::string l;
    while (std::getline(f, l)) if (l.rfind(key, 0) == 0) return atol(l.c_str() + strlen(key) + 1) / 1024;
    return -1;
}

struct hp { int width, layers, heads, patch, image_size, embed_dim; float eps; bool quick; };

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model.gguf backend input.bin out.bin [warmup] [iters] [threads] [fa]\n", argv[0]); return 1; }
    const char * mpath = argv[1]; std::string bname = argv[2]; const char * inpath = argv[3]; const char * outpath = argv[4];
    int warmup = argc > 5 ? atoi(argv[5]) : 2, iters = argc > 6 ? atoi(argv[6]) : 5, threads = argc > 7 ? atoi(argv[7]) : 4;
    bool use_fa = argc > 8 ? atoi(argv[8]) != 0 : false;

    // ---- load gguf metadata + tensors (no_alloc) ----
    ggml_context * wctx = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, &wctx };
    gguf_context * gg = gguf_init_from_file(mpath, gp);
    if (!gg) { fprintf(stderr, "failed to load %s\n", mpath); return 1; }
    auto geti = [&](const char * k) { return (int) gguf_get_val_u32(gg, gguf_find_key(gg, k)); };
    hp H;
    H.width = geti("clip-vit.width"); H.layers = geti("clip-vit.layers"); H.heads = geti("clip-vit.heads");
    H.patch = geti("clip-vit.patch_size"); H.image_size = geti("clip-vit.image_size"); H.embed_dim = geti("clip-vit.embed_dim");
    H.eps = gguf_get_val_f32(gg, gguf_find_key(gg, "clip-vit.ln_eps"));
    H.quick = strcmp(gguf_get_val_str(gg, gguf_find_key(gg, "clip-vit.gelu")), "quick") == 0;
    const int n_px = H.image_size / H.patch, n_patches = n_px * n_px, n_pos = n_patches + 1, d_head = H.width / H.heads;
    fprintf(stderr, "model: width=%d layers=%d heads=%d d_head=%d patch=%d img=%d n_pos=%d embed=%d gelu=%s eps=%g\n",
            H.width, H.layers, H.heads, d_head, H.patch, H.image_size, n_pos, H.embed_dim, H.quick ? "quick" : "erf", H.eps);

    // ---- backends ----
    ggml_backend_t backend = nullptr, cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, threads);
    if (bname == "CPU") backend = cpu;
    else {
        backend = ggml_backend_init_by_name(bname.c_str(), nullptr);
        if (!backend) { fprintf(stderr, "backend %s not found; available:", bname.c_str());
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) fprintf(stderr, " %s", ggml_backend_dev_name(ggml_backend_dev_get(i)));
            fprintf(stderr, "\n"); return 1; }
    }
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        size_t fr = 0, tot = 0; if (dev) ggml_backend_dev_memory(dev, &fr, &tot);
        fprintf(stderr, "backend: %s (%s) mem free %.0f MB / total %.0f MB\n", ggml_backend_name(backend), dev ? ggml_backend_dev_description(dev) : "?", fr / 1e6, tot / 1e6);
    }

    // ---- weights to backend ----
    double t0 = now_ms();
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    if (!wbuf) { fprintf(stderr, "weight alloc failed\n"); return 1; }
    {
        FILE * f = fopen(mpath, "rb"); std::vector<uint8_t> tmp;
        const size_t base = gguf_get_data_offset(gg);
        for (int64_t i = 0; i < gguf_get_n_tensors(gg); i++) {
            const char * name = gguf_get_tensor_name(gg, i);
            ggml_tensor * t = ggml_get_tensor(wctx, name);
            size_t nb = ggml_nbytes(t); tmp.resize(nb);
            fseek(f, base + gguf_get_tensor_offset(gg, i), SEEK_SET);
            if (fread(tmp.data(), 1, nb, f) != nb) { fprintf(stderr, "short read %s\n", name); return 1; }
            ggml_backend_tensor_set(t, tmp.data(), 0, nb);
        }
        fclose(f);
    }
    fprintf(stderr, "weights: %.1f MB in %s buffer, loaded in %.0f ms\n", ggml_backend_buffer_get_size(wbuf) / 1e6, ggml_backend_buffer_name(wbuf), now_ms() - t0);
    auto W = [&](const std::string & n) { ggml_tensor * t = ggml_get_tensor(wctx, n.c_str()); if (!t) { fprintf(stderr, "missing tensor %s\n", n.c_str()); exit(1); } return t; };

    // ---- input ----
    std::vector<float> input((size_t) 3 * H.image_size * H.image_size);
    { FILE * f = fopen(inpath, "rb"); if (!f || fread(input.data(), 4, input.size(), f) != input.size()) { fprintf(stderr, "bad input %s\n", inpath); return 1; } fclose(f); }

    // ---- scheduler ----
    ggml_backend_t backends[2] = { backend, cpu }; int nb = backend == cpu ? 1 : 2;
    const size_t graph_size = 4096 + 64 * H.layers;
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, nb, graph_size, false, true);
    std::vector<uint8_t> cbuf(ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false));

    ggml_tensor * inp_t = nullptr, * out_t = nullptr;
    auto build = [&]() -> ggml_cgraph * {
        ggml_init_params ip = { cbuf.size(), cbuf.data(), true };
        ggml_context * ctx = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);
        // input W,H,C,N (ggml ne0 = W)
        inp_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, H.image_size, H.image_size, 3, 1);
        ggml_set_name(inp_t, "inp"); ggml_set_input(inp_t);
        ggml_tensor * x = ggml_conv_2d(ctx, W("patch_w"), inp_t, H.patch, H.patch, 0, 0, 1, 1);   // [n_px, n_px, width, 1]
        x = ggml_reshape_3d(ctx, x, n_patches, H.width, 1);
        x = ggml_cont(ctx, ggml_transpose(ctx, x));                                              // [width, n_patches]
        x = ggml_reshape_2d(ctx, x, H.width, n_patches);
        x = ggml_concat(ctx, ggml_reshape_2d(ctx, W("class_embd"), H.width, 1), x, 1);           // [width, n_pos], CLS first
        x = ggml_add(ctx, x, W("pos_embd"));
        auto ln = [&](ggml_tensor * t, const std::string & p) { t = ggml_norm(ctx, t, H.eps); t = ggml_mul(ctx, t, W(p + "_w")); return ggml_add(ctx, t, W(p + "_b")); };
        x = ln(x, "ln_pre");
        const float scale = 1.0f / sqrtf((float) d_head);
        for (int il = 0; il < H.layers; il++) {
            std::string b = "blk." + std::to_string(il) + ".";
            ggml_tensor * cur = ln(x, b + "ln1");
            ggml_tensor * qkv = ggml_add(ctx, ggml_mul_mat(ctx, W(b + "qkv_w"), cur), W(b + "qkv_b"));     // [3*width, n_pos]
            const size_t es = ggml_element_size(qkv);
            ggml_tensor * q = ggml_view_3d(ctx, qkv, d_head, H.heads, n_pos, d_head * es, qkv->nb[1], 0);
            ggml_tensor * k = ggml_view_3d(ctx, qkv, d_head, H.heads, n_pos, d_head * es, qkv->nb[1], (size_t) H.width * es);
            ggml_tensor * v = ggml_view_3d(ctx, qkv, d_head, H.heads, n_pos, d_head * es, qkv->nb[1], (size_t) 2 * H.width * es);
            q = ggml_permute(ctx, q, 0, 2, 1, 3);                                                          // [d_head, n_pos, heads]
            k = ggml_permute(ctx, k, 0, 2, 1, 3);
            if (use_fa) {
                v = ggml_permute(ctx, v, 0, 2, 1, 3);
                cur = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);                    // [d_head, heads, n_pos]
                ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
                cur = ggml_reshape_2d(ctx, cur, H.width, n_pos);
            } else {
                v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));                                    // [n_pos, d_head, heads]
                ggml_tensor * kq = ggml_mul_mat(ctx, k, q);                                              // [n_pos_k, n_pos_q, heads]
                kq = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);
                ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);                                            // [d_head, n_pos, heads]
                cur = ggml_cont_2d(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3), H.width, n_pos);
            }
            cur = ggml_add(ctx, ggml_mul_mat(ctx, W(b + "out_w"), cur), W(b + "out_b"));
            x = ggml_add(ctx, x, cur);
            cur = ln(x, b + "ln2");
            cur = ggml_add(ctx, ggml_mul_mat(ctx, W(b + "fc1_w"), cur), W(b + "fc1_b"));
            cur = H.quick ? ggml_gelu_quick(ctx, cur) : ggml_gelu_erf(ctx, cur);
            cur = ggml_add(ctx, ggml_mul_mat(ctx, W(b + "fc2_w"), cur), W(b + "fc2_b"));
            x = ggml_add(ctx, x, cur);
        }
        ggml_tensor * cls = ggml_view_2d(ctx, x, H.width, 1, x->nb[1], 0);
        cls = ln(cls, "ln_post");
        out_t = ggml_mul_mat(ctx, W("proj_w"), cls);                                                    // [embed_dim, 1]
        ggml_set_name(out_t, "embedding"); ggml_set_output(out_t);
        ggml_build_forward_expand(gf, out_t);
        ggml_free(ctx);   // tensors live in cbuf (no_alloc); context struct no longer needed
        return gf;
    };

    // op support report
    {
        ggml_cgraph * gf = build();
        std::map<std::string, int> unsupported;
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * n = ggml_graph_node(gf, i);
            if (!ggml_backend_supports_op(backend, n)) unsupported[std::string(ggml_op_desc(n)) + "(" + ggml_type_name(n->src[0] ? n->src[0]->type : n->type) + ")"]++;
        }
        fprintf(stderr, "graph: %d nodes; unsupported on %s:", ggml_graph_n_nodes(gf), ggml_backend_name(backend));
        if (unsupported.empty()) fprintf(stderr, " none");
        for (auto & kv : unsupported) fprintf(stderr, " %s x%d", kv.first.c_str(), kv.second);
        fprintf(stderr, "\n");
    }

    std::vector<float> out(H.embed_dim);
    auto run = [&](bool verbose) {
        double a = now_ms();
        ggml_backend_sched_reset(sched);
        ggml_cgraph * gf = build();
        if (!ggml_backend_sched_alloc_graph(sched, gf)) { fprintf(stderr, "alloc graph failed\n"); exit(1); }
        ggml_backend_tensor_set(inp_t, input.data(), 0, input.size() * 4);
        double b = now_ms();
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
        ggml_backend_sched_synchronize(sched);
        double c = now_ms();
        ggml_backend_tensor_get(out_t, out.data(), 0, out.size() * 4);
        double d = now_ms();
        if (verbose) fprintf(stderr, "  build+alloc+set %.1f ms, compute %.1f ms, get %.1f ms, total %.1f ms\n", b - a, c - b, d - c, d - a);
        return d - a;
    };
    for (int i = 0; i < warmup; i++) { double t = run(true); fprintf(stderr, "warmup %d: %.1f ms\n", i, t); }
    std::vector<double> ts;
    for (int i = 0; i < iters; i++) ts.push_back(run(false));
    std::sort(ts.begin(), ts.end());
    double med = ts.empty() ? 0 : ts[ts.size() / 2], mean = 0; for (double t : ts) mean += t; mean /= std::max<size_t>(1, ts.size());
    for (int i = 0; i < nb; i++) fprintf(stderr, "sched buffer %s: %.1f MB\n", ggml_backend_name(backends[i]), ggml_backend_sched_get_buffer_size(sched, backends[i]) / 1e6);
    // L2 normalise (the ONNX graph's ReduceL2/Div)
    double ss = 0; for (float v : out) ss += (double) v * v; float inv = 1.0f / std::max(1e-12f, (float) sqrt(ss));
    for (float & v : out) v *= inv;
    { FILE * f = fopen(outpath, "wb"); fwrite(out.data(), 4, out.size(), f); fclose(f); }
    printf("RESULT backend=%s fa=%d iters=%d median_ms=%.1f min_ms=%.1f mean_ms=%.1f weights_MB=%.0f rss_MB=%ld hwm_MB=%ld\n",
           ggml_backend_name(backend), (int) use_fa, iters, med, ts.empty() ? 0 : ts[0], mean, ggml_backend_buffer_get_size(wbuf) / 1e6, rss_mb("VmRSS:"), rss_mb("VmHWM:"));
    ggml_backend_sched_free(sched); ggml_backend_buffer_free(wbuf); if (backend != cpu) ggml_backend_free(backend); ggml_backend_free(cpu); gguf_free(gg); ggml_free(wctx);
    return 0;
}
