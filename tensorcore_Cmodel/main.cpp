#include "otc_driver.h"

struct TestData {
    std::vector<double> a, b, c;
};

TestData gen_ones(int M, int K, int N) {
    TestData t;
    t.a.assign(M * K, 1.0);
    t.b.assign(K * N, 1.0);
    t.c.assign(M * N, 0.0);
    return t;
}

TestData gen_identity(int M, int K, int N) {
    TestData t;
    t.a.assign(M * K, 0.0);
    t.b.assign(K * N, 0.0);
    t.c.assign(M * N, 0.0);
    int mn = std::min({M, K, N});
    for (int i = 0; i < mn; i++) {
        t.a[i * K + i] = 1.0;
        t.b[i * N + i] = 1.0;
    }
    return t;
}

TestData gen_random(int M, int K, int N) {
    TestData t;
    t.a.resize(M * K);
    t.b.resize(K * N);
    t.c.resize(M * N);
    srand(42);
    for (auto& v : t.a) v = (rand() % 200 - 100) / 100.0;
    for (auto& v : t.b) v = (rand() % 200 - 100) / 100.0;
    for (auto& v : t.c) v = (rand() % 100 - 50) / 100.0;
    return t;
}

TestData gen_simple(int, int, int) {
    TestData t;
    t.a = {1, 2, 3, 4};
    t.b = {5, 6, 7, 8};
    t.c = {0, 0, 0, 0};
    return t;
}

std::vector<uint32_t> pack_ab(const std::vector<double>& vals, int type_ab, int sub) {
    int eb = FPConvert::elem_bits(type_ab);
    int eperw = 32 / eb;
    int nw = ((int)vals.size() + eperw - 1) / eperw;
    std::vector<uint32_t> words(nw, 0);

    for (int i = 0; i < (int)vals.size(); i++) {
        int wi = i / eperw, ei = i % eperw;
        uint32_t packed = 0;
        switch (type_ab) {
            case TYPE_FP8: {
                uint16_t fp9 = SoftFloat::f64_to_fp9(vals[i]);
                int s = (fp9 >> 8) & 1, e = (fp9 >> 3) & 0x1F, m = fp9 & 7;
                if (sub == SUB_FP8E5M2)
                    packed = (s << 7) | (e << 2) | (m >> 1);
                else
                    packed = (s << 7) | ((e & 0xF) << 3) | m;
                break;
            }
            case TYPE_FP4: {
                double v = vals[i];
                int s = (v < 0) ? 1 : 0;
                double av = fabs(v);
                int e, m;
                if (av == 0.0) {
                    e = 0;
                    m = 0;
                } else if (av >= 4.0) {
                    e = 2;
                    m = 1;
                } else {
                    int exp;
                    double frac = frexp(av, &exp);
                    int biased_e = exp;
                    if (biased_e <= 0) {
                        e = 0;
                        m = (av >= 0.5) ? 1 : 0;
                    } else if (biased_e >= 3) {
                        e = 2;
                        m = 1;
                    } else {
                        e = biased_e;
                        m = (2.0 * frac - 1.0 >= 0.5) ? 1 : 0;
                    }
                }
                packed = (s << 3) | (e << 1) | m;
                break;
            }
            case TYPE_FP16:
                packed = SoftFloat::f64_to_fp16(vals[i]);
                break;
        }
        words[wi] |= (packed << (ei * eb));
    }
    return words;
}

std::vector<uint32_t> pack_c_fp16(const std::vector<double>& vals) {
    int nw = ((int)vals.size() + 1) / 2;
    std::vector<uint32_t> words(nw, 0);
    for (int i = 0; i < (int)vals.size(); i++) {
        uint16_t h = SoftFloat::f64_to_fp16(vals[i]);
        words[i / 2] |= ((uint32_t)h << ((i % 2) * 16));
    }
    return words;
}

std::vector<double> golden_gemm(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& c, int M, int K,
                                int N) {
    std::vector<double> d(M * N, 0.0);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            d[i * N + j] = sum + c[i * N + j];
        }
    }
    return d;
}

bool verify(const std::vector<double>& got, const std::vector<double>& ref, double rtol, double atol, int M, int N) {
    bool pass = true;
    int nerr = 0;
    double max_err = 0, sum_err = 0;
    for (int i = 0; i < M * N; i++) {
        double err = fabs(got[i] - ref[i]);
        double thr = rtol * fabs(ref[i]) + atol;
        max_err = std::max(max_err, err);
        sum_err += err;
        if (err > thr) {
            if (nerr < 5) {
                std::cout << "  MISMATCH D[" << i / N << "][" << i % N << "]: got=" << got[i] << "  ref=" << ref[i]
                          << "  err=" << err << " (thr=" << thr << ")\n";
            }
            pass = false;
            nerr++;
        }
    }
    if (nerr > 5) std::cout << "  ... and " << (nerr - 5) << " more mismatches\n";
    int n = M * N;
    std::cout << "\n  Error stats: max=" << std::fixed << std::setprecision(6) << max_err << "  avg=" << (sum_err / n)
              << "  (atol=" << std::setprecision(3) << atol << " rtol=" << std::setprecision(0) << (rtol * 100) << "%)\n";
    return pass;
}

struct Args {
    OTC_Config cfg;
    std::string test = "ones";
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        if (s.find("--M=") == 0)
            a.cfg.M = std::stoi(s.substr(4));
        else if (s.find("--K=") == 0)
            a.cfg.K = std::stoi(s.substr(4));
        else if (s.find("--N=") == 0)
            a.cfg.N = std::stoi(s.substr(4));
        else if (s == "--type_ab=fp4")
            a.cfg.type_ab = TYPE_FP4;
        else if (s == "--type_ab=fp8e5m2") {
            a.cfg.type_ab = TYPE_FP8;
            a.cfg.type_ab_sub = SUB_FP8E5M2;
        } else if (s == "--type_ab=fp8e4m3") {
            a.cfg.type_ab = TYPE_FP8;
            a.cfg.type_ab_sub = SUB_FP8E4M3;
        } else if (s == "--type_ab=fp16")
            a.cfg.type_ab = TYPE_FP16;
        else if (s == "--type_cd=fp16")
            a.cfg.type_cd = TYPE_FP16;
        else if (s == "--type_cd=fp32")
            a.cfg.type_cd = TYPE_FP32;
        else if (s.find("--debug=") == 0)
            a.cfg.debug_level = std::stoi(s.substr(8));
        else if (s == "--trace")
            a.cfg.trace_en = true;
        else if (s.find("--test=") == 0)
            a.test = s.substr(7);
        else if (s == "--help" || s == "-h") {
            std::cout << "OpenTensorCore SimX — Cycle-Approximate Simulator\n"
                      << "Mirrors Vortex GPGPU simX architecture for OpenTensorCore RTL\n\n"
                      << "Usage: ./otc_simx [options]\n"
                      << "  --M=N             Matrix M dim (default: 8)\n"
                      << "  --K=N             Matrix K dim (default: 8, must be power of 2)\n"
                      << "  --N=N             Matrix N dim (default: 8)\n"
                      << "  --type_ab=TYPE    Input: fp4|fp8e5m2|fp8e4m3|fp16 (default: fp8e5m2)\n"
                      << "  --type_cd=TYPE    Output: fp16|fp32 (default: fp32)\n"
                      << "  --debug=LEVEL     0=off 1=summary 2=pipeline 3=full\n"
                      << "  --trace           Write trace to otc_run.log\n"
                      << "  --test=NAME       ones|identity|random|simple (default: ones)\n";
            exit(0);
        }
    }
    return a;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    OTC_Config& cfg = args.cfg;
    if (args.test == "simple") {
        cfg.M = 2;
        cfg.K = 2;
        cfg.N = 2;
    }

    std::cout << "Configuration:\n";
    std::cout << "  Compute:   D[" << cfg.M << "x" << cfg.N << "] = A[" << cfg.M << "x" << cfg.K << "] × B[" << cfg.K << "x"
              << cfg.N << "] + C[" << cfg.M << "x" << cfg.N << "]\n";

    const char* type_str;
    switch (cfg.type_ab) {
        case TYPE_FP4:
            type_str = "FP4";
            break;
        case TYPE_FP8:
            type_str = cfg.type_ab_sub == SUB_FP8E5M2 ? "FP8(E5M2)" : "FP8(E4M3)";
            break;
        case TYPE_FP16:
            type_str = "FP16";
            break;
        default:
            type_str = "?";
            break;
    }

    std::cout << "  Input AB:  " << type_str << " → internal FP9(E5M3)\n";
    std::cout << "  Output CD: " << (cfg.type_cd == TYPE_FP16 ? "FP16" : "FP32") << "  (accumulator: FP22(E8M13))\n";
    std::cout << "  Pipeline:  mul=" << cfg.mul_latency << "cyc + tree=" << cfg.add_latency << "cyc×" << cfg.tree_depth()
              << "lvl + acc=" << cfg.add_latency << "cyc = " << cfg.pipeline_depth() << " total\n";
    std::cout << "  Debug:     level=" << cfg.debug_level << "  trace=" << (cfg.debug_level > 0 ? "otc_run.log" : "off") << "\n";
    std::cout << "  Test:      " << args.test << "\n\n";

    TestData td;
    if (args.test == "identity")
        td = gen_identity(cfg.M, cfg.K, cfg.N);
    else if (args.test == "random")
        td = gen_random(cfg.M, cfg.K, cfg.N);
    else if (args.test == "simple")
        td = gen_simple(cfg.M, cfg.K, cfg.N);
    else
        td = gen_ones(cfg.M, cfg.K, cfg.N);

    auto gold = golden_gemm(td.a, td.b, td.c, cfg.M, cfg.K, cfg.N);
    std::cout << "Golden D (first row): ";
    for (int j = 0; j < std::min(8, cfg.N); j++) std::cout << std::fixed << std::setprecision(4) << gold[j] << " ";
    if (cfg.N > 8) std::cout << "...";
    std::cout << "\n";

    auto pa = pack_ab(td.a, cfg.type_ab, cfg.type_ab_sub);
    auto pb = pack_ab(td.b, cfg.type_ab, cfg.type_ab_sub);
    auto pc = pack_c_fp16(td.c);

    OTC_Device* dev;
    otc_dev_open(&dev);
    otc_configure(dev, cfg);
    otc_upload(dev, pa.data(), pa.size(), pb.data(), pb.size(), pc.data(), pc.size());

    int ret = otc_run(dev);
    if (ret != 0) {
        std::cout << "FAIL: simulation timed out!\n";
        otc_dev_close(dev);
        return 1;
    }

    std::vector<double> result(cfg.M * cfg.N);
    otc_download_f64(dev, result.data(), result.size());

    std::cout << "SimX D  (first row): ";
    for (int j = 0; j < std::min(8, cfg.N); j++) std::cout << std::fixed << std::setprecision(4) << result[j] << " ";
    if (cfg.N > 8) std::cout << "...";
    std::cout << "\n";

    double rtol, atol;
    switch (cfg.type_ab) {
        case TYPE_FP4:
            rtol = 0.5;
            atol = 2.0;
            break;
        case TYPE_FP8:
            rtol = 0.30;
            atol = 0.50;
            break;
        case TYPE_FP16:
            rtol = 0.01;
            atol = 0.001;
            break;
        default:
            rtol = 0.30;
            atol = 0.50;
            break;
    }

    bool pass = verify(result, gold, rtol, atol, cfg.M, cfg.N);
    std::cout << "\nVerification: " << (pass ? "PASSED ✓" : "FAILED ✗") << " (rtol=" << (rtol * 100) << "% atol=" << atol
              << ")\n\n";

    otc_stats(dev).print(std::cout);

    if (cfg.M <= 8 && cfg.N <= 8) {
        std::cout << "\n=== Full Output Matrix D[" << cfg.M << "x" << cfg.N << "] ===\n";
        for (int i = 0; i < cfg.M; i++) {
            std::cout << "  [";
            for (int j = 0; j < cfg.N; j++) std::cout << std::setw(8) << std::setprecision(3) << result[i * cfg.N + j];
            std::cout << " ]\n";
        }
        std::cout << "\n=== Golden Reference ===\n";
        for (int i = 0; i < cfg.M; i++) {
            std::cout << "  [";
            for (int j = 0; j < cfg.N; j++) std::cout << std::setw(8) << std::setprecision(3) << gold[i * cfg.N + j];
            std::cout << " ]\n";
        }
    }

    otc_dev_close(dev);
    return pass ? 0 : 1;
}
