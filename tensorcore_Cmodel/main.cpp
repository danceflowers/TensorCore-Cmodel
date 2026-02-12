#include "otc_decode.h"
#include "otc_driver.h"

struct TestData {
    std::vector<double> a, b, c;
};

struct Args {
    OTC_Config cfg;
    std::string test = "ones";
    int batches = 1;
    int random_runs = 5;
};

static TestData gen_ones(int M, int K, int N) {
    TestData t;
    t.a.assign(M * K, 1.0);
    t.b.assign(K * N, 1.0);
    t.c.assign(M * N, 0.0);
    return t;
}

static TestData gen_identity(int M, int K, int N) {
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

static TestData gen_random(int M, int K, int N, unsigned seed) {
    TestData t;
    t.a.resize(M * K);
    t.b.resize(K * N);
    t.c.resize(M * N);
    srand(seed);
    for (auto& v : t.a) v = (rand() % 200 - 100) / 100.0;
    for (auto& v : t.b) v = (rand() % 200 - 100) / 100.0;
    for (auto& v : t.c) v = (rand() % 100 - 50) / 100.0;
    return t;
}

static TestData gen_simple() {
    TestData t;
    t.a = {1, 2, 3, 4};
    t.b = {5, 6, 7, 8};
    t.c = {0, 0, 0, 0};
    return t;
}

static uint8_t quantize_fp8(double v, int sub) {
    return (sub == SUB_FP8E4M3) ? FPConvert::f64_to_fp8e4m3(v) : FPConvert::f64_to_fp8e5m2(v);
}

static double dequantize_fp8(uint8_t bits, int sub) {
    return (sub == SUB_FP8E4M3) ? FPConvert::fp8e4m3_to_f64(bits) : FPConvert::fp8e5m2_to_f64(bits);
}

static std::vector<uint32_t> pack_ab(const std::vector<double>& vals, int type_ab, int sub) {
    int eb = FPConvert::elem_bits(type_ab);
    int eperw = 32 / eb;
    int nw = ((int)vals.size() + eperw - 1) / eperw;
    std::vector<uint32_t> words(nw, 0);

    for (int i = 0; i < (int)vals.size(); i++) {
        int wi = i / eperw, ei = i % eperw;
        uint32_t packed = 0;
        if (type_ab == TYPE_FP8) {
            packed = quantize_fp8(vals[i], sub);
        } else if (type_ab == TYPE_FP4) {
            double v = vals[i];
            int s = (v < 0) ? 1 : 0;
            double av = fabs(v);
            int e = 0, m = 0;
            if (av >= 4.0) {
                e = 2;
                m = 1;
            } else if (av > 0.0) {
                int exp;
                double frac = frexp(av, &exp);
                if (exp <= 0) {
                    e = 0;
                    m = (av >= 0.5) ? 1 : 0;
                } else if (exp >= 3) {
                    e = 2;
                    m = 1;
                } else {
                    e = exp;
                    m = (2.0 * frac - 1.0 >= 0.5) ? 1 : 0;
                }
            }
            packed = (s << 3) | (e << 1) | m;
        } else {
            packed = SoftFloat::f64_to_fp16(vals[i]);
        }
        words[wi] |= (packed << (ei * eb));
    }
    return words;
}

static std::vector<uint32_t> pack_c_fp16(const std::vector<double>& vals) {
    int nw = ((int)vals.size() + 1) / 2;
    std::vector<uint32_t> words(nw, 0);
    for (int i = 0; i < (int)vals.size(); i++) {
        uint16_t h = SoftFloat::f64_to_fp16(vals[i]);
        words[i / 2] |= ((uint32_t)h << ((i % 2) * 16));
    }
    return words;
}

static double quantize_output(double v, const OTC_Config& cfg) {
    if (cfg.type_cd == TYPE_FP32) return SoftFloat::fp32_to_f64(SoftFloat::f64_to_fp32(v));
    if (cfg.type_cd == TYPE_FP16) return SoftFloat::fp16_to_f64(SoftFloat::f64_to_fp16(v));
    if (cfg.type_cd == TYPE_FP8) return dequantize_fp8(quantize_fp8(v, cfg.type_cd_sub), cfg.type_cd_sub);
    return v;
}

static std::vector<double> golden_gemm_fp32(const TestData& td, const OTC_Config& cfg) {
    // Golden path: input quantization follows model front-end, accumulation uses FP32.
    auto pa = pack_ab(td.a, cfg.type_ab, cfg.type_ab_sub);
    auto pb = pack_ab(td.b, cfg.type_ab, cfg.type_ab_sub);
    auto pc = pack_c_fp16(td.c);

    int eb = FPConvert::elem_bits(cfg.type_ab);
    int eperw = 32 / eb;
    std::vector<double> a(cfg.M * cfg.K), b(cfg.K * cfg.N), c(cfg.M * cfg.N);

    for (int i = 0; i < cfg.M * cfg.K; i++) {
        int wi = i / eperw, ei = i % eperw;
        a[i] = FPConvert::elem_to_f64(wi < (int)pa.size() ? pa[wi] : 0, ei, cfg.type_ab, cfg.type_ab_sub);
    }
    for (int i = 0; i < cfg.K * cfg.N; i++) {
        int wi = i / eperw, ei = i % eperw;
        b[i] = FPConvert::elem_to_f64(wi < (int)pb.size() ? pb[wi] : 0, ei, cfg.type_ab, cfg.type_ab_sub);
    }
    for (int i = 0; i < cfg.M * cfg.N; i++) {
        int wi = i / 2, ei = i % 2;
        uint32_t w = wi < (int)pc.size() ? pc[wi] : 0;
        uint16_t half = (w >> (ei * 16)) & 0xFFFF;
        c[i] = SoftFloat::fp16_to_f64(half);
    }

    std::vector<double> d(cfg.M * cfg.N, 0.0);
    for (int i = 0; i < cfg.M; i++) {
        for (int j = 0; j < cfg.N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < cfg.K; k++) {
                sum += (float)a[i * cfg.K + k] * (float)b[k * cfg.N + j];
            }
            d[i * cfg.N + j] = quantize_output((double)(sum + (float)c[i * cfg.N + j]), cfg);
        }
    }
    return d;
}

static bool verify(const std::vector<double>& got, const std::vector<double>& ref, double rtol, double atol, int M, int N) {
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
                std::cout << "  MISMATCH D[" << i / N << "][" << i % N << "]: got=" << got[i]
                          << " ref=" << ref[i] << " err=" << err << " (thr=" << thr << ")\n";
            }
            pass = false;
            nerr++;
        }
    }
    if (nerr > 5) std::cout << "  ... and " << (nerr - 5) << " more mismatches\n";
    std::cout << "  Error stats: max=" << std::fixed << std::setprecision(6) << max_err
              << " avg=" << (sum_err / (M * N)) << "\n";
    return pass;
}

static uint32_t build_inst(uint8_t opcode, uint8_t funct3, uint8_t rd = 0, uint8_t rs1 = 0, uint8_t rs2 = 0, uint8_t funct7 = 0) {
    return ((uint32_t)funct7 << 25) | ((uint32_t)rs2 << 20) | ((uint32_t)rs1 << 15) | ((uint32_t)funct3 << 12)
           | ((uint32_t)rd << 7) | opcode;
}

static bool execute_program(OTC_Device* dev, const std::vector<uint32_t>& pa, const std::vector<uint32_t>& pb,
                            const std::vector<uint32_t>& pc, int batches, std::vector<double>& result) {
    OTC_Decoder decoder;
    decoder.init();

    // Fetch/Decode/Execute loop (main control loop)
    std::vector<uint32_t> program = {
        build_inst(0x23, 0x01),  // TCU_LOAD A/B/C
        build_inst(0x21, 0x01),  // TCU_WMMA
        build_inst(0x27, 0x01),  // TCU_STORE
    };

    size_t pc_idx = 0;
    while (pc_idx < program.size()) {
        uint32_t inst_word = program[pc_idx++];             // Fetch
        DecodedInst inst = decoder.decode(inst_word);       // Decode
        if (!inst.valid) {
            std::cout << "Decode error at pc=" << (pc_idx - 1) << "\n";
            return false;
        }

        switch (inst.op) {                                  // Execute
            case OTC_OpType::TCU_LOAD:
                break;
            case OTC_OpType::TCU_WMMA:
                for (int b = 0; b < batches; b++) {
                    if (otc_submit(dev, pa.data(), pa.size(), pb.data(), pb.size(), pc.data(), pc.size()) != 0) {
                        std::cout << "Submit failed at batch " << b << "\n";
                        return false;
                    }
                }
                if (otc_run(dev) != 0) {
                    std::cout << "Execution timeout\n";
                    return false;
                }
                break;
            case OTC_OpType::TCU_STORE:
                if (otc_pop_result_f64(dev, result.data(), result.size()) <= 0) {
                    std::cout << "No output popped from FIFO\n";
                    return false;
                }
                break;
            default:
                break;
        }
    }
    return true;
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        if (s.find("--M=") == 0) a.cfg.M = std::stoi(s.substr(4));
        else if (s.find("--K=") == 0) a.cfg.K = std::stoi(s.substr(4));
        else if (s.find("--N=") == 0) a.cfg.N = std::stoi(s.substr(4));
        else if (s == "--type_ab=fp4") a.cfg.type_ab = TYPE_FP4;
        else if (s == "--type_ab=fp8e5m2") { a.cfg.type_ab = TYPE_FP8; a.cfg.type_ab_sub = SUB_FP8E5M2; }
        else if (s == "--type_ab=fp8e4m3") { a.cfg.type_ab = TYPE_FP8; a.cfg.type_ab_sub = SUB_FP8E4M3; }
        else if (s == "--type_ab=fp16") a.cfg.type_ab = TYPE_FP16;
        else if (s == "--type_cd=fp8e5m2") { a.cfg.type_cd = TYPE_FP8; a.cfg.type_cd_sub = SUB_FP8E5M2; }
        else if (s == "--type_cd=fp8e4m3") { a.cfg.type_cd = TYPE_FP8; a.cfg.type_cd_sub = SUB_FP8E4M3; }
        else if (s == "--type_cd=fp16") a.cfg.type_cd = TYPE_FP16;
        else if (s == "--type_cd=fp32") a.cfg.type_cd = TYPE_FP32;
        else if (s.find("--debug=") == 0) a.cfg.debug_level = std::stoi(s.substr(8));
        else if (s.find("--dispatch_width=") == 0) a.cfg.dispatch_width = std::stoi(s.substr(17));
        else if (s.find("--in_fifo_depth=") == 0) a.cfg.input_fifo_depth = std::stoi(s.substr(16));
        else if (s.find("--out_fifo_depth=") == 0) a.cfg.output_fifo_depth = std::stoi(s.substr(17));
        else if (s.find("--mem_bw=") == 0) a.cfg.mem_bandwidth_bytes_per_cycle = std::stoi(s.substr(9));
        else if (s.find("--batches=") == 0) a.batches = std::stoi(s.substr(10));
        else if (s.find("--random_runs=") == 0) a.random_runs = std::stoi(s.substr(14));
        else if (s == "--trace") a.cfg.trace_en = true;
        else if (s.find("--test=") == 0) a.test = s.substr(7);
    }
    return a;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    if (args.test == "simple") {
        args.cfg.M = 2;
        args.cfg.K = 2;
        args.cfg.N = 2;
    }

    const int runs = (args.test == "random") ? std::max(1, args.random_runs) : 1;
    bool all_pass = true;

    for (int run_id = 0; run_id < runs; run_id++) {
        TestData td;
        if (args.test == "identity") td = gen_identity(args.cfg.M, args.cfg.K, args.cfg.N);
        else if (args.test == "random") td = gen_random(args.cfg.M, args.cfg.K, args.cfg.N, 42 + run_id);
        else if (args.test == "simple") td = gen_simple();
        else td = gen_ones(args.cfg.M, args.cfg.K, args.cfg.N);

        auto pa = pack_ab(td.a, args.cfg.type_ab, args.cfg.type_ab_sub);
        auto pb = pack_ab(td.b, args.cfg.type_ab, args.cfg.type_ab_sub);
        auto pc = pack_c_fp16(td.c);

        OTC_Device* dev;
        otc_dev_open(&dev);
        if (otc_configure(dev, args.cfg) != 0) {
            std::cout << "Invalid config\n";
            otc_dev_close(dev);
            return 1;
        }

        std::vector<double> result(args.cfg.M * args.cfg.N, 0.0);
        bool exec_ok = execute_program(dev, pa, pb, pc, args.batches, result);
        if (!exec_ok) {
            otc_dev_close(dev);
            return 1;
        }

        auto gold_fp32 = golden_gemm_fp32(td, args.cfg);
        double rtol = (args.cfg.type_ab == TYPE_FP16) ? 0.05 : 0.10;
        double atol = (args.cfg.type_cd == TYPE_FP8) ? 0.30 : 0.08;

        std::cout << "[Run " << run_id << "] verify vs FP32 golden\n";
        bool pass = verify(result, gold_fp32, rtol, atol, args.cfg.M, args.cfg.N);
        all_pass = all_pass && pass;

        if (run_id == runs - 1) {
            otc_stats(dev).print(std::cout);
        }
        otc_dev_close(dev);
    }

    std::cout << "\nOverall: " << (all_pass ? "PASSED" : "FAILED") << "\n";
    return all_pass ? 0 : 1;
}
