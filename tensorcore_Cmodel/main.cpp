// ============================================================================
// OpenTensorCore SimX — Main Entry Point & Test Harness
//
// Mirrors:  tests/regression/basic/main.cpp   (Vortex test pattern)
//           ci/blackbox.sh                    (CLI arg interface)
//
// Usage:
//   ./otc_simx --test=ones --M=8 --K=8 --N=8 --type_ab=fp8e5m2 --debug=1
//   ./otc_simx --test=random --M=4 --K=4 --N=4 --debug=2 --trace
//   ./otc_simx --help
// ============================================================================
#include "../runtime/otc_driver.h"
#include <cstdarg>

// Global trace instance
TraceLog DT;

// ==================== Test Data Generators ====================

struct TestData {
    std::vector<double> a, b, c;  // FP64 reference matrices
};

// All ones: D[i][j] = K + c[i][j]
TestData gen_ones(int M, int K, int N) {
    TestData t;
    t.a.assign(M*K, 1.0);
    t.b.assign(K*N, 1.0);
    t.c.assign(M*N, 0.0);
    return t;
}

// Identity-like: A=eye, B=eye, C=0
TestData gen_identity(int M, int K, int N) {
    TestData t;
    t.a.assign(M*K, 0.0);
    t.b.assign(K*N, 0.0);
    t.c.assign(M*N, 0.0);
    int mn = std::min({M, K, N});
    for (int i = 0; i < mn; i++) {
        t.a[i*K + i] = 1.0;
        t.b[i*N + i] = 1.0;
    }
    return t;
}

// Random small values
TestData gen_random(int M, int K, int N) {
    TestData t;
    t.a.resize(M*K); t.b.resize(K*N); t.c.resize(M*N);
    srand(42);
    for (auto& v : t.a) v = (rand()%200-100) / 100.0;
    for (auto& v : t.b) v = (rand()%200-100) / 100.0;
    for (auto& v : t.c) v = (rand()%100-50)  / 100.0;
    return t;
}

// Simple 2x2: known result
TestData gen_simple(int, int, int) {
    // A = [1,2; 3,4], B = [5,6; 7,8], C = [0,0; 0,0]
    // D = [1*5+2*7, 1*6+2*8; 3*5+4*7, 3*6+4*8] = [19,22; 43,50]
    TestData t;
    t.a = {1,2, 3,4};
    t.b = {5,6, 7,8};
    t.c = {0,0, 0,0};
    return t;
}

// ==================== Pack FP64 → hardware packed words ====================

std::vector<uint32_t> pack_ab(const std::vector<double>& vals,
                               int type_ab, int sub) {
    int eb = FPConvert::elem_bits(type_ab);
    int eperw = 32 / eb;
    int nw = ((int)vals.size() + eperw - 1) / eperw;
    std::vector<uint32_t> words(nw, 0);

    for (int i = 0; i < (int)vals.size(); i++) {
        int wi = i / eperw, ei = i % eperw;
        uint32_t packed = 0;
        switch (type_ab) {
        case TYPE_FP8: {
            // Convert to FP9 first, then pack to FP8
            uint16_t fp9 = SoftFloat::f64_to_fp9(vals[i]);
            int s=(fp9>>8)&1, e=(fp9>>3)&0x1F, m=fp9&7;
            if (sub == SUB_FP8E5M2)
                packed = (s<<7)|(e<<2)|(m>>1);
            else // E4M3
                packed = (s<<7)|((e&0xF)<<3)|m;
            break;
        }
        case TYPE_FP4: {
            // Direct double→FP4(S1E2M1) conversion, bias=1
            double v = vals[i];
            int s = (v < 0) ? 1 : 0;
            double av = fabs(v);
            int e, m;
            if (av == 0.0) {
                e = 0; m = 0;
            } else if (av >= 4.0) {
                // overflow → saturate to max normal (e=2,m=1) = 3.0
                e = 2; m = 1;
            } else {
                // find biased exponent: av = 1.m * 2^(e-1)
                int exp;
                double frac = frexp(av, &exp); // av = frac * 2^exp, 0.5<=frac<1
                // IEEE-style: 1.m * 2^(exp-1), so biased_e = exp-1+bias = exp
                int biased_e = exp; // bias=1
                if (biased_e <= 0) {
                    // subnormal: 0.m * 2^0
                    e = 0;
                    m = (av >= 0.5) ? 1 : 0;
                } else if (biased_e >= 3) {
                    e = 2; m = 1; // saturate
                } else {
                    e = biased_e;
                    // mantissa: frac is [0.5, 1.0), significand = 2*frac = [1.0, 2.0)
                    // 1 mantissa bit: round (2*frac - 1.0) to 0 or 1
                    m = (2.0*frac - 1.0 >= 0.5) ? 1 : 0;
                }
            }
            packed = (s<<3)|(e<<1)|m;
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
        words[i/2] |= ((uint32_t)h << ((i%2)*16));
    }
    return words;
}

// ==================== Golden Reference GEMM ====================

std::vector<double> golden_gemm(const std::vector<double>& a,
                                 const std::vector<double>& b,
                                 const std::vector<double>& c,
                                 int M, int K, int N) {
    std::vector<double> d(M*N, 0.0);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < K; k++)
                sum += a[i*K+k] * b[k*N+j];
            d[i*N+j] = sum + c[i*N+j];
        }
    return d;
}

// ==================== Verification ====================

bool verify(const std::vector<double>& got,
            const std::vector<double>& ref,
            double rtol, double atol, int M, int N) {
    bool pass = true;
    int nerr = 0;
    double max_err = 0, sum_err = 0;
    for (int i = 0; i < M*N; i++) {
        double err = fabs(got[i] - ref[i]);
        double thr = rtol * fabs(ref[i]) + atol;
        max_err = std::max(max_err, err);
        sum_err += err;
        if (err > thr) {
            if (nerr < 5)
                printf("  MISMATCH D[%d][%d]: got=%f  ref=%f  err=%f (thr=%f)\n",
                       i/N, i%N, got[i], ref[i], err, thr);
            pass = false;
            nerr++;
        }
    }
    if (nerr > 5) printf("  ... and %d more mismatches\n", nerr-5);
    int n = M*N;
    printf("\n  Error stats: max=%.6f  avg=%.6f  (atol=%.3f rtol=%.0f%%)\n",
           max_err, sum_err/n, atol, rtol*100);
    return pass;
}

// ==================== CLI Parsing ====================

struct Args {
    OTC_Config cfg;
    std::string test = "ones";
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        if      (s.find("--M=") == 0) a.cfg.M = std::stoi(s.substr(4));
        else if (s.find("--K=") == 0) a.cfg.K = std::stoi(s.substr(4));
        else if (s.find("--N=") == 0) a.cfg.N = std::stoi(s.substr(4));
        else if (s == "--type_ab=fp4")     { a.cfg.type_ab=TYPE_FP4; }
        else if (s == "--type_ab=fp8e5m2") { a.cfg.type_ab=TYPE_FP8; a.cfg.type_ab_sub=SUB_FP8E5M2; }
        else if (s == "--type_ab=fp8e4m3") { a.cfg.type_ab=TYPE_FP8; a.cfg.type_ab_sub=SUB_FP8E4M3; }
        else if (s == "--type_ab=fp16")    { a.cfg.type_ab=TYPE_FP16; }
        else if (s == "--type_cd=fp16")    { a.cfg.type_cd=TYPE_FP16; }
        else if (s == "--type_cd=fp32")    { a.cfg.type_cd=TYPE_FP32; }
        else if (s.find("--debug=") == 0)  a.cfg.debug_level = std::stoi(s.substr(8));
        else if (s == "--trace")           a.cfg.trace_en = true;
        else if (s.find("--test=") == 0)   a.test = s.substr(7);
        else if (s == "--help" || s == "-h") {
            printf(
"OpenTensorCore SimX — Cycle-Approximate Simulator\n"
"Mirrors Vortex GPGPU simX architecture for OpenTensorCore RTL\n\n"
"Usage: ./otc_simx [options]\n"
"  --M=N             Matrix M dim (default: 8)\n"
"  --K=N             Matrix K dim (default: 8, must be power of 2)\n"
"  --N=N             Matrix N dim (default: 8)\n"
"  --type_ab=TYPE    Input: fp4|fp8e5m2|fp8e4m3|fp16 (default: fp8e5m2)\n"
"  --type_cd=TYPE    Output: fp16|fp32 (default: fp32)\n"
"  --debug=LEVEL     0=off 1=summary 2=pipeline 3=full\n"
"  --trace           Write trace to otc_run.log\n"
"  --test=NAME       ones|identity|random|simple (default: ones)\n"
            );
            exit(0);
        }
    }
    return a;
}

// ==================== Main ====================

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    OTC_Config& cfg = args.cfg;

    // Override for "simple" test
    if (args.test == "simple") { cfg.M=2; cfg.K=2; cfg.N=2; }

    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║    OpenTensorCore SimX — Cycle-Level Simulator   ║\n");
    printf("║    Architecture: Vortex GPGPU simX Pattern       ║\n");
    printf("║    RTL Target:   chenweiphd/OpenTensorCore       ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");

    printf("Configuration:\n");
    printf("  Compute:   D[%dx%d] = A[%dx%d] × B[%dx%d] + C[%dx%d]\n",
           cfg.M, cfg.N, cfg.M, cfg.K, cfg.K, cfg.N, cfg.M, cfg.N);
    const char* type_str;
    switch (cfg.type_ab) {
        case TYPE_FP4:  type_str = "FP4"; break;
        case TYPE_FP8:  type_str = cfg.type_ab_sub==SUB_FP8E5M2 ? "FP8(E5M2)" : "FP8(E4M3)"; break;
        case TYPE_FP16: type_str = "FP16"; break;
        default:        type_str = "?"; break;
    }
    printf("  Input AB:  %s → internal FP9(E5M3)\n", type_str);
    printf("  Output CD: %s  (accumulator: FP22(E8M13))\n",
           cfg.type_cd==TYPE_FP16 ? "FP16" : "FP32");
    printf("  Pipeline:  mul=%dcyc + tree=%dcyc×%dlvl + acc=%dcyc = %d total\n",
           cfg.mul_latency, cfg.add_latency, cfg.tree_depth(),
           cfg.add_latency, cfg.pipeline_depth());
    printf("  Debug:     level=%d  trace=%s\n", cfg.debug_level,
           cfg.trace_en ? "otc_run.log" : "off");
    printf("  Test:      %s\n\n", args.test.c_str());

    // ---- Generate test data ----
    TestData td;
    if      (args.test == "identity") td = gen_identity(cfg.M, cfg.K, cfg.N);
    else if (args.test == "random")   td = gen_random(cfg.M, cfg.K, cfg.N);
    else if (args.test == "simple")   td = gen_simple(cfg.M, cfg.K, cfg.N);
    else                              td = gen_ones(cfg.M, cfg.K, cfg.N);

    // ---- Golden reference ----
    auto gold = golden_gemm(td.a, td.b, td.c, cfg.M, cfg.K, cfg.N);
    printf("Golden D (first row): ");
    for (int j = 0; j < std::min(8, cfg.N); j++) printf("%.4f ", gold[j]);
    if (cfg.N > 8) printf("...");
    printf("\n");

    // ---- Pack data ----
    auto pa = pack_ab(td.a, cfg.type_ab, cfg.type_ab_sub);
    auto pb = pack_ab(td.b, cfg.type_ab, cfg.type_ab_sub);
    auto pc = pack_c_fp16(td.c);

    // ---- Open device & run (Vortex driver pattern) ----
    OTC_Device* dev;
    otc_dev_open(&dev);
    otc_configure(dev, cfg);
    otc_upload(dev, pa.data(), pa.size(), pb.data(), pb.size(), pc.data(), pc.size());

    printf("\nRunning simulation...\n");
    int ret = otc_run(dev);

    if (ret != 0) {
        printf("FAIL: simulation timed out!\n");
        otc_dev_close(dev);
        return 1;
    }

    // ---- Download results ----
    std::vector<double> result(cfg.M * cfg.N);
    otc_download_f64(dev, result.data(), result.size());

    printf("\nSimX D  (first row): ");
    for (int j = 0; j < std::min(8, cfg.N); j++) printf("%.4f ", result[j]);
    if (cfg.N > 8) printf("...");
    printf("\n");

    // ---- Verify ----
    // FP8(E5M2) has only 2 mantissa bits → large quantization steps
    // Accumulated over K mults, absolute error can be ~0.5
    double rtol, atol;
    switch (cfg.type_ab) {
        case TYPE_FP4:  rtol = 0.5;  atol = 2.0;   break;
        case TYPE_FP8:  rtol = 0.30; atol = 0.50;   break;
        case TYPE_FP16: rtol = 0.01; atol = 0.001;  break;
        default:        rtol = 0.30; atol = 0.50;   break;
    }

    bool pass = verify(result, gold, rtol, atol, cfg.M, cfg.N);
    printf("\nVerification: %s (rtol=%.0f%% atol=%.3f)\n", pass ? "PASSED ✓" : "FAILED ✗",
           rtol * 100, atol);

    // ---- Statistics ----
    printf("\n");
    otc_stats(dev).print(std::cout);

    printf("\n=== Hardware Resource Estimate (RTL) ===\n");
    printf("  tc_mul_pipe instances: %d  (K=%d per dp × %d dp units)\n",
           cfg.total_dp() * cfg.K, cfg.K, cfg.total_dp());
    printf("  tc_add_pipe instances: %d  (tree) + %d (final_add)\n",
           cfg.total_dp() * (cfg.K - 1), cfg.total_dp());
    printf("  Total FP operators:    %d\n",
           cfg.total_dp() * cfg.K + cfg.total_dp() * cfg.K);

    // ---- Print full output matrix for small tests ----
    if (cfg.M <= 8 && cfg.N <= 8) {
        printf("\n=== Full Output Matrix D[%dx%d] ===\n", cfg.M, cfg.N);
        for (int i = 0; i < cfg.M; i++) {
            printf("  [");
            for (int j = 0; j < cfg.N; j++)
                printf("%8.3f", result[i*cfg.N + j]);
            printf(" ]\n");
        }
        printf("\n=== Golden Reference ===\n");
        for (int i = 0; i < cfg.M; i++) {
            printf("  [");
            for (int j = 0; j < cfg.N; j++)
                printf("%8.3f", gold[i*cfg.N + j]);
            printf(" ]\n");
        }
    }

    otc_dev_close(dev);
    return pass ? 0 : 1;
}
