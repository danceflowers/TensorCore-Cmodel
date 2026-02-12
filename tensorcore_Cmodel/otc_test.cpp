// otc_test.cpp — Comprehensive test suite for OpenTensorCore SimX
// Tests every precision mode × multiple data patterns × multiple sizes
// Reports per-test pass/fail with detailed error stats.

#include "otc_driver.h"
#include "otc_decode.h"

// ============================================================================
// Test infrastructure
// ============================================================================

struct TestResult {
    std::string name;
    bool        pass;
    double      max_err;
    double      avg_err;
    int         mismatches;
};

static int g_total = 0, g_passed = 0, g_failed = 0;
static std::vector<TestResult> g_results;

// Pack helpers (same as main.cpp — duplicated here to keep test self-contained)
std::vector<uint32_t> test_pack_ab(const std::vector<double>& vals, int type_ab, int sub) {
    int eb = FPConvert::elem_bits(type_ab);
    int eperw = 32 / eb;
    int nw = ((int)vals.size() + eperw - 1) / eperw;
    std::vector<uint32_t> words(nw, 0);

    for (int i = 0; i < (int)vals.size(); i++) {
        int wi = i / eperw, ei = i % eperw;
        uint32_t packed = 0;
        switch (type_ab) {
            case TYPE_FP8: {
                if (sub == SUB_FP8E5M2) {
                    uint16_t fp9 = SoftFloat::f64_to_fp9(vals[i]);
                    int s9 = (fp9 >> 8) & 1, e9 = (fp9 >> 3) & 0x1F, m9 = fp9 & 7;
                    packed = (s9 << 7) | (e9 << 2) | (m9 >> 1);
                } else {
                    double v = vals[i];
                    int s = (v < 0) ? 1 : 0;
                    double av = fabs(v);
                    if (std::isnan(v)) {
                        packed = 0x7F;
                    } else if (av == 0.0) {
                        packed = (s << 7);
                    } else {
                        int exp;
                        double frac = frexp(av, &exp);
                        frac *= 2.0; exp--;
                        int be = exp + 7;
                        int m;
                        if (be >= 15) {
                            packed = (s << 7) | (0x0E << 3) | 0x07;
                        } else if (be <= 0) {
                            m = (int)(av / ldexp(1.0, -9) + 0.5) & 0x07;
                            packed = (s << 7) | m;
                        } else {
                            m = (int)((frac - 1.0) * 8.0 + 0.5) & 0x07;
                            packed = (s << 7) | (be << 3) | m;
                        }
                    }
                }
                break;
            }
            case TYPE_FP4: {
                double v = vals[i];
                int s = (v < 0) ? 1 : 0;
                double av = fabs(v);
                int e, m;
                if (av == 0.0) { e = 0; m = 0; }
                else if (av >= 4.0) { e = 2; m = 1; }
                else {
                    int ex; double fr = frexp(av, &ex);
                    int be = ex;
                    if (be <= 0) { e = 0; m = (av >= 0.5) ? 1 : 0; }
                    else if (be >= 3) { e = 2; m = 1; }
                    else { e = be; m = (2.0 * fr - 1.0 >= 0.5) ? 1 : 0; }
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

std::vector<uint32_t> test_pack_c_fp16(const std::vector<double>& vals) {
    int nw = ((int)vals.size() + 1) / 2;
    std::vector<uint32_t> words(nw, 0);
    for (int i = 0; i < (int)vals.size(); i++) {
        uint16_t h = SoftFloat::f64_to_fp16(vals[i]);
        words[i / 2] |= ((uint32_t)h << ((i % 2) * 16));
    }
    return words;
}

double quantize_output_ref(double v, int type_cd, int type_cd_sub);

// Quantized golden: same precision path as simulator
std::vector<double> quantized_golden(const std::vector<double>& a, const std::vector<double>& b,
                                      const std::vector<double>& c, int M, int K, int N,
                                      int type_ab, int sub, int type_cd, int type_cd_sub) {
    auto pa = test_pack_ab(a, type_ab, sub);
    auto pb = test_pack_ab(b, type_ab, sub);
    auto pc = test_pack_c_fp16(c);

    int eb = FPConvert::elem_bits(type_ab);
    int eperw = 32 / eb;

    std::vector<double> aq(M * K), bq(K * N), cq(M * N);
    for (int i = 0; i < M * K; i++) {
        int wi = i / eperw, ei = i % eperw;
        aq[i] = FPConvert::elem_to_f64(wi < (int)pa.size() ? pa[wi] : 0, ei, type_ab, sub);
    }
    for (int i = 0; i < K * N; i++) {
        int wi = i / eperw, ei = i % eperw;
        bq[i] = FPConvert::elem_to_f64(wi < (int)pb.size() ? pb[wi] : 0, ei, type_ab, sub);
    }
    for (int i = 0; i < M * N; i++) {
        int wi = i / 2, ei = i % 2;
        uint32_t w = wi < (int)pc.size() ? pc[wi] : 0;
        double c16 = SoftFloat::fp16_to_f64((w >> (ei * 16)) & 0xFFFF);
        cq[i] = SoftFloat::fp22_to_f64(SoftFloat::f64_to_fp22(c16));
    }

    // Golden path is aligned to the current simulator datapath implementation:
    //   1) A/B are quantized by pack+unpack above (aq/bq)
    //   2) dot-product accumulation is performed in host high precision
    //   3) C participates as fp22-quantized input (cq)
    //   4) final output is quantized by configured output type
    std::vector<double> d(M * N, 0.0);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double dot = 0.0;
            for (int k = 0; k < K; k++) {
                dot += aq[i * K + k] * bq[k * N + j];
            }
            double acc_fp22 = SoftFloat::fp22_to_f64(SoftFloat::f64_to_fp22(dot + cq[i * N + j]));
            d[i * N + j] = quantize_output_ref(acc_fp22, type_cd, type_cd_sub);
        }
    }
    return d;
}



double quantize_output_ref(double v, int type_cd, int type_cd_sub) {
    if (type_cd == TYPE_FP32) {
        return SoftFloat::fp32_to_f64(SoftFloat::f64_to_fp32(v));
    }
    if (type_cd == TYPE_FP16) {
        return SoftFloat::fp16_to_f64(SoftFloat::f64_to_fp16(v));
    }
    if (type_cd == TYPE_FP8) {
        uint8_t fp8 = (type_cd_sub == SUB_FP8E4M3) ? FPConvert::f64_to_fp8e4m3(v) : FPConvert::f64_to_fp8e5m2(v);
        return (type_cd_sub == SUB_FP8E4M3) ? FPConvert::fp8e4m3_to_f64(fp8) : FPConvert::fp8e5m2_to_f64(fp8);
    }
    return v;
}

std::vector<double> fp32_golden(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& c,
                                int M, int K, int N) {
    std::vector<double> d(M * N, 0.0);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += (float)a[i * K + k] * (float)b[k * N + j];
            d[i * N + j] = (double)(sum + (float)c[i * N + j]);
        }
    }
    return d;
}

void print_matrix(const char* tag, const std::vector<double>& m, int R, int C) {
    printf("%s\n", tag);
    for (int i = 0; i < R; i++) {
        printf("  ");
        for (int j = 0; j < C; j++) {
            printf("%10.6f ", m[i * C + j]);
        }
        printf("\n");
    }
}

// Run a single GEMM test, return TestResult
TestResult run_one_test(const std::string& name, int M, int K, int N,
                         int type_ab, int sub, int type_cd, int type_cd_sub,
                         const std::vector<double>& a, const std::vector<double>& b,
                         const std::vector<double>& c, double rtol, double atol, bool dump_matrix = false) {
    OTC_Config cfg;
    cfg.M = M; cfg.K = K; cfg.N = N;
    cfg.type_ab = type_ab; cfg.type_ab_sub = sub; cfg.type_cd = type_cd; cfg.type_cd_sub = type_cd_sub;
    cfg.debug_level = 0;

    auto pa = test_pack_ab(a, type_ab, sub);
    auto pb = test_pack_ab(b, type_ab, sub);
    auto pc = test_pack_c_fp16(c);

    auto gold_q = quantized_golden(a, b, c, M, K, N, type_ab, sub, type_cd, type_cd_sub);

    OTC_Device* dev;
    otc_dev_open(&dev);
    otc_configure(dev, cfg);
    otc_upload(dev, pa.data(), pa.size(), pb.data(), pb.size(), pc.data(), pc.size());
    int ret = otc_run(dev);

    TestResult tr;
    tr.name = name;

    if (ret != 0) {
        tr.pass = false; tr.max_err = 1e30; tr.avg_err = 1e30; tr.mismatches = M * N;
        otc_dev_close(dev);
        return tr;
    }

    std::vector<double> result(M * N);
    otc_download_f64(dev, result.data(), result.size());
    otc_dev_close(dev);

    std::vector<double> gold_fp32;
    if ((M == 8 && K == 8 && N == 8) || dump_matrix) {
        gold_fp32 = fp32_golden(a, b, c, M, K, N);
        printf("\n  [Matrix dump] %s\n", name.c_str());
        print_matrix("  Result matrix:", result, M, N);
        print_matrix("  Golden matrix (fp32):", gold_fp32, M, N);
        print_matrix("  Golden matrix (quantized output):", gold_q, M, N);
    }

    tr.pass = true; tr.max_err = 0; tr.avg_err = 0; tr.mismatches = 0;
    double sum_err = 0;
    for (int i = 0; i < M * N; i++) {
        double err = fabs(result[i] - gold_q[i]);
        double thr = rtol * fabs(gold_q[i]) + atol;
        sum_err += err;
        if (err > tr.max_err) tr.max_err = err;
        if (err > thr) { tr.pass = false; tr.mismatches++; }
    }
    tr.avg_err = sum_err / (M * N);
    return tr;
}

void report(const TestResult& r) {
    g_total++;
    if (r.pass) g_passed++; else g_failed++;
    g_results.push_back(r);

    printf("  %-50s %s  max_err=%.6e  avg_err=%.6e",
           r.name.c_str(), r.pass ? "PASS" : "FAIL", r.max_err, r.avg_err);
    if (!r.pass) printf("  (%d mismatches)", r.mismatches);
    printf("\n");
}

// ============================================================================
// Data generators
// ============================================================================

std::vector<double> gen_const(int n, double val) {
    return std::vector<double>(n, val);
}

std::vector<double> gen_zeros(int n) { return gen_const(n, 0.0); }

std::vector<double> gen_rand(int n, unsigned seed, double lo = -1.0, double hi = 1.0) {
    std::vector<double> v(n);
    srand(seed);
    for (auto& x : v) x = lo + (hi - lo) * (rand() % 10000) / 9999.0;
    return v;
}

std::vector<double> gen_identity(int rows, int cols) {
    std::vector<double> v(rows * cols, 0.0);
    int mn = std::min(rows, cols);
    for (int i = 0; i < mn; i++) v[i * cols + i] = 1.0;
    return v;
}

std::vector<double> gen_small_ints(int n, unsigned seed) {
    std::vector<double> v(n);
    srand(seed);
    for (auto& x : v) x = (rand() % 5) - 2;  // -2 to 2
    return v;
}

// ============================================================================
// Test suites
// ============================================================================

struct TypeSpec {
    const char* name;
    int type_ab;
    int sub;
    double rtol;
    double atol;
};

static const TypeSpec ALL_TYPES[] = {
    {"fp8e5m2", TYPE_FP8,  SUB_FP8E5M2, 0.05, 0.01},
    {"fp8e4m3", TYPE_FP8,  SUB_FP8E4M3, 0.05, 0.01},
    {"fp16",    TYPE_FP16, 0,            0.02, 0.005},
    {"fp4",     TYPE_FP4,  0,            0.05, 0.01},
};

void test_ones_suite() {
    printf("\n=== Suite: All-ones matrices ===\n");
    int dims[][3] = {{2,2,2},{4,4,4},{8,8,8},{16,16,16},{8,4,8},{4,8,4}};
    for (auto& ts : ALL_TYPES) {
        for (auto& d : dims) {
            int M = d[0], K = d[1], N = d[2];
            std::string name = std::string("ones_") + ts.name + "_" +
                               std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N);
            auto a = gen_const(M * K, 1.0);
            auto b = gen_const(K * N, 1.0);
            auto c = gen_zeros(M * N);
            report(run_one_test(name, M, K, N, ts.type_ab, ts.sub, TYPE_FP32, SUB_FP8E5M2, a, b, c, ts.rtol, ts.atol));
        }
    }
}

void test_identity_suite() {
    printf("\n=== Suite: Identity matrix ===\n");
    int dims[][3] = {{2,2,2},{4,4,4},{8,8,8}};
    for (auto& ts : ALL_TYPES) {
        for (auto& d : dims) {
            int M = d[0], K = d[1], N = d[2];
            std::string name = std::string("ident_") + ts.name + "_" +
                               std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N);
            auto a = gen_identity(M, K);
            auto b = gen_identity(K, N);
            auto c = gen_zeros(M * N);
            report(run_one_test(name, M, K, N, ts.type_ab, ts.sub, TYPE_FP32, SUB_FP8E5M2, a, b, c, ts.rtol, ts.atol));
        }
    }
}

void test_random_suite() {
    printf("\n=== Suite: Random matrices (multiple seeds) ===\n");
    int dims[][3] = {{8,8,8},{4,4,4},{8,4,8},{4,8,4},{16,16,16}};
    unsigned seeds[] = {42, 123, 256, 999, 1337, 2024, 31415, 65535, 77777, 88888};
    for (auto& ts : ALL_TYPES) {
        for (auto& d : dims) {
            for (auto seed : seeds) {
                int M = d[0], K = d[1], N = d[2];
                std::string name = std::string("rand_") + ts.name + "_" +
                                   std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N) +
                                   "_s" + std::to_string(seed);
                auto a = gen_rand(M * K, seed, -1.0, 1.0);
                auto b = gen_rand(K * N, seed + 100, -1.0, 1.0);
                auto c = gen_rand(M * N, seed + 200, -0.5, 0.5);
                report(run_one_test(name, M, K, N, ts.type_ab, ts.sub, TYPE_FP32, SUB_FP8E5M2, a, b, c, ts.rtol, ts.atol));
            }
        }
    }
}

void test_small_ints_suite() {
    printf("\n=== Suite: Small integer values ===\n");
    int dims[][3] = {{2,2,2},{4,4,4},{8,8,8}};
    for (auto& ts : ALL_TYPES) {
        for (auto& d : dims) {
            int M = d[0], K = d[1], N = d[2];
            std::string name = std::string("smallint_") + ts.name + "_" +
                               std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N);
            auto a = gen_small_ints(M * K, 77);
            auto b = gen_small_ints(K * N, 88);
            auto c = gen_zeros(M * N);
            report(run_one_test(name, M, K, N, ts.type_ab, ts.sub, TYPE_FP32, SUB_FP8E5M2, a, b, c, ts.rtol, ts.atol));
        }
    }
}

void test_with_bias_c_suite() {
    printf("\n=== Suite: Non-zero C bias ===\n");
    int dims[][3] = {{4,4,4},{8,8,8}};
    for (auto& ts : ALL_TYPES) {
        for (auto& d : dims) {
            int M = d[0], K = d[1], N = d[2];
            std::string name = std::string("bias_") + ts.name + "_" +
                               std::to_string(M) + "x" + std::to_string(K) + "x" + std::to_string(N);
            auto a = gen_const(M * K, 0.5);
            auto b = gen_const(K * N, 0.5);
            auto c = gen_const(M * N, 1.0);  // D = K*0.25 + 1.0
            report(run_one_test(name, M, K, N, ts.type_ab, ts.sub, TYPE_FP32, SUB_FP8E5M2, a, b, c, ts.rtol, ts.atol));
        }
    }
}

void test_edge_values_suite() {
    printf("\n=== Suite: Edge values (zeros, negatives, near-overflow) ===\n");
    for (auto& ts : ALL_TYPES) {
        int M = 4, K = 4, N = 4;
        {   // All zeros
            std::string name = std::string("zeros_") + ts.name;
            auto a = gen_zeros(M * K);
            auto b = gen_zeros(K * N);
            auto c = gen_zeros(M * N);
            report(run_one_test(name, M, K, N, ts.type_ab, ts.sub, TYPE_FP32, SUB_FP8E5M2, a, b, c, ts.rtol, ts.atol));
        }
        {   // Negative ones
            std::string name = std::string("negones_") + ts.name;
            auto a = gen_const(M * K, -1.0);
            auto b = gen_const(K * N, 1.0);
            auto c = gen_zeros(M * N);
            report(run_one_test(name, M, K, N, ts.type_ab, ts.sub, TYPE_FP32, SUB_FP8E5M2, a, b, c, ts.rtol, ts.atol));
        }
        {   // Mixed sign
            std::string name = std::string("mixsign_") + ts.name;
            std::vector<double> a(M * K), b(K * N);
            for (int i = 0; i < M * K; i++) a[i] = (i % 2 == 0) ? 1.0 : -1.0;
            for (int i = 0; i < K * N; i++) b[i] = (i % 3 == 0) ? -0.5 : 0.5;
            auto c = gen_zeros(M * N);
            report(run_one_test(name, M, K, N, ts.type_ab, ts.sub, TYPE_FP32, SUB_FP8E5M2, a, b, c, ts.rtol, ts.atol));
        }
    }
}


void test_precision_cross_suite() {
    printf("\n=== Suite: 8x8x8 random cross precision AB->CD (multi-run) ===\n");

    struct OutSpec { const char* name; int type_cd; int sub_cd; double rtol; double atol; };
    const OutSpec out_specs[] = {
        {"fp8e5m2", TYPE_FP8, SUB_FP8E5M2, 0.10, 0.20},
        {"fp8e4m3", TYPE_FP8, SUB_FP8E4M3, 0.25, 1.20},
        {"fp16", TYPE_FP16, SUB_FP8E5M2, 0.05, 0.02},
        {"fp32", TYPE_FP32, SUB_FP8E5M2, 0.05, 0.02},
    };

    const unsigned seeds[] = {11, 29, 47, 71, 97, 123, 211, 307};
    const int M = 8, K = 8, N = 8;

    for (const auto& in : ALL_TYPES) {
        for (const auto& out : out_specs) {
            for (unsigned seed : seeds) {
                auto a = gen_rand(M * K, seed, -1.0, 1.0);
                auto b = gen_rand(K * N, seed + 1000, -1.0, 1.0);
                auto c = gen_rand(M * N, seed + 2000, -0.5, 0.5);
                std::string name = std::string("cross_") + in.name + "_to_" + out.name + "_s" + std::to_string(seed);
                report(run_one_test(name, M, K, N, in.type_ab, in.sub, out.type_cd, out.sub_cd,
                                    a, b, c, out.rtol, out.atol, true));
            }
        }
    }
}


// ============================================================================
// Decode framework unit tests
// ============================================================================

void test_decode_framework() {
    printf("\n=== Suite: Decode framework ===\n");

    OTC_Decoder dec;
    dec.init();

    int dec_pass = 0, dec_fail = 0;

    auto check = [&](const char* name, bool cond) {
        g_total++;
        if (cond) { g_passed++; dec_pass++; }
        else { g_failed++; dec_fail++; }
        printf("  %-50s %s\n", name, cond ? "PASS" : "FAIL");
        TestResult tr; tr.name = name; tr.pass = cond;
        tr.max_err = tr.avg_err = 0; tr.mismatches = cond ? 0 : 1;
        g_results.push_back(tr);
    };

    // Test table loaded
    check("decode_table_loaded", dec.table_size() == 10);

    // Test all instruction types decode correctly from their encoding
    auto test_decode_type = [&](const char* label, uint8_t opcode, uint8_t funct3,
                                 OTC_OpType expected_op) {
        // Build a minimal 32-bit word with opcode in [6:0] and funct3 in [14:12]
        uint32_t inst = (uint32_t)opcode | ((uint32_t)funct3 << 12);
        DecodedInst d = dec.decode(inst);
        std::string nm = std::string("decode_") + label;
        check(nm.c_str(), d.valid && d.op == expected_op);
    };

    test_decode_type("TCU_WMMA",    0x21, 0x01, OTC_OpType::TCU_WMMA);
    test_decode_type("TCU_LOAD",    0x23, 0x01, OTC_OpType::TCU_LOAD);
    test_decode_type("TCU_STORE",   0x27, 0x01, OTC_OpType::TCU_STORE);
    test_decode_type("LOAD",        0x03, 0x02, OTC_OpType::LOAD);
    test_decode_type("STORE",       0x23, 0x02, OTC_OpType::STORE);
    test_decode_type("TCU_BARRIER", 0x33, 0x01, OTC_OpType::TCU_BARRIER);
    test_decode_type("TCU_SP",      0x43, 0x01, OTC_OpType::TCU_SP);
    test_decode_type("TCU_INT",     0x53, 0x00, OTC_OpType::TCU_INT);
    test_decode_type("TCU_DP",      0x63, 0x01, OTC_OpType::TCU_DP);
    test_decode_type("TCU_SFU",     0x73, 0x01, OTC_OpType::TCU_SFU);

    // Test invalid instruction
    {
        uint32_t bad = 0xFFFFFFFF;
        DecodedInst d = dec.decode(bad);
        check("decode_invalid_returns_nop", !d.valid && d.op == OTC_OpType::NOP);
    }

    // Test control flags
    {
        uint32_t wmma_inst = 0x21 | (0x01 << 12);  // TCU_WMMA
        DecodedInst d = dec.decode(wmma_inst);
        check("decode_wmma_is_tcu", d.is_tcu);
        check("decode_wmma_not_mem", !d.is_mem);
    }
    {
        uint32_t load_inst = 0x23 | (0x01 << 12);  // TCU_LOAD
        DecodedInst d = dec.decode(load_inst);
        check("decode_tcu_load_is_mem", d.is_mem);
    }
    {
        uint32_t barrier_inst = 0x33 | (0x01 << 12);  // TCU_BARRIER
        DecodedInst d = dec.decode(barrier_inst);
        check("decode_barrier_is_sync", d.is_sync);
    }

    // Test ISA table hot-swap
    {
        std::vector<ISA_Entry> custom = {
            { OTC_OpType::TCU_WMMA, 0x3F, 0x01, 0x03, ExecUnit::TCU, 0x07 },
        };
        OTC_Decoder dec2;
        dec2.init();
        dec2.load_isa_table(custom);
        uint32_t inst = 0x3F | (0x03 << 12);
        DecodedInst d = dec2.decode(inst);
        check("decode_custom_table", d.valid && d.op == OTC_OpType::TCU_WMMA);

        // Old encoding should no longer match
        uint32_t old_inst = 0x21 | (0x01 << 12);
        DecodedInst d2 = dec2.decode(old_inst);
        check("decode_custom_rejects_old", !d2.valid);
    }

    // Test register extraction
    {
        // rd=5 in bits [11:7], rs1=10 in bits [19:15], rs2=20 in bits [24:20]
        uint32_t inst = 0x21 | (0x01 << 12);  // TCU_WMMA base
        inst |= (5 << 7);    // rd=5
        inst |= (10 << 15);  // rs1=10
        inst |= (20 << 20);  // rs2=20
        DecodedInst d = dec.decode(inst);
        // Note: rd field overlaps unit_id extraction in our scheme,
        // so we just check rs1 and rs2
        check("decode_rs1_extraction", d.rs1 == 10);
        check("decode_rs2_extraction", d.rs2 == 20);
    }

    printf("  Decode tests: %d passed, %d failed\n", dec_pass, dec_fail);
}

// ============================================================================
// FP conversion round-trip tests
// ============================================================================

void test_fp_roundtrips() {
    printf("\n=== Suite: FP conversion round-trips ===\n");

    int fp_pass = 0, fp_fail = 0;
    auto check = [&](const char* name, bool cond) {
        g_total++;
        if (cond) { g_passed++; fp_pass++; }
        else { g_failed++; fp_fail++; }
        printf("  %-50s %s\n", name, cond ? "PASS" : "FAIL");
        TestResult tr; tr.name = name; tr.pass = cond;
        tr.max_err = tr.avg_err = 0; tr.mismatches = cond ? 0 : 1;
        g_results.push_back(tr);
    };

    // FP16 round-trip
    {
        double vals[] = {0.0, 1.0, -1.0, 0.5, 65504.0, -65504.0, 0.00006103515625};
        bool ok = true;
        for (double v : vals) {
            uint16_t h = SoftFloat::f64_to_fp16(v);
            double back = SoftFloat::fp16_to_f64(h);
            if (fabs(back - v) > 1e-10) ok = false;
        }
        check("fp16_roundtrip_exact_values", ok);
    }

    // FP9 round-trip
    {
        double vals[] = {0.0, 1.0, -1.0, 0.5, 2.0};
        bool ok = true;
        for (double v : vals) {
            uint16_t fp9 = SoftFloat::f64_to_fp9(v);
            double back = SoftFloat::fp9_to_f64(fp9);
            if (fabs(back - v) > 1e-10) ok = false;
        }
        check("fp9_roundtrip_exact_values", ok);
    }

    // FP22 round-trip
    {
        double vals[] = {0.0, 1.0, -1.0, 100.0, -100.0, 0.001};
        bool ok = true;
        for (double v : vals) {
            uint32_t fp22 = SoftFloat::f64_to_fp22(v);
            double back = SoftFloat::fp22_to_f64(fp22);
            if (fabs(back - v) / (fabs(v) + 1e-30) > 0.001) ok = false;
        }
        check("fp22_roundtrip_close", ok);
    }

    // FP8 E4M3: 1.0 round-trip
    {
        double v = 1.0;
        int s = 0; double av = 1.0;
        int exp; double frac = frexp(av, &exp); frac *= 2; exp--;
        int be = exp + 7;  // bias=7, so 1.0 has exp=0, be=7
        int m = (int)((frac - 1.0) * 8 + 0.5) & 7;
        uint8_t packed = (s << 7) | (be << 3) | m;
        double back = FPConvert::fp8e4m3_to_f64(packed);
        check("fp8e4m3_1.0_roundtrip", fabs(back - 1.0) < 1e-10);
    }

    // FP8 E4M3: -0.5 round-trip
    {
        double v = -0.5;
        int s = 1; double av = 0.5;
        int exp; double frac = frexp(av, &exp); frac *= 2; exp--;
        int be = exp + 7;
        int m = (int)((frac - 1.0) * 8 + 0.5) & 7;
        uint8_t packed = (s << 7) | (be << 3) | m;
        double back = FPConvert::fp8e4m3_to_f64(packed);
        check("fp8e4m3_-0.5_roundtrip", fabs(back - (-0.5)) < 1e-10);
    }

    // FP4 round-trip for representable values
    {
        double vals[] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0};
        bool ok = true;
        for (double v : vals) {
            // Pack and unpack
            std::vector<double> vv = {v};
            auto packed = test_pack_ab(vv, TYPE_FP4, 0);
            double back = FPConvert::fp4_to_f64(packed[0] & 0xF);
            if (fabs(back - v) > 0.5) ok = false;
        }
        check("fp4_roundtrip_representable", ok);
    }

    printf("  FP round-trip tests: %d passed, %d failed\n", fp_pass, fp_fail);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║   OpenTensorCore SimX — Comprehensive Test Suite           ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    test_fp_roundtrips();
    test_decode_framework();
    test_ones_suite();
    test_identity_suite();
    test_random_suite();
    test_small_ints_suite();
    test_with_bias_c_suite();
    test_edge_values_suite();
    test_precision_cross_suite();

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║   SUMMARY                                                  ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║   Total:  %4d                                             ║\n", g_total);
    printf("║   Passed: %4d                                             ║\n", g_passed);
    printf("║   Failed: %4d                                             ║\n", g_failed);
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    if (g_failed > 0) {
        printf("\nFailed tests:\n");
        for (auto& r : g_results) {
            if (!r.pass) {
                printf("  FAIL: %s  max_err=%.6e  mismatches=%d\n",
                       r.name.c_str(), r.max_err, r.mismatches);
            }
        }
    }

    return g_failed > 0 ? 1 : 0;
}
