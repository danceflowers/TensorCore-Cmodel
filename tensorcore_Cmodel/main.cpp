#include "otc_decode.h"
#include "otc_driver.h"

struct TestData {
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> c;
};

struct PrecCase {
    int type_ab;
    int type_ab_sub;
    int type_cd;
    int type_cd_sub;
    const char* name;
};

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

static std::vector<uint32_t> pack_ab(const std::vector<double>& vals, int type_ab, int sub) {
    int eb = FPConvert::elem_bits(type_ab);
    int eperw = 32 / eb;
    std::vector<uint32_t> words((vals.size() + eperw - 1) / eperw, 0);
    for (int i = 0; i < (int)vals.size(); i++) {
        int wi = i / eperw;
        int ei = i % eperw;
        uint32_t packed = 0;
        if (type_ab == TYPE_FP8) {
            packed = (sub == SUB_FP8E4M3) ? FPConvert::f64_to_fp8e4m3(vals[i]) : FPConvert::f64_to_fp8e5m2(vals[i]);
        } else if (type_ab == TYPE_FP4) {
            packed = (uint32_t)(SoftFloat::f64_to_fp9(vals[i]) >> 5) & 0xF;
        } else {
            packed = SoftFloat::f64_to_fp16(vals[i]);
        }
        words[wi] |= (packed << (ei * eb));
    }
    return words;
}

static std::vector<uint32_t> pack_c_fp16(const std::vector<double>& vals) {
    std::vector<uint32_t> words((vals.size() + 1) / 2, 0);
    for (int i = 0; i < (int)vals.size(); i++) {
        words[i / 2] |= ((uint32_t)SoftFloat::f64_to_fp16(vals[i]) << ((i % 2) * 16));
    }
    return words;
}

static void unpack_quantized_inputs(const TestData& td, const OTC_Config& cfg,
                                    std::vector<double>& aq, std::vector<double>& bq, std::vector<double>& cq) {
    auto pa = pack_ab(td.a, cfg.type_ab, cfg.type_ab_sub);
    auto pb = pack_ab(td.b, cfg.type_ab, cfg.type_ab_sub);
    auto pc = pack_c_fp16(td.c);

    int eb = FPConvert::elem_bits(cfg.type_ab);
    int eperw = 32 / eb;

    aq.resize(cfg.M * cfg.K);
    bq.resize(cfg.K * cfg.N);
    cq.resize(cfg.M * cfg.N);

    for (int i = 0; i < cfg.M * cfg.K; i++) {
        int wi = i / eperw;
        int ei = i % eperw;
        uint32_t w = (wi < (int)pa.size()) ? pa[wi] : 0;
        aq[i] = FPConvert::elem_to_f64(w, ei, cfg.type_ab, cfg.type_ab_sub);
    }
    for (int i = 0; i < cfg.K * cfg.N; i++) {
        int wi = i / eperw;
        int ei = i % eperw;
        uint32_t w = (wi < (int)pb.size()) ? pb[wi] : 0;
        bq[i] = FPConvert::elem_to_f64(w, ei, cfg.type_ab, cfg.type_ab_sub);
    }
    for (int i = 0; i < cfg.M * cfg.N; i++) {
        int wi = i / 2;
        int ei = i % 2;
        uint32_t w = (wi < (int)pc.size()) ? pc[wi] : 0;
        uint16_t h = (w >> (ei * 16)) & 0xFFFF;
        cq[i] = SoftFloat::fp16_to_f64(h);
    }
}


static std::vector<double> golden_model_quantized(const std::vector<double>& aq, const std::vector<double>& bq,
                                                  const std::vector<double>& cq, const OTC_Config& cfg) {
    std::vector<double> d(cfg.M * cfg.N, 0.0);
    for (int i = 0; i < cfg.M; i++) {
        for (int j = 0; j < cfg.N; j++) {
            std::vector<uint16_t> ts(cfg.K);
            for (int k = 0; k < cfg.K; k++) {
                uint16_t a9 = SoftFloat::f64_to_fp9(aq[i * cfg.K + k]);
                uint16_t b9 = SoftFloat::f64_to_fp9(bq[k * cfg.N + j]);
                ts[k] = SoftFloat::f64_to_fp13(SoftFloat::fp9_to_f64(FPEmu::fp9_mul(a9, b9)));
            }
            int w = cfg.K;
            while (w > 1) {
                for (int x = 0; x < w / 2; x++) ts[x] = FPEmu::fp13_add(ts[2 * x], ts[2 * x + 1]);
                w >>= 1;
            }
            uint32_t c22 = SoftFloat::f64_to_fp22(cq[i * cfg.N + j]);
            uint32_t out22 = FPEmu::fp22_add(FPEmu::fp9_to_fp22(FPEmu::fp13_to_fp9(ts[0])), c22);
            if (cfg.type_cd == TYPE_FP16) {
                d[i * cfg.N + j] = SoftFloat::fp16_to_f64(FPEmu::fp22_to_fp16(out22));
            } else if (cfg.type_cd == TYPE_FP32) {
                d[i * cfg.N + j] = SoftFloat::fp32_to_f64(SoftFloat::f64_to_fp32(SoftFloat::fp22_to_f64(out22)));
            } else {
                uint8_t fp8 = (cfg.type_cd_sub == SUB_FP8E4M3) ? FPEmu::fp22_to_fp8(out22, SUB_FP8E4M3)
                                                                : FPEmu::fp22_to_fp8(out22, SUB_FP8E5M2);
                d[i * cfg.N + j] = (cfg.type_cd_sub == SUB_FP8E4M3) ? FPConvert::fp8e4m3_to_f64(fp8)
                                                                     : FPConvert::fp8e5m2_to_f64(fp8);
            }
        }
    }
    return d;
}

static std::vector<double> golden_fp32_unquantized(const std::vector<double>& aq, const std::vector<double>& bq,
                                                    const std::vector<double>& cq, const OTC_Config& cfg) {
    std::vector<double> d(cfg.M * cfg.N, 0.0);
    for (int i = 0; i < cfg.M; i++) {
        for (int j = 0; j < cfg.N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < cfg.K; k++) {
                sum += (float)aq[i * cfg.K + k] * (float)bq[k * cfg.N + j];
            }
            d[i * cfg.N + j] = (double)(sum + (float)cq[i * cfg.N + j]);
        }
    }
    return d;
}

static std::vector<double> golden_fp32_quantized(const std::vector<double>& unq, const OTC_Config& cfg) {
    std::vector<double> q(unq.size(), 0.0);
    for (size_t i = 0; i < unq.size(); i++) {
        double v = unq[i];
        if (cfg.type_cd == TYPE_FP16) {
            q[i] = SoftFloat::fp16_to_f64(SoftFloat::f64_to_fp16(v));
        } else if (cfg.type_cd == TYPE_FP32) {
            q[i] = SoftFloat::fp32_to_f64(SoftFloat::f64_to_fp32(v));
        } else {
            uint8_t fp8 = (cfg.type_cd_sub == SUB_FP8E4M3) ? FPConvert::f64_to_fp8e4m3(v) : FPConvert::f64_to_fp8e5m2(v);
            q[i] = (cfg.type_cd_sub == SUB_FP8E4M3) ? FPConvert::fp8e4m3_to_f64(fp8) : FPConvert::fp8e5m2_to_f64(fp8);
        }
    }
    return q;
}

static void print_matrix(const char* tag, const std::vector<double>& m, int R, int C) {
    printf("%s\n", tag);
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) printf("%9.5f ", m[i * C + j]);
        printf("\n");
    }
}

int main() {
    std::vector<PrecCase> cases = {
        {TYPE_FP4,  SUB_FP8E5M2, TYPE_FP16, SUB_FP8E5M2, "ab=fp4 -> out=fp16"},
        {TYPE_FP8,  SUB_FP8E4M3, TYPE_FP16, SUB_FP8E5M2, "ab=fp8e4m3 -> out=fp16"},
        {TYPE_FP8,  SUB_FP8E5M2, TYPE_FP16, SUB_FP8E5M2, "ab=fp8e5m2 -> out=fp16"},
        {TYPE_FP16, SUB_FP8E5M2, TYPE_FP16, SUB_FP8E5M2, "ab=fp16 -> out=fp16"},
        {TYPE_FP8,  SUB_FP8E4M3, TYPE_FP8,  SUB_FP8E4M3, "ab=fp8e4m3 -> out=fp8e4m3"},
        {TYPE_FP8,  SUB_FP8E5M2, TYPE_FP8,  SUB_FP8E5M2, "ab=fp8e5m2 -> out=fp8e5m2"},
        {TYPE_FP16, SUB_FP8E5M2, TYPE_FP32, SUB_FP8E5M2, "ab=fp16 -> out=fp32"},
    };

    constexpr int kRepeat = 6;
    bool all = true;

    for (const auto& tc : cases) {
        for (int run = 0; run < kRepeat; ++run) {
            OTC_Config cfg;
            cfg.M = 8;
            cfg.K = 8;
            cfg.N = 8;
            cfg.type_ab = tc.type_ab;
            cfg.type_ab_sub = tc.type_ab_sub;
            cfg.type_cd = tc.type_cd;
            cfg.type_cd_sub = tc.type_cd_sub;

            auto td = gen_random(8, 8, 8, 1000 + run + (int)(&tc - &cases[0]) * 100);
            auto pa = pack_ab(td.a, cfg.type_ab, cfg.type_ab_sub);
            auto pb = pack_ab(td.b, cfg.type_ab, cfg.type_ab_sub);
            auto pc = pack_c_fp16(td.c);

            OTC_Device* dev = nullptr;
            otc_dev_open(&dev);
            otc_configure(dev, cfg);
            otc_submit(dev, pa.data(), (int)pa.size(), pb.data(), (int)pb.size(), pc.data(), (int)pc.size());
            otc_run(dev);

            std::vector<double> out(64);
            otc_pop_result_f64(dev, out.data(), 64);

            std::vector<double> aq, bq, cq;
            unpack_quantized_inputs(td, cfg, aq, bq, cq);
            auto gold_fp32 = golden_fp32_unquantized(aq, bq, cq, cfg);
            auto gold_quant = golden_fp32_quantized(gold_fp32, cfg);
            auto gold_model = golden_model_quantized(aq, bq, cq, cfg);

            printf("\n=== %s, run=%d ===\n", tc.name, run);
            print_matrix("result", out, 8, 8);
            print_matrix("golden_fp32_unquantized", gold_fp32, 8, 8);
            print_matrix("golden_quantized", gold_quant, 8, 8);

            double maxe_fp32q = 0.0;
            double maxe_model = 0.0;
            for (int i = 0; i < 64; i++) {
                maxe_fp32q = std::max(maxe_fp32q, fabs(out[i] - gold_quant[i]));
                maxe_model = std::max(maxe_model, fabs(out[i] - gold_model[i]));
            }
            printf("max_err_vs_fp32_quantized=%f\n", maxe_fp32q);
            printf("max_err_vs_model_quantized=%f\n", maxe_model);
            all = all && (maxe_model < 1e-6);
            otc_dev_close(dev);
        }
    }

    printf("\nOverall: %s\n", all ? "PASSED" : "FAILED");
    return all ? 0 : 1;
}
