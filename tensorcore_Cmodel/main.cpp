#include "otc_decode.h"
#include "otc_driver.h"

struct TestData { std::vector<double> a, b, c; };

static TestData gen_random(int M, int K, int N, unsigned seed) {
    TestData t; t.a.resize(M*K); t.b.resize(K*N); t.c.resize(M*N); srand(seed);
    for (auto& v: t.a) v = (rand()%200-100)/100.0;
    for (auto& v: t.b) v = (rand()%200-100)/100.0;
    for (auto& v: t.c) v = (rand()%100-50)/100.0;
    return t;
}

static std::vector<uint32_t> pack_ab(const std::vector<double>& vals, int type_ab, int sub) {
    int eb = FPConvert::elem_bits(type_ab), eperw = 32 / eb;
    std::vector<uint32_t> words((vals.size()+eperw-1)/eperw, 0);
    for (int i = 0; i < (int)vals.size(); i++) {
        int wi=i/eperw, ei=i%eperw; uint32_t packed=0;
        if (type_ab == TYPE_FP8) packed = (sub == SUB_FP8E4M3) ? FPConvert::f64_to_fp8e4m3(vals[i]) : FPConvert::f64_to_fp8e5m2(vals[i]);
        else if (type_ab == TYPE_FP4) packed = (uint32_t)(SoftFloat::f64_to_fp9(vals[i]) >> 5) & 0xF;
        else packed = SoftFloat::f64_to_fp16(vals[i]);
        words[wi] |= (packed << (ei * eb));
    }
    return words;
}

static std::vector<uint32_t> pack_c_fp16(const std::vector<double>& vals) {
    std::vector<uint32_t> words((vals.size()+1)/2,0);
    for (int i = 0; i < (int)vals.size(); i++) words[i/2] |= ((uint32_t)SoftFloat::f64_to_fp16(vals[i]) << ((i%2)*16));
    return words;
}

static std::vector<double> golden(const TestData& td, const OTC_Config& cfg) {
    auto pa = pack_ab(td.a, cfg.type_ab, cfg.type_ab_sub), pb = pack_ab(td.b, cfg.type_ab, cfg.type_ab_sub), pc = pack_c_fp16(td.c);
    int eb = FPConvert::elem_bits(cfg.type_ab), eperw = 32 / eb;
    std::vector<uint16_t> aq(cfg.M*cfg.K), bq(cfg.K*cfg.N); std::vector<uint32_t> cq(cfg.M*cfg.N);
    for (int i=0;i<cfg.M*cfg.K;i++){int wi=i/eperw,ei=i%eperw;uint32_t w=wi<(int)pa.size()?pa[wi]:0;aq[i]=(cfg.type_ab==TYPE_FP4)?FPEmu::fp4_to_fp9((w>>(ei*4))&0xF):((cfg.type_ab==TYPE_FP8)?((cfg.type_ab_sub==SUB_FP8E4M3)?FPEmu::fp8e4m3_to_fp9((w>>(ei*8))&0xFF):FPEmu::fp8e5m2_to_fp9((w>>(ei*8))&0xFF)):FPEmu::fp16_to_fp9((w>>(ei*16))&0xFFFF));}
    for (int i=0;i<cfg.K*cfg.N;i++){int wi=i/eperw,ei=i%eperw;uint32_t w=wi<(int)pb.size()?pb[wi]:0;bq[i]=(cfg.type_ab==TYPE_FP4)?FPEmu::fp4_to_fp9((w>>(ei*4))&0xF):((cfg.type_ab==TYPE_FP8)?((cfg.type_ab_sub==SUB_FP8E4M3)?FPEmu::fp8e4m3_to_fp9((w>>(ei*8))&0xFF):FPEmu::fp8e5m2_to_fp9((w>>(ei*8))&0xFF)):FPEmu::fp16_to_fp9((w>>(ei*16))&0xFFFF));}
    for (int i=0;i<cfg.M*cfg.N;i++){int wi=i/2,ei=i%2;uint16_t h=((wi<(int)pc.size()?pc[wi]:0)>>(ei*16))&0xFFFF;cq[i]=SoftFloat::f64_to_fp22(SoftFloat::fp16_to_f64(h));}
    std::vector<double> d(cfg.M*cfg.N,0.0);
    for (int i=0;i<cfg.M;i++) for (int j=0;j<cfg.N;j++) {
        std::vector<uint16_t> ts(cfg.K);
        for (int k=0;k<cfg.K;k++) ts[k]=SoftFloat::f64_to_fp13(SoftFloat::fp9_to_f64(FPEmu::fp9_mul(aq[i*cfg.K+k],bq[k*cfg.N+j])));
        int w=cfg.K; while(w>1){for(int x=0;x<w/2;x++) ts[x]=FPEmu::fp13_add(ts[2*x],ts[2*x+1]); w>>=1;}
        uint32_t out22 = FPEmu::fp22_add(FPEmu::fp9_to_fp22(FPEmu::fp13_to_fp9(ts[0])), cq[i*cfg.N+j]);
        if (cfg.type_cd == TYPE_FP16) d[i*cfg.N+j]=SoftFloat::fp16_to_f64(FPEmu::fp22_to_fp16(out22));
        else if (cfg.type_cd == TYPE_FP8) { uint8_t x=(uint8_t)FPEmu::fp22_to_fp8(out22,cfg.type_cd_sub); d[i*cfg.N+j]=(cfg.type_cd_sub==SUB_FP8E4M3)?FPConvert::fp8e4m3_to_f64(x):FPConvert::fp8e5m2_to_f64(x);} 
        else d[i*cfg.N+j]=SoftFloat::fp32_to_f64(SoftFloat::f64_to_fp32(SoftFloat::fp22_to_f64(out22)));
    }
    return d;
}

static void print_matrix(const char* tag, const std::vector<double>& m, int R, int C) {
    printf("%s\n", tag); for(int i=0;i<R;i++){for(int j=0;j<C;j++) printf("%9.5f ", m[i*C+j]); printf("\n");}
}

int main() {
    std::vector<std::pair<int,int>> ab = {{TYPE_FP4,SUB_FP8E5M2},{TYPE_FP8,SUB_FP8E4M3},{TYPE_FP8,SUB_FP8E5M2},{TYPE_FP16,SUB_FP8E5M2}};
    bool all = true;
    for (int run=0; run<4; ++run) {
        OTC_Config cfg; cfg.M=8; cfg.K=8; cfg.N=8; cfg.type_ab=ab[run].first; cfg.type_ab_sub=ab[run].second; cfg.type_cd=TYPE_FP16;
        auto td = gen_random(8,8,8,42+run);
        auto pa=pack_ab(td.a,cfg.type_ab,cfg.type_ab_sub), pb=pack_ab(td.b,cfg.type_ab,cfg.type_ab_sub), pc=pack_c_fp16(td.c);
        OTC_Device* dev; otc_dev_open(&dev); otc_configure(dev,cfg);
        otc_submit(dev,pa.data(),pa.size(),pb.data(),pb.size(),pc.data(),pc.size()); otc_run(dev);
        std::vector<double> out(64); otc_pop_result_f64(dev,out.data(),64);
        auto gold = golden(td,cfg);
        print_matrix("result", out, 8, 8); print_matrix("golden", gold, 8, 8);
        double maxe=0; for(int i=0;i<64;i++) maxe=std::max(maxe, fabs(out[i]-gold[i]));
        printf("run %d max_err=%f\n", run, maxe); all = all && maxe < 1e-6; otc_dev_close(dev);
    }
    printf("Overall: %s\n", all?"PASSED":"FAILED");
    return all?0:1;
}
