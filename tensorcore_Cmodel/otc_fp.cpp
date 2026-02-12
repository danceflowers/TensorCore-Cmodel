#include "otc_fp.h"

namespace {

struct RawFP {
    bool sign;
    int exp;
    uint32_t sig;
    bool is_zero;
    bool is_inf;
    bool is_nan;
};

inline RawFP decode(uint32_t bits, int exp_bits, int mant_bits) {
    RawFP r{};
    uint32_t s = (bits >> (exp_bits + mant_bits)) & 1u;
    uint32_t e = (bits >> mant_bits) & ((1u << exp_bits) - 1);
    uint32_t m = bits & ((1u << mant_bits) - 1);
    r.sign = s;
    r.is_zero = (e == 0 && m == 0);
    r.is_inf = (e == ((1u << exp_bits) - 1) && m == 0);
    r.is_nan = (e == ((1u << exp_bits) - 1) && m != 0);
    if (e == 0) {
        r.exp = 1;
        r.sig = m;
    } else {
        r.exp = (int)e;
        r.sig = (1u << mant_bits) | m;
    }
    return r;
}

inline uint32_t encode(bool sign, int exp, uint32_t mant, int exp_bits, int mant_bits) {
    uint32_t emax = (1u << exp_bits) - 1;
    if (exp <= 0) return ((uint32_t)sign << (exp_bits + mant_bits));
    if ((uint32_t)exp >= emax) return ((uint32_t)sign << (exp_bits + mant_bits)) | (emax << mant_bits);
    return ((uint32_t)sign << (exp_bits + mant_bits)) | ((uint32_t)exp << mant_bits) | (mant & ((1u << mant_bits) - 1));
}

uint32_t add_core(uint32_t a_bits, uint32_t b_bits, int exp_bits, int mant_bits) {
    RawFP a = decode(a_bits, exp_bits, mant_bits);
    RawFP b = decode(b_bits, exp_bits, mant_bits);
    uint32_t emax = (1u << exp_bits) - 1;
    if (a.is_nan || b.is_nan) return (emax << mant_bits) | (1u << (mant_bits - 1));
    if (a.is_inf && b.is_inf && a.sign != b.sign) return (emax << mant_bits) | (1u << (mant_bits - 1));
    if (a.is_inf) return ((uint32_t)a.sign << (exp_bits + mant_bits)) | (emax << mant_bits);
    if (b.is_inf) return ((uint32_t)b.sign << (exp_bits + mant_bits)) | (emax << mant_bits);
    if (a.is_zero) return b_bits;
    if (b.is_zero) return a_bits;

    if (a.exp < b.exp || (a.exp == b.exp && a.sig < b.sig)) std::swap(a, b);
    int de = a.exp - b.exp;
    uint64_t as = ((uint64_t)a.sig) << 3;
    uint64_t bs = ((uint64_t)b.sig) << 3;
    if (de > mant_bits + 4) bs = 1;
    else {
        uint64_t lost = bs & ((1ull << de) - 1ull);
        bs >>= de;
        if (lost) bs |= 1;
    }
    uint64_t rs;
    bool sign;
    int exp = a.exp;
    if (a.sign == b.sign) { rs = as + bs; sign = a.sign; }
    else {
        if (as >= bs) { rs = as - bs; sign = a.sign; }
        else { rs = bs - as; sign = b.sign; }
    }
    if (rs == 0) return 0;
    while (rs >= ((uint64_t)1 << (mant_bits + 5))) { rs >>= 1; exp++; }
    while (rs < ((uint64_t)1 << (mant_bits + 3)) && exp > 1) { rs <<= 1; exp--; }

    uint32_t mant = (rs >> 3) & ((1u << mant_bits) - 1);
    uint32_t g = (rs >> 2) & 1u, r = (rs >> 1) & 1u, s = rs & 1u;
    if (g && (r || s || (mant & 1u))) {
        mant++;
        if (mant == (1u << mant_bits)) { mant = 0; exp++; }
    }
    return encode(sign, exp, mant, exp_bits, mant_bits);
}

uint32_t mul_core(uint32_t a_bits, uint32_t b_bits, int exp_bits, int mant_bits) {
    RawFP a = decode(a_bits, exp_bits, mant_bits);
    RawFP b = decode(b_bits, exp_bits, mant_bits);
    uint32_t emax = (1u << exp_bits) - 1;
    bool sign = a.sign ^ b.sign;
    if (a.is_nan || b.is_nan) return (emax << mant_bits) | (1u << (mant_bits - 1));
    if ((a.is_inf && b.is_zero) || (b.is_inf && a.is_zero)) return (emax << mant_bits) | (1u << (mant_bits - 1));
    if (a.is_inf || b.is_inf) return ((uint32_t)sign << (exp_bits + mant_bits)) | (emax << mant_bits);
    if (a.is_zero || b.is_zero) return ((uint32_t)sign << (exp_bits + mant_bits));

    uint64_t prod = (uint64_t)a.sig * (uint64_t)b.sig;
    int exp = a.exp + b.exp - ((1 << (exp_bits - 1)) - 1);
    int top = 2 * mant_bits + 1;
    if (prod & (1ull << top)) {
        prod >>= 1;
        exp++;
    }
    uint64_t shifted = prod >> (mant_bits - 3);
    uint32_t mant = (shifted >> 3) & ((1u << mant_bits) - 1);
    uint32_t g = (shifted >> 2) & 1u, r = (shifted >> 1) & 1u, s = (shifted & 1u) | ((prod & ((1ull << (mant_bits - 3)) - 1ull)) ? 1u : 0u);
    if (g && (r || s || (mant & 1u))) {
        mant++;
        if (mant == (1u << mant_bits)) { mant = 0; exp++; }
    }
    return encode(sign, exp, mant, exp_bits, mant_bits);
}

double fp_to_f64_generic(uint32_t bits, int sign_bit, int exp_bits, int mant_bits, int exp_bias) {
    int s = (bits >> sign_bit) & 1;
    int e = (bits >> mant_bits) & ((1 << exp_bits) - 1);
    int m = bits & ((1 << mant_bits) - 1);
    int e_max = (1 << exp_bits) - 1;

    if (e == e_max) return m ? NAN : (s ? -INFINITY : INFINITY);
    if (e == 0 && m == 0) return s ? -0.0 : 0.0;

    double sig = (e == 0) ? (double)m / (1 << mant_bits) : 1.0 + (double)m / (1 << mant_bits);
    int exp = (e == 0) ? (1 - exp_bias) : (e - exp_bias);
    return (s ? -1.0 : 1.0) * ldexp(sig, exp);
}

uint32_t f64_to_fp_generic(double v, int sign_bit, int exp_bits, int mant_bits, int exp_bias) {
    uint32_t e_max = (1u << exp_bits) - 1;
    uint32_t mant_mask = (1u << mant_bits) - 1;

    if (std::isnan(v)) return (e_max << mant_bits) | (1u << (mant_bits - 1));
    if (std::isinf(v)) return (v < 0 ? (1u << sign_bit) : 0u) | (e_max << mant_bits);

    uint32_t s = (v < 0) ? 1u : 0u;
    v = fabs(v);
    if (v == 0.0) return s << sign_bit;

    int e;
    double f = frexp(v, &e);
    f *= 2.0;
    e--;

    int be = e + exp_bias;
    if (be >= (int)e_max) return (s << sign_bit) | (e_max << mant_bits);
    if (be <= 0) return (s << sign_bit);

    int m = (int)((f - 1.0) * (1 << mant_bits) + 0.5);
    if (m == (1 << mant_bits)) {
        m = 0;
        be++;
        if (be >= (int)e_max) return (s << sign_bit) | (e_max << mant_bits);
    }
    return (s << sign_bit) | ((uint32_t)be << mant_bits) | ((uint32_t)m & mant_mask);
}

}  // namespace

namespace SoftFloat {
// unchanged conversion helpers
double fp16_to_f64(uint16_t h) { return fp_to_f64_generic(h, 15, 5, 10, 15); }
uint16_t f64_to_fp16(double v) { return (uint16_t)f64_to_fp_generic(v, 15, 5, 10, 15); }
double fp32_to_f64(uint32_t w) { union { uint32_t u; float f; } u{w}; return (double)u.f; }
uint32_t f64_to_fp32(double v) { union { float f; uint32_t u; } u{(float)v}; return u.u; }
double fp9_to_f64(uint16_t bits9) { return fp_to_f64_generic(bits9 & 0x1FF, 8, 5, 3, 15); }
uint16_t f64_to_fp9(double v) { return (uint16_t)(f64_to_fp_generic(v, 8, 5, 3, 15) & 0x1FF); }
double fp13_to_f64(uint16_t bits13) { return fp_to_f64_generic(bits13 & 0x1FFF, 12, 5, 7, 15); }
uint16_t f64_to_fp13(double v) { return (uint16_t)(f64_to_fp_generic(v, 12, 5, 7, 15) & 0x1FFF); }
double fp22_to_f64(uint32_t bits22) { return fp_to_f64_generic(bits22 & 0x3FFFFF, 21, 8, 13, 127); }
uint32_t f64_to_fp22(double v) { return f64_to_fp_generic(v, 21, 8, 13, 127) & 0x3FFFFF; }
} // namespace SoftFloat

namespace FPEmu {
uint16_t fp4_to_fp9(uint8_t fp4) {
    uint8_t s = (fp4 >> 3) & 1, e = (fp4 >> 1) & 0x3, m = fp4 & 1;
    if (e == 0x3) return (s << 8) | (0x1F << 3) | (m ? 1 : 0);
    if (e == 0x0) return (s << 8) | (m << 2);
    if (e == 0x1) return (s << 8) | (0x0F << 3) | (m << 2);
    return (s << 8) | (0x1F << 3);
}
uint16_t fp8e4m3_to_fp9(uint8_t fp8) { return (uint16_t)(((fp8 >> 7) << 8) | (((fp8 >> 3) & 0xF) << 3) | (fp8 & 0x7)); }
uint16_t fp8e5m2_to_fp9(uint8_t fp8) { return (uint16_t)(((fp8 >> 7) << 8) | (((fp8 >> 2) & 0x1F) << 3) | ((fp8 & 0x3) << 1)); }
uint16_t fp16_to_fp9(uint16_t fp16) {
    uint16_t s = (fp16 >> 15) & 1, e = (fp16 >> 10) & 0x1F, m = fp16 & 0x3FF;
    uint16_t m3 = (m >> 7) + ((m & 0x40) && ((m & 0x3F) || ((m >> 7) & 1)));
    uint16_t eout = e;
    if (m3 >= 8) { m3 = 0; eout++; }
    if (eout >= 0x1F) return (s << 8) | (0x1F << 3);
    return (s << 8) | (eout << 3) | (m3 & 0x7);
}
uint16_t fp9_mul(uint16_t a, uint16_t b) { return (uint16_t)(mul_core(a & 0x1FF, b & 0x1FF, 5, 3) & 0x1FF); }
uint16_t fp13_add(uint16_t a, uint16_t b) { return (uint16_t)(add_core(a & 0x1FFF, b & 0x1FFF, 5, 7) & 0x1FFF); }
uint32_t fp22_add(uint32_t a, uint32_t b) { return add_core(a & 0x3FFFFF, b & 0x3FFFFF, 8, 13) & 0x3FFFFF; }
uint32_t fp9_to_fp22(uint16_t a) {
    uint16_t s = (a >> 8) & 1, e = (a >> 3) & 0x1F, m = a & 0x7;
    if (e == 0x1F) return ((uint32_t)s << 21) | (0xFFu << 13) | (m ? 1u : 0u);
    if (e == 0) return ((uint32_t)s << 21) | (m << 10);
    uint32_t e22 = e - 15 + 127;
    return ((uint32_t)s << 21) | (e22 << 13) | ((uint32_t)m << 10);
}
uint16_t fp13_to_fp9(uint16_t a) {
    uint16_t s=(a>>12)&1,e=(a>>7)&0x1F,m=a&0x7F;
    uint16_t m3=(m>>4)+((m&0x8)&&((m&0x7)||((m>>4)&1)));
    if (m3>=8){m3=0;e++;}
    if (e>=0x1F) return (s<<8)|(0x1F<<3);
    return (s<<8)|(e<<3)|(m3&0x7);
}
uint16_t fp22_to_fp8(uint32_t a, int sub) {
    uint32_t s=(a>>21)&1,e=(a>>13)&0xFF,m=a&0x1FFF;
    if (sub == SUB_FP8E5M2) {
        if (e==0xFF) return (s<<7)|(0x1F<<2)|((m?1:0)<<1);
        int ee=(int)e-127+15; if (ee<=0) return (s<<7); if (ee>=0x1F) return (s<<7)|(0x1F<<2);
        uint32_t m2=(m>>11)+((m&0x400)&&((m&0x3FF)||((m>>11)&1))); if (m2>=4){m2=0;ee++;}
        if (ee>=0x1F) return (s<<7)|(0x1F<<2);
        return (s<<7)|((ee&0x1F)<<2)|(m2&0x3);
    }
    if (e==0xFF) return (s<<7)|(0xF<<3)|(m?7:0);
    int ee=(int)e-127+7; if (ee<=0) return (s<<7); if (ee>=0xF) return (s<<7)|(0xE<<3)|0x7;
    uint32_t m3=(m>>10)+((m&0x200)&&((m&0x1FF)||((m>>10)&1))); if (m3>=8){m3=0;ee++;}
    if (ee>=0xF) return (s<<7)|(0xE<<3)|0x7;
    return (s<<7)|((ee&0xF)<<3)|(m3&0x7);
}
uint16_t fp22_to_fp16(uint32_t a) {
    uint32_t s=(a>>21)&1,e=(a>>13)&0xFF,m=a&0x1FFF;
    if (e==0xFF) return (s<<15)|(0x1F<<10)|(m?1:0);
    int ee=(int)e-127+15; if (ee<=0) return (s<<15); if (ee>=0x1F) return (s<<15)|(0x1F<<10);
    uint32_t m10=(m>>3)+((m&0x4)&&((m&0x3)||((m>>3)&1))); if (m10>=1024){m10=0;ee++;}
    if (ee>=0x1F) return (s<<15)|(0x1F<<10);
    return (s<<15)|((ee&0x1F)<<10)|(m10&0x3FF);
}
} // namespace FPEmu

namespace FPConvert {
double fp4_to_f64(uint8_t fp4) { return SoftFloat::fp9_to_f64(FPEmu::fp4_to_fp9(fp4)); }
double fp8e5m2_to_f64(uint8_t fp8) { return SoftFloat::fp9_to_f64(FPEmu::fp8e5m2_to_fp9(fp8)); }
double fp8e4m3_to_f64(uint8_t fp8) { return SoftFloat::fp9_to_f64(FPEmu::fp8e4m3_to_fp9(fp8)); }
uint8_t f64_to_fp8e5m2(double v) { return (uint8_t)FPEmu::fp22_to_fp8(SoftFloat::f64_to_fp22(v), SUB_FP8E5M2); }
uint8_t f64_to_fp8e4m3(double v) { return (uint8_t)FPEmu::fp22_to_fp8(SoftFloat::f64_to_fp22(v), SUB_FP8E4M3); }
double fp16_to_f64_via_fp9(uint16_t fp16) { return SoftFloat::fp9_to_f64(FPEmu::fp16_to_fp9(fp16)); }
double elem_to_f64(uint32_t word, int elem_idx, int type_ab, int sub) {
    if (type_ab == TYPE_FP4) return SoftFloat::fp9_to_f64(FPEmu::fp4_to_fp9((word >> (elem_idx * 4)) & 0xF));
    if (type_ab == TYPE_FP8) {
        uint8_t byte = (word >> (elem_idx * 8)) & 0xFF;
        return SoftFloat::fp9_to_f64(sub == SUB_FP8E4M3 ? FPEmu::fp8e4m3_to_fp9(byte) : FPEmu::fp8e5m2_to_fp9(byte));
    }
    return SoftFloat::fp9_to_f64(FPEmu::fp16_to_fp9((word >> (elem_idx * 16)) & 0xFFFF));
}
int elem_bits(int type_ab) { return type_ab == TYPE_FP4 ? 4 : (type_ab == TYPE_FP16 ? 16 : 8); }
} // namespace FPConvert
