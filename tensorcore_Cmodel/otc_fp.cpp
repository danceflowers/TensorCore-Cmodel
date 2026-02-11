#include "otc_fp.h"

namespace {

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
    double f = frexp(v, &e);  // v = f * 2^e, f in [0.5,1)
    f *= 2.0;
    e--;

    int be = e + exp_bias;
    if (be >= (int)e_max) return (s << sign_bit) | (e_max << mant_bits);

    if (be <= 0) {
        int m = (int)(v / ldexp(1.0, 1 - exp_bias) * (1 << mant_bits) + 0.5);
        if (m > (int)mant_mask) m = mant_mask;
        return (s << sign_bit) | (m & mant_mask);
    }

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

double fp16_to_f64(uint16_t h) {
    return fp_to_f64_generic(h, 15, 5, 10, 15);
}

uint16_t f64_to_fp16(double v) {
    return (uint16_t)f64_to_fp_generic(v, 15, 5, 10, 15);
}

double fp32_to_f64(uint32_t w) {
    union {
        uint32_t u;
        float f;
    } u;
    u.u = w;
    return (double)u.f;
}

uint32_t f64_to_fp32(double v) {
    union {
        float f;
        uint32_t u;
    } u;
    u.f = (float)v;
    return u.u;
}

double fp9_to_f64(uint16_t bits9) {
    return fp_to_f64_generic(bits9 & 0x1FF, 8, 5, 3, 15);
}

uint16_t f64_to_fp9(double v) {
    return (uint16_t)(f64_to_fp_generic(v, 8, 5, 3, 15) & 0x1FF);
}

double fp13_to_f64(uint16_t bits13) {
    return fp_to_f64_generic(bits13 & 0x1FFF, 12, 5, 7, 15);
}

uint16_t f64_to_fp13(double v) {
    return (uint16_t)(f64_to_fp_generic(v, 12, 5, 7, 15) & 0x1FFF);
}

double fp22_to_f64(uint32_t bits22) {
    return fp_to_f64_generic(bits22 & 0x3FFFFF, 21, 8, 13, 127);
}

uint32_t f64_to_fp22(double v) {
    return f64_to_fp_generic(v, 21, 8, 13, 127) & 0x3FFFFF;
}

} // namespace SoftFloat

namespace FPConvert {

double fp4_to_f64(uint8_t fp4) {
    int s = (fp4 >> 3) & 1;
    int e = (fp4 >> 1) & 3;
    int m = fp4 & 1;
    if (e == 3 && m) return NAN;
    if (e == 3) return s ? -INFINITY : INFINITY;
    if (e == 0) {
        if (m == 0) return s ? -0.0 : 0.0;
        return (s ? -1 : 1) * 0.5;
    }
    double sig = 1.0 + m * 0.5;
    return (s ? -1 : 1) * ldexp(sig, e - 1);
}

double fp8e5m2_to_f64(uint8_t fp8) {
    int s = (fp8 >> 7) & 1;
    int e = (fp8 >> 2) & 0x1F;
    int m = fp8 & 3;
    if (e == 0x1F) return m ? NAN : (s ? -INFINITY : INFINITY);
    if (e == 0 && m == 0) return s ? -0.0 : 0.0;
    double sig = (e == 0) ? m / 4.0 : 1.0 + m / 4.0;
    int exp = (e == 0) ? (1 - 15) : (e - 15);
    return (s ? -1 : 1) * ldexp(sig, exp);
}

double fp8e4m3_to_f64(uint8_t fp8) {
    int s = (fp8 >> 7) & 1;
    int e = (fp8 >> 3) & 0xF;
    int m = fp8 & 7;
    if (e == 0xF && m == 7) return NAN;
    if (e == 0 && m == 0) return s ? -0.0 : 0.0;
    double sig = (e == 0) ? m / 8.0 : 1.0 + m / 8.0;
    int exp = (e == 0) ? (1 - 7) : (e - 7);
    return (s ? -1 : 1) * ldexp(sig, exp);
}


uint8_t f64_to_fp8e5m2(double v) {
    uint16_t fp9 = SoftFloat::f64_to_fp9(v);
    int s9 = (fp9 >> 8) & 1;
    int e9 = (fp9 >> 3) & 0x1F;
    int m9 = fp9 & 0x7;
    return (uint8_t)((s9 << 7) | (e9 << 2) | (m9 >> 1));
}

uint8_t f64_to_fp8e4m3(double v) {
    int s = (v < 0) ? 1 : 0;
    double av = fabs(v);
    if (std::isnan(v)) return 0x7F;
    if (av == 0.0) return (uint8_t)(s << 7);

    int exp;
    double frac = frexp(av, &exp);
    frac *= 2.0;
    exp--;
    int be = exp + 7;
    int m = 0;

    if (be >= 15) {
        return (uint8_t)((s << 7) | (0x0E << 3) | 0x07);
    }
    if (be <= 0) {
        m = (int)(av / ldexp(1.0, -9) + 0.5) & 0x07;
        return (uint8_t)((s << 7) | m);
    }

    m = (int)((frac - 1.0) * 8.0 + 0.5) & 0x07;
    return (uint8_t)((s << 7) | (be << 3) | m);
}

double fp16_to_f64_via_fp9(uint16_t fp16) {
    double v = SoftFloat::fp16_to_f64(fp16);
    uint16_t fp9 = SoftFloat::f64_to_fp9(v);
    return SoftFloat::fp9_to_f64(fp9);
}

double elem_to_f64(uint32_t word, int elem_idx, int type_ab, int sub) {
    switch (type_ab) {
        case TYPE_FP4: {
            uint8_t nibble = (word >> (elem_idx * 4)) & 0xF;
            return SoftFloat::fp9_to_f64(SoftFloat::f64_to_fp9(fp4_to_f64(nibble)));
        }
        case TYPE_FP8: {
            uint8_t byte = (word >> (elem_idx * 8)) & 0xFF;
            double v = (sub == SUB_FP8E4M3) ? fp8e4m3_to_f64(byte) : fp8e5m2_to_f64(byte);
            return SoftFloat::fp9_to_f64(SoftFloat::f64_to_fp9(v));
        }
        case TYPE_FP16: {
            uint16_t half = (word >> (elem_idx * 16)) & 0xFFFF;
            return fp16_to_f64_via_fp9(half);
        }
        default:
            return 0.0;
    }
}

int elem_bits(int type_ab) {
    switch (type_ab) {
        case TYPE_FP4: return 4;
        case TYPE_FP8: return 8;
        case TYPE_FP16: return 16;
        default: return 8;
    }
}

} // namespace FPConvert
