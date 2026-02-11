#include "otc_fp.h"

namespace SoftFloat {

double fp16_to_f64(uint16_t h) {
    int s = (h >> 15) & 1;
    int e = (h >> 10) & 0x1F;
    int m = h & 0x3FF;
    if (e == 0x1F) {
        return m ? NAN : (s ? -INFINITY : INFINITY);
    }
    if (e == 0) {
        if (m == 0) return s ? -0.0 : 0.0;
        return (s ? -1.0 : 1.0) * ldexp((double)m / 1024.0, -14);
    }
    return (s ? -1.0 : 1.0) * ldexp(1.0 + (double)m / 1024.0, e - 15);
}

uint16_t f64_to_fp16(double v) {
    if (std::isnan(v)) return 0x7E00;
    if (std::isinf(v)) return v < 0 ? 0xFC00 : 0x7C00;
    uint16_t s = (v < 0) ? 1 : 0;
    v = fabs(v);
    if (v == 0) return s << 15;
    int e;
    double f = frexp(v, &e);
    f *= 2;
    e--;
    int be = e + 15;
    if (be >= 31) return (s << 15) | (0x1F << 10);
    if (be <= 0) {
        int m = (int)(v / ldexp(1.0, 1 - 15) * 1024 + 0.5) & 0x3FF;
        return (s << 15) | m;
    }
    int m = (int)((f - 1.0) * 1024 + 0.5) & 0x3FF;
    return (s << 15) | (be << 10) | m;
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
    int s = (bits9 >> 8) & 1;
    int e = (bits9 >> 3) & 0x1F;
    int m = bits9 & 7;
    if (e == 0x1F) return m ? NAN : (s ? -INFINITY : INFINITY);
    if (e == 0 && m == 0) return s ? -0.0 : 0.0;
    double sig = (e == 0) ? m / 8.0 : 1.0 + m / 8.0;
    int exp = (e == 0) ? (1 - 15) : (e - 15);
    return (s ? -1 : 1) * ldexp(sig, exp);
}

uint16_t f64_to_fp9(double v) {
    if (std::isnan(v)) return 0x1F9;
    if (std::isinf(v)) return v < 0 ? 0x1F8 : 0x0F8;
    uint16_t s = (v < 0) ? 1 : 0;
    v = fabs(v);
    if (v == 0) return s << 8;
    int e;
    double f = frexp(v, &e);
    f *= 2;
    e--;
    int be = e + 15;
    if (be >= 31) return (s << 8) | (0x1F << 3);
    if (be <= 0) {
        int m = (int)(v / ldexp(1.0, 1 - 15) * 8 + 0.5) & 7;
        return (s << 8) | m;
    }
    int m = (int)((f - 1.0) * 8 + 0.5) & 7;
    return (s << 8) | (be << 3) | m;
}

double fp22_to_f64(uint32_t bits22) {
    int s = (bits22 >> 21) & 1;
    int e = (bits22 >> 13) & 0xFF;
    int m = bits22 & 0x1FFF;
    if (e == 0xFF) return m ? NAN : (s ? -INFINITY : INFINITY);
    if (e == 0 && m == 0) return s ? -0.0 : 0.0;
    double sig = (e == 0) ? m / 8192.0 : 1.0 + m / 8192.0;
    int exp = (e == 0) ? (1 - 127) : (e - 127);
    return (s ? -1 : 1) * ldexp(sig, exp);
}

uint32_t f64_to_fp22(double v) {
    if (std::isnan(v)) return 0x3FFFFF;
    if (std::isinf(v)) return v < 0 ? 0x3FE000 : 0x1FE000;
    uint32_t s = (v < 0) ? 1 : 0;
    v = fabs(v);
    if (v == 0) return s << 21;
    int e;
    double f = frexp(v, &e);
    f *= 2;
    e--;
    int be = e + 127;
    if (be >= 255) return (s << 21) | (0xFF << 13);
    if (be <= 0) {
        int m = (int)(v / ldexp(1.0, 1 - 127) * 8192 + 0.5) & 0x1FFF;
        return (s << 21) | m;
    }
    int m = (int)((f - 1.0) * 8192 + 0.5) & 0x1FFF;
    return (s << 21) | (be << 13) | m;
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

double fp16_to_f64_via_fp9(uint16_t fp16) {
    double v = SoftFloat::fp16_to_f64(fp16);
    uint16_t fp9 = SoftFloat::f64_to_fp9(v);
    return SoftFloat::fp9_to_f64(fp9);
}

double elem_to_f64(uint32_t word, int elem_idx, int type_ab, int sub) {
    switch (type_ab) {
        case TYPE_FP4: {
            uint8_t nibble = (word >> (elem_idx * 4)) & 0xF;
            return fp4_to_f64(nibble);
        }
        case TYPE_FP8: {
            uint8_t byte = (word >> (elem_idx * 8)) & 0xFF;
            return (sub == SUB_FP8E4M3) ? fp8e4m3_to_f64(byte) : fp8e5m2_to_f64(byte);
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
