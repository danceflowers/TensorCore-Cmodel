// ============================================================================
// OpenTensorCore SimX — Software Float Models
// Mirrors: to_fp8_con_core.v  (FP8→FP9 conversion)
//          fp4_to_fp9fp4.v    (FP4→FP9 conversion)
//          fp16_to_fp9.v      (FP16→FP9, stub in RTL)
//          fmul_s1/s2/s3.v    (FP multiply functional model)
//          fadd_s1/s2.v       (FP add functional model)
//
// Design: Use double as the golden-reference computation type.
//         Quantize to each custom format for bit-accurate modeling.
// ============================================================================
#pragma once
#include "otc_types.h"

// ==================== Soft-float helpers (double ↔ custom format) ==========
namespace SoftFloat {

// ---- FP16 (IEEE 754 half) ----
inline double fp16_to_f64(uint16_t h) {
    int s = (h >> 15) & 1;
    int e = (h >> 10) & 0x1F;
    int m = h & 0x3FF;
    if (e==0x1F) return m ? NAN : (s ? -INFINITY : INFINITY);
    if (e==0 && m==0) return s ? -0.0 : 0.0;
    double sig = (e==0) ? m/1024.0 : 1.0+m/1024.0;
    int exp = (e==0) ? (1-15) : (e-15);
    return (s ? -1:1) * ldexp(sig, exp);
}

inline uint16_t f64_to_fp16(double v) {
    if (std::isnan(v))  return 0x7E00;
    if (std::isinf(v))  return v<0 ? 0xFC00 : 0x7C00;
    uint16_t s = (v<0)?1:0; v=fabs(v);
    if (v==0) return s<<15;
    int e; double f = frexp(v, &e); f*=2; e--;
    int be = e + 15;
    if (be >= 31) return (s<<15)|(0x1F<<10);
    if (be <= 0)  { int m=(int)(v/ldexp(1.0,1-15)*1024+0.5)&0x3FF; return (s<<15)|m; }
    int m = (int)((f-1.0)*1024+0.5)&0x3FF;
    return (s<<15)|(be<<10)|m;
}

// ---- FP32 (IEEE 754 single) ----
inline double fp32_to_f64(uint32_t w) {
    union { uint32_t u; float f; } u; u.u = w;
    return (double)u.f;
}
inline uint32_t f64_to_fp32(double v) {
    union { float f; uint32_t u; } u; u.f = (float)v;
    return u.u;
}

// ---- FP9 (E5M3): sign(1)+exp(5)+man(3), bias=15 ----
inline double fp9_to_f64(uint16_t bits9) {
    int s = (bits9>>8)&1, e = (bits9>>3)&0x1F, m = bits9&7;
    if (e==0x1F) return m ? NAN : (s ? -INFINITY : INFINITY);
    if (e==0 && m==0) return s ? -0.0 : 0.0;
    double sig = (e==0) ? m/8.0 : 1.0+m/8.0;
    int exp = (e==0) ? (1-15) : (e-15);
    return (s?-1:1) * ldexp(sig, exp);
}
inline uint16_t f64_to_fp9(double v) {
    if (std::isnan(v))  return 0x1F9;
    if (std::isinf(v))  return v<0 ? 0x1F8 : 0x0F8;
    uint16_t s=(v<0)?1:0; v=fabs(v);
    if (v==0) return s<<8;
    int e; double f=frexp(v,&e); f*=2; e--;
    int be = e+15;
    if (be>=31) return (s<<8)|(0x1F<<3);
    if (be<=0) { int m=(int)(v/ldexp(1.0,1-15)*8+0.5)&7; return (s<<8)|m; }
    int m=(int)((f-1.0)*8+0.5)&7;
    return (s<<8)|(be<<3)|m;
}

// ---- FP22 (E8M13): sign(1)+exp(8)+man(13), bias=127 ----
//      accumulator format used in tc_dot_product final_add
inline double fp22_to_f64(uint32_t bits22) {
    int s=(bits22>>21)&1, e=(bits22>>13)&0xFF, m=bits22&0x1FFF;
    if (e==0xFF) return m?NAN:(s?-INFINITY:INFINITY);
    if (e==0&&m==0) return s?-0.0:0.0;
    double sig=(e==0)?m/8192.0:1.0+m/8192.0;
    int exp=(e==0)?(1-127):(e-127);
    return (s?-1:1)*ldexp(sig,exp);
}
inline uint32_t f64_to_fp22(double v) {
    if (std::isnan(v)) return 0x3FFFFF;
    if (std::isinf(v)) return v<0?0x3FE000:0x1FE000;
    uint32_t s=(v<0)?1:0; v=fabs(v);
    if (v==0) return s<<21;
    int e; double f=frexp(v,&e); f*=2; e--;
    int be=e+127;
    if (be>=255) return (s<<21)|(0xFF<<13);
    if (be<=0) { int m=(int)(v/ldexp(1.0,1-127)*8192+0.5)&0x1FFF; return (s<<21)|m; }
    int m=(int)((f-1.0)*8192+0.5)&0x1FFF;
    return (s<<21)|(be<<13)|m;
}

} // namespace SoftFloat

// ==================== Input Format Conversion (mirrors to_fp8_con_core.v) ==
namespace FPConvert {

// FP4 (S1 E2 M1) → double  (mirrors fp4_to_fp9fp4.v)
inline double fp4_to_f64(uint8_t fp4) {
    int s=(fp4>>3)&1, e=(fp4>>1)&3, m=fp4&1;
    if (e==3 && m) return NAN;
    if (e==3)      return s ? -INFINITY : INFINITY;
    if (e==0) {
        if (m==0) return s ? -0.0 : 0.0;
        // subnormal FP4: 0.m * 2^(1-bias), bias for E2 = 1
        return (s?-1:1) * 0.5;  // 0.1 * 2^0 = 0.5
    }
    // normal: 1.m * 2^(e-bias), bias=1
    double sig = 1.0 + m * 0.5;
    return (s?-1:1) * ldexp(sig, e-1);
}

// FP8 E5M2 → double (mirrors to_fp8_con_core.v case FP8E5M2)
inline double fp8e5m2_to_f64(uint8_t fp8) {
    int s=(fp8>>7)&1, e=(fp8>>2)&0x1F, m=fp8&3;
    if (e==0x1F) return m?NAN:(s?-INFINITY:INFINITY);
    if (e==0&&m==0) return s?-0.0:0.0;
    double sig=(e==0)?m/4.0:1.0+m/4.0;
    int exp=(e==0)?(1-15):(e-15);
    return (s?-1:1)*ldexp(sig,exp);
}

// FP8 E4M3 → double (mirrors to_fp8_con_core.v case FP8E4M3)
inline double fp8e4m3_to_f64(uint8_t fp8) {
    int s=(fp8>>7)&1, e=(fp8>>3)&0xF, m=fp8&7;
    if (e==0xF && m==7) return NAN;
    if (e==0&&m==0) return s?-0.0:0.0;
    double sig=(e==0)?m/8.0:1.0+m/8.0;
    int exp=(e==0)?(1-7):(e-7);
    return (s?-1:1)*ldexp(sig,exp);
}

// FP16 → FP9 (truncation) — RTL file fp16_to_fp9.v is empty stub
inline double fp16_to_f64_via_fp9(uint16_t fp16) {
    // model: truncate FP16 to FP9 precision
    double v = SoftFloat::fp16_to_f64(fp16);
    uint16_t fp9 = SoftFloat::f64_to_fp9(v);
    return SoftFloat::fp9_to_f64(fp9);
}

// Generic element → double conversion
inline double elem_to_f64(uint32_t word, int elem_idx, int type_ab, int sub) {
    switch (type_ab) {
    case TYPE_FP4: {
        uint8_t nibble = (word >> (elem_idx * 4)) & 0xF;
        return fp4_to_f64(nibble);
    }
    case TYPE_FP8: {
        uint8_t byte = (word >> (elem_idx * 8)) & 0xFF;
        return (sub == SUB_FP8E4M3) ? fp8e4m3_to_f64(byte)
                                     : fp8e5m2_to_f64(byte);
    }
    case TYPE_FP16: {
        uint16_t half = (word >> (elem_idx * 16)) & 0xFFFF;
        return fp16_to_f64_via_fp9(half);
    }
    default: return 0.0;
    }
}

// Element bit-width
inline int elem_bits(int type_ab) {
    switch (type_ab) {
    case TYPE_FP4:  return 4;
    case TYPE_FP8:  return 8;
    case TYPE_FP16: return 16;
    default: return 8;
    }
}

} // namespace FPConvert
