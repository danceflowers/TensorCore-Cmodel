#pragma once
// =============================================================================
// fp_types.h — Bit-accurate floating-point types for OpenTensorCore simulator
// Matches RTL: FP4(E2M1), FP8(E4M3/E5M2), FP9(E5M3), FP16, FP22(E8M13), FP32
// =============================================================================
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>

// Rounding modes (matching define.v)
enum RoundingMode : uint8_t { RNE=0, RTZ=1, RDN=2, RUP=3, RMM=4 };

// Precision type identifiers
enum PrecisionType { PREC_FP4_E2M1, PREC_FP8_E4M3, PREC_FP8_E5M2, PREC_FP16, PREC_FP32 };

// ─────────────────────────────────────────────────────────────
//  Leading-zero counter (matches RTL lzc module)
// ─────────────────────────────────────────────────────────────
inline int clz(uint32_t val, int width) {
    if (val == 0) return width;
    int c = 0;
    for (int i = width - 1; i >= 0; i--) {
        if (val & (1u << i)) break;
        c++;
    }
    return c;
}

// ─────────────────────────────────────────────────────────────
//  RTL-accurate rounding module (matches rounding.v)
//  in[WIDTH-1:0]  — value to round
//  sign, roundin, stickyin — rounding context
//  Returns: out[WIDTH-1:0], inexact, cout (carry out), r_up
// ─────────────────────────────────────────────────────────────
struct RoundResult {
    uint32_t out;
    bool inexact;
    bool cout;
    bool r_up;
};

inline RoundResult do_rounding(uint32_t in, int WIDTH, bool sign,
                               bool roundin, bool stickyin, RoundingMode rm)
{
    RoundResult r;
    uint32_t mask = (1u << WIDTH) - 1;
    in &= mask;

    r.inexact = roundin || stickyin;

    switch (rm) {
        case RNE: r.r_up = roundin && (stickyin || (in & 1)); break;
        case RTZ: r.r_up = false; break;
        case RDN: r.r_up = sign && r.inexact; break;
        case RUP: r.r_up = !sign && r.inexact; break;
        case RMM: r.r_up = roundin; break;
        default:  r.r_up = false;
    }

    uint32_t sum = in + (r.r_up ? 1 : 0);
    r.cout = (sum >> WIDTH) & 1;
    r.out  = sum & mask;
    return r;
}

// ─────────────────────────────────────────────────────────────
//  Double ↔ various format conversions (for test harness)
// ─────────────────────────────────────────────────────────────
inline double fp9_to_double(uint16_t fp9) {
    bool s = (fp9 >> 8) & 1;
    int e   = (fp9 >> 3) & 0x1F;
    int m   = fp9 & 0x7;
    if (e == 31) return m ? NAN : (s ? -INFINITY : INFINITY);
    if (e == 0 && m == 0) return s ? -0.0 : 0.0;
    double v = (e == 0) ? (m / 8.0) * pow(2, -14) : (1.0 + m / 8.0) * pow(2, e - 15);
    return s ? -v : v;
}

inline double fp22_to_double(uint32_t fp22) {
    bool s    = (fp22 >> 21) & 1;
    int e     = (fp22 >> 13) & 0xFF;
    int m     = fp22 & 0x1FFF;
    if (e == 255) return m ? NAN : (s ? -INFINITY : INFINITY);
    if (e == 0 && m == 0) return s ? -0.0 : 0.0;
    double v = (e == 0) ? (m / 8192.0) * pow(2, -126) : (1.0 + m / 8192.0) * pow(2, e - 127);
    return s ? -v : v;
}

inline double fp16_to_double(uint16_t fp16) {
    bool s = (fp16 >> 15) & 1;
    int e  = (fp16 >> 10) & 0x1F;
    int m  = fp16 & 0x3FF;
    if (e == 31) return m ? NAN : (s ? -INFINITY : INFINITY);
    if (e == 0 && m == 0) return s ? -0.0 : 0.0;
    double v = (e == 0) ? (m / 1024.0) * pow(2, -14) : (1.0 + m / 1024.0) * pow(2, e - 15);
    return s ? -v : v;
}

inline double fp8_e4m3_to_double(uint8_t v) {
    bool s = (v >> 7) & 1; int e = (v >> 3) & 0xF; int m = v & 0x7;
    if (e == 15) return NAN;
    if (e == 0 && m == 0) return s ? -0.0 : 0.0;
    double r = (e == 0) ? (m / 8.0) * pow(2, -6) : (1.0 + m / 8.0) * pow(2, e - 7);
    return s ? -r : r;
}

inline double fp8_e5m2_to_double(uint8_t v) {
    bool s = (v >> 7) & 1; int e = (v >> 2) & 0x1F; int m = v & 0x3;
    if (e == 31) return m ? NAN : (s ? -INFINITY : INFINITY);
    if (e == 0 && m == 0) return s ? -0.0 : 0.0;
    double r = (e == 0) ? (m / 4.0) * pow(2, -14) : (1.0 + m / 4.0) * pow(2, e - 15);
    return s ? -r : r;
}

inline double fp4_to_double(uint8_t v) {
    bool s = (v >> 3) & 1; int e = (v >> 1) & 0x3; int m = v & 0x1;
    if (e == 3 && m == 1) return NAN;
    if (e == 3 && m == 0) return s ? -INFINITY : INFINITY;
    if (e == 0 && m == 0) return s ? -0.0 : 0.0;
    double r = (e == 0) ? (m / 2.0) * pow(2, 0) : (1.0 + m / 2.0) * pow(2, e - 1);
    return s ? -r : r;
}

// Double → format (approximate, for test data generation)
inline uint16_t double_to_fp16(double val) {
    if (std::isnan(val)) return 0x7E00;
    if (std::isinf(val)) return val > 0 ? 0x7C00 : 0xFC00;
    if (val == 0.0) return std::signbit(val) ? 0x8000 : 0;
    bool s = val < 0; val = fabs(val);
    int e; double m = frexp(val, &e); e--; m *= 2;
    int b = e + 15;
    if (b >= 31) return (s << 15) | 0x7C00;
    if (b <= 0) { int sh = 1 - b; if (sh > 11) return s << 15; return (s << 15) | (((int)(m * 1024) >> sh) & 0x3FF); }
    int mt = (int)((m - 1.0) * 1024 + 0.5); if (mt >= 1024) { mt = 0; b++; }
    if (b >= 31) return (s << 15) | 0x7C00;
    return (s << 15) | (b << 10) | (mt & 0x3FF);
}

inline uint8_t double_to_fp8_e4m3(double val) {
    if (std::isnan(val) || std::isinf(val)) return (val < 0 ? 0x80 : 0) | 0x76;
    if (val == 0.0) return std::signbit(val) ? 0x80 : 0;
    bool s = val < 0; val = fabs(val);
    int e; double m = frexp(val, &e); e--; m *= 2;
    int b = e + 7;
    if (b >= 15) return (s << 7) | 0x76;
    if (b <= 0) { int sh = 1 - b; if (sh > 4) return s << 7; return (s << 7) | (((int)(m * 8) >> sh) & 0x7); }
    int mt = (int)((m - 1.0) * 8 + 0.5); if (mt >= 8) { mt = 0; b++; }
    if (b >= 15) return (s << 7) | 0x76;
    return (s << 7) | (b << 3) | (mt & 0x7);
}

inline uint8_t double_to_fp8_e5m2(double val) {
    if (std::isnan(val)) return 0x7F;
    if (std::isinf(val)) return val > 0 ? 0x7C : 0xFC;
    if (val == 0.0) return std::signbit(val) ? 0x80 : 0;
    bool s = val < 0; val = fabs(val);
    int e; double m = frexp(val, &e); e--; m *= 2;
    int b = e + 15;
    if (b >= 31) return (s << 7) | 0x7C;
    if (b <= 0) { int sh = 1 - b; if (sh > 3) return s << 7; return (s << 7) | (((int)(m * 4) >> sh) & 0x3); }
    int mt = (int)((m - 1.0) * 4 + 0.5); if (mt >= 4) { mt = 0; b++; }
    if (b >= 31) return (s << 7) | 0x7C;
    return (s << 7) | (b << 2) | (mt & 0x3);
}

inline uint8_t double_to_fp4(double val) {
    if (std::isnan(val)) return 0xF;
    if (std::isinf(val)) return val > 0 ? 0x6 : 0xE;
    if (val == 0.0) return std::signbit(val) ? 0x8 : 0;
    bool s = val < 0; val = fabs(val);
    int e; double m = frexp(val, &e); e--; m *= 2;
    int b = e + 1;
    if (b >= 3) return (s << 3) | 0x6;
    if (b <= 0) { if (val >= 0.25) return (s << 3) | 1; return s << 3; }
    int mt = (int)((m - 1.0) * 2 + 0.5); if (mt >= 2) { mt = 0; b++; }
    if (b >= 3) return (s << 3) | 0x6;
    return (s << 3) | (b << 1) | (mt & 1);
}

// ─────────────────────────────────────────────────────────────
//  Input → FP9 conversions (used at tensor core entry)
// ─────────────────────────────────────────────────────────────
inline uint16_t fp4_to_fp9(uint8_t fp4) {
    bool s = (fp4 >> 3) & 1; int e = (fp4 >> 1) & 3; int m = fp4 & 1;
    if (e == 3 && m == 1) return (s << 8) | (0x1F << 3) | 4; // NaN
    if (e == 3 && m == 0) return (s << 8) | (0x1F << 3);      // Inf
    if (e == 0 && m == 0) return (s << 8);                      // Zero
    if (e == 0) return (s << 8) | (14 << 3);                    // subnorm → 1.0 * 2^(-1)
    return (s << 8) | ((e + 14) << 3) | (m << 2);               // normal: rebias
}

inline uint16_t fp8_e4m3_to_fp9(uint8_t fp8) {
    bool s = (fp8 >> 7) & 1; int e = (fp8 >> 3) & 0xF; int m = fp8 & 7;
    if (e == 15) return (s << 8) | (0x1F << 3) | 4;             // NaN
    if (e == 0 && m == 0) return (s << 8);
    if (e == 0) { // subnormal normalize
        int lz = clz(m, 3);
        int ne = 9 - lz;
        if (ne <= 0) return (s << 8) | (((m << (1+lz)) & 7));
        return (s << 8) | (ne << 3) | (((m << (1+lz)) & 7));
    }
    int ne = e + 8;
    if (ne >= 31) return (s << 8) | (0x1F << 3); // overflow → Inf
    return (s << 8) | (ne << 3) | m;
}

inline uint16_t fp8_e5m2_to_fp9(uint8_t fp8) {
    bool s = (fp8 >> 7) & 1; int e = (fp8 >> 2) & 0x1F; int m = fp8 & 3;
    if (e == 31) {
        if (m) return (s << 8) | (0x1F << 3) | 4;
        return (s << 8) | (0x1F << 3);
    }
    return (s << 8) | (e << 3) | (m << 1); // same bias, extend mantissa
}

inline uint16_t fp16_to_fp9(uint16_t fp16) {
    bool s = (fp16 >> 15) & 1; int e = (fp16 >> 10) & 0x1F; int m = fp16 & 0x3FF;
    if (e == 0x1F) { if (m) return (s << 8) | (0x1F << 3) | 4; return (s << 8) | (0x1F << 3); }
    if (e == 0 && m == 0) return (s << 8);
    if (e == 0) {
        int lz = clz(m, 10);
        int ne = 1 - lz;
        if (ne <= 0) { return (s << 8) | ((m >> 7) & 7); }
        uint32_t nm = (m << (1+lz)) & 0x3FF;
        return (s << 8) | (ne << 3) | ((nm >> 7) & 7);
    }
    // Round-to-nearest-even for mantissa truncation 10→3
    int fp9m = (m >> 7) & 7;
    bool g = (m >> 6) & 1, r = (m >> 5) & 1, st = (m & 0x1F) != 0;
    if (g && (r || st || (fp9m & 1))) { fp9m++; if (fp9m >= 8) { fp9m = 0; e++; if (e >= 31) return (s << 8) | (0x1F << 3); } }
    return (s << 8) | (e << 3) | fp9m;
}

// Convert any input to FP9 based on precision type
inline uint16_t convert_to_fp9(uint32_t raw_bits, PrecisionType prec) {
    switch (prec) {
        case PREC_FP4_E2M1: return fp4_to_fp9(raw_bits & 0xF);
        case PREC_FP8_E4M3: return fp8_e4m3_to_fp9(raw_bits & 0xFF);
        case PREC_FP8_E5M2: return fp8_e5m2_to_fp9(raw_bits & 0xFF);
        case PREC_FP16:     return fp16_to_fp9(raw_bits & 0xFFFF);
        default: return 0;
    }
}

// ─────────────────────────────────────────────────────────────
//  FP9 → FP22 and FP16 → FP22 (for accumulator)
// ─────────────────────────────────────────────────────────────
inline uint32_t fp9_to_fp22(uint16_t fp9) {
    bool s = (fp9 >> 8) & 1; int e = (fp9 >> 3) & 0x1F; int m = fp9 & 7;
    if (e == 0 && m == 0) return (s << 21);
    if (e == 0x1F) {
        return (s << 21) | (0xFF << 13) | (m ? (0x1000 | (m << 10)) : 0);
    }
    if (e == 0) { // subnormal: normalize for FP22
        int lz = clz(m, 3);
        int ne = -14 - lz + 127; // rebias
        if (ne <= 0) return (s << 21) | ((m << (10+1+lz)) & 0x1FFF);
        return (s << 21) | (ne << 13) | ((((m << (1+lz)) & 7) << 10) & 0x1FFF);
    }
    return (s << 21) | ((e + 112) << 13) | (m << 10);
}

inline uint32_t fp16_to_fp22(uint16_t fp16) {
    bool s = (fp16 >> 15) & 1; int e = (fp16 >> 10) & 0x1F; int m = fp16 & 0x3FF;
    if (e == 0 && m == 0) return (s << 21);
    if (e == 0x1F) return (s << 21) | (0xFF << 13) | (m ? 0x1000 : 0);
    if (e == 0) {
        int lz = clz(m, 10);
        int ne = -14 - lz + 127;
        if (ne <= 0) return (s << 21) | ((m << (3+1+lz)) & 0x1FFF);
        return (s << 21) | (ne << 13) | ((((m << (1+lz)) & 0x3FF) << 3) & 0x1FFF);
    }
    return (s << 21) | ((e + 112) << 13) | ((m << 3) & 0x1FFF);
}

// ─────────────────────────────────────────────────────────────
//  FP22 → output format conversions
// ─────────────────────────────────────────────────────────────
inline uint8_t fp22_to_fp8_e4m3(uint32_t fp22, RoundingMode rm = RNE) {
    bool s = (fp22 >> 21) & 1; int e = (fp22 >> 13) & 0xFF; int m = fp22 & 0x1FFF;
    if (e == 0xFF) return (s << 7) | (14 << 3) | 7; // max (E4M3 no inf)
    if (e == 0) return (s << 7);
    int ne = (int)e - 120;
    uint32_t fm = (1u << 13) | m;
    if (ne >= 15) return (s << 7) | (14 << 3) | 7;
    if (ne <= 0) {
        int sh = 1 - ne; if (sh > 14) return (s << 7);
        fm >>= sh;
        int o = (fm >> 10) & 7;
        bool g = (fm >> 9) & 1, r = (fm >> 8) & 1, st = (fm & 0xFF) != 0;
        bool up = false;
        switch(rm) { case RNE: up = g && (r||st||(o&1)); break; case RTZ: break; case RDN: up = s&&(g||r||st); break; case RUP: up = !s&&(g||r||st); break; case RMM: up = g; break; }
        if (up) { o++; if (o >= 8) { o = 0; ne = 1; } else ne = 0; }
        else ne = 0;
        return (s << 7) | (ne << 3) | (o & 7);
    }
    int o = (m >> 10) & 7;
    bool g = (m >> 9) & 1, r = (m >> 8) & 1, st = (m & 0xFF) != 0;
    bool up = false;
    switch(rm) { case RNE: up = g && (r||st||(o&1)); break; case RTZ: break; case RDN: up = s&&(g||r||st); break; case RUP: up = !s&&(g||r||st); break; case RMM: up = g; break; }
    if (up) { o++; if (o >= 8) { o = 0; ne++; if (ne >= 15) return (s << 7) | (14 << 3) | 7; } }
    return (s << 7) | (ne << 3) | (o & 7);
}

inline uint8_t fp22_to_fp8_e5m2(uint32_t fp22, RoundingMode rm = RNE) {
    bool s = (fp22 >> 21) & 1; int e = (fp22 >> 13) & 0xFF; int m = fp22 & 0x1FFF;
    if (e == 0xFF) { if (m) return (s << 7) | (0x1F << 2) | 1; return (s << 7) | (0x1F << 2); }
    if (e == 0) return (s << 7);
    int ne = (int)e - 112;
    if (ne >= 31) { bool rmin = (rm==RTZ)||(rm==RDN&&!s)||(rm==RUP&&s); return rmin ? ((s<<7)|(30<<2)|3) : ((s<<7)|(0x1F<<2)); }
    uint32_t fm = (1u << 13) | m;
    if (ne <= 0) {
        int sh = 1 - ne; if (sh > 14) return (s << 7);
        fm >>= sh;
        int o = (fm >> 11) & 3;
        bool g = (fm >> 10) & 1, r = (fm >> 9) & 1, st = (fm & 0x1FF) != 0;
        bool up = false;
        switch(rm) { case RNE: up = g && (r||st||(o&1)); break; case RTZ: break; case RDN: up = s&&(g||r||st); break; case RUP: up = !s&&(g||r||st); break; case RMM: up = g; break; }
        if (up) { o++; if (o >= 4) { o = 0; ne = 1; } else ne = 0; }
        else ne = 0;
        return (s << 7) | (ne << 2) | (o & 3);
    }
    int o = (m >> 11) & 3;
    bool g = (m >> 10) & 1, r = (m >> 9) & 1, st = (m & 0x1FF) != 0;
    bool up = false;
    switch(rm) { case RNE: up = g && (r||st||(o&1)); break; case RTZ: break; case RDN: up = s&&(g||r||st); break; case RUP: up = !s&&(g||r||st); break; case RMM: up = g; break; }
    if (up) { o++; if (o >= 4) { o = 0; ne++; if (ne >= 31) { bool rmin=(rm==RTZ)||(rm==RDN&&!s)||(rm==RUP&&s); return rmin?((s<<7)|(30<<2)|3):((s<<7)|(0x1F<<2)); } } }
    return (s << 7) | (ne << 2) | (o & 3);
}

inline uint16_t fp22_to_fp16(uint32_t fp22, RoundingMode rm = RNE) {
    bool s = (fp22 >> 21) & 1; int e = (fp22 >> 13) & 0xFF; int m = fp22 & 0x1FFF;
    if (e == 0xFF) { if (m) return (s << 15) | (0x1F << 10) | 0x200; return (s << 15) | (0x1F << 10); }
    if (e == 0) return (s << 15);
    int ne = (int)e - 112;
    if (ne >= 31) { bool rmin = (rm==RTZ)||(rm==RDN&&!s)||(rm==RUP&&s); return rmin ? ((s<<15)|(30<<10)|0x3FF) : ((s<<15)|(0x1F<<10)); }
    uint32_t fm = (1u << 13) | m;
    if (ne <= 0) {
        int sh = 1 - ne; if (sh > 14) return (s << 15);
        fm >>= sh;
        int o = (fm >> 3) & 0x3FF;
        bool g = (fm >> 2) & 1, r = (fm >> 1) & 1, st = fm & 1;
        bool up = false;
        switch(rm) { case RNE: up = g && (r||st||(o&1)); break; case RTZ: break; case RDN: up = s&&(g||r||st); break; case RUP: up = !s&&(g||r||st); break; case RMM: up = g; break; }
        if (up) { o++; if (o >= 1024) { o = 0; ne = 1; } else ne = 0; }
        else ne = 0;
        return (s << 15) | (ne << 10) | (o & 0x3FF);
    }
    int o = (m >> 3) & 0x3FF;
    bool g = (m >> 2) & 1, r = (m >> 1) & 1, st = m & 1;
    bool up = false;
    switch(rm) { case RNE: up = g && (r||st||(o&1)); break; case RTZ: break; case RDN: up = s&&(g||r||st); break; case RUP: up = !s&&(g||r||st); break; case RMM: up = g; break; }
    if (up) { o++; if (o >= 1024) { o = 0; ne++; if (ne >= 31) { bool rmin=(rm==RTZ)||(rm==RDN&&!s)||(rm==RUP&&s); return rmin?((s<<15)|(30<<10)|0x3FF):((s<<15)|(0x1F<<10)); } } }
    return (s << 15) | (ne << 10) | (o & 0x3FF);
}

inline uint32_t fp22_to_fp32(uint32_t fp22) {
    bool s = (fp22 >> 21) & 1; int e = (fp22 >> 13) & 0xFF; int m = fp22 & 0x1FFF;
    return ((uint32_t)s << 31) | ((uint32_t)e << 23) | ((uint32_t)m << 10);
}

// Convert C bias to FP22 based on output format
inline uint32_t convert_c_to_fp22(uint32_t raw_bits, PrecisionType prec) {
    switch (prec) {
        case PREC_FP8_E4M3: return fp9_to_fp22(fp8_e4m3_to_fp9(raw_bits & 0xFF));
        case PREC_FP8_E5M2: return fp9_to_fp22(fp8_e5m2_to_fp9(raw_bits & 0xFF));
        case PREC_FP16:     return fp16_to_fp22(raw_bits & 0xFFFF);
        case PREC_FP4_E2M1: return fp9_to_fp22(fp4_to_fp9(raw_bits & 0xF));
        default: return 0;
    }
}
