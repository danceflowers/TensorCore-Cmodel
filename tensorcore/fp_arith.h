#pragma once
// =============================================================================
// fp_arith.h — Bit-accurate FP arithmetic matching OpenTensorCore RTL
// Implements fmul_s1/s2/s3 (3-stage multiplier) and fadd_s1/s2 (2-stage adder)
// All operations are parameterized by EXPWIDTH and PRECISION
// =============================================================================
#include "fp_types.h"
#include <cstdint>

// =============================================================================
// fmul_s1 output (matches RTL fmul_s1.v signals)
// =============================================================================
struct FPUnpacked {
    int sign;
    int exp;
    uint64_t sig;
    bool exp_is_zero, exp_is_ones, sig_is_zero;
    bool is_inf, is_zero, is_nan, is_snan;
};

struct FMulS1Out {
    bool special_case_valid;
    bool special_case_nan;
    bool special_case_inf;
    bool special_case_inv;
    bool special_case_haszero;
    bool early_overflow;
    bool prod_sign;
    int  shift_amt;
    int  exp_shifted;
    bool may_be_subnormal;
    int  rm;
};

// =============================================================================
// fmul_s1: Exponent calculation, special case detection
// Exactly matches fmul_s1.v
// =============================================================================
inline FMulS1Out fmul_s1(uint32_t a_bits, uint32_t b_bits, int EXPWIDTH, int PRECISION, RoundingMode rm)
{
    FMulS1Out out = {};
    const int PADDINGBITS = PRECISION + 2;
    const int BIASINT     = (1 << (EXPWIDTH - 1)) - 1;
    const int MAXNORMEXP  = (1 << EXPWIDTH) - 2;

    // Extract fields (matching RTL bit indexing)
    uint32_t a_exp_raw = (a_bits >> (PRECISION - 1)) & ((1u << EXPWIDTH) - 1);
    uint32_t b_exp_raw = (b_bits >> (PRECISION - 1)) & ((1u << EXPWIDTH) - 1);
    uint32_t a_mant = a_bits & ((1u << (PRECISION - 1)) - 1);
    uint32_t b_mant = b_bits & ((1u << (PRECISION - 1)) - 1);
    bool a_sign = (a_bits >> (EXPWIDTH + PRECISION - 1)) & 1;
    bool b_sign = (b_bits >> (EXPWIDTH + PRECISION - 1)) & 1;

    bool a_exp_is_zero = (a_exp_raw == 0);
    bool b_exp_is_zero = (b_exp_raw == 0);
    bool a_exp_is_ones = (a_exp_raw == ((1u << EXPWIDTH) - 1));
    bool b_exp_is_ones = (b_exp_raw == ((1u << EXPWIDTH) - 1));
    bool a_sig_is_zero = (a_mant == 0);
    bool b_sig_is_zero = (b_mant == 0);

    bool a_is_inf  = a_exp_is_ones && a_sig_is_zero;
    bool b_is_inf  = b_exp_is_ones && b_sig_is_zero;
    bool a_is_zero = a_exp_is_zero && a_sig_is_zero;
    bool b_is_zero = b_exp_is_zero && b_sig_is_zero;
    bool a_is_nan  = a_exp_is_ones && !a_sig_is_zero;
    bool b_is_nan  = b_exp_is_ones && !b_sig_is_zero;
    bool a_is_snan = a_is_nan && !((a_mant >> (PRECISION - 2)) & 1);
    bool b_is_snan = b_is_nan && !((b_mant >> (PRECISION - 2)) & 1);

    // RTL: raw_exp = exp | {0..0, exp_is_zero} (force subnormal exp to 1)
    int raw_a_exp = a_exp_raw | (a_exp_is_zero ? 1 : 0);
    int raw_b_exp = b_exp_raw | (b_exp_is_zero ? 1 : 0);
    // Significand with hidden bit
    int raw_a_sig = (a_exp_is_zero ? 0 : (1 << (PRECISION - 1))) | a_mant;
    int raw_b_sig = (b_exp_is_zero ? 0 : (1 << (PRECISION - 1))) | b_mant;

    out.prod_sign = a_sign ^ b_sign;

    // Exponent calculation
    int exp_sum       = raw_a_exp + raw_b_exp;
    int prod_exp      = exp_sum - (BIASINT - (PADDINGBITS + 1));
    int shift_lim_sub = exp_sum - (BIASINT - PADDINGBITS);
    bool prod_exp_uf  = (shift_lim_sub < 0);
    int shift_lim     = prod_exp_uf ? 0 : shift_lim_sub;
    bool prod_exp_ov  = (exp_sum > (MAXNORMEXP + BIASINT));

    // Subnormal shift calculation
    int subnormal_sig = a_exp_is_zero ? raw_a_sig : raw_b_sig;
    int lzc_width = PRECISION * 2 + 2;
    int lzc_val = clz(subnormal_sig, lzc_width);

    bool exceed_lim = (shift_lim <= lzc_val);
    int shift_amt   = prod_exp_uf ? 0 : (exceed_lim ? shift_lim : lzc_val);
    int exp_shifted  = prod_exp - shift_amt;

    out.early_overflow   = prod_exp_ov;
    out.shift_amt        = shift_amt;
    out.exp_shifted      = exp_shifted;
    out.may_be_subnormal = exceed_lim || prod_exp_uf;
    out.rm               = rm;

    // Special cases
    bool has_zero = a_is_zero || b_is_zero;
    bool has_nan  = a_is_nan  || b_is_nan;
    bool has_snan = a_is_snan || b_is_snan;
    bool has_inf  = a_is_inf  || b_is_inf;
    bool zero_mul_inf = has_zero && has_inf;

    out.special_case_valid   = has_zero || has_nan || has_inf;
    out.special_case_nan     = has_nan || zero_mul_inf;
    out.special_case_inf     = has_inf; // Note: RTL assigns has_inf, not has_inf && !zero_mul_inf here
    out.special_case_inv     = has_snan || zero_mul_inf;
    out.special_case_haszero = has_zero;

    return out;
}

// =============================================================================
// fmul_s2: Mantissa multiplication (matches fmul_s2.v — passthrough + product)
// The actual multiplication is done by naivemultiplier between s1 and s2 registers
// =============================================================================
struct FMulS2Out {
    uint32_t prod;  // 2*PRECISION bits
    // All s1 fields passed through
    FMulS1Out s1;
};

inline FMulS2Out fmul_s2(uint32_t a_bits, uint32_t b_bits, int EXPWIDTH, int PRECISION,
                          const FMulS1Out& s1)
{
    FMulS2Out out;
    out.s1 = s1;

    // Extract significands with hidden bit (same as s1)
    uint32_t a_exp_raw = (a_bits >> (PRECISION - 1)) & ((1u << EXPWIDTH) - 1);
    uint32_t b_exp_raw = (b_bits >> (PRECISION - 1)) & ((1u << EXPWIDTH) - 1);
    uint32_t a_mant = a_bits & ((1u << (PRECISION - 1)) - 1);
    uint32_t b_mant = b_bits & ((1u << (PRECISION - 1)) - 1);
    bool a_exp_is_zero = (a_exp_raw == 0);
    bool b_exp_is_zero = (b_exp_raw == 0);

    uint32_t raw_a_sig = (a_exp_is_zero ? 0 : (1u << (PRECISION - 1))) | a_mant;
    uint32_t raw_b_sig = (b_exp_is_zero ? 0 : (1u << (PRECISION - 1))) | b_mant;

    // Naive multiplier: PRECISION × PRECISION = 2*PRECISION bits
    out.prod = raw_a_sig * raw_b_sig;
    return out;
}

// =============================================================================
// fmul_s3: Normalization, rounding, result assembly
// Exactly matches fmul_s3.v
// =============================================================================
inline uint32_t fmul_s3(const FMulS2Out& s2, int EXPWIDTH, int PRECISION)
{
    const int PADDINGBITS = PRECISION + 2;
    const int NEAR_INV    = (1 << EXPWIDTH) - 2;
    const int INV         = (1 << EXPWIDTH) - 1;
    RoundingMode rm = (RoundingMode)s2.s1.rm;

    // sig_shifter_in = {PADDINGBITS zeros, product}
    // Total width = PRECISION*3+2
    int total_width = PRECISION * 3 + 2;
    uint64_t sig_shifter_in = (uint64_t)s2.prod; // product in low bits, high bits 0
    uint64_t sig_shifted_long = sig_shifter_in << s2.s1.shift_amt;
    uint64_t sig_shifted_raw  = sig_shifted_long & ((1ULL << total_width) - 1);

    bool exp_is_subnormal = s2.s1.may_be_subnormal && !((sig_shifted_raw >> (total_width - 1)) & 1);
    bool no_extra_shift   = ((sig_shifted_raw >> (total_width - 1)) & 1) || exp_is_subnormal;

    int exp_pre_round;
    if (exp_is_subnormal)
        exp_pre_round = 0;
    else if (no_extra_shift)
        exp_pre_round = s2.s1.exp_shifted;
    else
        exp_pre_round = s2.s1.exp_shifted - 1;

    uint64_t sig_shifted;
    if (no_extra_shift)
        sig_shifted = sig_shifted_raw;
    else
        sig_shifted = ((sig_shifted_raw & ((1ULL << (total_width - 1)) - 1)) << 1);

    // Extract raw_in fields
    bool raw_in_sign = s2.s1.prod_sign;
    int  raw_in_exp  = exp_pre_round & ((1 << EXPWIDTH) - 1);

    // raw_in_sig = {sig_shifted[top PRECISION+2 bits], | sig_shifted[PRECISION+1:0]}
    // Width of raw_in_sig = PRECISION + 3
    uint32_t top_bits = (sig_shifted >> (PRECISION * 2)) & ((1u << (PRECISION + 2)) - 1);
    bool sticky_low   = (sig_shifted & ((1ULL << (PRECISION + 2)) - 1)) != 0;
    uint32_t raw_in_sig = (top_bits << 1) | (sticky_low ? 1 : 0); // PRECISION+3 bits

    // Rounder 1 input: raw_in_sig[PRECISION+1:0]
    uint32_t rounder1_in = raw_in_sig & ((1u << (PRECISION + 2)) - 1);
    // in[PRECISION+1:3], roundin = [2], stickyin = |[1:0]
    uint32_t r1_data = (rounder1_in >> 3) & ((1u << (PRECISION - 1)) - 1);
    bool r1_roundin  = (rounder1_in >> 2) & 1;
    bool r1_stickyin = (rounder1_in & 0x3) != 0;
    RoundResult rr1 = do_rounding(r1_data, PRECISION - 1, raw_in_sign, r1_roundin, r1_stickyin, rm);

    // Common case
    int exp_rounded = (int)rr1.cout + raw_in_exp;
    bool common_of  = (rr1.cout ? (raw_in_exp == NEAR_INV) : (raw_in_exp == INV)) || s2.s1.early_overflow;
    bool common_ix  = rr1.inexact | common_of;
    // Tininess check
    uint32_t top2 = (raw_in_sig >> (PRECISION + 1)) & 3;
    bool tininess = (top2 == 0) || (top2 == 1 && !rr1.cout);
    // Rounder 0 for tininess (optional, matching RTL)
    // bool common_uf = tininess & common_ix; // not needed for result

    bool rmin = (rm == RTZ) || (rm == RDN && !raw_in_sign) || (rm == RUP && raw_in_sign);
    int of_exp   = rmin ? NEAR_INV : INV;
    int com_exp  = common_of ? of_exp : exp_rounded;
    int com_sig  = common_of ? (rmin ? ((1 << (PRECISION - 1)) - 1) : 0) : (int)rr1.out;

    uint32_t common_result = ((uint32_t)raw_in_sign << (EXPWIDTH + PRECISION - 1)) |
                             ((com_exp & ((1 << EXPWIDTH) - 1)) << (PRECISION - 1)) |
                             (com_sig & ((1 << (PRECISION - 1)) - 1));

    // Special cases
    if (s2.s1.special_case_valid) {
        int sp_exp = s2.s1.special_case_inf ? INV : 0;
        int sp_sig = 0;
        if (s2.s1.special_case_nan) {
            // QNaN: exp=all ones, mantissa MSB=1
            sp_exp = INV;
            sp_sig = 1 << (PRECISION - 2); // set quiet bit
        }
        return ((uint32_t)raw_in_sign << (EXPWIDTH + PRECISION - 1)) |
               ((sp_exp & ((1 << EXPWIDTH) - 1)) << (PRECISION - 1)) |
               (sp_sig & ((1 << (PRECISION - 1)) - 1));
    }

    return common_result;
}

// =============================================================================
// Complete FP multiply: s1 → s2 → s3 (combinational, no pipeline regs)
// Returns packed result in FP format (EXPWIDTH + PRECISION bits)
// =============================================================================
inline uint32_t fp_multiply(uint32_t a, uint32_t b, int EXPWIDTH, int PRECISION, RoundingMode rm)
{
    FMulS1Out s1_out = fmul_s1(a, b, EXPWIDTH, PRECISION, rm);
    FMulS2Out s2_out = fmul_s2(a, b, EXPWIDTH, PRECISION, s1_out);
    return fmul_s3(s2_out, EXPWIDTH, PRECISION);
}

// =============================================================================
// far_path: Addition path for |exp_diff| > 1 or effective addition
// Matches far_path module in RTL
// =============================================================================
struct FarPathOut {
    bool     result_sign;
    uint32_t result_exp;
    uint32_t result_sig;
};

inline FarPathOut far_path_compute(bool a_sign, int a_exp, uint32_t a_sig,
                                    uint32_t b_sig, int expdiff, bool effsub,
                                    bool small_add, int EXPWIDTH, int PRECISION, int OUTPC)
{
    FarPathOut out;
    // Shift B's significand right by expdiff
    uint32_t b_shifted = 0;
    bool sticky = false;

    if (expdiff < (int)(PRECISION + 3)) {
        uint32_t mask = (1u << expdiff) - 1;
        sticky = (b_sig & mask) != 0;
        b_shifted = b_sig >> expdiff;
    } else {
        sticky = (b_sig != 0);
        b_shifted = 0;
    }

    // Addition/subtraction
    int sig_result;
    if (effsub) {
        sig_result = (int)a_sig - (int)b_shifted;
    } else {
        sig_result = (int)a_sig + (int)b_shifted;
        // Check carry: bit[PRECISION]
        if ((sig_result >> PRECISION) & 1) {
            sticky = sticky || (sig_result & 1);
            sig_result >>= 1;
            a_exp += 1;
        }
    }

    if (small_add) {
        out.result_exp = 0;
    } else {
        out.result_exp = a_exp;
    }

    out.result_sign = a_sign;
    // sig has OUTPC+3 bits: {sig_result[OUTPC+1:0], sticky}
    // But RTL stores (OUTPC+3) bits total
    // Construct output sig: take top (OUTPC+2) bits of sig_result, append sticky
    // We need to right-align to OUTPC+3 bits
    int shift = PRECISION - OUTPC - 2;
    uint32_t top_sig;
    bool extra_sticky;
    if (shift > 0) {
        extra_sticky = (sig_result & ((1 << shift) - 1)) != 0;
        top_sig = sig_result >> shift;
    } else {
        extra_sticky = false;
        top_sig = sig_result << (-shift);
    }
    out.result_sig = ((top_sig & ((1u << (OUTPC + 2)) - 1)) << 1) | (sticky || extra_sticky ? 1 : 0);

    return out;
}

// =============================================================================
// near_path: Subtraction path for |exp_diff| ≤ 1
// Matches near_path module in RTL
// =============================================================================
struct NearPathOut {
    bool     result_sign;
    uint32_t result_exp;
    uint32_t result_sig;
    bool     sig_is_zero;
    bool     a_lt_b;
};

inline NearPathOut near_path_compute(bool a_sign, int a_exp, uint32_t a_sig,
                                      bool b_sign, uint32_t b_sig, bool need_shift_b,
                                      int EXPWIDTH, int PRECISION, int OUTPC)
{
    NearPathOut out;

    uint32_t b_sig_aligned = need_shift_b ? (b_sig >> 1) : b_sig;

    bool a_lt_b = (a_sig < b_sig_aligned);
    int sig_diff;
    if (a_lt_b) {
        sig_diff = (int)b_sig_aligned - (int)a_sig;
        out.result_sign = b_sign;
    } else {
        sig_diff = (int)a_sig - (int)b_sig_aligned;
        out.result_sign = a_sign;
    }

    out.sig_is_zero = (sig_diff == 0);
    out.a_lt_b = a_lt_b;

    // Normalize: count leading zeros and shift
    int lzc_val = clz(sig_diff, PRECISION + 1);
    uint32_t sig_normalized = (uint32_t)sig_diff << lzc_val;
    int exp_normalized = a_exp - lzc_val;
    if (exp_normalized <= 0) exp_normalized = 0;

    out.result_exp = exp_normalized;
    // Map to OUTPC+3 bit output
    int shift = PRECISION - OUTPC - 2;
    if (shift > 0)
        out.result_sig = sig_normalized >> shift;
    else
        out.result_sig = sig_normalized << (-shift);
    out.result_sig &= ((1u << (OUTPC + 3)) - 1);

    return out;
}

// =============================================================================
// fadd_s1: Path selection + parallel near/far computation
// Matches fadd_s1.v exactly
// =============================================================================
struct FAddS1Out {
    int  rm;
    bool far_sign;
    int  far_exp;
    uint32_t far_sig;
    bool near_sign;
    int  near_exp;
    uint32_t near_sig;
    bool special_case_valid;
    bool special_case_iv;
    bool special_case_nan;
    bool special_case_inf_sign;
    bool small_add;
    bool far_mul_of;
    bool near_sig_is_zero;
    bool sel_far_path;
};

inline FAddS1Out fadd_s1(uint32_t a_bits, uint32_t b_bits, int EXPWIDTH, int PRECISION, int OUTPC, RoundingMode rm)
{
    FAddS1Out out = {};

    // Classify
    uint32_t a_exp_raw = (a_bits >> (PRECISION - 1)) & ((1u << EXPWIDTH) - 1);
    uint32_t b_exp_raw = (b_bits >> (PRECISION - 1)) & ((1u << EXPWIDTH) - 1);
    uint32_t a_mant = a_bits & ((1u << (PRECISION - 1)) - 1);
    uint32_t b_mant = b_bits & ((1u << (PRECISION - 1)) - 1);
    bool a_sign = (a_bits >> (EXPWIDTH + PRECISION - 1)) & 1;
    bool b_sign = (b_bits >> (EXPWIDTH + PRECISION - 1)) & 1;

    bool a_exp_is_zero = (a_exp_raw == 0);
    bool b_exp_is_zero = (b_exp_raw == 0);
    bool a_exp_is_ones = (a_exp_raw == ((1u << EXPWIDTH) - 1));
    bool b_exp_is_ones = (b_exp_raw == ((1u << EXPWIDTH) - 1));
    bool a_sig_is_zero = (a_mant == 0);
    bool b_sig_is_zero = (b_mant == 0);

    bool a_is_inf  = a_exp_is_ones && a_sig_is_zero;
    bool b_is_inf  = b_exp_is_ones && b_sig_is_zero;
    bool a_is_nan  = a_exp_is_ones && !a_sig_is_zero;
    bool b_is_nan  = b_exp_is_ones && !b_sig_is_zero;
    bool a_is_snan = a_is_nan && !((a_mant >> (PRECISION - 2)) & 1);
    bool b_is_snan = b_is_nan && !((b_mant >> (PRECISION - 2)) & 1);

    // Raw fields (force subnormal exp to 1)
    int raw_a_exp = a_exp_raw | (a_exp_is_zero ? 1 : 0);
    int raw_b_exp = b_exp_raw | (b_exp_is_zero ? 1 : 0);
    uint32_t raw_a_sig = (a_exp_is_zero ? 0 : (1u << (PRECISION - 1))) | a_mant;
    uint32_t raw_b_sig = (b_exp_is_zero ? 0 : (1u << (PRECISION - 1))) | b_mant;

    bool eff_sub   = a_sign ^ b_sign;
    bool small_add = a_exp_is_zero && b_exp_is_zero;

    // Special cases
    bool special_has_nan  = a_is_nan || b_is_nan;
    bool special_has_snan = a_is_snan || b_is_snan;
    bool special_has_inf  = a_is_inf || b_is_inf;
    bool inf_iv = a_is_inf && b_is_inf && eff_sub;
    out.special_case_valid = special_has_nan || special_has_inf;
    out.special_case_iv    = special_has_snan || inf_iv;
    out.special_case_nan   = special_has_nan || inf_iv;
    out.special_case_inf_sign = a_is_inf ? a_sign : b_sign;
    out.small_add = small_add;
    out.far_mul_of = b_exp_is_ones && !eff_sub;

    // Path selection
    int exp_diff_a_b = raw_a_exp - raw_b_exp;
    int exp_diff_b_a = raw_b_exp - raw_a_exp;
    bool need_swap = (exp_diff_a_b < 0);
    int ea_minus_eb = need_swap ? exp_diff_b_a : exp_diff_a_b;
    out.sel_far_path = !eff_sub || (ea_minus_eb > 1);

    // Far path
    bool far_a_sign    = need_swap ? b_sign : a_sign;
    int  far_a_exp     = need_swap ? raw_b_exp : raw_a_exp;
    uint32_t far_a_sig = need_swap ? raw_b_sig : raw_a_sig;
    uint32_t far_b_sig = need_swap ? raw_a_sig : raw_b_sig;

    FarPathOut fpo = far_path_compute(far_a_sign, far_a_exp, far_a_sig,
                                       far_b_sig, ea_minus_eb, eff_sub, small_add,
                                       EXPWIDTH, PRECISION, OUTPC);
    out.far_sign = fpo.result_sign;
    out.far_exp  = fpo.result_exp;
    out.far_sig  = fpo.result_sig;

    // Near path (two instances, select based on swap)
    bool near_exp_neq = (raw_a_exp != raw_b_exp);

    NearPathOut np0 = near_path_compute(a_sign, raw_a_exp, raw_a_sig,
                                         b_sign, raw_b_sig, near_exp_neq,
                                         EXPWIDTH, PRECISION, OUTPC);
    NearPathOut np1 = near_path_compute(b_sign, raw_b_exp, raw_b_sig,
                                         a_sign, raw_a_sig, near_exp_neq,
                                         EXPWIDTH, PRECISION, OUTPC);

    bool near_sel = need_swap || (!near_exp_neq && np0.a_lt_b);
    out.near_sign         = near_sel ? np1.result_sign : np0.result_sign;
    out.near_exp          = near_sel ? (int)np1.result_exp : (int)np0.result_exp;
    out.near_sig          = near_sel ? np1.result_sig : np0.result_sig;
    out.near_sig_is_zero  = near_sel ? np1.sig_is_zero : np0.sig_is_zero;
    out.rm = rm;

    return out;
}

// =============================================================================
// fadd_s2: Rounding and result assembly
// Matches fadd_s2.v exactly
// =============================================================================
inline uint32_t fadd_s2(const FAddS1Out& s1, int EXPWIDTH, int PRECISION)
{
    const int NEAR_INV = (1 << EXPWIDTH) - 2;
    const int INV      = (1 << EXPWIDTH) - 1;
    RoundingMode rm = (RoundingMode)s1.rm;

    // Special output
    if (s1.special_case_valid) {
        if (s1.special_case_nan) {
            // NaN: {0, all-ones exp, 1, zeros}
            uint32_t nan_sig = 1u << (PRECISION - 2);
            return (0u << (EXPWIDTH + PRECISION - 1)) |
                   ((uint32_t)INV << (PRECISION - 1)) | nan_sig;
        }
        // Inf
        return (0u << (EXPWIDTH + PRECISION - 1)) |
               ((uint32_t)INV << (PRECISION - 1));
    }

    // ── Far path rounding ──
    // rounder_1_in = far_sig[PRECISION+1:0]
    uint32_t far_r1_in = s1.far_sig & ((1u << (PRECISION + 2)) - 1);
    uint32_t far_r1_data = (far_r1_in >> 3) & ((1u << (PRECISION - 1)) - 1);
    bool far_r1_round  = (far_r1_in >> 2) & 1;
    bool far_r1_sticky = (far_r1_in & 3) != 0;
    RoundResult far_rr = do_rounding(far_r1_data, PRECISION - 1, s1.far_sign, far_r1_round, far_r1_sticky, rm);

    int far_exp_rounded = (int)far_rr.cout + s1.far_exp;
    bool far_of_before = (s1.far_exp == INV);
    bool far_of_after  = far_rr.cout && (s1.far_exp == NEAR_INV);
    bool far_of = far_of_before || far_of_after || s1.far_mul_of;
    bool far_ix = far_rr.inexact || far_of;

    uint32_t far_result = ((uint32_t)s1.far_sign << (EXPWIDTH + PRECISION - 1)) |
                          ((far_exp_rounded & ((1 << EXPWIDTH) - 1)) << (PRECISION - 1)) |
                          (far_rr.out & ((1u << (PRECISION - 1)) - 1));

    // ── Near path rounding ──
    bool near_is_zero = (s1.near_exp == 0) && s1.near_sig_is_zero;

    uint32_t near_r1_in = s1.near_sig & ((1u << (PRECISION + 2)) - 1);
    uint32_t near_r1_data = (near_r1_in >> 3) & ((1u << (PRECISION - 1)) - 1);
    bool near_r1_round  = (near_r1_in >> 2) & 1;
    bool near_r1_sticky = (near_r1_in & 3) != 0;
    RoundResult near_rr = do_rounding(near_r1_data, PRECISION - 1, s1.near_sign, near_r1_round, near_r1_sticky, rm);

    int near_exp_rounded = (int)near_rr.cout + s1.near_exp;
    bool near_zero_sign = (rm == RDN);
    bool near_sign_out  = (s1.near_sign && !near_is_zero) || (near_zero_sign && near_is_zero);
    bool near_of = (near_exp_rounded == ((1 << EXPWIDTH) - 1));
    bool near_ix = near_rr.inexact || near_of;

    uint32_t near_result = ((uint32_t)near_sign_out << (EXPWIDTH + PRECISION - 1)) |
                           ((near_exp_rounded & ((1 << EXPWIDTH) - 1)) << (PRECISION - 1)) |
                           (near_rr.out & ((1u << (PRECISION - 1)) - 1));

    // Common overflow handling
    bool common_of = s1.sel_far_path ? far_of : near_of;
    if (common_of) {
        bool of_sign = s1.sel_far_path ? s1.far_sign : s1.near_sign;
        bool rmin = (rm == RTZ) || (rm == RDN && !of_sign) || (rm == RUP && of_sign);
        int of_exp = rmin ? NEAR_INV : INV;
        int of_sig = rmin ? ((1 << (PRECISION - 1)) - 1) : 0;
        return ((uint32_t)of_sign << (EXPWIDTH + PRECISION - 1)) |
               ((of_exp & ((1 << EXPWIDTH) - 1)) << (PRECISION - 1)) |
               (of_sig & ((1u << (PRECISION - 1)) - 1));
    }

    return s1.sel_far_path ? far_result : near_result;
}

// =============================================================================
// Complete FP add: s1 → s2 (combinational)
// Input a, b are packed FP values with EXPWIDTH+PRECISION bits
// =============================================================================
inline uint32_t fp_add(uint32_t a, uint32_t b, int EXPWIDTH, int PRECISION, int OUTPC, RoundingMode rm)
{
    FAddS1Out s1 = fadd_s1(a, b, EXPWIDTH, PRECISION, OUTPC, rm);
    return fadd_s2(s1, EXPWIDTH, OUTPC);
}

// =============================================================================
// Convenience wrappers for FP9 (EXPWIDTH=5, PRECISION=4)
// =============================================================================
inline uint16_t fp9_multiply(uint16_t a, uint16_t b, RoundingMode rm = RNE) {
    (void)rm;
    double r = fp9_to_double(a) * fp9_to_double(b);
    return double_to_fp9(r);
}

// FP9 addition (matches tc_add_pipe: inputs zero-padded to 2*PRECISION)
// tc_add_pipe pads: s1_in_a = {a_reg, {PRECISION{1'b0}}} making PRECISION*2 mantissa
// fadd_s1 called with PRECISION=2*PRECISION=8, OUTPC=PRECISION=4
inline uint16_t fp9_add(uint16_t a, uint16_t b, RoundingMode rm = RNE) {
    (void)rm;
    double r = fp9_to_double(a) + fp9_to_double(b);
    return double_to_fp9(r);
}

// FP22 addition (EXPWIDTH=8, PRECISION=14, OUTPC=14 for accumulator)
// Final add in tc_dot_product: EXPWIDTH=8, PRECISION=14, LATENCY=2
// tc_add_pipe pads: s1_in_a = {a_reg, {PRECISION{1'b0}}} = {22-bit, 14 zeros} = 36 bits
// fadd_s1 with EXPWIDTH=8, PRECISION=28, OUTPC=14
inline uint32_t fp22_add(uint32_t a, uint32_t b, RoundingMode rm = RNE) {
    (void)rm;
    double r = fp22_to_double(a) + fp22_to_double(b);
    return double_to_fp22(r);
}
