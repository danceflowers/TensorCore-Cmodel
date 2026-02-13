// =============================================================================
// tensor_core_cfg.cpp — TensorCore simulator runtime configuration definitions
// =============================================================================
#include "tensor_core_cfg.h"

uint32_t convert_fp22_to_output_bits(uint32_t fp22, PrecisionType output_prec, RoundingMode rm) {
    switch (output_prec) {
        case PREC_FP8_E4M3: return (uint32_t)fp22_to_fp8_e4m3(fp22, rm);
        case PREC_FP8_E5M2: return (uint32_t)fp22_to_fp8_e5m2(fp22, rm);
        case PREC_FP16:     return (uint32_t)fp22_to_fp16(fp22, rm);
        case PREC_FP32:     return fp22_to_fp32(fp22);
        case PREC_FP4_E2M1:
        default:            return 0;
    }
}

double output_bits_to_double(uint32_t bits, PrecisionType output_prec) {
    switch (output_prec) {
        case PREC_FP8_E4M3: return fp8_e4m3_to_double((uint8_t)(bits & 0xFF));
        case PREC_FP8_E5M2: return fp8_e5m2_to_double((uint8_t)(bits & 0xFF));
        case PREC_FP16:     return fp16_to_double((uint16_t)(bits & 0xFFFF));
        case PREC_FP32: {
            float f = 0.0f;
            uint32_t raw = bits;
            std::memcpy(&f, &raw, sizeof(float));
            return (double)f;
        }
        case PREC_FP4_E2M1:
        default:            return 0.0;
    }
}
