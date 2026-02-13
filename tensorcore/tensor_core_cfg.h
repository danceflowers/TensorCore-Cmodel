#pragma once
// =============================================================================
// tensor_core_cfg.h — TensorCore simulator runtime configuration declarations
// =============================================================================
#include "fp_types.h"

struct TensorCoreCfg {
    PrecisionType input_prec  = PREC_FP8_E4M3;
    PrecisionType output_prec = PREC_FP8_E4M3;
    RoundingMode  rm          = RNE;
};

uint32_t convert_fp22_to_output_bits(uint32_t fp22, PrecisionType output_prec, RoundingMode rm);
double output_bits_to_double(uint32_t bits, PrecisionType output_prec);
