#pragma once

#include "../fp_types.h"
#include <cstdint>

namespace otc {

uint16_t convert_input_to_fp9(uint32_t raw, PrecisionType prec);
uint32_t convert_bias_to_fp22(uint32_t raw, PrecisionType prec);

} // namespace otc
