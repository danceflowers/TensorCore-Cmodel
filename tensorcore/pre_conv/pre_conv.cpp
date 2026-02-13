#include "pre_conv.h"

namespace otc {

uint16_t convert_input_to_fp9(uint32_t raw, PrecisionType prec) {
    return convert_to_fp9(raw, prec);
}

uint32_t convert_bias_to_fp22(uint32_t raw, PrecisionType prec) {
    return convert_c_to_fp22(raw, prec);
}

} // namespace otc
