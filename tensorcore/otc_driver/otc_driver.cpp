#include "otc_driver.h"
#include "../pre_conv/pre_conv.h"
#include "../pipeline/pipeline.h"

namespace otc {

void run_identity_case(PrecisionType prec, uint32_t out[8][8]) {
    Pipeline pipeline;

    uint16_t a[8][8] = {};
    uint16_t b[8][8] = {};
    uint32_t c[8][8] = {};

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            const uint32_t a_raw = (i == j) ? double_to_fp16(1.0) : double_to_fp16(0.0);
            const uint32_t b_raw = double_to_fp16(static_cast<double>(i * 8 + j));
            a[i][j] = convert_input_to_fp9(a_raw, prec);
            b[i][j] = convert_input_to_fp9(b_raw, prec);
            c[i][j] = convert_bias_to_fp22(double_to_fp16(0.0), prec);
        }
    }

    pipeline.sim().load_inputs(a, b, c, PREC_FP16, RNE);
    pipeline.sim().run_to_completion();

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            out[i][j] = pipeline.sim().d_out[i][j];
        }
    }
}

} // namespace otc
