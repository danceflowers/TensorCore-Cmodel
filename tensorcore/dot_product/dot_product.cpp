#include "dot_product.h"
#include "../fp_arith.h"

namespace otc {

uint32_t dot_product_fp22(const uint16_t a[8], const uint16_t b[8]) {
    uint32_t acc = 0;
    for (int k = 0; k < 8; ++k) {
        const uint16_t mul = fp9_multiply(a[k], b[k], RNE);
        acc = fp22_add(acc, fp9_to_fp22(mul), RNE);
    }
    return acc;
}

} // namespace otc
