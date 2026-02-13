#include "test.h"
#include "../otc_driver/otc_driver.h"
#include "../fp_types.h"
#include <cstdio>

namespace otc {

int run_smoke_test() {
    uint32_t out[8][8] = {};
    run_identity_case(PREC_FP16, out);

    int non_zero = 0;
    for (auto & row : out) {
        for (auto cell : row) {
            non_zero += (cell != 0);
        }
    }

    std::printf("[test] identity case completed, non_zero=%d\n", non_zero);
    return non_zero > 0 ? 0 : 1;
}

} // namespace otc
