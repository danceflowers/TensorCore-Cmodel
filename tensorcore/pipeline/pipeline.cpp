#include "pipeline.h"

namespace otc {

Pipeline::Pipeline() : sim_() {}

void Pipeline::step(bool) {
    sim_.tick();
}

TensorCoreSim& Pipeline::sim() {
    return sim_;
}

} // namespace otc
