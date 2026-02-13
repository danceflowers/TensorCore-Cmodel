#pragma once

#include "../tensor_core_sim.h"

namespace otc {

class Pipeline {
public:
    Pipeline();
    void step(bool valid);
    TensorCoreSim& sim();

private:
    TensorCoreSim sim_;
};

} // namespace otc
