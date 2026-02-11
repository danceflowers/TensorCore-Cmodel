// ============================================================================
// OpenTensorCore SimX — Cycle-Level Pipeline Model
//
// RTL Module Mapping:
//   PipeStage<T>    → elastic pipeline register (valid/ready/stall)
//   DotProductUnit  → tc_dot_product.v (K muls + log2K add-tree + final_add)
//   TensorCoreUnit  → tensor_core.v (format conversion + MxN DP array)
//
// Modeling strategy follows Vortex SimX cycle-approximate style.
// ============================================================================
#pragma once
#include "otc_fp.h"
#include "otc_types.h"

template <typename T>
class PipeStage {
public:
    int latency_;
    int timer_;
    T data_;
    bool occupied_;
    bool output_ready_;

    explicit PipeStage(int lat = 1) : latency_(lat), timer_(0), occupied_(false), output_ready_(false) {}

    void reset() {
        timer_ = 0;
        occupied_ = false;
        output_ready_ = false;
    }

    bool can_accept() const { return !occupied_; }

    void push(const T& d) {
        assert(!occupied_);
        data_ = d;
        timer_ = latency_;
        occupied_ = true;
        output_ready_ = false;
    }

    bool tick(bool downstream_stall = false) {
        if (!occupied_) return false;
        if (output_ready_ || downstream_stall) return false;
        timer_--;
        if (timer_ <= 0) {
            output_ready_ = true;
            return true;
        }
        return false;
    }

    T pop() {
        assert(output_ready_);
        output_ready_ = false;
        occupied_ = false;
        return data_;
    }

    bool has_output() const { return output_ready_; }
};

struct DPInput {
    std::vector<double> a;
    std::vector<double> b;
    double c;
    int row, col;
};

struct DPResult {
    double value;
    int row, col;
};

class DotProductUnit {
public:
    const OTC_Config* cfg_ = nullptr;
    int total_latency_ = 0;

    struct InFlight {
        DPResult result;
        int remaining;
        bool valid;
    };

    std::vector<InFlight> pipe_;
    DPResult output_;
    bool output_valid_ = false;

    void init(const OTC_Config* cfg);
    void reset();
    bool can_accept() const;
    void push(const DPInput& in, OTC_Stats& stats);
    void tick();
    bool busy() const;
};

class TensorCoreUnit {
public:
    OTC_Config cfg_;
    OTC_Stats stats_;
    std::vector<DotProductUnit> dp_units_;

    enum State { IDLE, FORMAT_CONV, DISPATCH, DRAIN, DONE };
    State state_ = IDLE;
    int phase_timer_ = 0;
    int dispatch_idx_ = 0;
    uint64_t cycle_ = 0;

    std::vector<uint32_t> raw_a_, raw_b_, raw_c_;
    std::vector<double> conv_a_, conv_b_, conv_c_;
    std::vector<double> output_d_;
    int results_collected_ = 0;

    void init(const OTC_Config& cfg);
    void reset();
    void load(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, const std::vector<uint32_t>& c);
    void start();
    void do_format_conversion();
    void dispatch_all(OTC_Stats& stats);
    void tick();
    uint64_t run(int max_cycles = 100000);
    bool is_done() const;
    bool is_busy() const;
    std::vector<double> get_result_f64() const;
    std::vector<uint16_t> get_result_fp16() const;
    std::vector<uint32_t> get_result_fp32() const;
};
