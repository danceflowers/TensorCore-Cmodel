#pragma once
#include <deque>
#include "otc_fp.h"
#include "otc_types.h"

struct BatchJob {
    int id = -1;
    std::vector<uint32_t> raw_a;
    std::vector<uint32_t> raw_b;
    std::vector<uint32_t> raw_c;
};

struct BatchWork {
    int id = -1;
    std::vector<double> conv_a;
    std::vector<double> conv_b;
    std::vector<double> conv_c;
};

struct BatchResult {
    int id = -1;
    std::vector<double> d;
    uint64_t start_cycle = 0;
    uint64_t done_cycle = 0;
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

    enum State { IDLE, RUNNING, DRAIN, DONE };
    State state_ = IDLE;
    uint64_t cycle_ = 0;

    std::deque<BatchJob> input_fifo_;
    std::deque<BatchWork> format_fifo_;
    std::deque<BatchResult> output_fifo_;

    struct ActiveBatch {
        bool valid = false;
        int id = -1;
        std::vector<double> conv_a;
        std::vector<double> conv_b;
        std::vector<double> conv_c;
        std::vector<double> output_d;
        int dispatch_idx = 0;
        int results_collected = 0;
        uint64_t start_cycle = 0;
    } active_;

    int next_batch_id_ = 0;
    int total_dp_busy_ = 0;

    std::vector<double> last_output_d_;

    void init(const OTC_Config& cfg);
    void reset();
    void load(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, const std::vector<uint32_t>& c);
    bool enqueue_job(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, const std::vector<uint32_t>& c);
    void start();
    void do_format_conversion_stage();
    bool load_active_from_format();
    void dispatch_some(OTC_Stats& stats);
    void collect_results();
    bool push_output_result(const BatchResult& br);
    bool pop_output_result(BatchResult& br);
    bool can_accept_job() const;
    bool has_pending_work() const;
    void tick();
    uint64_t run(int max_cycles = 100000);
    bool is_done() const;
    bool is_busy() const;
    std::vector<double> get_result_f64() const;
    std::vector<uint16_t> get_result_fp16() const;
    std::vector<uint32_t> get_result_fp32() const;
};
