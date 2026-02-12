#pragma once
#include <deque>

#include "otc_ac_float.h"
#include "otc_fp.h"
#include "otc_types.h"

// ---------------------------
// Batch payloads (FIFO flits)
// ---------------------------
struct BatchJob {
    int batch_id = -1;
    std::vector<uint32_t> raw_a;
    std::vector<uint32_t> raw_b;
    std::vector<uint32_t> raw_c;
};

struct BatchWork {
    int batch_id = -1;
    std::vector<double> a_f64;
    std::vector<double> b_f64;
    std::vector<double> c_f64;
};

struct BatchResult {
    int batch_id = -1;
    std::vector<double> d_f64;
    uint64_t start_cycle = 0;
    uint64_t done_cycle = 0;
};

// Per-dot-product input/output packet.
struct DPInput {
    std::vector<double> a_f64;
    std::vector<double> b_f64;
    double c_f64;
    int row;
    int col;
};

struct DPResult {
    double value_f64;
    int row;
    int col;
};

class DotProductUnit {
public:
    const OTC_Config* cfg_ = nullptr;
    int latency_total_ = 0;

    struct PipeEntry {
        DPResult result;
        int latency_countdown;
        bool entry_valid;
    };

    std::vector<PipeEntry> pipe_q_;
    DPResult output_data_;
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
        bool batch_valid = false;
        int batch_id = -1;
        std::vector<double> a_f64;
        std::vector<double> b_f64;
        std::vector<double> c_f64;
        std::vector<double> d_f64;
        int dispatch_ptr = 0;
        int results_collected = 0;
        uint64_t start_cycle = 0;
    } active_batch_;

    int next_batch_id_ = 0;
    int dp_busy_acc_cycles_ = 0;

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
