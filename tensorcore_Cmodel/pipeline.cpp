#include "pipeline.h"

namespace {

using fp8_e4m3_t = ac_std_float<8, 4>;
using fp8_e5m2_t = ac_std_float<8, 5>;
using fp9_t = ac_std_float<9, 5>;
using fp13_t = ac_std_float<13, 5>;
using fp16_t = ac_std_float<16, 5>;
using fp22_t = ac_std_float<22, 8>;
using fp32_t = ac_std_float<32, 8>;

// Architect note:
// Quantization is modeled as an explicit type-conversion boundary,
// equivalent to register format truncation at a hardware interface.
inline double quantize_output_f64(double in_f64, const OTC_Config& cfg) {
    const fp22_t acc_val(in_f64);

    if (cfg.type_cd == TYPE_FP32) {
        const fp32_t out = acc_val.template convert<32, 8, AC_RND_CONV>();
        return out.to_double();
    }
    if (cfg.type_cd == TYPE_FP16) {
        const fp16_t out = acc_val.template convert<16, 5, AC_RND_CONV>();
        return out.to_double();
    }
    if (cfg.type_cd == TYPE_FP8) {
        if (cfg.type_cd_sub == SUB_FP8E4M3) {
            const fp8_e4m3_t out = acc_val.template convert<8, 4, AC_RND_CONV>();
            return out.to_double();
        }
        const fp8_e5m2_t out = acc_val.template convert<8, 5, AC_RND_CONV>();
        return out.to_double();
    }

    return acc_val.to_double();
}

}  // namespace

void DotProductUnit::init(const OTC_Config* cfg) {
    cfg_ = cfg;
    latency_total_ = cfg->mul_latency + cfg->tree_depth() * cfg->add_latency + cfg->add_latency + 1;
}

void DotProductUnit::reset() {
    pipe_q_.clear();
    output_valid_ = false;
}

bool DotProductUnit::can_accept() const { return true; }

void DotProductUnit::push(const DPInput& in, OTC_Stats& stats) {
    // Architect note:
    // We intentionally keep every arithmetic stage in explicit reduced precision
    // to mimic datapath truncation between pipeline stages:
    //   mul -> fp9, tree -> fp13, dot export -> fp9, accum -> fp22.

    std::vector<fp13_t> tree_vals(cfg_->K);

    for (int k = 0; k < cfg_->K; ++k) {
        const fp9_t a_fp9(in.a_f64[k]);
        const fp9_t b_fp9(in.b_f64[k]);
        const fp9_t p_fp9 = a_fp9 * b_fp9;
        tree_vals[k] = p_fp9.template convert<13, 5, AC_RND_CONV>();
        stats.mul_ops++;
    }

    int level_width = cfg_->K;
    while (level_width > 1) {
        for (int i = 0; i < level_width / 2; ++i) {
            tree_vals[i] = tree_vals[2 * i] + tree_vals[2 * i + 1];
            stats.add_ops++;
        }
        level_width >>= 1;
    }

    const fp9_t dot_fp9 = tree_vals[0].template convert<9, 5, AC_RND_CONV>();
    const fp22_t c_fp22(in.c_f64);
    const fp22_t out_fp22 = dot_fp9.template convert<22, 8, AC_RND_CONV>() + c_fp22;
    stats.add_ops++;

    PipeEntry pipe_entry;
    pipe_entry.result = {out_fp22.to_double(), in.row, in.col};
    pipe_entry.latency_countdown = latency_total_;
    pipe_entry.entry_valid = true;
    pipe_q_.push_back(pipe_entry);
}

void DotProductUnit::tick() {
    output_valid_ = false;
    for (auto it = pipe_q_.begin(); it != pipe_q_.end();) {
        it->latency_countdown--;
        if (it->latency_countdown <= 0) {
            output_data_ = it->result;
            output_valid_ = true;
            it = pipe_q_.erase(it);
        } else {
            ++it;
        }
    }
}

bool DotProductUnit::busy() const { return !pipe_q_.empty(); }

void TensorCoreUnit::init(const OTC_Config& cfg) {
    cfg_ = cfg;
    assert(cfg_.validate());

    dp_units_.resize(cfg_.M * cfg_.N);
    for (auto& dp : dp_units_) {
        dp.init(&cfg_);
    }

    last_output_d_.resize(cfg_.M * cfg_.N, 0.0);

    stats_.dp_capacity_units = cfg_.total_dp();
    stats_.peak_bw_bytes_per_cycle = cfg_.mem_bandwidth_bytes_per_cycle;

    DT.init(cfg_.debug_level, cfg_.trace_en);
    DT.log(1, "OTC init: D[%dx%d] = A[%dx%d] x B[%dx%d] + C", cfg_.M, cfg_.N, cfg_.M, cfg_.K, cfg_.K, cfg_.N);
    DT.log(1, "  pipeline depth=%d, dp_units=%d, dispatch_width=%d", cfg_.pipeline_depth(), cfg_.total_dp(),
           cfg_.dispatch_width);
}

void TensorCoreUnit::reset() {
    state_ = IDLE;
    cycle_ = 0;
    stats_ = {};
    stats_.dp_capacity_units = cfg_.total_dp();
    stats_.peak_bw_bytes_per_cycle = cfg_.mem_bandwidth_bytes_per_cycle;

    output_fifo_.clear();
    active_batch_ = {};

    next_batch_id_ = 0;
    dp_busy_acc_cycles_ = 0;

    for (auto& dp : dp_units_) {
        dp.reset();
    }
    std::fill(last_output_d_.begin(), last_output_d_.end(), 0.0);
}

bool TensorCoreUnit::try_start_batch(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b,
                                    const std::vector<uint32_t>& c) {
    if (active_batch_.batch_valid) {
        return false;
    }

    active_batch_.batch_valid = true;
    active_batch_.batch_id = next_batch_id_++;
    active_batch_.a_f64.resize(cfg_.M * cfg_.K);
    active_batch_.b_f64.resize(cfg_.K * cfg_.N);
    active_batch_.c_f64.resize(cfg_.M * cfg_.N, 0.0);
    active_batch_.d_f64.assign(cfg_.M * cfg_.N, 0.0);
    active_batch_.dispatch_ptr = 0;
    active_batch_.results_collected = 0;
    active_batch_.start_cycle = cycle_;

    const int elem_bits = FPConvert::elem_bits(cfg_.type_ab);
    const int elems_per_word = 32 / elem_bits;

    for (int i = 0; i < cfg_.M * cfg_.K; ++i) {
        const int wi = i / elems_per_word;
        const int ei = i % elems_per_word;
        const uint32_t w = (wi < (int)a.size()) ? a[wi] : 0;
        active_batch_.a_f64[i] = FPConvert::elem_to_f64(w, ei, cfg_.type_ab, cfg_.type_ab_sub);
    }

    for (int i = 0; i < cfg_.K * cfg_.N; ++i) {
        const int wi = i / elems_per_word;
        const int ei = i % elems_per_word;
        const uint32_t w = (wi < (int)b.size()) ? b[wi] : 0;
        active_batch_.b_f64[i] = FPConvert::elem_to_f64(w, ei, cfg_.type_ab, cfg_.type_ab_sub);
    }

    for (int i = 0; i < cfg_.M * cfg_.N; ++i) {
        const int wi = i / 2;
        const int ei = i % 2;
        const uint32_t w = (wi < (int)c.size()) ? c[wi] : 0;
        const uint16_t half = (w >> (ei * 16)) & 0xFFFF;
        active_batch_.c_f64[i] = SoftFloat::fp22_to_f64(SoftFloat::f64_to_fp22(SoftFloat::fp16_to_f64(half)));
    }

    stats_.dram_read_bytes += (uint64_t)(a.size() + b.size() + c.size()) * 4;
    stats_.batches_enqueued++;
    stats_.format_active_cycles++;
    DT.log(2, "ACTIVATE batch#%d", active_batch_.batch_id);
    return true;
}

bool TensorCoreUnit::enqueue_job(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b,
                                 const std::vector<uint32_t>& c) {
    return try_start_batch(a, b, c);
}

void TensorCoreUnit::load(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, const std::vector<uint32_t>& c) {
    if (!enqueue_job(a, b, c)) {
        DT.log(1, "WARNING: core busy, batch dropped (only output FIFO is buffered)");
    }
}

void TensorCoreUnit::start() {
    assert(state_ == IDLE || state_ == DONE);
    state_ = RUNNING;
    DT.log(1, "START: cycle-stepping pipeline enabled");
}

void TensorCoreUnit::dispatch_some(OTC_Stats& stats) {
    if (!active_batch_.batch_valid) {
        return;
    }

    int launch_budget = cfg_.dispatch_width;
    int launch_count = 0;

    while (launch_budget > 0 && active_batch_.dispatch_ptr < cfg_.total_dp()) {
        const int dp_index = active_batch_.dispatch_ptr;
        const int row = dp_index / cfg_.N;
        const int col = dp_index % cfg_.N;

        DPInput dp_in;
        dp_in.a_f64.resize(cfg_.K);
        dp_in.b_f64.resize(cfg_.K);
        dp_in.row = row;
        dp_in.col = col;

        for (int k = 0; k < cfg_.K; ++k) {
            dp_in.a_f64[k] = active_batch_.a_f64[row * cfg_.K + k];
            if (cfg_.transpose_b) {
                dp_in.b_f64[k] = active_batch_.b_f64[col * cfg_.K + k];
            } else {
                dp_in.b_f64[k] = active_batch_.b_f64[k * cfg_.N + col];
            }
        }
        dp_in.c_f64 = active_batch_.c_f64[row * cfg_.N + col];

        dp_units_[dp_index].push(dp_in, stats);
        active_batch_.dispatch_ptr++;
        launch_budget--;
        launch_count++;
    }

    stats_.dp_issue_slots += cfg_.dispatch_width;
    if (launch_count > 0) {
        stats_.dispatch_active_cycles++;
    }
}

void TensorCoreUnit::collect_results() {
    int dp_busy_units = 0;

    for (auto& dp : dp_units_) {
        dp.tick();
        if (dp.output_valid_ && active_batch_.batch_valid) {
            const int out_idx = dp.output_data_.row * cfg_.N + dp.output_data_.col;
            active_batch_.d_f64[out_idx] = quantize_output_f64(dp.output_data_.value_f64, cfg_);
            active_batch_.results_collected++;
        }
        if (dp.busy()) {
            dp_busy_units++;
        }
    }

    dp_busy_acc_cycles_ += dp_busy_units;

    if (active_batch_.batch_valid && active_batch_.results_collected >= cfg_.total_dp()) {
        BatchResult br;
        br.batch_id = active_batch_.batch_id;
        br.d_f64 = active_batch_.d_f64;
        br.start_cycle = active_batch_.start_cycle;
        br.done_cycle = cycle_;

        if (push_output_result(br)) {
            stats_.matrices_done++;
            stats_.total_latency_cycles += (br.done_cycle - br.start_cycle + 1);
            last_output_d_ = br.d_f64;
            active_batch_ = {};
        }
    }
}

bool TensorCoreUnit::push_output_result(const BatchResult& br) {
    if ((int)output_fifo_.size() >= cfg_.output_fifo_depth) {
        stats_.output_backpressure_cycles++;
        return false;
    }

    output_fifo_.push_back(br);
    stats_.output_fifo_max_occupancy = std::max(stats_.output_fifo_max_occupancy, (uint64_t)output_fifo_.size());
    stats_.dram_write_bytes += (uint64_t)br.d_f64.size() * 4;
    return true;
}

bool TensorCoreUnit::pop_output_result(BatchResult& br) {
    if (output_fifo_.empty()) {
        return false;
    }
    br = output_fifo_.front();
    output_fifo_.pop_front();
    return true;
}

bool TensorCoreUnit::can_accept_job() const { return !active_batch_.batch_valid; }

bool TensorCoreUnit::has_pending_work() const {
    if (active_batch_.batch_valid) return true;
    for (const auto& dp : dp_units_) {
        if (dp.busy()) return true;
    }
    return false;
}

void TensorCoreUnit::tick() {
    cycle_++;
    stats_.total_cycles++;
    DT.set_cycle(cycle_);

    if (state_ == IDLE || state_ == DONE) {
        return;
    }

    bool stage_enable = false;

    if (active_batch_.batch_valid) {
        stage_enable = true;
        dispatch_some(stats_);
    }

    collect_results();

    if (stage_enable) {
        stats_.busy_cycles++;
    } else {
        stats_.stall_cycles++;
    }

    stats_.dp_busy_unit_cycles = dp_busy_acc_cycles_;

    if (!has_pending_work()) {
        state_ = DONE;
    }
}

uint64_t TensorCoreUnit::run(int max_cycles) {
    if (state_ == IDLE || state_ == DONE) {
        start();
    }
    while (state_ != DONE && (int)cycle_ < max_cycles) {
        tick();
    }
    return cycle_;
}

bool TensorCoreUnit::is_done() const { return state_ == DONE; }

bool TensorCoreUnit::is_busy() const { return state_ != IDLE && state_ != DONE; }

std::vector<double> TensorCoreUnit::get_result_f64() const {
    if (!output_fifo_.empty()) {
        return output_fifo_.front().d_f64;
    }
    return last_output_d_;
}

std::vector<uint16_t> TensorCoreUnit::get_result_fp16() const {
    std::vector<double> src = get_result_f64();
    std::vector<uint16_t> out(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        out[i] = SoftFloat::f64_to_fp16(src[i]);
    }
    return out;
}

std::vector<uint32_t> TensorCoreUnit::get_result_fp32() const {
    std::vector<double> src = get_result_f64();
    std::vector<uint32_t> out(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        out[i] = SoftFloat::f64_to_fp32(src[i]);
    }
    return out;
}
