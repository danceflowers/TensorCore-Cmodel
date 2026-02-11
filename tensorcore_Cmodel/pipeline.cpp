#include "pipeline.h"

namespace {

double quantize_output(double v, const OTC_Config& cfg) {
    if (cfg.type_cd == TYPE_FP32) {
        return SoftFloat::fp32_to_f64(SoftFloat::f64_to_fp32(v));
    }
    if (cfg.type_cd == TYPE_FP16) {
        return SoftFloat::fp16_to_f64(SoftFloat::f64_to_fp16(v));
    }
    if (cfg.type_cd == TYPE_FP8) {
        uint8_t fp8 = (cfg.type_cd_sub == SUB_FP8E4M3) ? FPConvert::f64_to_fp8e4m3(v) : FPConvert::f64_to_fp8e5m2(v);
        return (cfg.type_cd_sub == SUB_FP8E4M3) ? FPConvert::fp8e4m3_to_f64(fp8) : FPConvert::fp8e5m2_to_f64(fp8);
    }
    return v;
}

}  // namespace


void DotProductUnit::init(const OTC_Config* cfg) {
    cfg_ = cfg;
    total_latency_ = cfg->mul_latency + cfg->tree_depth() * cfg->add_latency + cfg->add_latency + 1;
}

void DotProductUnit::reset() {
    pipe_.clear();
    output_valid_ = false;
}

bool DotProductUnit::can_accept() const { return true; }

void DotProductUnit::push(const DPInput& in, OTC_Stats& stats) {
    double products[64];
    for (int k = 0; k < cfg_->K; k++) {
        double p = in.a[k] * in.b[k];
        uint16_t p9_bits = SoftFloat::f64_to_fp9(p);
        double p9 = SoftFloat::fp9_to_f64(p9_bits);
        uint16_t p13_bits = SoftFloat::f64_to_fp13(p9);
        products[k] = SoftFloat::fp13_to_f64(p13_bits);
        stats.mul_ops++;
    }

    int n = cfg_->K;
    while (n > 1) {
        for (int i = 0; i < n / 2; i++) {
            double s = products[2 * i] + products[2 * i + 1];
            uint16_t s13_bits = SoftFloat::f64_to_fp13(s);
            products[i] = SoftFloat::fp13_to_f64(s13_bits);
            stats.add_ops++;
        }
        n /= 2;
    }

    uint16_t dot9_bits = SoftFloat::f64_to_fp9(products[0]);
    double dot9 = SoftFloat::fp9_to_f64(dot9_bits);

    double sum_with_c = dot9 + in.c;
    uint32_t fp22_bits = SoftFloat::f64_to_fp22(sum_with_c);
    double fp22_val = SoftFloat::fp22_to_f64(fp22_bits);
    stats.add_ops++;

    InFlight entry;
    entry.result = {fp22_val, in.row, in.col};
    entry.remaining = total_latency_;
    entry.valid = true;
    pipe_.push_back(entry);
}

void DotProductUnit::tick() {
    output_valid_ = false;
    for (auto it = pipe_.begin(); it != pipe_.end();) {
        it->remaining--;
        if (it->remaining <= 0) {
            output_ = it->result;
            output_valid_ = true;
            it = pipe_.erase(it);
        } else {
            ++it;
        }
    }
}

bool DotProductUnit::busy() const { return !pipe_.empty(); }

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
    DT.log(1, "  pipeline depth=%d, dp_units=%d, dispatch_width=%d", cfg_.pipeline_depth(), cfg_.total_dp(), cfg_.dispatch_width);
}

void TensorCoreUnit::reset() {
    state_ = IDLE;
    cycle_ = 0;
    stats_ = {};
    stats_.dp_capacity_units = cfg_.total_dp();
    stats_.peak_bw_bytes_per_cycle = cfg_.mem_bandwidth_bytes_per_cycle;
    input_fifo_.clear();
    format_fifo_.clear();
    output_fifo_.clear();
    active_ = {};
    next_batch_id_ = 0;
    total_dp_busy_ = 0;
    for (auto& dp : dp_units_) {
        dp.reset();
    }
    std::fill(last_output_d_.begin(), last_output_d_.end(), 0.0);
}

bool TensorCoreUnit::enqueue_job(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, const std::vector<uint32_t>& c) {
    if ((int)input_fifo_.size() >= cfg_.input_fifo_depth) {
        return false;
    }
    BatchJob job;
    job.id = next_batch_id_++;
    job.raw_a = a;
    job.raw_b = b;
    job.raw_c = c;
    input_fifo_.push_back(job);
    stats_.batches_enqueued++;
    return true;
}

void TensorCoreUnit::load(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, const std::vector<uint32_t>& c) {
    bool ok = enqueue_job(a, b, c);
    if (!ok) {
        DT.log(1, "WARNING: input FIFO full (depth=%d), batch dropped", cfg_.input_fifo_depth);
    }
}

void TensorCoreUnit::start() {
    assert(state_ == IDLE || state_ == DONE);
    state_ = RUNNING;
    DT.log(1, "START: cycle-stepping pipeline enabled");
}

void TensorCoreUnit::do_format_conversion_stage() {
    if (input_fifo_.empty()) {
        return;
    }

    const BatchJob& job = input_fifo_.front();
    BatchWork work;
    work.id = job.id;

    int eb = FPConvert::elem_bits(cfg_.type_ab);
    int elems_per_word = 32 / eb;

    int total_a = cfg_.M * cfg_.K;
    work.conv_a.resize(total_a);
    for (int i = 0; i < total_a; i++) {
        int wi = i / elems_per_word;
        int ei = i % elems_per_word;
        uint32_t w = (wi < (int)job.raw_a.size()) ? job.raw_a[wi] : 0;
        work.conv_a[i] = FPConvert::elem_to_f64(w, ei, cfg_.type_ab, cfg_.type_ab_sub);
    }

    int total_b = cfg_.K * cfg_.N;
    work.conv_b.resize(total_b);
    for (int i = 0; i < total_b; i++) {
        int wi = i / elems_per_word;
        int ei = i % elems_per_word;
        uint32_t w = (wi < (int)job.raw_b.size()) ? job.raw_b[wi] : 0;
        work.conv_b[i] = FPConvert::elem_to_f64(w, ei, cfg_.type_ab, cfg_.type_ab_sub);
    }

    int total_c = cfg_.M * cfg_.N;
    work.conv_c.resize(total_c, 0.0);
    for (int i = 0; i < total_c; i++) {
        int wi = i / 2;
        int ei = i % 2;
        uint32_t w = (wi < (int)job.raw_c.size()) ? job.raw_c[wi] : 0;
        uint16_t half = (w >> (ei * 16)) & 0xFFFF;
        double c_f64 = SoftFloat::fp16_to_f64(half);
        uint32_t c_fp22 = SoftFloat::f64_to_fp22(c_f64);
        work.conv_c[i] = SoftFloat::fp22_to_f64(c_fp22);
    }

    uint64_t bytes_in = (uint64_t)(job.raw_a.size() + job.raw_b.size() + job.raw_c.size()) * 4;
    stats_.dram_read_bytes += bytes_in;

    format_fifo_.push_back(work);
    input_fifo_.pop_front();
    DT.log(2, "FORMAT batch#%d done", work.id);
}

bool TensorCoreUnit::load_active_from_format() {
    if (active_.valid || format_fifo_.empty()) {
        return false;
    }
    const BatchWork& work = format_fifo_.front();
    active_.valid = true;
    active_.id = work.id;
    active_.conv_a = work.conv_a;
    active_.conv_b = work.conv_b;
    active_.conv_c = work.conv_c;
    active_.output_d.assign(cfg_.M * cfg_.N, 0.0);
    active_.dispatch_idx = 0;
    active_.results_collected = 0;
    active_.start_cycle = cycle_;
    format_fifo_.pop_front();
    DT.log(2, "ACTIVATE batch#%d", active_.id);
    return true;
}

void TensorCoreUnit::dispatch_some(OTC_Stats& stats) {
    if (!active_.valid) {
        return;
    }

    int total = cfg_.total_dp();
    int launch_budget = cfg_.dispatch_width;
    int launched = 0;

    while (launch_budget > 0 && active_.dispatch_idx < total) {
        int idx = active_.dispatch_idx;
        int i = idx / cfg_.N;
        int j = idx % cfg_.N;

        DPInput in;
        in.a.resize(cfg_.K);
        in.b.resize(cfg_.K);
        in.row = i;
        in.col = j;

        for (int k = 0; k < cfg_.K; k++) {
            in.a[k] = active_.conv_a[i * cfg_.K + k];
        }

        for (int k = 0; k < cfg_.K; k++) {
            if (cfg_.transpose_b) {
                in.b[k] = active_.conv_b[j * cfg_.K + k];
            } else {
                in.b[k] = active_.conv_b[k * cfg_.N + j];
            }
        }

        in.c = active_.conv_c[i * cfg_.N + j];
        dp_units_[idx].push(in, stats);

        active_.dispatch_idx++;
        launch_budget--;
        launched++;
    }

    stats_.dp_issue_slots += cfg_.dispatch_width;
    if (launched > 0) {
        stats_.dispatch_active_cycles++;
    }
}

void TensorCoreUnit::collect_results() {
    int busy_units = 0;
    for (auto& dp : dp_units_) {
        dp.tick();
        if (dp.output_valid_ && active_.valid) {
            int idx = dp.output_.row * cfg_.N + dp.output_.col;
            active_.output_d[idx] = quantize_output(dp.output_.value, cfg_);
            active_.results_collected++;
        }
        if (dp.busy()) {
            busy_units++;
        }
    }
    total_dp_busy_ += busy_units;

    if (active_.valid && active_.results_collected >= cfg_.total_dp()) {
        BatchResult br;
        br.id = active_.id;
        br.d = active_.output_d;
        br.start_cycle = active_.start_cycle;
        br.done_cycle = cycle_;

        if (push_output_result(br)) {
            stats_.matrices_done++;
            stats_.total_latency_cycles += (br.done_cycle - br.start_cycle + 1);
            last_output_d_ = br.d;
            active_ = {};
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
    stats_.dram_write_bytes += (uint64_t)br.d.size() * 4;
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

bool TensorCoreUnit::can_accept_job() const { return (int)input_fifo_.size() < cfg_.input_fifo_depth; }

bool TensorCoreUnit::has_pending_work() const {
    if (!input_fifo_.empty() || !format_fifo_.empty()) {
        return true;
    }
    if (active_.valid) {
        return true;
    }
    for (const auto& dp : dp_units_) {
        if (dp.busy()) {
            return true;
        }
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

    bool did_work = false;

    if (!input_fifo_.empty()) {
        stats_.format_active_cycles++;
        did_work = true;
        do_format_conversion_stage();
    }

    if (load_active_from_format()) {
        did_work = true;
    }

    if (active_.valid) {
        did_work = true;
        dispatch_some(stats_);
    }

    collect_results();

    if (did_work) {
        stats_.busy_cycles++;
    } else {
        stats_.stall_cycles++;
        if (!can_accept_job()) {
            stats_.input_fifo_stall_cycles++;
        }
    }

    stats_.dp_busy_unit_cycles = total_dp_busy_;

    if (!has_pending_work()) {
        state_ = DRAIN;
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
        return output_fifo_.front().d;
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
