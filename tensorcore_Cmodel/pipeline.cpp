#include "pipeline.h"

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
        // fp9 multiply (e5m3)
        double p = in.a[k] * in.b[k];
        uint16_t p9_bits = SoftFloat::f64_to_fp9(p);
        double p9 = SoftFloat::fp9_to_f64(p9_bits);

        // accumulate in 2x fp9 precision: fp13 (e5m7)
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

    // reduce dot product back to fp9
    uint16_t dot9_bits = SoftFloat::f64_to_fp9(products[0]);
    double dot9 = SoftFloat::fp9_to_f64(dot9_bits);

    // final accumulation with C in fp22
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

    output_d_.resize(cfg_.M * cfg_.N, 0.0);

    DT.init(cfg_.debug_level, cfg_.trace_en);
    DT.log(1, "OTC init: D[%dx%d] = A[%dx%d] x B[%dx%d] + C", cfg_.M, cfg_.N, cfg_.M, cfg_.K, cfg_.K, cfg_.N);
    DT.log(1, "  pipeline depth=%d, dp_units=%d", cfg_.pipeline_depth(), cfg_.total_dp());
    DT.log(1, "  HW resources: %d muls, %d add-tree adders, %d accumulators", cfg_.total_dp() * cfg_.K,
           cfg_.total_dp() * (cfg_.K - 1), cfg_.total_dp());
}

void TensorCoreUnit::reset() {
    state_ = IDLE;
    cycle_ = 0;
    phase_timer_ = 0;
    dispatch_idx_ = 0;
    results_collected_ = 0;
    stats_ = {};
    for (auto& dp : dp_units_) {
        dp.reset();
    }
}

void TensorCoreUnit::load(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, const std::vector<uint32_t>& c) {
    raw_a_ = a;
    raw_b_ = b;
    raw_c_ = c;
}

void TensorCoreUnit::start() {
    assert(state_ == IDLE || state_ == DONE);
    state_ = FORMAT_CONV;
    phase_timer_ = cfg_.conv_latency;
    dispatch_idx_ = 0;
    results_collected_ = 0;
    std::fill(output_d_.begin(), output_d_.end(), 0.0);
    for (auto& dp : dp_units_) {
        dp.reset();
    }

    DT.log(1, "START: begin format conversion (%d cycles)", phase_timer_);
}

void TensorCoreUnit::do_format_conversion() {
    int eb = FPConvert::elem_bits(cfg_.type_ab);
    int elems_per_word = 32 / eb;

    int total_a = cfg_.M * cfg_.K;
    conv_a_.resize(total_a);
    for (int i = 0; i < total_a; i++) {
        int wi = i / elems_per_word;
        int ei = i % elems_per_word;
        uint32_t w = (wi < (int)raw_a_.size()) ? raw_a_[wi] : 0;
        conv_a_[i] = FPConvert::elem_to_f64(w, ei, cfg_.type_ab, cfg_.type_ab_sub);
    }

    int total_b = cfg_.K * cfg_.N;
    conv_b_.resize(total_b);
    for (int i = 0; i < total_b; i++) {
        int wi = i / elems_per_word;
        int ei = i % elems_per_word;
        uint32_t w = (wi < (int)raw_b_.size()) ? raw_b_[wi] : 0;
        conv_b_[i] = FPConvert::elem_to_f64(w, ei, cfg_.type_ab, cfg_.type_ab_sub);
    }

    int total_c = cfg_.M * cfg_.N;
    conv_c_.resize(total_c, 0.0);
    for (int i = 0; i < total_c; i++) {
        int wi = i / 2;
        int ei = i % 2;
        uint32_t w = (wi < (int)raw_c_.size()) ? raw_c_[wi] : 0;
        uint16_t half = (w >> (ei * 16)) & 0xFFFF;
        double c_f64 = SoftFloat::fp16_to_f64(half);
        uint32_t c_fp22 = SoftFloat::f64_to_fp22(c_f64);
        conv_c_[i] = SoftFloat::fp22_to_f64(c_fp22);
    }

    DT.log(2, "FmtConv done: A[0]=%f A[1]=%f B[0]=%f C[0]=%f", conv_a_[0], total_a > 1 ? conv_a_[1] : 0.0, conv_b_[0],
           conv_c_[0]);
}

void TensorCoreUnit::dispatch_all(OTC_Stats& stats) {
    for (int i = 0; i < cfg_.M; i++) {
        for (int j = 0; j < cfg_.N; j++) {
            DPInput in;
            in.a.resize(cfg_.K);
            in.b.resize(cfg_.K);
            in.row = i;
            in.col = j;

            for (int k = 0; k < cfg_.K; k++) {
                in.a[k] = conv_a_[i * cfg_.K + k];
            }

            for (int k = 0; k < cfg_.K; k++) {
                if (cfg_.transpose_b) {
                    in.b[k] = conv_b_[j * cfg_.K + k];
                } else {
                    in.b[k] = conv_b_[k * cfg_.N + j];
                }
            }

            in.c = conv_c_[i * cfg_.N + j];

            int dp_idx = i * cfg_.N + j;
            dp_units_[dp_idx].push(in, stats);

            DT.log(3, "  DISPATCH dp[%d][%d]: a[0]=%f b[0]=%f c=%f", i, j, in.a[0], in.b[0], in.c);
        }
    }
}

void TensorCoreUnit::tick() {
    cycle_++;
    stats_.total_cycles++;
    DT.set_cycle(cycle_);

    switch (state_) {
        case IDLE:
            break;

        case FORMAT_CONV:
            stats_.busy_cycles++;
            stats_.conv_cycles++;
            phase_timer_--;
            if (phase_timer_ <= 0) {
                do_format_conversion();
                state_ = DISPATCH;
                DT.log(1, "FORMAT_CONV → DISPATCH");
            }
            break;

        case DISPATCH:
            stats_.busy_cycles++;
            dispatch_all(stats_);
            state_ = DRAIN;
            DT.log(1, "DISPATCH → DRAIN (all %d DPs launched)", cfg_.total_dp());
            break;

        case DRAIN: {
            stats_.busy_cycles++;
            bool any_busy = false;
            for (auto& dp : dp_units_) {
                dp.tick();
                if (dp.output_valid_) {
                    int idx = dp.output_.row * cfg_.N + dp.output_.col;
                    output_d_[idx] = dp.output_.value;
                    results_collected_++;
                    DT.log(2, "  RESULT dp[%d][%d] = %f (collected %d/%d)", dp.output_.row, dp.output_.col, dp.output_.value,
                           results_collected_, cfg_.total_dp());
                }
                if (dp.busy()) {
                    any_busy = true;
                }
            }

            if (results_collected_ >= cfg_.total_dp()) {
                state_ = DONE;
                stats_.matrices_done++;
                DT.log(1, "DRAIN → DONE in %lu cycles (pipeline=%d)", cycle_, cfg_.pipeline_depth());
            } else if (!any_busy && results_collected_ < cfg_.total_dp()) {
                DT.log(1, "WARNING: no busy DPs but only %d/%d results", results_collected_, cfg_.total_dp());
                state_ = DONE;
            }
            break;
        }

        case DONE:
            break;
    }
}

uint64_t TensorCoreUnit::run(int max_cycles) {
    start();
    while (state_ != DONE && (int)cycle_ < max_cycles) {
        tick();
    }
    return cycle_;
}

bool TensorCoreUnit::is_done() const { return state_ == DONE; }

bool TensorCoreUnit::is_busy() const { return state_ != IDLE && state_ != DONE; }

std::vector<double> TensorCoreUnit::get_result_f64() const { return output_d_; }

std::vector<uint16_t> TensorCoreUnit::get_result_fp16() const {
    std::vector<uint16_t> out(output_d_.size());
    for (size_t i = 0; i < output_d_.size(); i++) {
        out[i] = SoftFloat::f64_to_fp16(output_d_[i]);
    }
    return out;
}

std::vector<uint32_t> TensorCoreUnit::get_result_fp32() const {
    std::vector<uint32_t> out(output_d_.size());
    for (size_t i = 0; i < output_d_.size(); i++) {
        out[i] = SoftFloat::f64_to_fp32(output_d_[i]);
    }
    return out;
}
