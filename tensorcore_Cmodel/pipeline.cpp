#include "pipeline.h"

namespace {

struct BankMem {
    std::vector<uint16_t> data;
    void init(int n) { data.assign(n, 0); }
    void write(int idx, uint16_t v) { if (idx >= 0 && idx < (int)data.size()) data[idx] = v; }
    uint16_t read(int idx) const { return (idx >= 0 && idx < (int)data.size()) ? data[idx] : 0; }
};

inline double quantize_output_f64(uint32_t fp22, const OTC_Config& cfg) {
    if (cfg.type_cd == TYPE_FP32) return SoftFloat::fp32_to_f64(SoftFloat::f64_to_fp32(SoftFloat::fp22_to_f64(fp22)));
    if (cfg.type_cd == TYPE_FP16) return SoftFloat::fp16_to_f64(FPEmu::fp22_to_fp16(fp22));
    if (cfg.type_cd == TYPE_FP8) {
        uint8_t fp8 = (uint8_t)FPEmu::fp22_to_fp8(fp22, cfg.type_cd_sub);
        return cfg.type_cd_sub == SUB_FP8E4M3 ? FPConvert::fp8e4m3_to_f64(fp8) : FPConvert::fp8e5m2_to_f64(fp8);
    }
    return SoftFloat::fp22_to_f64(fp22);
}

}  // namespace

void DotProductUnit::init(const OTC_Config* cfg) { cfg_ = cfg; latency_total_ = 6; }
void DotProductUnit::reset() { pipe_q_.clear(); output_valid_ = false; }
bool DotProductUnit::can_accept() const { return (int)pipe_q_.size() < 8; }

void DotProductUnit::push(const DPInput& in, OTC_Stats& stats) {
    std::vector<uint16_t> tree_vals(cfg_->K);
    for (int k = 0; k < cfg_->K; ++k) {
        uint16_t p9 = FPEmu::fp9_mul(in.a_fp9[k], in.b_fp9[k]);
        uint16_t p13 = SoftFloat::f64_to_fp13(SoftFloat::fp9_to_f64(p9));
        tree_vals[k] = p13;
        stats.mul_ops++;
    }
    int w = cfg_->K;
    while (w > 1) {
        for (int i = 0; i < w / 2; ++i) {
            tree_vals[i] = FPEmu::fp13_add(tree_vals[2 * i], tree_vals[2 * i + 1]);
            stats.add_ops++;
        }
        w >>= 1;
    }
    uint16_t dot9 = FPEmu::fp13_to_fp9(tree_vals[0]);
    uint32_t out22 = FPEmu::fp22_add(FPEmu::fp9_to_fp22(dot9), in.c_fp22);
    stats.add_ops++;
    pipe_q_.push_back({{out22, in.row, in.col}, latency_total_, true});
}

void DotProductUnit::tick() {
    output_valid_ = false;
    for (auto it = pipe_q_.begin(); it != pipe_q_.end();) {
        it->latency_countdown--;
        if (it->latency_countdown <= 0) { output_data_ = it->result; output_valid_ = true; it = pipe_q_.erase(it); }
        else ++it;
    }
}

bool DotProductUnit::busy() const { return !pipe_q_.empty(); }

void TensorCoreUnit::init(const OTC_Config& cfg) {
    cfg_ = cfg; assert(cfg_.validate());
    dp_units_.resize(cfg_.M * cfg_.N);
    for (auto& dp : dp_units_) dp.init(&cfg_);
    last_output_d_.resize(cfg_.M * cfg_.N, 0.0);
    stats_.dp_capacity_units = cfg_.total_dp();
    stats_.peak_bw_bytes_per_cycle = cfg_.mem_bandwidth_bytes_per_cycle;
    DT.init(cfg_.debug_level, cfg_.trace_en);
}

void TensorCoreUnit::reset() {
    state_ = IDLE; cycle_ = 0; stats_ = {}; stats_.dp_capacity_units = cfg_.total_dp();
    stats_.peak_bw_bytes_per_cycle = cfg_.mem_bandwidth_bytes_per_cycle;
    output_fifo_.clear(); active_batch_ = {}; next_batch_id_ = 0; dp_busy_acc_cycles_ = 0;
    for (auto& dp : dp_units_) dp.reset();
    std::fill(last_output_d_.begin(), last_output_d_.end(), 0.0);
}

bool TensorCoreUnit::try_start_batch(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, const std::vector<uint32_t>& c) {
    if (active_batch_.batch_valid) return false;
    active_batch_.batch_valid = true; active_batch_.batch_id = next_batch_id_++;
    active_batch_.a_fp9.resize(cfg_.M * cfg_.K); active_batch_.b_fp9.resize(cfg_.K * cfg_.N);
    active_batch_.c_fp22.resize(cfg_.M * cfg_.N, 0); active_batch_.d_f64.assign(cfg_.M * cfg_.N, 0.0);
    active_batch_.dispatch_ptr = 0; active_batch_.results_collected = 0; active_batch_.start_cycle = cycle_;
    int eb = FPConvert::elem_bits(cfg_.type_ab), eperw = 32 / eb;
    for (int i = 0; i < cfg_.M * cfg_.K; ++i) {
        int wi = i / eperw, ei = i % eperw; uint32_t w = wi < (int)a.size() ? a[wi] : 0;
        if (cfg_.type_ab == TYPE_FP4) active_batch_.a_fp9[i] = FPEmu::fp4_to_fp9((w >> (ei * 4)) & 0xF);
        else if (cfg_.type_ab == TYPE_FP8) { uint8_t x = (w >> (ei * 8)) & 0xFF; active_batch_.a_fp9[i] = (cfg_.type_ab_sub == SUB_FP8E4M3) ? FPEmu::fp8e4m3_to_fp9(x) : FPEmu::fp8e5m2_to_fp9(x); }
        else active_batch_.a_fp9[i] = FPEmu::fp16_to_fp9((w >> (ei * 16)) & 0xFFFF);
    }
    for (int i = 0; i < cfg_.K * cfg_.N; ++i) {
        int wi = i / eperw, ei = i % eperw; uint32_t w = wi < (int)b.size() ? b[wi] : 0;
        if (cfg_.type_ab == TYPE_FP4) active_batch_.b_fp9[i] = FPEmu::fp4_to_fp9((w >> (ei * 4)) & 0xF);
        else if (cfg_.type_ab == TYPE_FP8) { uint8_t x = (w >> (ei * 8)) & 0xFF; active_batch_.b_fp9[i] = (cfg_.type_ab_sub == SUB_FP8E4M3) ? FPEmu::fp8e4m3_to_fp9(x) : FPEmu::fp8e5m2_to_fp9(x); }
        else active_batch_.b_fp9[i] = FPEmu::fp16_to_fp9((w >> (ei * 16)) & 0xFFFF);
    }
    for (int i = 0; i < cfg_.M * cfg_.N; ++i) {
        int wi = i / 2, ei = i % 2; uint16_t h = ((wi < (int)c.size() ? c[wi] : 0) >> (ei * 16)) & 0xFFFF;
        active_batch_.c_fp22[i] = SoftFloat::f64_to_fp22(SoftFloat::fp16_to_f64(h));
    }
    stats_.dram_read_bytes += (uint64_t)(a.size() + b.size() + c.size()) * 4; stats_.batches_enqueued++;
    return true;
}

bool TensorCoreUnit::enqueue_job(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, const std::vector<uint32_t>& c) { return try_start_batch(a, b, c); }
void TensorCoreUnit::load(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, const std::vector<uint32_t>& c) { (void)enqueue_job(a,b,c); }
void TensorCoreUnit::start() { state_ = RUNNING; }

void TensorCoreUnit::dispatch_some(OTC_Stats& stats) {
    if (!active_batch_.batch_valid) return;
    int budget = cfg_.dispatch_width;
    while (budget-- > 0 && active_batch_.dispatch_ptr < cfg_.total_dp()) {
        int dp_index = active_batch_.dispatch_ptr, row = dp_index / cfg_.N, col = dp_index % cfg_.N;
        DPInput in; in.a_fp9.resize(cfg_.K); in.b_fp9.resize(cfg_.K); in.row = row; in.col = col;
        for (int k = 0; k < cfg_.K; ++k) {
            in.a_fp9[k] = active_batch_.a_fp9[row * cfg_.K + k];
            in.b_fp9[k] = cfg_.transpose_b ? active_batch_.b_fp9[col * cfg_.K + k] : active_batch_.b_fp9[k * cfg_.N + col];
        }
        in.c_fp22 = active_batch_.c_fp22[row * cfg_.N + col];
        if (dp_units_[dp_index].can_accept()) { dp_units_[dp_index].push(in, stats); active_batch_.dispatch_ptr++; }
    }
    stats_.dp_issue_slots += cfg_.dispatch_width;
}

void TensorCoreUnit::collect_results() {
    int busy = 0;
    for (auto& dp : dp_units_) {
        dp.tick();
        if (dp.output_valid_ && active_batch_.batch_valid) {
            int out_idx = dp.output_data_.row * cfg_.N + dp.output_data_.col;
            active_batch_.d_f64[out_idx] = quantize_output_f64(dp.output_data_.value_fp22, cfg_);
            active_batch_.results_collected++;
        }
        if (dp.busy()) busy++;
    }
    dp_busy_acc_cycles_ += busy;
    if (active_batch_.batch_valid && active_batch_.results_collected >= cfg_.total_dp()) {
        BatchResult br{active_batch_.batch_id, active_batch_.d_f64, active_batch_.start_cycle, cycle_};
        if (push_output_result(br)) { stats_.matrices_done++; last_output_d_ = br.d_f64; active_batch_ = {}; }
    }
}

bool TensorCoreUnit::push_output_result(const BatchResult& br) {
    if ((int)output_fifo_.size() >= cfg_.output_fifo_depth) return false;
    output_fifo_.push_back(br); stats_.dram_write_bytes += (uint64_t)br.d_f64.size() * 4; return true;
}

bool TensorCoreUnit::pop_output_result(BatchResult& br) { if (output_fifo_.empty()) return false; br = output_fifo_.front(); output_fifo_.pop_front(); return true; }
bool TensorCoreUnit::can_accept_job() const { return !active_batch_.batch_valid; }
bool TensorCoreUnit::has_pending_work() const { if (active_batch_.batch_valid) return true; for (const auto& d: dp_units_) if (d.busy()) return true; return false; }
void TensorCoreUnit::tick() { cycle_++; stats_.total_cycles++; if (state_==IDLE||state_==DONE) return; dispatch_some(stats_); collect_results(); if (has_pending_work()) stats_.busy_cycles++; else state_=DONE; stats_.dp_busy_unit_cycles = dp_busy_acc_cycles_; }
uint64_t TensorCoreUnit::run(int max_cycles) { if (state_==IDLE||state_==DONE) start(); while(state_!=DONE && (int)cycle_<max_cycles) tick(); return cycle_; }
bool TensorCoreUnit::is_done() const { return state_==DONE; }
bool TensorCoreUnit::is_busy() const { return state_!=IDLE && state_!=DONE; }
std::vector<double> TensorCoreUnit::get_result_f64() const { return !output_fifo_.empty()?output_fifo_.front().d_f64:last_output_d_; }
std::vector<uint16_t> TensorCoreUnit::get_result_fp16() const { auto src=get_result_f64(); std::vector<uint16_t> out(src.size()); for(size_t i=0;i<src.size();++i) out[i]=SoftFloat::f64_to_fp16(src[i]); return out; }
std::vector<uint32_t> TensorCoreUnit::get_result_fp32() const { auto src=get_result_f64(); std::vector<uint32_t> out(src.size()); for(size_t i=0;i<src.size();++i) out[i]=SoftFloat::f64_to_fp32(src[i]); return out; }
