// ============================================================================
// OpenTensorCore SimX — Cycle-Level Pipeline Model
//
// RTL Module Mapping:
//   PipeStage<T>    →  elastic pipeline register (valid/ready/stall)
//   MulPipe         →  tc_mul_pipe.v (fmul_s1 + naivemul + fmul_s2 + fmul_s3)
//   AddPipe         →  tc_add_pipe.v (fadd_s1 + fadd_s2)
//   DotProductUnit  →  tc_dot_product.v (K muls + log2K add-tree + final_add)
//   TensorCoreUnit  →  tensor_core.v (to_fp8_con + mm_mul_add + stream_fifo)
//
// Modeling Strategy (Vortex SimX "cycle-approximate"):
//   Each pipeline stage is modeled as a FIFO with configurable latency.
//   Data enters, counts down, and exits after 'latency' cycles.
//   Back-pressure (stall) freezes the countdown.
//   This matches Vortex's approach: ~6% cycle accuracy vs RTL.
// ============================================================================
#pragma once
#include "otc_types.h"
#include "otc_fp.h"

// ==================== Generic Pipeline Stage ====================
// Models one elastic pipeline stage with N-cycle latency.
// Mirrors Vortex's VX_pipe_register / VX_elastic_buffer pattern.
template<typename T>
class PipeStage {
public:
    int  latency_;       // total cycles to traverse this stage
    int  timer_;         // remaining cycles (0 = output ready)
    T    data_;
    bool occupied_;
    bool output_ready_;  // data available at output port

    PipeStage(int lat = 1) : latency_(lat), timer_(0),
                             occupied_(false), output_ready_(false) {}

    void reset() { timer_=0; occupied_=false; output_ready_=false; }

    bool can_accept() const { return !occupied_; }

    void push(const T& d) {
        assert(!occupied_);
        data_     = d;
        timer_    = latency_;
        occupied_ = true;
        output_ready_ = false;
    }

    // Advance one cycle. Returns true if output becomes ready this cycle.
    bool tick(bool downstream_stall = false) {
        if (!occupied_) return false;
        if (output_ready_) {
            // Waiting for downstream to consume
            return false;
        }
        if (downstream_stall) return false;
        timer_--;
        if (timer_ <= 0) {
            output_ready_ = true;
            return true;
        }
        return false;
    }

    // Consume the output (downstream accepted it)
    T pop() {
        assert(output_ready_);
        output_ready_ = false;
        occupied_      = false;
        return data_;
    }

    bool has_output() const { return output_ready_; }
};

// ==================== Dot Product Unit (tc_dot_product.v) ====================
// Computes: result = Σ(a[k]*b[k], k=0..K-1) + c
//
// RTL pipeline structure:
//   K × tc_mul_pipe (2 cycles each, parallel)
//     → log₂K levels of tc_add_pipe (2 cycles each)
//       → 1 × final_add (FP9+FP16 → FP22, 2 cycles)
//         → fp22_to_fp8_con → SRAM output buffer
//
// Cycle-approximate model:
//   We treat the entire DP as a multi-stage pipe:
//   [mul_stage] → [add_tree_stage × depth] → [final_add_stage] → [output]
//
//   Since all K muls fire in parallel and are fully pipelined,
//   we model them as a single 2-cycle stage.
//   The add tree has depth = log₂K levels, each 2 cycles.
//   Final add is 2 cycles.

struct DPInput {
    std::vector<double> a;   // K elements (already converted to double)
    std::vector<double> b;   // K elements
    double c;                // bias
    int row, col;            // position in output matrix
};

struct DPResult {
    double value;            // accumulated FP22-precision result
    int    row, col;
};

class DotProductUnit {
public:
    const OTC_Config* cfg_ = nullptr;
    int total_latency_ = 0;

    // Model: single pipeline with total_latency_ stages
    // We use a shift-register approach: data enters and exits after N ticks
    struct InFlight {
        DPResult result;
        int      remaining;  // cycles until output
        bool     valid;
    };
    std::vector<InFlight> pipe_;  // in-flight operations

    DPResult  output_;
    bool      output_valid_ = false;

    void init(const OTC_Config* cfg) {
        cfg_ = cfg;
        total_latency_ = cfg->mul_latency
                        + cfg->tree_depth() * cfg->add_latency
                        + cfg->add_latency   // final_add
                        + 1;                  // output register
    }

    void reset() {
        pipe_.clear();
        output_valid_ = false;
    }

    bool can_accept() const {
        // pipelined: can always accept (one per cycle) unless output blocked
        return true;
    }

    // Push a new computation into the pipeline
    void push(const DPInput& in, OTC_Stats& stats) {
        // === Functional computation (mirrors fmul + fadd chain) ===
        // Mul stage: K parallel multiplies
        double products[64]; // max K
        for (int k = 0; k < cfg_->K; k++) {
            products[k] = in.a[k] * in.b[k];
            stats.mul_ops++;
        }

        // Add-tree reduction: log₂K levels
        int n = cfg_->K;
        while (n > 1) {
            for (int i = 0; i < n/2; i++) {
                products[i] = products[2*i] + products[2*i+1];
                stats.add_ops++;
            }
            n /= 2;
        }

        // Final add: sum + bias C (FP22 precision)
        double result = products[0] + in.c;
        stats.add_ops++;

        // Quantize through FP22 to model precision loss
        uint32_t fp22_bits = SoftFloat::f64_to_fp22(result);
        double fp22_val    = SoftFloat::fp22_to_f64(fp22_bits);

        // Push into pipeline
        InFlight entry;
        entry.result = {fp22_val, in.row, in.col};
        entry.remaining = total_latency_;
        entry.valid = true;
        pipe_.push_back(entry);
    }

    // Advance one cycle
    void tick() {
        output_valid_ = false;
        for (auto it = pipe_.begin(); it != pipe_.end(); ) {
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

    bool busy() const { return !pipe_.empty(); }
};

// ==================== Tensor Core Unit (tensor_core.v) ====================
// Top-level: FormatConv → M×N DotProduct array → Output FIFO
//
// Mirrors: tensor_core.v     (top wrapper, AXI-Stream interface)
//          to_fp8_con.v      (input format conversion, instantiates 2 cores)
//          mm_mul_add.v      (M×N generate loop of tc_dot_product)
//          config_registers.v(shape/type validation)
//          stream_fifo_pipe_true (output buffering)
//
// Execution model:
//   1. FORMAT_CONV: convert all input elements (conv_latency cycles)
//   2. DISPATCH:    feed M×N dot products (1 cycle per DP, pipelined)
//   3. DRAIN:       wait for all pipeline results
//   4. OUTPUT:      results available in output_d_ buffer

class TensorCoreUnit {
public:
    // ---- Configuration ----
    OTC_Config   cfg_;
    OTC_Stats    stats_;

    // ---- Sub-modules (mirrors mm_mul_add generate loop) ----
    //   In RTL: M×N = 64 instances of tc_dot_product, all parallel.
    //   In SimX: We model them as a pool. Since all share the same
    //            pipeline depth, we just use a global cycle counter.
    std::vector<DotProductUnit> dp_units_; // M×N units

    // ---- State machine ----
    enum State { IDLE, FORMAT_CONV, DISPATCH, DRAIN, DONE };
    State    state_ = IDLE;
    int      phase_timer_ = 0;
    int      dispatch_idx_ = 0;
    uint64_t cycle_ = 0;

    // ---- Data buffers ----
    // Input (raw packed words, from host/driver)
    std::vector<uint32_t> raw_a_, raw_b_, raw_c_;
    // Converted doubles (after format conversion stage)
    std::vector<double> conv_a_, conv_b_, conv_c_;
    // Output matrix D[M×N]
    std::vector<double> output_d_;

    int results_collected_ = 0;

    // ---- Initialize (mirrors processor.init() in Vortex SimX) ----
    void init(const OTC_Config& cfg) {
        cfg_ = cfg;
        assert(cfg_.validate());

        dp_units_.resize(cfg_.M * cfg_.N);
        for (auto& dp : dp_units_) dp.init(&cfg_);

        output_d_.resize(cfg_.M * cfg_.N, 0.0);

        DT.init(cfg_.debug_level, cfg_.trace_en);
        DT.log(1, "OTC init: D[%dx%d] = A[%dx%d] x B[%dx%d] + C",
               cfg_.M, cfg_.N, cfg_.M, cfg_.K, cfg_.K, cfg_.N);
        DT.log(1, "  pipeline depth=%d, dp_units=%d",
               cfg_.pipeline_depth(), cfg_.total_dp());
        DT.log(1, "  HW resources: %d muls, %d add-tree adders, %d accumulators",
               cfg_.total_dp() * cfg_.K,
               cfg_.total_dp() * (cfg_.K - 1),
               cfg_.total_dp());
    }

    // ---- Reset ----
    void reset() {
        state_ = IDLE;
        cycle_ = 0;
        phase_timer_ = 0;
        dispatch_idx_ = 0;
        results_collected_ = 0;
        stats_ = {};
        for (auto& dp : dp_units_) dp.reset();
    }

    // ---- Load matrices (mirrors vx_copy_to_dev) ----
    void load(const std::vector<uint32_t>& a,
              const std::vector<uint32_t>& b,
              const std::vector<uint32_t>& c) {
        raw_a_ = a; raw_b_ = b; raw_c_ = c;
    }

    // ---- Start execution (mirrors vx_start) ----
    void start() {
        assert(state_ == IDLE || state_ == DONE);
        state_ = FORMAT_CONV;
        phase_timer_ = cfg_.conv_latency;
        dispatch_idx_ = 0;
        results_collected_ = 0;
        std::fill(output_d_.begin(), output_d_.end(), 0.0);
        for (auto& dp : dp_units_) dp.reset();

        DT.log(1, "START: begin format conversion (%d cycles)", phase_timer_);
    }

    // ---- Format conversion (mirrors to_fp8_con.v + to_fp9.v) ----
    void do_format_conversion() {
        int eb = FPConvert::elem_bits(cfg_.type_ab);
        int elems_per_word = 32 / eb;

        // Convert A: M rows × K cols
        int total_a = cfg_.M * cfg_.K;
        conv_a_.resize(total_a);
        for (int i = 0; i < total_a; i++) {
            int wi = i / elems_per_word;
            int ei = i % elems_per_word;
            uint32_t w = (wi < (int)raw_a_.size()) ? raw_a_[wi] : 0;
            conv_a_[i] = FPConvert::elem_to_f64(w, ei, cfg_.type_ab, cfg_.type_ab_sub);
        }

        // Convert B: K rows × N cols (stored column-major for dot product)
        int total_b = cfg_.K * cfg_.N;
        conv_b_.resize(total_b);
        for (int i = 0; i < total_b; i++) {
            int wi = i / elems_per_word;
            int ei = i % elems_per_word;
            uint32_t w = (wi < (int)raw_b_.size()) ? raw_b_[wi] : 0;
            conv_b_[i] = FPConvert::elem_to_f64(w, ei, cfg_.type_ab, cfg_.type_ab_sub);
        }

        // Convert C (bias): M×N values in FP16
        int total_c = cfg_.M * cfg_.N;
        conv_c_.resize(total_c, 0.0);
        for (int i = 0; i < total_c; i++) {
            int wi = i / 2;
            int ei = i % 2;
            uint32_t w = (wi < (int)raw_c_.size()) ? raw_c_[wi] : 0;
            uint16_t half = (w >> (ei * 16)) & 0xFFFF;
            conv_c_[i] = SoftFloat::fp16_to_f64(half);
        }

        DT.log(2, "FmtConv done: A[0]=%f A[1]=%f B[0]=%f C[0]=%f",
               conv_a_[0], total_a>1?conv_a_[1]:0.0, conv_b_[0], conv_c_[0]);
    }

    // ---- Dispatch all M×N dot products ----
    //   In RTL: all 64 units fire simultaneously (single-cycle dispatch).
    //   In SimX: we model this as 1-cycle dispatch (all parallel).
    void dispatch_all(OTC_Stats& stats) {
        for (int i = 0; i < cfg_.M; i++) {
            for (int j = 0; j < cfg_.N; j++) {
                DPInput in;
                in.a.resize(cfg_.K);
                in.b.resize(cfg_.K);
                in.row = i;
                in.col = j;

                // A[i][k] for k=0..K-1  (row i of A)
                for (int k = 0; k < cfg_.K; k++)
                    in.a[k] = conv_a_[i * cfg_.K + k];

                // B[k][j] for k=0..K-1  (column j of B)
                // B stored as B[col][row] in RTL (mm_mul_add indexing)
                for (int k = 0; k < cfg_.K; k++) {
                    if (cfg_.transpose_b)
                        in.b[k] = conv_b_[j * cfg_.K + k]; // B already col-major
                    else
                        in.b[k] = conv_b_[k * cfg_.N + j]; // B row-major → extract col j
                }

                in.c = conv_c_[i * cfg_.N + j];

                int dp_idx = i * cfg_.N + j;
                dp_units_[dp_idx].push(in, stats);

                DT.log(3, "  DISPATCH dp[%d][%d]: a[0]=%f b[0]=%f c=%f",
                       i, j, in.a[0], in.b[0], in.c);
            }
        }
    }

    // ---- Main tick (core simulation loop) ----
    void tick() {
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

        case DRAIN:
            stats_.busy_cycles++;
            {
                bool any_busy = false;
                for (auto& dp : dp_units_) {
                    dp.tick();
                    if (dp.output_valid_) {
                        int idx = dp.output_.row * cfg_.N + dp.output_.col;
                        output_d_[idx] = dp.output_.value;
                        results_collected_++;
                        DT.log(2, "  RESULT dp[%d][%d] = %f (collected %d/%d)",
                               dp.output_.row, dp.output_.col, dp.output_.value,
                               results_collected_, cfg_.total_dp());
                    }
                    if (dp.busy()) any_busy = true;
                }

                if (results_collected_ >= cfg_.total_dp()) {
                    state_ = DONE;
                    stats_.matrices_done++;
                    DT.log(1, "DRAIN → DONE in %lu cycles (pipeline=%d)",
                           cycle_, cfg_.pipeline_depth());
                } else if (!any_busy && results_collected_ < cfg_.total_dp()) {
                    // Safety: if nothing is busy but not all collected,
                    // something went wrong
                    DT.log(1, "WARNING: no busy DPs but only %d/%d results",
                           results_collected_, cfg_.total_dp());
                    state_ = DONE;
                }
            }
            break;

        case DONE:
            break;
        }
    }

    // ---- Run to completion (mirrors vx_ready loop) ----
    uint64_t run(int max_cycles = 100000) {
        start();
        while (state_ != DONE && (int)cycle_ < max_cycles) {
            tick();
        }
        return cycle_;
    }

    // ---- Query state ----
    bool is_done() const { return state_ == DONE; }
    bool is_busy() const { return state_ != IDLE && state_ != DONE; }

    // ---- Get output as FP format values ----
    std::vector<double> get_result_f64() const { return output_d_; }

    std::vector<uint16_t> get_result_fp16() const {
        std::vector<uint16_t> out(output_d_.size());
        for (size_t i = 0; i < output_d_.size(); i++)
            out[i] = SoftFloat::f64_to_fp16(output_d_[i]);
        return out;
    }

    std::vector<uint32_t> get_result_fp32() const {
        std::vector<uint32_t> out(output_d_.size());
        for (size_t i = 0; i < output_d_.size(); i++)
            out[i] = SoftFloat::f64_to_fp32(output_d_[i]);
        return out;
    }
};
