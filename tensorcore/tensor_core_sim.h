#pragma once
// =============================================================================
// tensor_core_sim.h — Cycle-accurate TensorCore simulator
// Models the exact pipeline from OpenTensorCore Verilog RTL:
//   8× tc_mul_pipe (2-cycle each) → 3-level adder tree of tc_add_pipe (2-cycle each)
//   → final FP22 add (2-cycle) → FP22→output conversion (1-cycle)
// Pipeline depth: 2(mul) + 2+2+2(add tree) + 2(final add) + 1(convert) = 11 cycles
// =============================================================================
#include "fp_types.h"
#include "fp_arith.h"
#include <array>
#include <vector>
#include <functional>
#include <cstdio>

// =============================================================================
// PipeStage2: Models a 2-stage pipeline with valid/ready handshaking
// Exactly matches RTL's tc_mul_pipe / tc_add_pipe register control
// =============================================================================
template <typename T>
struct PipeStage2 {
    T     data1, data2;
    bool  valid1 = false, valid2 = false;

    // Returns: in_ready (can accept new data)
    // out_ready: downstream is ready to consume
    bool in_ready(bool out_ready) const {
        return !(!out_ready && valid1 && valid2);
    }

    bool out_valid() const { return valid2; }
    const T& out_data() const { return data2; }

    // Advance the pipeline by one clock cycle
    // compute1: function called when reg1 is enabled (with input data)
    //           produces data to store in data1
    // compute2: function called when reg2 is enabled
    //           transforms data1 → data2
    // Returns true if input was accepted
    bool tick(bool in_valid, const T& in_data, bool out_ready,
              std::function<T(const T&)> compute1 = nullptr,
              std::function<T(const T&)> compute2 = nullptr)
    {
        bool reg_en1 = in_valid && !(valid1 && valid2 && !out_ready);
        bool reg_en2 = valid1 && !(valid2 && !out_ready);

        // Update valid registers
        bool new_valid1 = valid1, new_valid2 = valid2;
        if (!(!out_ready && valid1 && valid2)) {
            new_valid1 = in_valid;
        }
        if (!(!out_ready && valid2)) {
            new_valid2 = valid1;
        }

        // Update data registers
        T new_data1 = data1, new_data2 = data2;
        if (reg_en1) {
            new_data1 = compute1 ? compute1(in_data) : in_data;
        }
        if (reg_en2) {
            new_data2 = compute2 ? compute2(data1) : data1;
        }

        valid1 = new_valid1;
        valid2 = new_valid2;
        data1  = new_data1;
        data2  = new_data2;

        return reg_en1;
    }

    void reset() { valid1 = valid2 = false; }
};

// =============================================================================
// Data tokens flowing through the pipeline
// =============================================================================
struct MulInput {
    uint16_t a;  // FP9
    uint16_t b;  // FP9
};

struct MulStage1Data {
    FMulS1Out s1;
    uint16_t  a_bits;  // preserved for s2
    uint16_t  b_bits;
};

struct FP9Token {
    uint16_t value;  // packed FP9
};

struct FP22Token {
    uint32_t value;  // packed FP22
};

// =============================================================================
// Single dot-product pipeline (one output element of the 8×8 matrix)
// Computes: D[i][j] = sum(A[i][k]*B[k][j] for k=0..7) + C[i][j]
// =============================================================================
struct DotProductPipeline {
    // Multiplier pipelines (8 parallel)
    PipeStage2<MulStage1Data> mul_pipe[8];
    // Multiplication products (held between mul output and add tree input)
    uint16_t mul_results[8];
    bool     mul_results_valid[8];

    // Adder tree: Level 0 (4 adders), Level 1 (2 adders), Level 2 (1 adder)
    // Each is a 2-stage pipeline
    PipeStage2<FP9Token> add_L0[4]; // pairs: (0,4),(1,5),(2,6),(3,7)
    PipeStage2<FP9Token> add_L1[2]; // pairs: (L0[0],L0[1]), (L0[2],L0[3])
    PipeStage2<FP9Token> add_L2;    // pair: (L1[0],L1[1])

    // Final FP22 add (tree result + bias C)
    PipeStage2<FP22Token> final_add;

    // Output conversion register
    bool   conv_valid = false;
    uint32_t conv_fp22 = 0;

    // Sideband: C bias and rounding mode propagated through pipeline
    // In RTL these are ctrl registers at each stage
    uint16_t c_bias;    // FP16 or FP8 bias
    RoundingMode rm;
    PrecisionType output_prec;

    // Intermediate storage for adder tree inputs
    uint16_t add_L0_a[4], add_L0_b[4];
    bool add_L0_input_valid[4];
    uint16_t add_L1_a[2], add_L1_b[2];
    bool add_L1_input_valid[2];
    uint16_t add_L2_a, add_L2_b;
    bool add_L2_input_valid;
    uint32_t final_add_a; // FP22 from tree
    uint32_t final_add_b; // FP22 from C
    bool final_add_input_valid;

    void reset() {
        for (int i = 0; i < 8; i++) { mul_pipe[i].reset(); mul_results_valid[i] = false; }
        for (int i = 0; i < 4; i++) { add_L0[i].reset(); add_L0_input_valid[i] = false; }
        for (int i = 0; i < 2; i++) { add_L1[i].reset(); add_L1_input_valid[i] = false; }
        add_L2.reset(); add_L2_input_valid = false;
        final_add.reset(); final_add_input_valid = false;
        conv_valid = false;
    }

    // Returns output valid and result (FP8/FP16/FP32 depending on output_prec)
    bool out_valid() const { return conv_valid; }
    uint32_t out_result() const { return conv_fp22; }
};

// =============================================================================
// Top-level Tensor Core simulator: 8×8 matrix of dot product pipelines
// Computes D[8×8] = A[8×8] × B[8×8] + C[8×8]
// =============================================================================
struct TensorCoreSim {
    static constexpr int M = 8, K = 8, N = 8;
    static constexpr int PIPELINE_DEPTH = 11;

    // 64 dot-product pipelines
    DotProductPipeline dp[M][N];

    // Configuration
    PrecisionType input_prec  = PREC_FP8_E4M3;
    PrecisionType output_prec = PREC_FP8_E4M3;
    RoundingMode  rm          = RNE;

    // Input data (all K elements arrive simultaneously)
    uint16_t a_fp9[M][K];  // A matrix in FP9
    uint16_t b_fp9[K][N];  // B matrix in FP9 (transposed for column access)
    uint32_t c_fp22[M][N]; // C matrix in FP22

    // Output
    uint32_t d_fp22[M][N]; // Raw FP22 results
    bool     d_valid[M][N];

    // Pipeline state
    bool input_loaded = false;
    bool output_ready = true;
    int  cycle_count  = 0;

    // Statistics
    int total_cycles = 0;
    int jobs_completed = 0;

    void reset() {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                dp[i][j].reset();
                d_valid[i][j] = false;
            }
        input_loaded = false;
        cycle_count = 0;
        total_cycles = 0;
        jobs_completed = 0;
    }

    // Load input matrices (already converted to FP9/FP22)
    void load_inputs(const uint16_t a[M][K], const uint16_t b[K][N],
                     const uint32_t c[M][N], PrecisionType prec, RoundingMode r = RNE)
    {
        input_prec = prec;
        output_prec = prec;
        rm = r;
        for (int i = 0; i < M; i++)
            for (int k = 0; k < K; k++)
                a_fp9[i][k] = a[i][k];
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++)
                b_fp9[k][j] = b[k][j];
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                c_fp22[i][j] = c[i][j];
                d_valid[i][j] = false;
            }
        input_loaded = true;
    }

    // Run the full pipeline until all outputs are valid
    // Returns total cycles taken
    int run_to_completion() {
        if (!input_loaded) return 0;

        int cycles = 0;
        bool all_done = false;

        while (!all_done && cycles < 100) {
            tick();
            cycles++;

            all_done = true;
            for (int i = 0; i < M && all_done; i++)
                for (int j = 0; j < N && all_done; j++)
                    if (!d_valid[i][j]) all_done = false;
        }

        total_cycles += cycles;
        jobs_completed++;
        input_loaded = false;
        return cycles;
    }

    // Single clock tick — advance all 64 pipelines by one cycle
    void tick() {
        cycle_count++;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                tick_dot_product(i, j);
            }
        }
    }

private:
    void tick_dot_product(int i, int j) {
        auto& p = dp[i][j];
        p.rm = rm;
        p.output_prec = output_prec;

        // ============================================================
        // Stage 11: Output conversion (FP22 → output format)
        // ============================================================
        bool conv_out_ready = true; // always ready to accept output
        if (p.final_add.out_valid() && !p.conv_valid) {
            p.conv_valid = true;
            p.conv_fp22 = p.final_add.out_data().value;
            d_fp22[i][j] = p.conv_fp22;
            d_valid[i][j] = true;
        }

        // ============================================================
        // Stages 9-10: Final FP22 add (tree result + C bias)
        // ============================================================
        bool final_out_ready = !p.conv_valid || conv_out_ready;
        {
            // Check if we have input for final add
            bool final_in_valid = p.add_L2.out_valid() && !p.final_add_input_valid;
            if (final_in_valid && !p.final_add_input_valid) {
                // Convert FP9 tree result to FP22
                p.final_add_a = fp9_to_fp22(p.add_L2.out_data().value);
                p.final_add_b = c_fp22[i][j];
                p.final_add_input_valid = true;
            }

            bool fa_in_valid = p.final_add_input_valid;
            FP22Token fa_in = {p.final_add_a};

            p.final_add.tick(fa_in_valid, fa_in, final_out_ready,
                // Stage 1: latch input, compute fadd_s1 (both paths stored implicitly)
                [&](const FP22Token& in) -> FP22Token {
                    return in; // data latched; s1 computed in stage 2
                },
                // Stage 2: full FP22 add
                [&](const FP22Token& in) -> FP22Token {
                    uint32_t result = fp22_add(in.value, p.final_add_b, rm);
                    return {result};
                });

            if (p.final_add.in_ready(final_out_ready) && p.final_add_input_valid) {
                p.final_add_input_valid = false;
            }
        }

        // ============================================================
        // Stages 7-8: Adder tree Level 2 (1 adder: L1[0] + L1[1])
        // ============================================================
        bool l2_out_ready = p.final_add.in_ready(final_out_ready);
        {
            bool l2_in_valid = p.add_L1[0].out_valid() && p.add_L1[1].out_valid()
                               && !p.add_L2_input_valid;
            if (l2_in_valid) {
                p.add_L2_a = p.add_L1[0].out_data().value;
                p.add_L2_b = p.add_L1[1].out_data().value;
                p.add_L2_input_valid = true;
            }

            FP9Token l2_in = {p.add_L2_a};
            p.add_L2.tick(p.add_L2_input_valid, l2_in, l2_out_ready,
                [](const FP9Token& in) -> FP9Token { return in; },
                [&](const FP9Token& in) -> FP9Token {
                    return {fp9_add(in.value, p.add_L2_b, rm)};
                });

            if (p.add_L2.in_ready(l2_out_ready) && p.add_L2_input_valid) {
                p.add_L2_input_valid = false;
            }
        }

        // ============================================================
        // Stages 5-6: Adder tree Level 1 (2 adders)
        // ============================================================
        bool l1_out_ready[2];
        l1_out_ready[0] = p.add_L2.in_ready(l2_out_ready);
        l1_out_ready[1] = l1_out_ready[0];
        for (int a = 0; a < 2; a++) {
            int src0 = a * 2, src1 = a * 2 + 1;
            bool l1_in_valid = p.add_L0[src0].out_valid() && p.add_L0[src1].out_valid()
                               && !p.add_L1_input_valid[a];
            if (l1_in_valid) {
                p.add_L1_a[a] = p.add_L0[src0].out_data().value;
                p.add_L1_b[a] = p.add_L0[src1].out_data().value;
                p.add_L1_input_valid[a] = true;
            }

            FP9Token l1_in = {p.add_L1_a[a]};
            p.add_L1[a].tick(p.add_L1_input_valid[a], l1_in, l1_out_ready[a],
                [](const FP9Token& in) -> FP9Token { return in; },
                [&, a](const FP9Token& in) -> FP9Token {
                    return {fp9_add(in.value, p.add_L1_b[a], rm)};
                });

            if (p.add_L1[a].in_ready(l1_out_ready[a]) && p.add_L1_input_valid[a]) {
                p.add_L1_input_valid[a] = false;
            }
        }

        // ============================================================
        // Stages 3-4: Adder tree Level 0 (4 adders: pairs (0,4),(1,5),(2,6),(3,7))
        // ============================================================
        bool l0_out_ready[4];
        l0_out_ready[0] = p.add_L1[0].in_ready(l1_out_ready[0]);
        l0_out_ready[1] = l0_out_ready[0]; // shares L1[0]
        l0_out_ready[2] = p.add_L1[1].in_ready(l1_out_ready[1]);
        l0_out_ready[3] = l0_out_ready[2]; // shares L1[1]

        for (int a = 0; a < 4; a++) {
            int src0 = a, src1 = a + 4; // RTL: muls_result[j] + muls_result[j+SHAPE_K/2]
            bool l0_in_valid = p.mul_results_valid[src0] && p.mul_results_valid[src1]
                               && !p.add_L0_input_valid[a];
            if (l0_in_valid) {
                p.add_L0_a[a] = p.mul_results[src0];
                p.add_L0_b[a] = p.mul_results[src1];
                p.add_L0_input_valid[a] = true;
            }

            FP9Token l0_in = {p.add_L0_a[a]};
            p.add_L0[a].tick(p.add_L0_input_valid[a], l0_in, l0_out_ready[a],
                [](const FP9Token& in) -> FP9Token { return in; },
                [&, a](const FP9Token& in) -> FP9Token {
                    return {fp9_add(in.value, p.add_L0_b[a], rm)};
                });

            if (p.add_L0[a].in_ready(l0_out_ready[a]) && p.add_L0_input_valid[a]) {
                p.add_L0_input_valid[a] = false;
                p.mul_results_valid[src0] = false;
                p.mul_results_valid[src1] = false;
            }
        }

        // ============================================================
        // Stages 1-2: Multipliers (8 parallel)
        // ============================================================
        for (int k = 0; k < K; k++) {
            bool mul_out_ready = true;
            // Determine if add tree L0 can accept our output
            int add_idx = k < 4 ? k : (k - 4);
            // Actually mul outputs feed into L0 via mul_results buffer
            // They're always "ready" as long as the buffer slot is free
            mul_out_ready = !p.mul_results_valid[k];

            bool mul_in_valid = input_loaded && !p.mul_results_valid[k];
            MulStage1Data mul_in;
            mul_in.a_bits = a_fp9[i][k];
            mul_in.b_bits = b_fp9[k][j];
            mul_in.s1 = fmul_s1(mul_in.a_bits, mul_in.b_bits, 5, 4, rm);

            p.mul_pipe[k].tick(mul_in_valid, mul_in, mul_out_ready,
                // Stage 1: latch + s1 computation
                [&](const MulStage1Data& in) -> MulStage1Data {
                    return in;
                },
                // Stage 2: multiplication + s3 normalization
                [&](const MulStage1Data& in) -> MulStage1Data {
                    FMulS2Out s2 = fmul_s2(in.a_bits, in.b_bits, 5, 4, in.s1);
                    MulStage1Data out = in;
                    // Store result in s1.shift_amt as temp (hacky but works)
                    // Actually, compute full result here
                    uint32_t result = fmul_s3(s2, 5, 4);
                    out.s1.shift_amt = result; // reuse field to carry result
                    return out;
                });

            // Capture multiplier output
            if (p.mul_pipe[k].out_valid() && !p.mul_results_valid[k]) {
                p.mul_results[k] = (uint16_t)(p.mul_pipe[k].out_data().s1.shift_amt & 0x1FF);
                p.mul_results_valid[k] = true;
            }
        }
    }
};

// =============================================================================
// Functional (non-pipelined) reference: compute D = A*B + C using same arithmetic
// =============================================================================
inline void reference_matmul(const uint16_t a_fp9[8][8], const uint16_t b_fp9[8][8],
                              const uint32_t c_fp22[8][8], uint32_t d_fp22[8][8],
                              RoundingMode rm = RNE)
{
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            // Step 1: 8 multiplications
            uint16_t products[8];
            for (int k = 0; k < 8; k++) {
                products[k] = fp9_multiply(a_fp9[i][k], b_fp9[k][j], rm);
            }

            // Step 2: Adder tree Level 0 (pairs: 0+4, 1+5, 2+6, 3+7)
            uint16_t sums_l0[4];
            for (int a = 0; a < 4; a++) {
                sums_l0[a] = fp9_add(products[a], products[a + 4], rm);
            }

            // Step 3: Adder tree Level 1
            uint16_t sums_l1[2];
            sums_l1[0] = fp9_add(sums_l0[0], sums_l0[1], rm);
            sums_l1[1] = fp9_add(sums_l0[2], sums_l0[3], rm);

            // Step 4: Adder tree Level 2
            uint16_t sum_l2 = fp9_add(sums_l1[0], sums_l1[1], rm);

            // Step 5: Convert to FP22 and add C
            uint32_t sum_fp22 = fp9_to_fp22(sum_l2);
            d_fp22[i][j] = fp22_add(sum_fp22, c_fp22[i][j], rm);
        }
    }
}
