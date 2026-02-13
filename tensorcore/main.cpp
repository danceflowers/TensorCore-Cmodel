// =============================================================================
// main.cpp — OpenTensorCore Cycle-Accurate Simulator Test Harness
// Tests: random 8×8×8 matrix multiplications across FP4/FP8/FP16 precisions
// Validates: pipelined results match combinational reference (bit-exact)
// Reports: pipeline latency, cycle counts, and output visualization
//
// Usage:
//   ./tensorcore_sim                          Run all tests, all precisions
//   ./tensorcore_sim --prec FP8_E4M3          Restrict to one precision
//   ./tensorcore_sim --test 3                 Run only test 3
//   ./tensorcore_sim --prec FP16 --test 1     Combine filters
//   ./tensorcore_sim --rm RTZ                 Set rounding mode
//   ./tensorcore_sim --seed 12345             Fixed RNG seed
//   ./tensorcore_sim --help                   Show usage
// =============================================================================
#include "tensor_core_sim.h"
#include "tensor_core_cfg.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>

// =============================================================================
// Global configuration (populated from command-line arguments)
// =============================================================================
struct Config {
    std::vector<PrecisionType> precisions;
    std::vector<PrecisionType> out_precisions;
    int  test_id    = 0;       // 0 = all, 1-6 = specific test
    RoundingMode rm = RNE;
    uint32_t seed   = 0;       // 0 = use time
    bool show_help  = false;
};

static Config g_cfg;

// =============================================================================
// Random matrix generation
// =============================================================================
static uint32_t rng_state = 42;
inline uint32_t xorshift32() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

inline double rand_double(double lo, double hi) {
    return lo + (hi - lo) * (double)(xorshift32() & 0xFFFF) / 65535.0;
}

struct MatrixSet {
    uint16_t a_fp9[8][8];
    uint16_t b_fp9[8][8];
    uint32_t c_fp22[8][8];
    uint32_t a_raw[8][8];
    uint32_t b_raw[8][8];
    uint32_t c_raw[8][8];
};

double raw_to_double(uint32_t bits, PrecisionType prec) {
    switch (prec) {
        case PREC_FP4_E2M1: return fp4_to_double((uint8_t)(bits & 0xF));
        case PREC_FP8_E4M3: return fp8_e4m3_to_double((uint8_t)(bits & 0xFF));
        case PREC_FP8_E5M2: return fp8_e5m2_to_double((uint8_t)(bits & 0xFF));
        case PREC_FP16:     return fp16_to_double((uint16_t)(bits & 0xFFFF));
        case PREC_FP32: {
            float f = 0.0f;
            uint32_t raw = bits;
            std::memcpy(&f, &raw, sizeof(float));
            return (double)f;
        }
        default: return 0.0;
    }
}

MatrixSet generate_random_matrices(PrecisionType prec) {
    MatrixSet ms = {};
    double range_lo, range_hi;

    switch (prec) {
        case PREC_FP4_E2M1: range_lo = -3.0; range_hi = 3.0; break;
        case PREC_FP8_E4M3: range_lo = -8.0; range_hi = 8.0; break;
        case PREC_FP8_E5M2: range_lo = -4.0; range_hi = 4.0; break;
        case PREC_FP16:     range_lo = -10.0; range_hi = 10.0; break;
        default:            range_lo = -1.0; range_hi = 1.0; break;
    }

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            double va = rand_double(range_lo, range_hi);
            double vb = rand_double(range_lo, range_hi);
            double vc = rand_double(range_lo * 0.5, range_hi * 0.5);

            switch (prec) {
                case PREC_FP4_E2M1:
                    ms.a_raw[i][j] = double_to_fp4(va);
                    ms.b_raw[i][j] = double_to_fp4(vb);
                    ms.c_raw[i][j] = double_to_fp4(vc);
                    ms.a_fp9[i][j] = fp4_to_fp9(ms.a_raw[i][j]);
                    ms.b_fp9[i][j] = fp4_to_fp9(ms.b_raw[i][j]);
                    ms.c_fp22[i][j] = fp9_to_fp22(fp4_to_fp9(ms.c_raw[i][j]));
                    break;
                case PREC_FP8_E4M3:
                    ms.a_raw[i][j] = double_to_fp8_e4m3(va);
                    ms.b_raw[i][j] = double_to_fp8_e4m3(vb);
                    ms.c_raw[i][j] = double_to_fp8_e4m3(vc);
                    ms.a_fp9[i][j] = fp8_e4m3_to_fp9(ms.a_raw[i][j]);
                    ms.b_fp9[i][j] = fp8_e4m3_to_fp9(ms.b_raw[i][j]);
                    ms.c_fp22[i][j] = fp9_to_fp22(fp8_e4m3_to_fp9(ms.c_raw[i][j]));
                    break;
                case PREC_FP8_E5M2:
                    ms.a_raw[i][j] = double_to_fp8_e5m2(va);
                    ms.b_raw[i][j] = double_to_fp8_e5m2(vb);
                    ms.c_raw[i][j] = double_to_fp8_e5m2(vc);
                    ms.a_fp9[i][j] = fp8_e5m2_to_fp9(ms.a_raw[i][j]);
                    ms.b_fp9[i][j] = fp8_e5m2_to_fp9(ms.b_raw[i][j]);
                    ms.c_fp22[i][j] = fp9_to_fp22(fp8_e5m2_to_fp9(ms.c_raw[i][j]));
                    break;
                case PREC_FP16:
                    ms.a_raw[i][j] = double_to_fp16(va);
                    ms.b_raw[i][j] = double_to_fp16(vb);
                    ms.c_raw[i][j] = double_to_fp16(vc);
                    ms.a_fp9[i][j] = fp16_to_fp9(ms.a_raw[i][j]);
                    ms.b_fp9[i][j] = fp16_to_fp9(ms.b_raw[i][j]);
                    ms.c_fp22[i][j] = fp16_to_fp22(ms.c_raw[i][j]);
                    break;
                default: break;
            }
        }
    }
    return ms;
}

// =============================================================================
// Precision name string
// =============================================================================
const char* prec_name(PrecisionType p) {
    switch (p) {
        case PREC_FP4_E2M1: return "FP4_E2M1";
        case PREC_FP8_E4M3: return "FP8_E4M3";
        case PREC_FP8_E5M2: return "FP8_E5M2";
        case PREC_FP16:     return "FP16";
        case PREC_FP32:     return "FP32";
        default:            return "UNKNOWN";
    }
}

void print_matrix_fp22(const char* title, const uint32_t m[8][8]) {
    printf("    %s\n", title);
    for (int i = 0; i < 8; i++) {
        printf("      ");
        for (int j = 0; j < 8; j++) {
            printf("%9.4f ", fp22_to_double(m[i][j]));
        }
        printf("\n");
    }
}

void print_matrix_output(const char* title, const uint32_t m[8][8], PrecisionType out_prec) {
    printf("    %s\n", title);
    for (int i = 0; i < 8; i++) {
        printf("      ");
        for (int j = 0; j < 8; j++) {
            printf("%9.4f ", output_bits_to_double(m[i][j], out_prec));
        }
        printf("\n");
    }
}

void golden_fp32_matmul(const MatrixSet& ms, PrecisionType in_prec, double out[8][8]) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            float acc = 0.0f;
            for (int k = 0; k < 8; k++) {
                float a = (float)raw_to_double(ms.a_raw[i][k], in_prec);
                float b = (float)raw_to_double(ms.b_raw[k][j], in_prec);
                acc += a * b;
            }
            float c = (float)raw_to_double(ms.c_raw[i][j], in_prec);
            out[i][j] = (double)(acc + c);
        }
    }
}

void quantized_golden_from_fp22(const uint32_t golden_fp22[8][8], PrecisionType out_prec, RoundingMode rm,
                                uint32_t out_bits[8][8]) {
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            out_bits[i][j] = convert_fp22_to_output_bits(golden_fp22[i][j], out_prec, rm);
}

const char* rm_name(RoundingMode rm) {
    switch (rm) {
        case RNE: return "RNE (Round to Nearest, ties to Even)";
        case RTZ: return "RTZ (Round toward Zero)";
        case RDN: return "RDN (Round Down / toward -Inf)";
        case RUP: return "RUP (Round Up / toward +Inf)";
        case RMM: return "RMM (Round to Nearest, ties to Max Magnitude)";
        default:  return "UNKNOWN";
    }
}

// =============================================================================
// Compare FP22 results
// =============================================================================
bool compare_fp22(uint32_t a, uint32_t b) {
    int ae = (a >> 13) & 0xFF, am = a & 0x1FFF;
    int be = (b >> 13) & 0xFF, bm = b & 0x1FFF;
    if (ae == 255 && am != 0 && be == 255 && bm != 0) return true;
    return a == b;
}

// =============================================================================
// Test 1: Single matmul per precision, verify bit-exact match
// =============================================================================
void test_single_matmul() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Test 1: Single 8×8×8 MatMul per Precision                 ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    for (auto prec : g_cfg.precisions) {
        for (auto out_prec : g_cfg.out_precisions) {
            if (!(out_prec == PREC_FP8_E4M3 || out_prec == PREC_FP8_E5M2 || out_prec == PREC_FP16 || out_prec == PREC_FP32))
                continue;
        MatrixSet ms = generate_random_matrices(prec);

        uint32_t ref[8][8];
        reference_matmul(ms.a_fp9, ms.b_fp9, ms.c_fp22, ref, g_cfg.rm);

        TensorCoreSim sim;
        sim.reset();
        TensorCoreCfg cfg;
        cfg.input_prec = prec;
        cfg.output_prec = out_prec;
        cfg.rm = g_cfg.rm;
        sim.load_inputs(ms.a_fp9, ms.b_fp9, ms.c_fp22, cfg);
        int cycles = sim.run_to_completion();

        uint32_t q_golden[8][8];
        quantized_golden_from_fp22(ref, out_prec, g_cfg.rm, q_golden);

        double fp32_golden[8][8];
        golden_fp32_matmul(ms, prec, fp32_golden);

        int mismatches = 0;
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                if (sim.d_out[i][j] != q_golden[i][j])
                    mismatches++;

        printf("  In %-10s -> Out %-8s: %2d cycles latency | %s\n",
               prec_name(prec), prec_name(out_prec), cycles,
               mismatches == 0 ? "✓ Bit-exact match (64/64 elements)" :
                                 "✗ MISMATCH");

        print_matrix_output("Result Matrix", sim.d_out, out_prec);
        print_matrix_output("Golden Matrix (Quantized)", q_golden, out_prec);
        printf("    Golden Matrix (Unquantized FP32)\n");
        for (int i = 0; i < 8; i++) {
            printf("      ");
            for (int j = 0; j < 8; j++) printf("%9.4f ", fp32_golden[i][j]);
            printf("\n");
        }

        if (mismatches > 0) {
            printf("    Mismatched elements:\n");
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 8; j++)
                    if (sim.d_out[i][j] != q_golden[i][j])
                        printf("      [%d][%d]: out=0x%08X qgold=0x%08X (%.6f vs %.6f, fp32=%.6f)\n",
                               i, j, sim.d_out[i][j], q_golden[i][j],
                               output_bits_to_double(sim.d_out[i][j], out_prec),
                               output_bits_to_double(q_golden[i][j], out_prec),
                               fp32_golden[i][j]);
        }
        }
    }
    printf("\n");
}

// =============================================================================
// Test 2: Back-to-back pipelined matmuls
// =============================================================================
void test_pipelined_throughput() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Test 2: Back-to-Back Pipelined MatMuls                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    TensorCoreSim sim;
    sim.reset();

    struct JobResult {
        uint32_t d_fp22[8][8];
        uint32_t ref[8][8];
        PrecisionType prec;
        int cycles;
    };

    std::vector<PrecisionType> job_precs;
    for (int round = 0; round < 2 && job_precs.size() < 8; round++)
        for (auto p : g_cfg.precisions)
            if (job_precs.size() < 8) job_precs.push_back(p);

    int num_jobs = (int)job_precs.size();
    std::vector<JobResult> results;
    int total_cycles = 0;

    for (int job = 0; job < num_jobs; job++) {
        MatrixSet ms = generate_random_matrices(job_precs[job]);

        JobResult jr;
        jr.prec = job_precs[job];
        reference_matmul(ms.a_fp9, ms.b_fp9, ms.c_fp22, jr.ref, g_cfg.rm);

        sim.reset();
        sim.load_inputs(ms.a_fp9, ms.b_fp9, ms.c_fp22, job_precs[job], g_cfg.rm);
        jr.cycles = sim.run_to_completion();
        total_cycles += jr.cycles;

        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                jr.d_fp22[i][j] = sim.d_fp22[i][j];

        results.push_back(jr);
    }

    int all_match = 0;
    for (int job = 0; job < num_jobs; job++) {
        auto& jr = results[job];
        bool match = true;
        for (int i = 0; i < 8 && match; i++)
            for (int j = 0; j < 8 && match; j++)
                if (!compare_fp22(jr.d_fp22[i][j], jr.ref[i][j]))
                    match = false;
        if (match) all_match++;

        printf("  Job %d [%-10s]: %2d cycles | %s\n",
               job, prec_name(jr.prec), jr.cycles,
               match ? "✓ bit-exact" : "✗ MISMATCH");
    }

    printf("\n  Total: %d cycles for %d jobs | %.1f cycles/matmul | %d/%d bit-exact\n\n",
           total_cycles, num_jobs, total_cycles / (double)num_jobs, all_match, num_jobs);
}

// =============================================================================
// Test 3: Stress test — many random matrices
// =============================================================================
void test_stress() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Test 3: Stress Test (100 random matrices per precision)   ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    int tests_per_prec = 100;

    for (auto prec : g_cfg.precisions) {
        for (auto out_prec : g_cfg.out_precisions) {
            if (!(out_prec == PREC_FP8_E4M3 || out_prec == PREC_FP8_E5M2 || out_prec == PREC_FP16 || out_prec == PREC_FP32))
                continue;
            int pass = 0, fail = 0;
            double max_rel_err_vs_fp32 = 0;
            int total_cycles = 0;

            for (int t = 0; t < tests_per_prec; t++) {
                MatrixSet ms = generate_random_matrices(prec);
                uint32_t ref[8][8];
                reference_matmul(ms.a_fp9, ms.b_fp9, ms.c_fp22, ref, g_cfg.rm);
                uint32_t q_golden[8][8];
                quantized_golden_from_fp22(ref, out_prec, g_cfg.rm, q_golden);
                double fp32_golden[8][8];
                golden_fp32_matmul(ms, prec, fp32_golden);

                TensorCoreSim sim;
                sim.reset();
                TensorCoreCfg cfg;
                cfg.input_prec = prec;
                cfg.output_prec = out_prec;
                cfg.rm = g_cfg.rm;
                sim.load_inputs(ms.a_fp9, ms.b_fp9, ms.c_fp22, cfg);
                int cycles = sim.run_to_completion();
                total_cycles += cycles;

                bool match = true;
                for (int i = 0; i < 8 && match; i++)
                    for (int j = 0; j < 8 && match; j++)
                        if (sim.d_out[i][j] != q_golden[i][j])
                            match = false;

                if (match) pass++; else fail++;

                for (int i = 0; i < 8; i++) {
                    for (int j = 0; j < 8; j++) {
                        double expected = fp32_golden[i][j];
                        double actual = output_bits_to_double(sim.d_out[i][j], out_prec);
                        if (expected != 0 && !std::isnan(expected) && !std::isinf(expected)) {
                            double rel = fabs(actual - expected) / fabs(expected);
                            if (rel > max_rel_err_vs_fp32) max_rel_err_vs_fp32 = rel;
                        }
                    }
                }
            }

            printf("  In %-10s -> Out %-8s: %d/%d bit-exact ✓ | avg %.1f cyc/matmul | max rel err vs FP32: %.2e\n",
                   prec_name(prec), prec_name(out_prec), pass, tests_per_prec,
                   total_cycles / (double)tests_per_prec, max_rel_err_vs_fp32);
        }
    }
    printf("\n");
}

// =============================================================================
// Test 4: Pipeline visualization — show stage occupancy cycle by cycle
// =============================================================================
void test_pipeline_visualization() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Test 4: Pipeline Stage Visualization (single dot product) ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    printf("  Pipeline architecture (per dot product, matching RTL):\n");
    printf("  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐\n");
    printf("  │ 8× MUL_PIPE  │→ │  ADD_TREE_L0 │→ │  ADD_TREE_L1 │→ │  ADD_TREE_L2 │\n");
    printf("  │  (2 cycles)  │  │  4× (2 cyc)  │  │  2× (2 cyc)  │  │  1× (2 cyc)  │\n");
    printf("  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘\n");
    printf("        ↓                                                       ↓\n");
    printf("  ┌──────────────┐                                    ┌──────────────┐\n");
    printf("  │  FP9→FP22    │ ← ── ── ── ── ── ── ── ── ── ─── │  FINAL_ADD   │\n");
    printf("  │  + C bias    │                                    │  (2 cycles)  │\n");
    printf("  └──────────────┘                                    └──────────────┘\n");
    printf("        ↓\n");
    printf("  ┌──────────────┐\n");
    printf("  │  FP22→OUT    │\n");
    printf("  │  (1 cycle)   │\n");
    printf("  └──────────────┘\n\n");

    PrecisionType viz_prec = g_cfg.precisions.empty() ? PREC_FP8_E4M3 : g_cfg.precisions[0];
    printf("  Using precision: %s\n\n", prec_name(viz_prec));

    MatrixSet ms = generate_random_matrices(viz_prec);
    TensorCoreSim sim;
    sim.reset();
    sim.load_inputs(ms.a_fp9, ms.b_fp9, ms.c_fp22, viz_prec, g_cfg.rm);

    printf("  Cycle-by-cycle stage occupancy for dot product [0][0]:\n\n");
    printf("  Cyc │ MUL1 MUL2 │ L0_1 L0_2 │ L1_1 L1_2 │ L2_1 L2_2 │ FA_1 FA_2 │ CONV │\n");
    printf("  ────┼───────────┼───────────┼───────────┼───────────┼───────────┼──────┤\n");

    for (int c = 0; c < 15; c++) {
        sim.tick();
        auto& p = sim.dp[0][0];

        bool mul_v1 = false, mul_v2 = false;
        for (int k = 0; k < 8; k++) {
            if (p.mul_pipe[k].valid1) mul_v1 = true;
            if (p.mul_pipe[k].valid2) mul_v2 = true;
        }
        bool l0_v1 = false, l0_v2 = false;
        for (int a = 0; a < 4; a++) {
            if (p.add_L0[a].valid1) l0_v1 = true;
            if (p.add_L0[a].valid2) l0_v2 = true;
        }
        bool l1_v1 = false, l1_v2 = false;
        for (int a = 0; a < 2; a++) {
            if (p.add_L1[a].valid1) l1_v1 = true;
            if (p.add_L1[a].valid2) l1_v2 = true;
        }

        printf("  %3d │  %c    %c   │  %c    %c   │  %c    %c   │  %c    %c   │  %c    %c   │  %c   │\n",
               c + 1,
               mul_v1 ? '#' : '.', mul_v2 ? '#' : '.',
               l0_v1 ? '#' : '.', l0_v2 ? '#' : '.',
               l1_v1 ? '#' : '.', l1_v2 ? '#' : '.',
               p.add_L2.valid1 ? '#' : '.', p.add_L2.valid2 ? '#' : '.',
               p.final_add.valid1 ? '#' : '.', p.final_add.valid2 ? '#' : '.',
               p.conv_valid ? '#' : '.');

        if (sim.d_valid[0][0]) {
            printf("\n  Output available at cycle %d\n", c + 1);
            break;
        }
    }

    uint32_t ref[8][8];
    reference_matmul(ms.a_fp9, ms.b_fp9, ms.c_fp22, ref, g_cfg.rm);
    bool match = compare_fp22(sim.d_fp22[0][0], ref[0][0]);
    printf("  Element [0][0]: pipe=0x%06X ref=0x%06X → %s\n\n",
           sim.d_fp22[0][0], ref[0][0], match ? "✓ match" : "✗ MISMATCH");
}

// =============================================================================
// Test 5: Output format conversion test
// =============================================================================
void test_output_conversion() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Test 5: FP22 → Output Format Conversion                   ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    PrecisionType conv_prec = g_cfg.precisions.empty() ? PREC_FP8_E4M3 : g_cfg.precisions[0];
    MatrixSet ms = generate_random_matrices(conv_prec);
    TensorCoreSim sim;
    sim.reset();
    sim.load_inputs(ms.a_fp9, ms.b_fp9, ms.c_fp22, conv_prec, g_cfg.rm);
    sim.run_to_completion();

    printf("  Input precision: %s\n", prec_name(conv_prec));
    printf("  Sample FP22 accumulator outputs → converted formats:\n\n");
    printf("  [i][j] │   FP22 (hex)  │  FP22 (dec)  │  →FP8_E4M3 │  →FP8_E5M2 │  →FP16   │\n");
    printf("  ───────┼───────────────┼──────────────┼────────────┼────────────┼──────────┤\n");

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            uint32_t fp22 = sim.d_fp22[i][j];
            uint8_t  fp8_e4m3 = fp22_to_fp8_e4m3(fp22);
            uint8_t  fp8_e5m2 = fp22_to_fp8_e5m2(fp22);
            uint16_t fp16     = fp22_to_fp16(fp22);

            printf("  [%d][%d] │  0x%06X     │ %+11.4f │    0x%02X    │    0x%02X    │  0x%04X  │\n",
                   i, j, fp22, fp22_to_double(fp22),
                   fp8_e4m3, fp8_e5m2, fp16);
        }
    }
    printf("\n");
}

// =============================================================================
// Test 6: Edge cases (zeros, identity)
// =============================================================================
void test_edge_cases() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Test 6: Edge Cases                                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    uint16_t a_fp9[8][8] = {};
    uint16_t b_fp9[8][8] = {};
    uint32_t c_fp22[8][8] = {};

    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            a_fp9[i][j] = (i == j) ? 0x078 : 0x000;

    double test_vals[] = {1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, 3.5};
    for (int k = 0; k < 8; k++)
        for (int j = 0; j < 8; j++)
            b_fp9[k][j] = fp16_to_fp9(double_to_fp16(test_vals[k]));

    TensorCoreSim sim;
    sim.reset();
    sim.load_inputs(a_fp9, b_fp9, c_fp22, PREC_FP16, g_cfg.rm);
    int cycles = sim.run_to_completion();

    uint32_t ref[8][8];
    reference_matmul(a_fp9, b_fp9, c_fp22, ref, g_cfg.rm);

    printf("  Identity × B test (D = I*B + 0 should equal B):\n");
    printf("  Cycles: %d\n\n", cycles);

    int match_count = 0;
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            if (compare_fp22(sim.d_fp22[i][j], ref[i][j])) match_count++;
    printf("  Bit-exact match with reference: %d/64\n", match_count);

    printf("\n  Row 0 results (should match B[k][0] for k=0..7):\n  ");
    for (int j = 0; j < 8; j++)
        printf("  D[0][%d]=%.3f", j, fp22_to_double(sim.d_fp22[0][j]));
    printf("\n  ");
    for (int j = 0; j < 8; j++)
        printf("  B[0][%d]=%.3f", j, fp9_to_double(b_fp9[0][j]));
    printf("\n\n");

    printf("  Zero matrix test (A=0, B=random, C=0 → D should be 0):\n");
    memset(a_fp9, 0, sizeof(a_fp9));
    sim.reset();
    sim.load_inputs(a_fp9, b_fp9, c_fp22, PREC_FP16, g_cfg.rm);
    sim.run_to_completion();

    bool all_zero = true;
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            if (fp22_to_double(sim.d_fp22[i][j]) != 0.0) all_zero = false;

    printf("  Result: %s\n\n", all_zero ? "✓ All zeros" : "✗ Non-zero values found");
}

// =============================================================================
// Architecture summary
// =============================================================================
void print_summary() {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Architecture Summary                                      ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    printf("  Tensor Core: 8×8×8 matrix multiply-add (D = A×B + C)\n");
    printf("  64 parallel dot-product units (one per output element)\n\n");
    printf("  Pipeline stages per dot product (matching Verilog RTL):\n");
    printf("  ┌─────────────────────────────────────────────────────────┐\n");
    printf("  │ Stage 1-2:  8× tc_mul_pipe (fmul_s1 → fmul_s2/s3)     │\n");
    printf("  │             FP9 × FP9 → FP9, 3-stage multiply          │\n");
    printf("  │             Pipeline latency: 2 cycles                  │\n");
    printf("  ├─────────────────────────────────────────────────────────┤\n");
    printf("  │ Stage 3-4:  4× tc_add_pipe Level 0                     │\n");
    printf("  │             Pairs: (0,4),(1,5),(2,6),(3,7)              │\n");
    printf("  │             FP9 + FP9 → FP9, near/far path adder       │\n");
    printf("  │             Pipeline latency: 2 cycles                  │\n");
    printf("  ├─────────────────────────────────────────────────────────┤\n");
    printf("  │ Stage 5-6:  2× tc_add_pipe Level 1                     │\n");
    printf("  │             Pipeline latency: 2 cycles                  │\n");
    printf("  ├─────────────────────────────────────────────────────────┤\n");
    printf("  │ Stage 7-8:  1× tc_add_pipe Level 2                     │\n");
    printf("  │             Pipeline latency: 2 cycles                  │\n");
    printf("  ├─────────────────────────────────────────────────────────┤\n");
    printf("  │ Stage 9-10: Final tc_add_pipe (FP22 accumulator)       │\n");
    printf("  │             FP22 + FP22 → FP22 (tree result + C bias)  │\n");
    printf("  │             Pipeline latency: 2 cycles                  │\n");
    printf("  ├─────────────────────────────────────────────────────────┤\n");
    printf("  │ Stage 11:   FP22 → output conversion                   │\n");
    printf("  │             FP22 → FP8(E4M3/E5M2) / FP16 / FP32       │\n");
    printf("  │             Latency: 1 cycle                            │\n");
    printf("  └─────────────────────────────────────────────────────────┘\n");
    printf("  Total pipeline depth: 11 cycles\n");
    printf("  Backpressure: valid/ready at each 2-stage boundary\n");
    printf("  Arithmetic: bit-accurate FP9 multiply/add, FP22 accumulate\n\n");
}

// =============================================================================
// Print active configuration
// =============================================================================
void print_config() {
    printf("  Configuration:\n");
    printf("    Precisions : ");
    for (size_t i = 0; i < g_cfg.precisions.size(); i++) {
        if (i > 0) printf(", ");
        printf("%s", prec_name(g_cfg.precisions[i]));
    }
    printf("\n");
    printf("    Out Prec   : ");
    for (size_t i = 0; i < g_cfg.out_precisions.size(); i++) {
        if (i > 0) printf(", ");
        printf("%s", prec_name(g_cfg.out_precisions[i]));
    }
    printf("\n");
    printf("    Rounding   : %s\n", rm_name(g_cfg.rm));
    printf("    RNG seed   : %u\n", rng_state);
    if (g_cfg.test_id == 0)
        printf("    Tests      : ALL (1-6)\n");
    else
        printf("    Tests      : %d only\n", g_cfg.test_id);
    printf("\n");
}

// =============================================================================
// Usage / help
// =============================================================================
void print_usage(const char* prog) {
    printf("\n");
    printf("  OpenTensorCore Cycle-Accurate Simulator v2.0\n\n");
    printf("  Usage: %s [OPTIONS]\n\n", prog);
    printf("  Options:\n");
    printf("    --prec <PRECISION>   Restrict to a single precision format\n");
    printf("                         Values: FP4_E2M1 | FP8_E4M3 | FP8_E5M2 | FP16\n");
    printf("                         Default: all precisions\n\n");
    printf("    --out-prec <PREC>    Restrict output precision format\n");
    printf("                         Values: FP8_E4M3 | FP8_E5M2 | FP16 | FP32\n");
    printf("                         Default: all supported output precisions\n\n");
    printf("    --test <ID>          Run only a specific test (1-6)\n");
    printf("                         1 = Single matmul per precision\n");
    printf("                         2 = Back-to-back pipelined matmuls\n");
    printf("                         3 = Stress test (20 random matrices/prec)\n");
    printf("                         4 = Pipeline stage visualization\n");
    printf("                         5 = Output format conversion table\n");
    printf("                         6 = Edge cases (identity, zero matrices)\n");
    printf("                         Default: all tests\n\n");
    printf("    --rm <MODE>          Rounding mode\n");
    printf("                         Values: RNE | RTZ | RDN | RUP | RMM\n");
    printf("                         Default: RNE\n\n");
    printf("    --seed <VALUE>       Fixed RNG seed (0 = use current time)\n");
    printf("                         Default: 0\n\n");
    printf("    --help               Show this help message\n\n");
    printf("  Examples:\n");
    printf("    %s                            Run all tests, all precisions\n", prog);
    printf("    %s --prec FP8_E4M3            Test FP8 E4M3 only\n", prog);
    printf("    %s --test 3 --prec FP16       Stress test FP16 only\n", prog);
    printf("    %s --prec FP16 --out-prec FP32  FP16 input, FP32 output\n", prog);
    printf("    %s --rm RTZ --seed 42         Fixed seed, round-toward-zero\n", prog);
    printf("\n");
}

// =============================================================================
// Argument parsing
// =============================================================================
PrecisionType parse_precision(const char* s) {
    if (strcmp(s, "FP4_E2M1") == 0 || strcmp(s, "FP4") == 0)     return PREC_FP4_E2M1;
    if (strcmp(s, "FP8_E4M3") == 0 || strcmp(s, "E4M3") == 0)    return PREC_FP8_E4M3;
    if (strcmp(s, "FP8_E5M2") == 0 || strcmp(s, "E5M2") == 0)    return PREC_FP8_E5M2;
    if (strcmp(s, "FP16") == 0)                                    return PREC_FP16;
    if (strcmp(s, "FP32") == 0)                                    return PREC_FP32;
    fprintf(stderr, "  Error: Unknown precision '%s'\n", s);
    fprintf(stderr, "  Valid: FP4_E2M1 | FP8_E4M3 | FP8_E5M2 | FP16 | FP32\n\n");
    exit(1);
}

RoundingMode parse_rounding(const char* s) {
    if (strcmp(s, "RNE") == 0) return RNE;
    if (strcmp(s, "RTZ") == 0) return RTZ;
    if (strcmp(s, "RDN") == 0) return RDN;
    if (strcmp(s, "RUP") == 0) return RUP;
    if (strcmp(s, "RMM") == 0) return RMM;
    fprintf(stderr, "  Error: Unknown rounding mode '%s'\n", s);
    fprintf(stderr, "  Valid: RNE | RTZ | RDN | RUP | RMM\n\n");
    exit(1);
}

bool parse_args(int argc, char* argv[]) {
    g_cfg.precisions.clear();
    g_cfg.out_precisions.clear();
    g_cfg.test_id  = 0;
    g_cfg.rm       = RNE;
    g_cfg.seed     = 0;
    g_cfg.show_help = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            g_cfg.show_help = true;
            return true;
        } else if (strcmp(argv[i], "--prec") == 0 && i + 1 < argc) {
            g_cfg.precisions.push_back(parse_precision(argv[++i]));
        } else if (strcmp(argv[i], "--out-prec") == 0 && i + 1 < argc) {
            g_cfg.out_precisions.push_back(parse_precision(argv[++i]));
        } else if (strcmp(argv[i], "--test") == 0 && i + 1 < argc) {
            g_cfg.test_id = atoi(argv[++i]);
            if (g_cfg.test_id < 1 || g_cfg.test_id > 6) {
                fprintf(stderr, "  Error: Test ID must be 1-6, got %d\n\n", g_cfg.test_id);
                return false;
            }
        } else if (strcmp(argv[i], "--rm") == 0 && i + 1 < argc) {
            g_cfg.rm = parse_rounding(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            g_cfg.seed = (uint32_t)atol(argv[++i]);
        } else {
            fprintf(stderr, "  Error: Unknown argument '%s'\n\n", argv[i]);
            return false;
        }
    }

    if (g_cfg.precisions.empty()) {
        g_cfg.precisions = {PREC_FP4_E2M1, PREC_FP8_E4M3, PREC_FP8_E5M2, PREC_FP16};
    }
    if (g_cfg.out_precisions.empty()) {
        g_cfg.out_precisions = {PREC_FP8_E4M3, PREC_FP8_E5M2, PREC_FP16, PREC_FP32};
    }

    return true;
}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char* argv[]) {
    if (!parse_args(argc, argv)) {
        print_usage(argv[0]);
        return 1;
    }

    if (g_cfg.show_help) {
        print_usage(argv[0]);
        return 0;
    }

    rng_state = g_cfg.seed ? g_cfg.seed : (uint32_t)time(nullptr);

    printf("\n");
    printf("  ╔════════════════════════════════════════════════════════════╗\n");
    printf("  ║  OpenTensorCore Cycle-Accurate Simulator v2.0            ║\n");
    printf("  ║  Matching Verilog RTL Pipeline Architecture              ║\n");
    printf("  ║  FP9 multiply (3-stage) + FP9 add (2-stage near/far)    ║\n");
    printf("  ║  FP22 accumulator + output format conversion             ║\n");
    printf("  ╚════════════════════════════════════════════════════════════╝\n\n");

    print_config();

    bool run_all = (g_cfg.test_id == 0);

    if (run_all) print_summary();

    if (run_all || g_cfg.test_id == 1) test_single_matmul();
    if (run_all || g_cfg.test_id == 2) test_pipelined_throughput();
    if (run_all || g_cfg.test_id == 3) test_stress();
    if (run_all || g_cfg.test_id == 4) test_pipeline_visualization();
    if (run_all || g_cfg.test_id == 5) test_output_conversion();
    if (run_all || g_cfg.test_id == 6) test_edge_cases();

    printf("  ════════════════════════════════════════════════════════════\n");
    printf("  All tests completed.\n\n");

    return 0;
}
