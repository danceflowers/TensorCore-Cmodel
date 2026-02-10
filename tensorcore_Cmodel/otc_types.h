// ============================================================================
// OpenTensorCore SimX — Type Definitions & Configuration
// Mirrors:  define.v          (global macros)
//           config_registers.v (parameter validation)
//           VX_types.vh        (Vortex shared type header)
// ============================================================================
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

// ==================== Float type encodings (define.v) ====================
#define TYPE_FP4        0x06
#define TYPE_FP8        0x02
#define TYPE_FP16       0x0A
#define TYPE_FP32       0x0E
#define SUB_FP8E5M2    0
#define SUB_FP8E4M3    1

// ==================== Architecture Config (mirrors config_registers.v) ====
struct OTC_Config {
    int M = 8, K = 8, N = 8;       // matrix dimensions
    uint8_t type_ab     = TYPE_FP8; // input format
    uint8_t type_ab_sub = SUB_FP8E5M2;
    uint8_t type_cd     = TYPE_FP32; // output format
    bool    transpose_b = false;

    // pipeline latency params (from RTL)
    int mul_latency  = 2;           // tc_mul_pipe stages
    int add_latency  = 2;           // tc_add_pipe stages
    int conv_latency = 1;           // format conversion

    // debug (mirrors --debug=N)
    int  debug_level = 0;           // 0=off, 1=summary, 2=pipeline, 3=full
    bool trace_en    = false;

    // derived helpers
    int  tree_depth() const { int d=0,k=K; while(k>1){k>>=1;d++;} return d; }
    int  total_dp()   const { return M * N; }
    int  pipeline_depth() const {
        return conv_latency + mul_latency + tree_depth() * add_latency
             + add_latency /* final_add */ + 1 /* output */;
    }
    bool validate() const {
        return M>0 && K>0 && N>0 && (K & (K-1))==0
            && (type_ab==TYPE_FP4||type_ab==TYPE_FP8||type_ab==TYPE_FP16)
            && (type_cd==TYPE_FP16||type_cd==TYPE_FP32);
    }
};

// ==================== Simulation Statistics (mirrors perf counters) ========
struct OTC_Stats {
    uint64_t total_cycles      = 0;
    uint64_t busy_cycles       = 0;
    uint64_t stall_cycles      = 0;
    uint64_t mul_ops           = 0;
    uint64_t add_ops           = 0;
    uint64_t matrices_done     = 0;
    uint64_t conv_cycles       = 0;

    void print(std::ostream& os) const {
        os << "=== OpenTensorCore SimX Performance Counters ===" << std::endl;
        os << "Total cycles:       " << total_cycles << std::endl;
        os << "Busy cycles:        " << busy_cycles << std::endl;
        os << "Stall cycles:       " << stall_cycles << std::endl;
        os << "MUL operations:     " << mul_ops << std::endl;
        os << "ADD operations:     " << add_ops << std::endl;
        os << "Matrices completed: " << matrices_done << std::endl;
        double util = total_cycles ? 100.0*busy_cycles/total_cycles : 0;
        os << "Utilization:        " << std::fixed << std::setprecision(1)
           << util << "%" << std::endl;
    }
};

// ==================== Debug Trace (mirrors --debug=N / run.log) ===========
class TraceLog {
public:
    int         level_ = 0;
    uint64_t    cycle_ = 0;
    std::ostream* os_  = &std::cout;
    std::ofstream fout_;

    void init(int level, bool to_file) {
        level_ = level;
        if (to_file) { fout_.open("otc_run.log"); if (fout_.is_open()) os_ = &fout_; }
    }
    void set_cycle(uint64_t c) { cycle_ = c; }

    void log(int lvl, const char* fmt, ...) {
        if (lvl > level_) return;
        char buf[1024];
        va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof(buf), fmt, ap); va_end(ap);
        *os_ << "[" << std::setw(6) << cycle_ << "] " << buf << "\n";
    }
};

extern TraceLog DT;  // global trace instance
