// ============================================================================
// OpenTensorCore SimX — Type Definitions & Configuration
// Mirrors:  define.v            (global macros)
//           config_registers.v  (parameter validation)
//           VX_types.vh         (Vortex shared type header)
// ============================================================================
#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#define TYPE_FP4 0x06
#define TYPE_FP8 0x02
#define TYPE_FP16 0x0A
#define TYPE_FP32 0x0E
#define SUB_FP8E5M2 0
#define SUB_FP8E4M3 1

struct OTC_Config {
    int M = 8, K = 8, N = 8;
    uint8_t type_ab = TYPE_FP8;
    uint8_t type_ab_sub = SUB_FP8E5M2;
    uint8_t type_cd = TYPE_FP32;
    bool transpose_b = false;

    int mul_latency = 2;
    int add_latency = 2;
    int conv_latency = 1;

    int debug_level = 0;
    bool trace_en = false;

    int tree_depth() const;
    int total_dp() const;
    int pipeline_depth() const;
    bool validate() const;
};

struct OTC_Stats {
    uint64_t total_cycles = 0;
    uint64_t busy_cycles = 0;
    uint64_t stall_cycles = 0;
    uint64_t mul_ops = 0;
    uint64_t add_ops = 0;
    uint64_t matrices_done = 0;
    uint64_t conv_cycles = 0;

    void print(std::ostream& os) const;
};

class TraceLog {
public:
    int level_ = 0;
    uint64_t cycle_ = 0;
    std::ostream* os_ = nullptr;
    std::ofstream fout_;

    void init(int level, bool to_file);
    void set_cycle(uint64_t c);
    void log(int lvl, const char* fmt, ...);
};

extern TraceLog DT;
