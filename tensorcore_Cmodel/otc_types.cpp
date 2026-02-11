// ============================================================================
// Configuration/statistics/logging implementation.
// ============================================================================
#include "otc_types.h"

int OTC_Config::tree_depth() const {
    int d = 0;
    int k = K;
    while (k > 1) {
        k >>= 1;
        d++;
    }
    return d;
}

int OTC_Config::total_dp() const { return M * N; }

int OTC_Config::pipeline_depth() const {
    return conv_latency + mul_latency + tree_depth() * add_latency + add_latency + 1;
}

bool OTC_Config::validate() const {
    return M > 0 && K > 0 && N > 0 && (K & (K - 1)) == 0 &&
           (type_ab == TYPE_FP4 || type_ab == TYPE_FP8 || type_ab == TYPE_FP16) &&
           (type_cd == TYPE_FP16 || type_cd == TYPE_FP32);
}

void OTC_Stats::print(std::ostream& os) const {
    os << "=== OpenTensorCore SimX Performance Counters ===" << std::endl;
    os << "Total cycles:       " << total_cycles << std::endl;
    os << "Busy cycles:        " << busy_cycles << std::endl;
    os << "Stall cycles:       " << stall_cycles << std::endl;
    os << "MUL operations:     " << mul_ops << std::endl;
    os << "ADD operations:     " << add_ops << std::endl;
    os << "Matrices completed: " << matrices_done << std::endl;
    double util = total_cycles ? 100.0 * busy_cycles / total_cycles : 0;
    os << "Utilization:        " << std::fixed << std::setprecision(1) << util << "%" << std::endl;
}

void TraceLog::init(int level, bool to_file) {
    level_ = level;
    cycle_ = 0;
    os_ = nullptr;
    if (level_ <= 0) return;

    if (to_file || level_ > 0) {
        fout_.open("otc_run.log");
        if (fout_.is_open()) {
            os_ = &fout_;
            return;
        }
    }
    os_ = &std::clog;
}

void TraceLog::set_cycle(uint64_t c) { cycle_ = c; }

void TraceLog::log(int lvl, const char* fmt, ...) {
    if (lvl > level_ || os_ == nullptr) return;
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    *os_ << "[" << std::setw(6) << cycle_ << "] " << buf << "\n";
}

TraceLog DT;
