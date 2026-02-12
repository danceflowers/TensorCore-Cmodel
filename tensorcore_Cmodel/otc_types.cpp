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
           (type_cd == TYPE_FP8 || type_cd == TYPE_FP16 || type_cd == TYPE_FP32) &&
           dispatch_width > 0 && output_fifo_depth > 0 &&
           mem_bandwidth_bytes_per_cycle > 0;
}

void OTC_Stats::print(std::ostream& os) const {
    os << "=== OpenTensorCore SimX Performance Counters ===" << std::endl;
    os << "Total cycles:             " << total_cycles << std::endl;
    os << "Busy cycles:              " << busy_cycles << std::endl;
    os << "Stall cycles:             " << stall_cycles << std::endl;
    os << "Format active cycles:     " << format_active_cycles << std::endl;
    os << "Dispatch active cycles:   " << dispatch_active_cycles << std::endl;
    os << "Output backpressure cyc:  " << output_backpressure_cycles << std::endl;
    os << "Output FIFO max occ:      " << output_fifo_max_occupancy << std::endl;
    os << "MUL operations:           " << mul_ops << std::endl;
    os << "ADD operations:           " << add_ops << std::endl;
    os << "Matrices completed:       " << matrices_done << std::endl;
    os << "Batches enqueued:         " << batches_enqueued << std::endl;
    os << "DRAM read bytes:          " << dram_read_bytes << std::endl;
    os << "DRAM write bytes:         " << dram_write_bytes << std::endl;

    double util = total_cycles ? 100.0 * busy_cycles / total_cycles : 0;
    double throughput = total_cycles ? (double)matrices_done / (double)total_cycles : 0;
    double avg_latency = matrices_done ? (double)total_latency_cycles / (double)matrices_done : 0;
    double avg_bw = total_cycles ? ((double)(dram_read_bytes + dram_write_bytes) / (double)total_cycles) : 0.0;
    double bw_util = peak_bw_bytes_per_cycle ? 100.0 * avg_bw / (double)peak_bw_bytes_per_cycle : 0.0;
    double dp_util = (total_cycles > 0 && dp_capacity_units > 0) ?
        100.0 * (double)dp_busy_unit_cycles / ((double)total_cycles * (double)dp_capacity_units) : 0.0;

    os << "Utilization:              " << std::fixed << std::setprecision(1) << util << "%" << std::endl;
    os << "Throughput (batch/cycle): " << std::fixed << std::setprecision(6) << throughput << std::endl;
    os << "Avg latency (cycles):     " << std::fixed << std::setprecision(2) << avg_latency << std::endl;
    os << "Avg BW (bytes/cycle):     " << std::fixed << std::setprecision(2) << avg_bw << std::endl;
    os << "BW utilization:           " << std::fixed << std::setprecision(2) << bw_util << "%" << std::endl;
    os << "Compute util:             " << std::fixed << std::setprecision(2) << dp_util << "%" << std::endl;
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
