// ============================================================================
// Driver API implementation (separated from header for .h/.cpp design).
// ============================================================================
#include "otc_driver.h"

int otc_dev_open(OTC_Device** dev) {
    *dev = new OTC_Device();
    return 0;
}

int otc_dev_close(OTC_Device* dev) {
    delete dev;
    return 0;
}

int otc_configure(OTC_Device* dev, const OTC_Config& cfg) {
    if (!cfg.validate()) {
        std::cerr << "ERROR: invalid config (M=" << cfg.M << " K=" << cfg.K << " N=" << cfg.N
                  << " type_ab=" << std::hex << (int)cfg.type_ab << std::dec << ")\n";
        return -1;
    }
    dev->tc.init(cfg);
    dev->tc.reset();
    dev->configured = true;
    return 0;
}

int otc_upload(OTC_Device* dev, const uint32_t* a, int na, const uint32_t* b, int nb, const uint32_t* c, int nc) {
    if (!dev->configured) return -1;
    dev->tc.load(std::vector<uint32_t>(a, a + na), std::vector<uint32_t>(b, b + nb), std::vector<uint32_t>(c, c + nc));
    return 0;
}

int otc_start(OTC_Device* dev) {
    if (!dev->configured) return -1;
    dev->tc.start();
    return 0;
}

int otc_ready(OTC_Device* dev) { return dev->tc.is_done() ? 1 : 0; }

int otc_tick(OTC_Device* dev) {
    dev->tc.tick();
    return 0;
}

int otc_run(OTC_Device* dev, int max_cycles) {
    dev->tc.run(max_cycles);
    return dev->tc.is_done() ? 0 : -1;
}

int otc_download_f64(OTC_Device* dev, double* dst, int n) {
    auto r = dev->tc.get_result_f64();
    int cnt = std::min(n, (int)r.size());
    memcpy(dst, r.data(), cnt * sizeof(double));
    return cnt;
}

int otc_download_fp32(OTC_Device* dev, uint32_t* dst, int n) {
    auto r = dev->tc.get_result_fp32();
    int cnt = std::min(n, (int)r.size());
    memcpy(dst, r.data(), cnt * sizeof(uint32_t));
    return cnt;
}

const OTC_Stats& otc_stats(OTC_Device* dev) { return dev->tc.stats_; }
