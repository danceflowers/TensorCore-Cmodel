// ============================================================================
// OpenTensorCore SimX — Driver API
// Mirrors:  runtime/simx/vortex.cpp    (Vortex unified driver interface)
//           include/vortex.h           (VX_* API declarations)
//
// API mapping:
//   vx_dev_open()     → otc_dev_open()
//   vx_dev_close()    → otc_dev_close()
//   vx_copy_to_dev()  → otc_upload()
//   vx_start()        → otc_start()
//   vx_ready()        → otc_ready()
//   vx_copy_from_dev()→ otc_download()
// ============================================================================
#pragma once
#include "pipeline.h"

// ==================== Device Handle ====================
struct OTC_Device {
    TensorCoreUnit  tc;
    bool            configured = false;
};

// ==================== C-style Driver API ====================

inline int otc_dev_open(OTC_Device** dev) {
    *dev = new OTC_Device();
    return 0;
}

inline int otc_dev_close(OTC_Device* dev) {
    delete dev;
    return 0;
}

inline int otc_configure(OTC_Device* dev, const OTC_Config& cfg) {
    if (!cfg.validate()) {
        fprintf(stderr, "ERROR: invalid config (M=%d K=%d N=%d type_ab=%02x)\n",
                cfg.M, cfg.K, cfg.N, cfg.type_ab);
        return -1;
    }
    dev->tc.init(cfg);
    dev->tc.reset();
    dev->configured = true;
    return 0;
}

// Upload packed matrix data
inline int otc_upload(OTC_Device* dev,
                      const uint32_t* a, int na,
                      const uint32_t* b, int nb,
                      const uint32_t* c, int nc) {
    if (!dev->configured) return -1;
    dev->tc.load(std::vector<uint32_t>(a, a+na),
                 std::vector<uint32_t>(b, b+nb),
                 std::vector<uint32_t>(c, c+nc));
    return 0;
}

// Start execution
inline int otc_start(OTC_Device* dev) {
    if (!dev->configured) return -1;
    dev->tc.start();
    return 0;
}

// Poll for completion (non-blocking)
inline int otc_ready(OTC_Device* dev) {
    return dev->tc.is_done() ? 1 : 0;
}

// Advance one cycle (for external stepping)
inline int otc_tick(OTC_Device* dev) {
    dev->tc.tick();
    return 0;
}

// Run to completion
inline int otc_run(OTC_Device* dev, int max_cycles = 100000) {
    dev->tc.run(max_cycles);
    return dev->tc.is_done() ? 0 : -1;
}

// Download results as doubles
inline int otc_download_f64(OTC_Device* dev, double* dst, int n) {
    auto r = dev->tc.get_result_f64();
    int cnt = std::min(n, (int)r.size());
    memcpy(dst, r.data(), cnt * sizeof(double));
    return cnt;
}

// Download results as FP32 packed words
inline int otc_download_fp32(OTC_Device* dev, uint32_t* dst, int n) {
    auto r = dev->tc.get_result_fp32();
    int cnt = std::min(n, (int)r.size());
    memcpy(dst, r.data(), cnt * sizeof(uint32_t));
    return cnt;
}

// Get stats reference
inline const OTC_Stats& otc_stats(OTC_Device* dev) {
    return dev->tc.stats_;
}
