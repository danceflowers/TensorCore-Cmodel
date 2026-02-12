#pragma once
#include "pipeline.h"

struct OTC_Device {
    TensorCoreUnit tc;
    bool configured = false;
};

int otc_dev_open(OTC_Device** dev);
int otc_dev_close(OTC_Device* dev);
int otc_configure(OTC_Device* dev, const OTC_Config& cfg);
int otc_upload(OTC_Device* dev, const uint32_t* a, int na, const uint32_t* b, int nb, const uint32_t* c, int nc);
int otc_start(OTC_Device* dev);
int otc_ready(OTC_Device* dev);
int otc_tick(OTC_Device* dev);
int otc_run(OTC_Device* dev, int max_cycles = 100000);
int otc_download_f64(OTC_Device* dev, double* dst, int n);
int otc_download_fp32(OTC_Device* dev, uint32_t* dst, int n);
int otc_submit(OTC_Device* dev, const uint32_t* a, int na, const uint32_t* b, int nb, const uint32_t* c, int nc);
int otc_pop_result_f64(OTC_Device* dev, double* dst, int n);
const OTC_Stats& otc_stats(OTC_Device* dev);
