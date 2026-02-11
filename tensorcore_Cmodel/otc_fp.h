#pragma once
#include "otc_types.h"

namespace SoftFloat {

double fp16_to_f64(uint16_t h);
uint16_t f64_to_fp16(double v);

double fp32_to_f64(uint32_t w);
uint32_t f64_to_fp32(double v);

double fp9_to_f64(uint16_t bits9);
uint16_t f64_to_fp9(double v);

double fp22_to_f64(uint32_t bits22);
uint32_t f64_to_fp22(double v);

} // namespace SoftFloat

namespace FPConvert {

double fp4_to_f64(uint8_t fp4);
double fp8e5m2_to_f64(uint8_t fp8);
double fp8e4m3_to_f64(uint8_t fp8);
double fp16_to_f64_via_fp9(uint16_t fp16);
double elem_to_f64(uint32_t word, int elem_idx, int type_ab, int sub);
int elem_bits(int type_ab);

} // namespace FPConvert
