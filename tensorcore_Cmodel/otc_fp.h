#pragma once
#include "otc_types.h"

namespace SoftFloat {

double fp16_to_f64(uint16_t h);
uint16_t f64_to_fp16(double v);

double fp32_to_f64(uint32_t w);
uint32_t f64_to_fp32(double v);

double fp9_to_f64(uint16_t bits9);
uint16_t f64_to_fp9(double v);

double fp13_to_f64(uint16_t bits13);
uint16_t f64_to_fp13(double v);

double fp22_to_f64(uint32_t bits22);
uint32_t f64_to_fp22(double v);

} // namespace SoftFloat

namespace FPEmu {

uint16_t fp4_to_fp9(uint8_t fp4);
uint16_t fp8e4m3_to_fp9(uint8_t fp8);
uint16_t fp8e5m2_to_fp9(uint8_t fp8);
uint16_t fp16_to_fp9(uint16_t fp16);

uint16_t fp9_mul(uint16_t a, uint16_t b);
uint16_t fp13_add(uint16_t a, uint16_t b);
uint32_t fp22_add(uint32_t a, uint32_t b);
uint16_t fp22_to_fp8(uint32_t a, int sub);
uint16_t fp22_to_fp16(uint32_t a);
uint32_t fp9_to_fp22(uint16_t a);
uint16_t fp13_to_fp9(uint16_t a);

} // namespace FPEmu

namespace FPConvert {

double fp4_to_f64(uint8_t fp4);
double fp8e5m2_to_f64(uint8_t fp8);
double fp8e4m3_to_f64(uint8_t fp8);
uint8_t f64_to_fp8e5m2(double v);
uint8_t f64_to_fp8e4m3(double v);
double fp16_to_f64_via_fp9(uint16_t fp16);
double elem_to_f64(uint32_t word, int elem_idx, int type_ab, int sub);
int elem_bits(int type_ab);

} // namespace FPConvert
