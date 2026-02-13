# OpenTensorCore Cycle-Accurate Simulator — 技术文档

## 1. 概述

本项目是 OpenTensorCore 硬件的**周期精确 (cycle-accurate)** C++ 模拟器，严格匹配 Verilog RTL 实现的流水线架构和浮点运算行为。模拟器实现了完整的 8×8×8 矩阵乘加运算 `D = A × B + C`，支持 FP4/FP8/FP16 多种输入精度，内部采用 FP9 乘法/加法、FP22 累加器，最终输出可转换为 FP8/FP16/FP32。

所有算术运算采用整数位操作实现（无浮点快捷方式），确保与 RTL 逐位匹配 (bit-exact)。

---

## 2. 项目结构

```
tensorcore_sim/
├── Makefile              构建系统：编译、测试、精度配置
├── fp_types.h            浮点格式定义与格式转换函数
├── fp_arith.h            RTL 精确的浮点乘法/加法运算
├── tensor_core_sim.h     周期精确流水线模拟器
├── main.cpp              测试框架与命令行接口
└── README.md             本文档
```

---

## 3. 硬件流水线架构

### 3.1 整体架构

每个输出元素 D[i][j] 由一个独立的点积单元 (Dot Product Unit) 计算，共 64 个并行单元组成 8×8 输出矩阵。每个点积单元的流水线结构如下：

```
Stage 1-2:   8× tc_mul_pipe    FP9 × FP9 → FP9    (2 cycles)
                    ↓
Stage 3-4:   4× tc_add_pipe    加法树 Level 0      (2 cycles)
             对: (0,4)(1,5)(2,6)(3,7)
                    ↓
Stage 5-6:   2× tc_add_pipe    加法树 Level 1      (2 cycles)
                    ↓
Stage 7-8:   1× tc_add_pipe    加法树 Level 2      (2 cycles)
                    ↓
Stage 9-10:  1× tc_add_pipe    FP22 累加器          (2 cycles)
             FP22(树结果) + FP22(C偏置) → FP22
                    ↓
Stage 11:    FP22 → 输出格式转换                     (1 cycle)
```

**总流水线深度：11 个时钟周期**

### 3.2 数据流

```
输入A[8×8] (FP4/FP8/FP16) ─→ 转换为 FP9 ─→ 乘法器
输入B[8×8] (FP4/FP8/FP16) ─→ 转换为 FP9 ─→ 乘法器
输入C[8×8] (FP8/FP16)     ─→ 转换为 FP22 ─→ 累加器
                                                 ↓
输出D[8×8] ←─ FP22→FP8/FP16/FP32 ←─ FP22累加结果
```

### 3.3 反压机制

每个 2 级流水线边界实现 valid/ready 握手协议，匹配 RTL 的 skid buffer 逻辑：
- `reg_en1 = in_valid && !(valid1 && valid2 && !out_ready)`
- `reg_en2 = valid1 && !(valid2 && !out_ready)`

---

## 4. 源文件详解

### 4.1 fp_types.h — 浮点格式定义与转换

#### 枚举类型

| 类型 | 说明 |
|------|------|
| `RoundingMode` | 舍入模式枚举：`RNE`(近偶)、`RTZ`(截断)、`RDN`(向下)、`RUP`(向上)、`RMM`(近大) |
| `PrecisionType` | 精度类型枚举：`PREC_FP4_E2M1`、`PREC_FP8_E4M3`、`PREC_FP8_E5M2`、`PREC_FP16`、`PREC_FP32` |

#### 浮点格式位宽

| 格式 | 总位宽 | 符号 | 指数 | 尾数 | 偏置 | 说明 |
|------|--------|------|------|------|------|------|
| FP4 (E2M1) | 4 | 1 | 2 | 1 | 1 | 最小精度输入格式 |
| FP8 (E4M3) | 8 | 1 | 4 | 3 | 7 | 标准 FP8 训练格式 |
| FP8 (E5M2) | 8 | 1 | 5 | 2 | 15 | 兼容 IEEE 754 的 FP8 |
| FP9 (E5M3) | 9 | 1 | 5 | 3 | 15 | 内部统一运算格式 |
| FP16 | 16 | 1 | 5 | 10 | 15 | IEEE 754 半精度 |
| FP22 (E8M13) | 22 | 1 | 8 | 13 | 127 | 累加器高精度格式 |
| FP32 | 32 | 1 | 8 | 23 | 127 | IEEE 754 单精度输出 |

#### 关键函数

**基础工具：**

| 函数 | 功能 |
|------|------|
| `clz(val, width)` | 前导零计数，匹配 RTL 的 `lzc` 模块 |
| `do_rounding(in, WIDTH, sign, rm, ...)` | 通用舍入模块，匹配 RTL 的 `rounding.v`。输入为扩展尾数，根据舍入模式和 guard/round/sticky 位产生进位和截断结果 |

**格式↔双精度转换（测试用）：**

| 函数 | 方向 | 说明 |
|------|------|------|
| `fp9_to_double()` | FP9 → double | 用于测试验证的值提取 |
| `fp22_to_double()` | FP22 → double | 用于测试验证的值提取 |
| `fp16_to_double()` | FP16 → double | |
| `fp8_e4m3_to_double()` | FP8 E4M3 → double | |
| `fp8_e5m2_to_double()` | FP8 E5M2 → double | |
| `fp4_to_double()` | FP4 → double | |
| `double_to_fp16()` | double → FP16 | 测试输入生成 |
| `double_to_fp8_e4m3()` | double → FP8 E4M3 | |
| `double_to_fp8_e5m2()` | double → FP8 E5M2 | |
| `double_to_fp4()` | double → FP4 | |

**输入格式 → FP9 转换（匹配 RTL `input_to_fp9` 逻辑）：**

| 函数 | 输入 | 说明 |
|------|------|------|
| `fp4_to_fp9(fp4)` | FP4 E2M1 | 指数扩展 (2→5 bit)，尾数零填充 (1→3 bit) |
| `fp8_e4m3_to_fp9(fp8)` | FP8 E4M3 | 指数扩展 (4→5 bit)，尾数保留 (3→3 bit) |
| `fp8_e5m2_to_fp9(fp8)` | FP8 E5M2 | 指数保留 (5 bit)，尾数零填充 (2→3 bit) |
| `fp16_to_fp9(fp16)` | FP16 | 指数保留 (5 bit)，尾数截断 (10→3 bit) + 舍入 |
| `convert_to_fp9(raw, prec)` | 通用 | 根据 `PrecisionType` 分发到上述函数 |

**FP9/FP16 → FP22 转换（累加器输入格式对齐）：**

| 函数 | 说明 |
|------|------|
| `fp9_to_fp22(fp9)` | 指数扩展 (5→8 bit, +112 偏置调整)，尾数零填充 (3→13 bit) |
| `fp16_to_fp22(fp16)` | 指数扩展 (5→8 bit, +112 偏置调整)，尾数截断 (10→13 bit) |
| `convert_c_to_fp22(raw, prec)` | C 矩阵偏置转换的统一入口 |

**FP22 → 输出格式转换（匹配 RTL `fp22_to_fp8_con`）：**

| 函数 | 输出 | 说明 |
|------|------|------|
| `fp22_to_fp8_e4m3(fp22, rm)` | FP8 E4M3 | 指数截断 + 尾数舍入，处理溢出→最大值/Inf |
| `fp22_to_fp8_e5m2(fp22, rm)` | FP8 E5M2 | 指数截断 + 尾数舍入 |
| `fp22_to_fp16(fp22, rm)` | FP16 | 指数调整 (-112 偏置)，尾数截断 + 舍入 |
| `fp22_to_fp32(fp22)` | FP32 | 无损扩展 |

---

### 4.2 fp_arith.h — RTL 精确浮点运算

本文件实现了与 Verilog RTL 逐位匹配的参数化浮点乘法和加法运算。

#### 数据结构

| 结构体 | 对应 RTL | 说明 |
|--------|----------|------|
| `FPUnpacked` | — | 解包后的浮点数：符号、指数、尾数、特殊值标志 |
| `FMulS1Out` | `fmul_s1.v` 输出 | 乘法第 1 级输出：指数计算结果、特殊值标志、亚正规移位量 |
| `FMulS2Out` | `fmul_s2.v` 输出 | 乘法第 2 级输出：尾数乘积 + S1 透传信号 |
| `FarPathOut` | `fadd_s1.v` 远路径 | 加法远路径结果：适用于指数差 >1 或有效加法 |
| `NearPathOut` | `fadd_s1.v` 近路径 | 加法近路径结果：适用于指数差 ≤1 的有效减法 |
| `FAddS1Out` | `fadd_s1.v` 输出 | 加法第 1 级完整输出：包含远/近两条路径结果和路径选择信号 |

#### 乘法流水线 (3 级，匹配 fmul_s1.v / fmul_s2.v / fmul_s3.v)

| 函数 | 对应 RTL | 功能 |
|------|----------|------|
| `fmul_s1(a, b, EW, P, rm)` | `fmul_s1.v` | **第 1 级**：输入分类 (零/Inf/NaN/亚正规)，指数相加并减偏置，用 LZC 计算亚正规移位量，检测溢出/下溢 |
| `fmul_s2(a, b, EW, P, s1)` | `fmul_s2.v` + `naivemultiplier` | **第 2 级**：`P×P → 2P` 位尾数乘法，透传 S1 控制信号 |
| `fmul_s3(s2, EW, P)` | `fmul_s3.v` | **第 3 级**：亚正规移位，LZC 归一化，调用 `do_rounding` 舍入，处理溢出 (→Inf/最大值) 和下溢 (→亚正规/零) |
| `fp_multiply(a, b, EW, P, rm)` | — | 组合调用 s1→s2→s3，用于非流水线参考模型 |

#### 加法流水线 (2 级，匹配 fadd_s1.v / fadd_s2.v)

| 函数 | 对应 RTL | 功能 |
|------|----------|------|
| `far_path_compute(...)` | `fadd_s1.v` 远路径 | 指数差 >1 或有效加法：对齐移位较小操作数，执行加/减法，跟踪 sticky 位，处理进位归一化 |
| `near_path_compute(...)` | `fadd_s1.v` 近路径 | 指数差 ≤1 且有效减法：双路计算 (A-B 和 B-A)，用 LZC 归一化消除前导零 |
| `fadd_s1(a, b, EW, P, OUTPC, rm)` | `fadd_s1.v` | **第 1 级**：输入分类，指数差计算，交换判定，并行执行远/近两条路径，根据条件选择路径 (`sel_far = !eff_sub \|\| exp_diff > 1`) |
| `fadd_s2(s1, EW, P)` | `fadd_s2.v` | **第 2 级**：对选定路径结果进行舍入，处理溢出和下溢，组装特殊值输出 (NaN/Inf) |
| `fp_add(a, b, EW, P, OUTPC, rm)` | — | 组合调用 s1→s2，用于非流水线参考模型 |

#### 便捷封装函数

| 函数 | 参数 | 说明 |
|------|------|------|
| `fp9_multiply(a, b, rm)` | EW=5, P=4 | FP9×FP9→FP9 乘法 |
| `fp9_add(a, b, rm)` | EW=5, P=8(零填充), OUTPC=4 | FP9+FP9→FP9 加法，匹配 `tc_add_pipe` 零填充行为 |
| `fp22_add(a, b, rm)` | EW=8, P=28(零填充), OUTPC=14 | FP22+FP22→FP22 累加，使用 64 位内联运算 |

---

### 4.3 tensor_core_sim.h — 周期精确流水线模拟器

#### PipeStage2\<T\> — 2 级流水线模板

```cpp
template <typename T>
struct PipeStage2 {
    T     data1, data2;          // 两级寄存器数据
    bool  valid1, valid2;        // 两级 valid 标志

    bool in_ready(out_ready);    // 反压查询：是否可接受新输入
    bool out_valid();            // 输出有效标志
    const T& out_data();         // 输出数据
    bool tick(in_valid, in_data, out_ready, compute1, compute2);  // 推进一个时钟周期
    void reset();
};
```

**核心逻辑**：精确匹配 RTL 的 `tc_mul_pipe` / `tc_add_pipe` 寄存器使能逻辑。每个 `tick()` 调用可传入 `compute1` 和 `compute2` 回调函数，分别对应第 1 级和第 2 级的计算逻辑。

#### 数据令牌 (Token) 结构

| 结构体 | 用途 |
|--------|------|
| `MulInput` | 乘法器输入：两个 FP9 操作数 |
| `MulStage1Data` | 乘法流水线内部数据：S1 输出 + 原始输入位 |
| `FP9Token` | FP9 格式数据令牌，在加法树中流动 |
| `FP22Token` | FP22 格式数据令牌，在累加器中流动 |

#### DotProductPipeline — 单点积流水线

```cpp
struct DotProductPipeline {
    PipeStage2<MulStage1Data> mul_pipe[8];   // 8 个并行乘法器
    PipeStage2<FP9Token>      add_L0[4];     // 加法树 Level 0: 4 个加法器
    PipeStage2<FP9Token>      add_L1[2];     // 加法树 Level 1: 2 个加法器
    PipeStage2<FP9Token>      add_L2;        // 加法树 Level 2: 1 个加法器
    PipeStage2<FP22Token>     final_add;     // FP22 最终累加器
    bool conv_valid;                          // 输出转换寄存器 valid
    uint32_t conv_fp22;                       // 转换后的 FP22 结果
};
```

管理一个完整点积的所有中间缓冲区和 valid 标志，包括乘法结果缓冲 (`mul_results`)、各级加法树输入缓冲、以及最终累加器输入缓冲。

#### TensorCoreSim — 顶层 Tensor Core 模拟器

```cpp
struct TensorCoreSim {
    static constexpr int M=8, K=8, N=8;
    static constexpr int PIPELINE_DEPTH = 11;

    DotProductPipeline dp[M][N];              // 64 个并行点积单元

    void reset();                              // 复位所有流水线状态
    void load_inputs(a, b, c, prec, rm);      // 加载输入矩阵 (已转换为 FP9/FP22)
    int  run_to_completion();                  // 运行至所有输出有效，返回周期数
    void tick();                               // 推进所有 64 个流水线一个时钟周期
};
```

**关键方法详解：**

- **`load_inputs()`**：将预转换的 FP9 格式 A/B 矩阵和 FP22 格式 C 矩阵加载到模拟器内部存储
- **`tick()`**：外层循环遍历 64 个点积单元，对每个调用 `tick_dot_product(i,j)`
- **`tick_dot_product(i,j)`**：模拟单个时钟周期内一个点积单元的行为。按从后往前的顺序更新各级流水线（避免数据冒险）：
  1. Stage 11: 输出格式转换 (FP22→目标格式)
  2. Stage 9-10: FP22 最终累加 (树结果 + C 偏置)
  3. Stage 7-8: 加法树 Level 2
  4. Stage 5-6: 加法树 Level 1
  5. Stage 3-4: 加法树 Level 0
  6. Stage 1-2: 8 路并行乘法
- **`run_to_completion()`**：循环调用 `tick()` 直到所有 64 个输出 valid，返回总周期数

#### reference_matmul() — 非流水线参考模型

```cpp
void reference_matmul(a_fp9[8][8], b_fp9[8][8], c_fp22[8][8], d_fp22[8][8], rm);
```

使用与流水线完全相同的算术逻辑（相同的 `fp9_multiply`、`fp9_add`、`fp22_add` 函数），但无流水线时序，按组合逻辑顺序计算。用于验证流水线结果的位精确性。

**加法树顺序**严格匹配 RTL：
```
Level 0: (prod[0]+prod[4]), (prod[1]+prod[5]), (prod[2]+prod[6]), (prod[3]+prod[7])
Level 1: (L0[0]+L0[1]), (L0[2]+L0[3])
Level 2: (L1[0]+L1[1])
```

---

### 4.4 main.cpp — 测试框架

#### 命令行参数

| 参数 | 取值 | 默认 | 说明 |
|------|------|------|------|
| `--prec` | `FP4_E2M1` / `FP8_E4M3` / `FP8_E5M2` / `FP16` | 全部 | 限定测试精度 |
| `--test` | `1` - `6` | 全部 | 限定测试编号 |
| `--rm` | `RNE` / `RTZ` / `RDN` / `RUP` / `RMM` | `RNE` | 舍入模式 |
| `--seed` | 正整数 | 0 (用当前时间) | RNG 种子，保证可复现 |
| `--help` | — | — | 显示帮助信息 |

#### 全局配置

```cpp
struct Config {
    std::vector<PrecisionType> precisions;  // 要测试的精度列表
    int  test_id;                            // 0=全部, 1-6=单测试
    RoundingMode rm;                         // 舍入模式
    uint32_t seed;                           // RNG 种子
};
```

#### 测试用例

| ID | 函数 | 说明 |
|----|------|------|
| 1 | `test_single_matmul()` | 每种精度执行一次 8×8×8 矩阵乘加，验证流水线与参考模型的位精确匹配，报告延迟周期数 |
| 2 | `test_pipelined_throughput()` | 连续执行多个不同精度的矩阵乘加任务，测量总周期和平均吞吐 |
| 3 | `test_stress()` | 每种精度 20 组随机矩阵压力测试，统计通过率和相对于 FP64 参考的最大相对误差 |
| 4 | `test_pipeline_visualization()` | 可视化单个点积单元的逐周期流水线级占用状态 (#=有效, .=空) |
| 5 | `test_output_conversion()` | 展示 FP22 累加结果到 FP8/FP16 各输出格式的转换表 |
| 6 | `test_edge_cases()` | 边界测试：单位矩阵 (I×B=B)，零矩阵 (0×B+0=0) |

#### 辅助函数

| 函数 | 说明 |
|------|------|
| `xorshift32()` | 轻量级伪随机数生成器 |
| `generate_random_matrices(prec)` | 生成随机输入矩阵，值范围根据精度格式适配 |
| `compare_fp22(a, b)` | FP22 位精确比较，正确处理 NaN (两个 NaN 视为匹配) |
| `parse_args(argc, argv)` | 命令行参数解析 |

---

## 5. 构建与使用

### 5.1 依赖

- **编译器**：g++ 或 clang++，支持 C++17
- **操作系统**：Linux / macOS / Windows (MinGW)

### 5.2 Makefile 目标

```bash
make                              # 编译 release 版本
make debug                        # 编译带 AddressSanitizer 的 debug 版本
make test                         # 编译并运行全部测试
make test PREC=FP8_E4M3           # 仅测试 FP8 E4M3 精度
make test TEST=3                  # 仅运行测试 3 (压力测试)
make test PREC=FP16 TEST=1        # 组合：测试 1 + FP16
make test RM_MODE=RTZ             # 使用 RTZ 舍入模式
make test SEED=42                 # 固定随机种子（可复现）
make stress                       # 快捷方式：运行压力测试
make stress PREC=FP16             # 压力测试 FP16
make viz                          # 快捷方式：流水线可视化
make clean                        # 清理构建产物
make help                         # 显示帮助
```

### 5.3 直接使用命令行

```bash
./tensorcore_sim                                # 全部测试
./tensorcore_sim --prec FP8_E4M3 --test 1       # 精度+测试过滤
./tensorcore_sim --rm RTZ --seed 42             # 舍入模式+固定种子
./tensorcore_sim --help                          # 帮助信息
```

### 5.4 Makefile 参数说明

| 参数 | 取值 | 默认 | 说明 |
|------|------|------|------|
| `PREC` | `FP4_E2M1` / `FP8_E4M3` / `FP8_E5M2` / `FP16` / `ALL` | `ALL` | 精度过滤 |
| `TEST` | `1`-`6` / `ALL` | `ALL` | 测试编号过滤 |
| `RM_MODE` | `RNE` / `RTZ` / `RDN` / `RUP` / `RMM` | `RNE` | 舍入模式 |
| `SEED` | 正整数 / `0` | `0` | RNG 种子 (0=用时间) |

---

## 6. RTL 对应关系

| 模拟器模块 | Verilog RTL 文件 | 说明 |
|-----------|-----------------|------|
| `fmul_s1()` | `fmul_s1.v` | 乘法第 1 级：指数计算、特殊值检测 |
| `fmul_s2()` | `fmul_s2.v` + `naivemultiplier` | 乘法第 2 级：尾数乘法 |
| `fmul_s3()` | `fmul_s3.v` | 乘法第 3 级：归一化、舍入 |
| `fadd_s1()` | `fadd_s1.v` | 加法第 1 级：双路径并行计算 |
| `fadd_s2()` | `fadd_s2.v` | 加法第 2 级：舍入、结果组装 |
| `clz()` | `lzc.v` | 前导零计数器 |
| `do_rounding()` | `rounding.v` | 通用舍入模块 |
| `PipeStage2<T>` | `tc_mul_pipe.v` / `tc_add_pipe.v` | 2 级流水线控制 |
| `DotProductPipeline` | `tc_dot_product.v` | 单点积单元 |
| `TensorCoreSim` | `tensor_core.v` + `mm_mul_add.v` | 顶层 Tensor Core |
| `fp4_to_fp9()` 等 | `define.v` 中的格式定义 | 输入格式转换 |
| `fp22_to_fp8_e4m3()` 等 | `fp22_to_fp8_con.v` | 输出格式转换 |

---

## 7. 验证策略

1. **位精确匹配验证**：流水线模拟器与组合逻辑参考模型使用相同的算术函数，结果必须逐位相同
2. **随机压力测试**：每种精度 20 组随机矩阵，100% 位精确通过
3. **边界值测试**：单位矩阵、零矩阵、特殊值 (NaN/Inf/亚正规)
4. **流水线行为验证**：可视化逐周期级占用，确认 11 周期延迟
5. **FP64 相对误差参考**：量化有限精度运算相对于双精度浮点的误差范围
6. **多舍入模式测试**：支持 5 种 IEEE 754 舍入模式，可通过 `--rm` 切换

---

## 8. 设计要点

### 8.1 为什么使用整数位操作而非 C++ 浮点？

RTL 硬件中的浮点运算由组合逻辑门实现，不依赖任何浮点硬件。模拟器必须精确匹配每一步中间值（指数计算、尾数乘法、移位、舍入），因此全部使用 `uint16_t` / `uint32_t` / `uint64_t` 整数运算，避免 C++ 编译器的浮点优化引入偏差。

### 8.2 加法器的近路径 vs 远路径

RTL 加法器实现了两条并行路径以优化关键路径延迟：

- **远路径 (Far Path)**：`|exp_diff| > 1` 或有效加法。对齐较小操作数（右移），执行加/减，仅需检查进位归一化
- **近路径 (Near Path)**：`|exp_diff| ≤ 1` 且有效减法。可能产生大量前导零，需要 LZC 归一化。双路计算 (A-B 和 B-A)，根据结果符号选择

### 8.3 FP9 加法的零填充

RTL 中 `tc_add_pipe` 将 FP9 输入零填充到 `2×PRECISION` 宽度：
```verilog
s1_in_a = {a_reg, {PRECISION{1'b0}}}  // 9-bit → 13-bit
```
这为加法运算提供了额外的精度位，减少中间舍入误差。模拟器中的 `fp9_add()` 精确复现此行为。

### 8.4 FP22 累加的 64 位运算

FP22 格式经零填充后尾数达到 28 位 (`PRECISION=14, 零填充→28`)，乘法中间结果为 56 位，超出 32 位整数范围。因此 `fp22_add()` 使用 64 位整数 (`uint64_t`) 内联实现，直接在 `fp_arith.h` 中展开完整的远/近双路径加法逻辑。
