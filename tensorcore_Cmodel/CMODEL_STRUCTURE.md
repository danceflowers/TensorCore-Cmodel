# TensorCore Cmodel 结构分析

本文面向 `tensorcore_Cmodel/` 目录，梳理该 Cmodel（cycle-approximate simulator）的代码分层、数据流与时序模型。

## 1. 顶层定位

该 Cmodel 目标是以 **近周期级（cycle-approximate）** 方式模拟 OpenTensorCore，设计模式参考 Vortex SimX。核心映射关系是：

- `TensorCoreUnit` 对应 RTL 顶层 `tensor_core.v`
- `DotProductUnit` 对应 RTL `tc_dot_product.v`
- `PipeStage<T>` 对应弹性流水寄存器模型
- `otc_*` 驱动接口对齐 `vx_*` 风格 API

## 2. 当前目录下的实际文件分工

> 代码都集中在当前目录内（并非真正拆成 `include/`、`runtime/`、`sim/`、`tests/` 子目录）。

- `main.cpp`
  - 可执行入口 + 测试 harness（数据生成、打包、运行、验收）。
  - 包含命令行参数解析、golden GEMM 参考计算、误差判定。
- `otc_driver.h`
  - C 风格设备 API 封装层（open/config/upload/run/download/stats）。
  - 对上屏蔽 `TensorCoreUnit` 内部状态机细节。
- `pipeline.h`
  - 核心仿真模型：`PipeStage<T>`、`DotProductUnit`、`TensorCoreUnit`。
  - 实现格式转换、并行 DP 发射、流水线排空（drain）与结果回收。
- `otc_types.h`
  - 配置结构 `OTC_Config`、统计结构 `OTC_Stats`、日志跟踪 `TraceLog`。
  - 给出类型编码、延迟参数、配置合法性检查。
- `otc_fp.h`
  - 浮点格式转换与量化（FP4/FP8/FP9/FP16/FP22/FP32 与 double 互转）。
- `Makefile`
  - 编译目标 `otc_simx` 与回归测试目标集合。

## 3. 架构分层（从调用关系看）

### 3.1 Host/Test 层

`main.cpp` 完成如下职责：

1. 生成 A/B/C 测试矩阵（ones/identity/random/simple）。
2. 用 `pack_ab`、`pack_c_fp16` 打包成硬件输入字格式。
3. 调用驱动 API：`otc_dev_open` → `otc_configure` → `otc_upload` → `otc_run`。
4. `otc_download_f64` 取回结果并与 `golden_gemm` 对比。
5. 输出统计与资源估算。

### 3.2 Driver 适配层

`otc_driver.h` 是很薄的一层，核心作用：

- 用 `OTC_Device` 持有一个 `TensorCoreUnit` 实例。
- 把 C 风格 API 映射到 C++ 对象方法。
- 保持“可替换后端”的接口习惯（和 Vortex 风格一致）。

### 3.3 Core Simulation 层

`pipeline.h` 中 `TensorCoreUnit` 是核心执行引擎，状态机是：

- `IDLE`
- `FORMAT_CONV`
- `DISPATCH`
- `DRAIN`
- `DONE`

其中：

- `FORMAT_CONV`：把打包输入解码到 `conv_a_ / conv_b_ / conv_c_`。
- `DISPATCH`：一次性为所有 `M×N` 输出元素构建 `DPInput`，并推入对应 `DotProductUnit`。
- `DRAIN`：按周期推进 DP 流水线并回收结果，直到全部输出就绪。

### 3.4 Numeric Model 层

`otc_fp.h` 负责位级格式语义，保证 Cmodel 的数值行为可控：

- 输入侧：按 `type_ab` + 子类型（E5M2/E4M3）做元素转换。
- 计算侧：乘加用 double 做函数级计算，再量化到 FP22（模拟累加器精度）。
- 输出侧：支持导出为 FP16/FP32 或 double。

## 4. 关键数据流

以一次 GEMM 为例，数据流如下：

1. Host 生成 double 矩阵 A/B/C。
2. `pack_ab` / `pack_c_fp16` 量化并打包成 `uint32_t` words。
3. `TensorCoreUnit::load` 缓存 raw words。
4. `do_format_conversion` 按元素宽度拆包并转成 double。
5. `dispatch_all` 为每个输出位置 `(i,j)` 构造一个 `DPInput`：
   - `A` 取第 `i` 行
   - `B` 取第 `j` 列
   - `C` 取偏置元素
6. `DotProductUnit::push` 内做 K 次乘法 + 加法树 + final add，结果量化至 FP22 并入飞行队列。
7. `tick` 周期推进，结果成熟后写回 `output_d_`。
8. Host 下载并验收。

## 5. 时序/性能模型要点

- DotProduct 延迟模型为：
  - `mul_latency`
  - `tree_depth * add_latency`
  - `final add_latency`
  - `+1` 输出寄存器
- `K` 需为 2 的幂（便于加法树规整）。
- `OTC_Stats` 统计：总周期、忙周期、乘加操作数、完成矩阵数等。

这意味着它不是“每个门级细节”的 RTL 仿真，而是按结构化阶段做周期近似，换取可读性和速度。

## 6. 代码组织上的一个注意点

`README.md` 和 `Makefile` 里展示了 `include/sim/runtime/tests` 的“逻辑路径”，但仓库当前 `tensorcore_Cmodel/` 目录实际是 **扁平放置**。阅读时可把这些路径理解为“概念分层”而非真实目录。

## 7. 一句话总结

这个 Cmodel 的结构是：

- **`main.cpp`（测试入口）**
  + **`otc_driver.h`（接口封装）**
  + **`pipeline.h`（状态机 + 并行 DP 核 + 周期推进）**
  + **`otc_fp.h`（数值格式模型）**
  + **`otc_types.h`（配置/统计/trace）**

共同构成一个“可跑回归、可观察流水、与 RTL 模块一一对照”的 TensorCore 周期近似仿真器。
