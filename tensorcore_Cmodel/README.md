# OpenTensorCore SimX — Cycle-Approximate Simulator

仿照 [Vortex GPGPU SimX](https://github.com/vortexgpgpu/vortex) 模拟器的运行方式和模式，为 [OpenTensorCore](https://github.com/chenweiphd/OpenTensorCore) 编写的 C model (cycle-approximate simulator)。

## 架构对应关系

| Vortex SimX | OpenTensorCore SimX | RTL 对应 |
|-------------|---------------------|----------|
| `Processor` | `TensorCoreUnit` | `tensor_core.v` |
| `Core` | `DotProductUnit` | `tc_dot_product.v` |
| `Pipeline` | `PipeStage<T>` | elastic pipeline reg |
| `vx_dev_open` | `otc_dev_open` | 设备初始化 |
| `vx_start` | `otc_start` | 启动计算 |
| `vx_ready` | `otc_ready` | 轮询完成 |
| `--debug=N` | `--debug=N` | trace 级别 |
| `run.log` | `otc_run.log` | 执行 trace |

## 项目结构

```
otc_simx/
├── include/
│   ├── otc_types.h          # 配置、统计、TraceLog (≈ VX_types.vh)
│   └── otc_fp.h             # SoftFloat 转换库 (FP4/8/9/16/22 ↔ double)
├── sim/simx/
│   └── pipeline.h           # 核心流水线模型 (PipeStage + DotProduct + TensorCore)
├── runtime/
│   └── otc_driver.h         # 驱动 API (≈ runtime/simx/vortex.cpp)
├── tests/
│   └── main.cpp             # 测试 harness (≈ tests/regression/)
├── Makefile                 # 构建系统 (make test-all)
└── README.md
```

## 快速开始

```bash
make                    # 编译
make test               # 默认测试 (ones 8×8×8 FP8)
make test-all           # 所有测试 (7 个用例)
```

## 命令行参数

```bash
./otc_simx --test=ones --M=8 --K=8 --N=8 --type_ab=fp8e5m2 --debug=1
./otc_simx --test=random --M=4 --K=4 --N=4 --debug=2 --trace
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--M/K/N=N` | 矩阵维度 | 8 |
| `--type_ab=` | `fp4`, `fp8e5m2`, `fp8e4m3`, `fp16` | `fp8e5m2` |
| `--test=` | `ones`, `identity`, `random`, `simple` | `ones` |
| `--debug=N` | 调试级别 0-3 | 0 |
| `--trace` | 写入 otc_run.log | off |
| `--batches=N` | 多 batch 提交数量 | 1 |
| `--dispatch_width=N` | 每周期下发 DP 数 | 8 |
| `--in_fifo_depth=N` | 输入 FIFO 深度 | 8 |
| `--out_fifo_depth=N` | 输出 FIFO 深度 | 8 |
| `--mem_bw=N` | 峰值带宽（B/cycle）用于带宽利用率估算 | 32 |
| `--random_runs=N` | random 测试重复次数 | 5 |

## 流水线延迟模型

K=8时: 1(conv) + 2(mul) + 3×2(tree) + 2(acc) + 1(out) = **12 周期**

## 测试结果

```
simple 2×2×2    PASSED ✓  (9 cyc,  8 MULs)
identity 4×4×4  PASSED ✓  (11 cyc, 64 MULs)
ones 8×8×8      PASSED ✓  (13 cyc, 512 MULs)
random 8×8×8    PASSED ✓  (13 cyc, max_err=0.51)
FP4 ones 8×8×8  PASSED ✓  (13 cyc)
FP16 ones 8×8×8 PASSED ✓  (13 cyc)
16×16×16        PASSED ✓  (15 cyc, 4096 MULs)
```


## 多 Batch Cycle-Stepping 与性能指标

当前模型支持通过 `otc_submit` 连续提交多个 batch，并在 `tick()` 中逐周期推进：

- 输入队列：`input_fifo_depth` 可配置，模拟前端请求缓存。
- 格式转换阶段与计算阶段并行推进（cycle-stepping）。
- 输出队列：`output_fifo_depth` 可配置，支持背压统计。
- 新增统计项：
  - 吞吐率：`Throughput (batch/cycle)`
  - 平均延迟：`Avg latency (cycles)`
  - 带宽利用率：`BW utilization`
  - 计算单元利用率：`Compute util`
  - FIFO 相关：`Output FIFO max occ`、`Output backpressure cyc`

示例：

```bash
./otc_simx --test=ones --batches=8 --dispatch_width=16 --in_fifo_depth=16 --out_fifo_depth=16 --mem_bw=64
```


## 主控制循环（Fetch/Decode/Execute）

`main.cpp` 现在通过 `OTC_Decoder` 驱动简单指令序列，按 **Fetch -> Decode -> Execute** 执行：

- `TCU_LOAD`：准备输入数据
- `TCU_WMMA`：通过 `otc_driver` 提交多 batch 并执行
- `TCU_STORE`：从输出 FIFO 弹出结果

这样实现了将 tensor core 计算模块封装在 `otc_driver` 内，并与 `otc_decode` 在主循环中集成。
