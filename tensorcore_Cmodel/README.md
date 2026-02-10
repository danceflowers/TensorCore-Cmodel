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
