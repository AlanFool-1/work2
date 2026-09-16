---
title: Method V0.4：Cora 实验分析
description: >-
  V0.4 在 Cora 十客户端上的 local、O0-O5 历史选择实验、当前唯一 O4 block-v
  联邦协议，以及任务性能与动力学闭合诊断。
tags:
  - Method V0.4
  - Cora
  - Local backbone
  - FedAvg
  - FedAdamW
  - Lie consistency
---

# Method V0.4：Cora 实验分析

版本：2026-09-11 UTC

!!! success "优化器状态错配已定位，轻量 FedAdamW-v 已跑通"
    完整 V0.4 local 100 轮达到 **79.53% ACC ± 11.69%**，相较正式 V0.3 local
    提升 5.41 个百分点。原始 persistent-Adam equal FedAvg 只有
    **70.77% ± 12.44%**；每轮重置状态并切换 AdamW 后恢复到
    **79.93% ± 13.16%**，进一步聚合 per-tensor block-mean 二阶矩后达到
    **81.47% ± 8.64%**。

!!! warning "任务差距来自优化器，不等于动力学对齐已经成功"
    O4 的任务均值和客户端方差最好，但末轮 Lie cosine 仍只有 0.031，latent norm
    ratio 为 0.642。AdamW-reset 的对应值为 0.693/0.797，更接近 local 的
    0.943/0.885。优化器修复解释并消除了任务准确率差距，但不能写成 operator
    federation 或跨客户端 Koopman 坐标对齐成功。

[查看 V0.4 算法设计](../methodv04.md){ .md-button .md-button--primary }
[查看 V0.3 实验分析](methodv03.md){ .md-button }

## 1. 实验协议

两组正式结果均使用：

| 项目 | 设置 |
| --- | --- |
| 入口 | `codev0/main.py` |
| 数据 | Cora disjoint partitions |
| 客户端 | 10 |
| 轮次 / local epoch | 100 / 1 |
| Seed | 42 |
| Worker / GPU | 10 workers / GPU 0,1 |
| 聚合 | equal aggregation |
| 选择规则 | 每个客户端验证集最佳轮，对应同轮测试 ACC/F1，再跨客户端汇总 |
| 状态维度 | $d_h=64$ |
| Observable | $[H,\rho(H)]$，$d_a=32$ |
| Generator | K=1 Bernstein，$\gamma=10^{-4}$ |
| ODE | RK4，$T=1.0$，step size $0.1$ |
| Lie | 1–10 轮关闭，11–30 轮 ramp 到 0.05 |

local 不是十次独立脚本。它保留 server、worker、client、持久 Adam、上传与服务器
聚合，只注释客户端接收全局参数与优化器状态的单行调用：

```python
# self._accept_global_state()
```

FedAvg 正式运行时恢复该行。当前代码保留的是 **FedAvg active + block-v
FedAdamW** 状态；注释这一行后，初始化时创建的 persistent Adam 不被替换，因此仍
保持原 local 对照语义。

当前可执行代码只保留 O4。`persistent`、`reset`、full-v 和 global-correction 不再是
配置或命令行选项；下文 O0-O5 是选择 O4 时已完成的历史对照及其原始 artifact。

## 2. Gate 0：数学对象与工程边界

`python -m models.v04.diagnostics` 通过：

| 检查 | 结果 |
| --- | ---: |
| observable identity | machine precision |
| JVP vs explicit Jacobian | 最大误差 $6.94\times10^{-17}$ |
| K=1 partition | $5.55\times10^{-17}$ |
| K=1 one-tap equivalence | $5.55\times10^{-17}$ |
| small full-generator max real part | $-2.0\times10^{-3}$ |
| RK4 vs matrix exponential | 相对误差 $1.67\times10^{-14}$ |
| field stop-gradient | 通过 |
| probe / field / main optimizer isolation | 通过 |
| FedAdamW first/second-moment clocks | $k=1$ / $t=101$ 分离通过 |
| equal/weighted block-v aggregation | 2.0 / 2.5 toy 通过 |
| toy anchor cosine | $0.308\rightarrow0.997$ |

这些结果只证明实现与公式对齐，不代表 Cora 性能。

## 3. 分阶段结果

| 阶段 | 轮次 | 改动 | ACC mean ± std | Macro-F1 |
| --- | ---: | --- | ---: | ---: |
| Gate 1 | 30 | K=1，reference/Lie off，high=1 | 77.77 ± 11.60% | 41.41% |
| Gate 2 | 30 | 增加 probe + field anchor，Lie off | 77.77 ± 11.60% | 41.41% |
| Gate 3 | 30 | Lie ramp 至 0.05，high=1 | 77.15 ± 11.67% | 40.66% |
| 初始完整配置 | 100 | high=1 | 77.75 ± 11.31% | 42.46% |
| Generator screen | 30 | high=2，Lie off | 79.10 ± 11.86% | 45.17% |
| Generator screen | 30 | high=4，Lie off | 78.95 ± 12.53% | 45.45% |
| Full screen | 30 | high=2，Lie=0.05 | 79.32 ± 11.96% | 46.00% |
| **Accepted local** | **100** | **high=2，完整 V0.4** | **79.53 ± 11.69%** | **46.15%** |
| Original O0 FedAvg | 100 | 恢复 global replacement，旧 Adam state persistent | 70.77 ± 12.44% | 26.58% |

Gate 2 与 Gate 1 的任务结果完全一致，说明 reference modules 在 Lie 关闭时确实不进入
forward，也没有通过共享 optimizer 污染主模型。Gate 2 的 field-task cosine 从 0.007
升至 0.821，field RMS 保持约 0.10，满足进入 Lie 的条件。

### 为什么 high-band damping 从 1 调到 2

K=1 初始化时，低频 band 近 identity-preserving，high band 的更强负生成元提供可学习
的 heat-kernel 式低通起点：

$$
\dot Z
=\mathcal B_0(\bar L)ZG_0^\top
+\mathcal B_1(\bar L)ZG_1^\top.
$$

只将 high damping 从 1 改为 2，30 轮结果由 77.77 提升至 79.10；继续增到 4 没有
进一步收益且方差增大。因此最终配置采用 2.0。该矩阵仍参与学习，并不是额外的固定
diffusion branch；稳定参数化和 K=1 结构均未改变。

## 4. 正式 local 100 轮

### 4.1 汇总

| 指标 | V0.3 正式 local | V0.4 正式 local | 变化 |
| --- | ---: | ---: | ---: |
| ACC mean | 74.12% | **79.53%** | **+5.41 pp** |
| Client std | 13.25% | **11.69%** | **-1.55 pp** |
| Macro-F1 | 29.53% | **46.15%** | **+16.62 pp** |

### 4.2 逐客户端 paired test

| Client | Best round | Val ACC | Test ACC | Macro-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 12 | 94.06% | 88.42% | 49.32% |
| 1 | 10 | 85.32% | 84.54% | 54.79% |
| 2 | 30 | 82.00% | 85.56% | 39.10% |
| 3 | 39 | 66.32% | 65.96% | 47.66% |
| 4 | 9 | 67.89% | 53.61% | 36.18% |
| 5 | 29 | 78.00% | 73.83% | 20.96% |
| 6 | 14 | 95.12% | 90.00% | 70.45% |
| 7 | 15 | 73.96% | 74.34% | 39.89% |
| 8 | 20 | 93.40% | 90.43% | 45.48% |
| 9 | 10 | 93.75% | 88.66% | 57.69% |

客户端 4 仍是最明显的弱点；不能用总体均值掩盖该子图的 53.61%。

### 4.3 动力学诊断

| 诊断 | Round 30 | Round 50 | Round 100 | 判断 |
| --- | ---: | ---: | ---: | --- |
| field-task cosine | 0.826 | 0.938 | **0.985** | reference anchor 成立 |
| Lie cosine | 0.347 | 0.726 | **0.943** | 闭合持续改善 |
| field RMS | 0.103 | 0.105 | **0.110** | 无零场坍缩 |
| latent norm ratio | 0.872 | 0.896 | **0.885** | 无爆炸/过度归零 |
| Lie speed ratio | 13.75 | 14.18 | **14.52** | 方向有效，速度未校准 |
| $G_0$ symmetric max eig | -0.0001 | -0.0001 | **-0.0001** | 稳定证书成立 |
| $G_1$ symmetric max eig | -0.0133 | -0.0097 | **-0.0018** | 稳定证书成立 |

因此 local 成功支持的是 direction-level Lie closure。speed ratio 仍约 14.5，不能声称
参考场与 Koopman RHS 的物理时间尺度已经匹配；speed loss 若加入，必须另做消融。

## 5. 原始 persistent-Adam equal FedAvg 100 轮

### 5.1 逐客户端结果

| Client | Best round | Test ACC | Macro-F1 |
| ---: | ---: | ---: | ---: |
| 0 | 19 | 83.16% | 30.27% |
| 1 | 58 | 68.04% | 38.51% |
| 2 | 34 | 83.33% | 27.03% |
| 3 | 38 | 47.87% | 23.78% |
| 4 | 65 | 50.52% | 26.02% |
| 5 | 1 | 73.83% | 16.99% |
| 6 | 1 | 84.55% | 30.54% |
| 7 | 50 | 64.60% | 16.83% |
| 8 | 36 | 74.47% | 23.30% |
| 9 | 35 | 77.32% | 32.53% |

### 5.2 为什么任务过线但方法诊断失败

| 诊断 | Local round 100 | FedAvg round 100 |
| --- | ---: | ---: |
| ACC | 79.53% | 70.77% |
| field-task cosine | 0.985 | 0.518 |
| Lie cosine | 0.943 | **0.006** |
| latent norm ratio | 0.885 | **0.375** |
| Lie speed ratio | 14.52 | **42.29** |

全参数平均同时混合 encoder/observable 坐标、field、probe 和 generator。各客户端本地
形成的 reference direction 与 latent basis 不必在参数平均后保持一致，因此服务器模型
广播回来后：

1. probe/field anchor 学习被每轮平均部分抵消；
2. $J_\Psi F$ 与 $\mathcal K\Psi$ 不再同向；
3. 聚合后的动力学表现出明显更强的有限时域收缩；
4. 分类头仍可获得约 70.77%，但 Macro-F1 和 hard-client 表现明显下降。

该结果后来被确认同时包含 optimizer-state mismatch：客户端参数每轮被服务器移动到
$\theta^r_{global}$，但三个 Adam 继续携带旧本地轨迹产生的 $m,v,t$。因此 70.77%
仍是可复现的原始协议基线，但不再是推荐的联邦优化默认值。

## 6. 联邦优化器分阶段修复

!!! info "历史选择梯度"
    本节的 O0-O5 已完成并保留用于解释为什么选择 O4；它们不是当前代码中的可切换
    optimizer 版本。当前正常联邦路径固定为 O4 block-v。

除优化器状态处理外，以下实验保持完全相同的 Cora partition、seed、backbone、
FedAvg 模型聚合、学习率、Lie、轮次和 local epoch。

| ID | Main optimizer protocol | ACC mean ± std | Macro-F1 | Worst client |
| --- | --- | ---: | ---: | ---: |
| O0 | Adam，旧 $m,v,t$ persistent | 70.77 ± 12.44% | 26.58% | 47.87% |
| O1 | Adam，每轮 reset $m,v,t$ | 73.92 ± 14.73% | 37.28% | 47.42% |
| O2 | AdamW，每轮 reset $m,v,t$ | 79.93 ± 13.16% | 52.08% | 52.13% |
| O3 | FedAdamW-v，聚合完整 $v$ | 79.27 ± 9.66% | 48.19% | 58.76% |
| **O4** | **FedAdamW-v，per-tensor block-mean $v$** | **81.47 ± 8.64%** | **53.25%** | **67.01%** |
| O5 | O4 + $\alpha\Delta_G,\alpha=0.25$ | 81.32 ± 9.16% | 53.67% | 64.95% |
| O5 | O4 + $\alpha\Delta_G,\alpha=0.5$ | 80.90 ± 11.56% | 52.83% | 54.64% |

### 6.1 因果判断

1. O1 相对 O0 提升 **3.15 pp**，达到预设 Gate A，证明 stale Adam state 是重要因素。
2. O2 再提升 **6.01 pp**，说明当前 $10^{-4}$ weight decay 下，AdamW 的 decoupled
   decay 也是主要因素；O2 已经消除 local/FedAvg 任务差距。
3. O4 相对 O2 再提升 **1.53 pp**，标准差下降 **4.52 pp**，最差客户端提高
   **14.88 pp**，通过 second-moment Gate B。
4. O5 两个系数都降低均值或最差客户端，未通过 Gate C；因此选中不带 correction
   的 O4，correction 实现现已删除。

按指南，O2 的 field/probe 也是 fresh AdamW，而 O3/O4 将辅助优化器改为 fresh Adam；
所以 O4-O2 是完整 FedAdamW-v 协议的净收益，不是“只聚合 $v$”的无混杂效应。
O3 与 O4 的 main/auxiliary 处理完全一致，二者才严格隔离 full-v 与 block-v 压缩。

因此，对当前 V0.4、当前 Cora partition 和 seed 42，可以回答：

$$
\boxed{\text{79→69 的任务差距主要是 federated optimizer protocol 问题。}}
$$

这是一组严格同配置单 seed 因果对照；在额外 seeds 和普通 GCN 控制完成前，不把它
外推为所有图联邦模型的普遍结论。

### 6.2 为什么选择 block-v

O3 每客户端每轮额外上传约 1,334,428 bytes 的完整 main second moment；O4 把每个
parameter tensor 视为一个 block，只上传 21 个 float64 标量，即 **168 bytes**。
O4 不仅通信量低，当前结果还同时获得最高均值、最低标准差和最高 worst-client。

自定义 optimizer 显式使用两个 bias-correction 时钟：

$$
\widehat m_k=\frac{m_k}{1-\beta_1^k},\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t},
$$

其中 $k$ 每轮从 1 开始，$t$ 跨 communication rounds 延续。它不能由标准 PyTorch
AdamW 的单一 `state["step"]` 精确代替。

### 6.3 任务与动力学诊断并不等价

| 协议 | Field-task cosine | Lie cosine | Speed ratio | Latent norm ratio |
| --- | ---: | ---: | ---: | ---: |
| Local Adam | 0.985 | **0.943** | 14.52 | **0.885** |
| O0 persistent Adam | 0.518 | 0.006 | 42.29 | 0.375 |
| O1 reset Adam | 0.407 | -0.003 | 23.44 | 0.592 |
| O2 reset AdamW | 0.751 | **0.693** | **12.95** | 0.797 |
| O3 full-v | 0.911 | 0.079 | 14.53 | 0.653 |
| **O4 block-v** | **0.882** | 0.031 | 17.03 | 0.642 |
| O5 $\alpha=0.25$ | 0.902 | 0.112 | 15.18 | 0.689 |
| O5 $\alpha=0.5$ | 0.911 | 0.532 | 8.71 | 0.856 |

O4 是任务性能与公平性默认；O2 则是当前更强的 Lie-alignment 对照。后续研究不能用
O4 的 81.47% 掩盖 Lie cosine 仍弱的事实。

## 7. O4 逐客户端 paired test

| Client | Best round | Val ACC | Test ACC | Macro-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 27 | 96.04% | 90.53% | 55.89% |
| 1 | 30 | 81.65% | 83.51% | 44.15% |
| 2 | 60 | 90.00% | 91.11% | 53.36% |
| 3 | 73 | 68.42% | 69.15% | 63.31% |
| 4 | 25 | 60.55% | 67.01% | 53.01% |
| 5 | 43 | 88.00% | 75.70% | 51.07% |
| 6 | 36 | 95.12% | 86.36% | 54.39% |
| 7 | 61 | 78.12% | 74.34% | 43.93% |
| 8 | 48 | 96.23% | 88.30% | 56.10% |
| 9 | 31 | 93.75% | 88.66% | 57.30% |

## 8. 产物

| 运行 | 目录 |
| --- | --- |
| Accepted local 100 | `logs/Cora_disjoint/clients_10/20260910_124512_v04_local100_final/` |
| O0 persistent Adam | `logs/Cora_disjoint/clients_10/20260910_124927_v04_fedavg100_final/` |
| O1 reset Adam | `logs/Cora_disjoint/clients_10/20260911_010516_v04_optimizer_o1_reset_adam_100r/` |
| O2 reset AdamW | `logs/Cora_disjoint/clients_10/20260911_010830_v04_optimizer_o2_reset_adamw_100r/` |
| O3 full-v oracle | `logs/Cora_disjoint/clients_10/20260911_011658_v04_optimizer_o3_fedadamw_fullv_100r/` |
| **O4 block-v default** | `logs/Cora_disjoint/clients_10/20260911_012034_v04_optimizer_o4_fedadamw_blockv_100r/` |
| O5 $\alpha=0.25$ | `logs/Cora_disjoint/clients_10/20260911_012402_v04_optimizer_o5_fedadamw_alpha025_100r/` |
| O5 $\alpha=0.5$ | `logs/Cora_disjoint/clients_10/20260911_012733_v04_optimizer_o5_fedadamw_alpha05_100r/` |

每个目录包含当时解析后的 `config.json`、逐轮 `metrics.csv`、逐客户端
`client_best.csv`、最终 `result.json` 和模型检查点。它们是不可变历史证据；当前
实现只保留 O4 block-v，且当前没有训练进程在运行。
