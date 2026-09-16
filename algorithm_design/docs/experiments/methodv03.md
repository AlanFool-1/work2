---
title: Method V0.3：实验分析
description: V0.3 在 Cora 十客户端上的正确 local 流程、失败诊断，以及转入 V0.4 的设计入口。
---

# Method V0.3：实验分析

版本：2026-09-10 UTC（暂停 V0.3 实验，转入 V0.4 设计）

!!! danger "V0.3 已暂停：正式 local 结果尚未过门"
    按正确的十客户端 `main.py` 流程完成 100 轮后，当前最好结果是
    **74.12% ACC ± 13.25%**（Macro-F1 **29.53%**），仍低于约 80% 的预期本地主干水平。
    因此 V0.3 没有进入正式 FedAvg 评估；本页以下新增的筛选结果只用于解释瓶颈，不能替代
    完整生命周期证据。

!!! info "当前状态"
    V0.3 实验已按要求暂停，后续工作转入 V0.4 设计。已有训练进程已停止；当前没有新的
    Cora local 或 FedAvg 任务在后台运行。

[查看 V0.3 算法设计](../methodv03.md){ .md-button .md-button--primary }

## 1. 本次先纠正了什么

早期调试把 local 错误地实现成“逐客户端调用独立 runner”。这不是本项目所要求的
local 对照，相关结果全部作废，不进入本页正式表格。

正确做法保留完整联邦运行链：

```text
main.py
  -> ParentProcess 启动持久 worker
  -> Server 每轮下发全局状态
  -> 10 个 Client 各训练 1 个 local epoch
  -> Client 上传模型
  -> Server 完成 FedAvg 聚合
```

local 与 federated 的唯一区别，是 V0.3 客户端接收消息时暂时注释下面这一行：

```python
# set_state_dict(self.model, self.sd['global'], self.gpu_id)
```

因此每个客户端在 100 轮中持续使用自己的模型和 Adam 状态；服务器调度、上传和聚合
仍完整执行，只是聚合后的全局参数不覆盖客户端本地参数。恢复这一行才是正常 FedAvg。

代码结构也已收敛为与 `xzc/Fedrated` 一致的入口和职责划分：

```text
codev0/
├── main.py
├── myparser.py
├── configs/v03.json
├── models/v03/{model.py,client.py,server.py}
├── modules/{federated.py,multiprocs.py,structured_logger.py}
├── data/
└── misc/
```

V0.3 不再导入 V0.1/V0.2/S0 的模型或训练代码。旧模型目录、旧配置、旧 runner、
proxy 与整个 tests 目录均已移除。

## 2. 统一实验设置

所有正式结果均直接由 `codev0/main.py` 产生：

| 项目 | 设置 |
| --- | --- |
| 数据集 | Cora disjoint |
| 客户端数 | 10 |
| 随机种子 | 42 |
| 训练 | 100 rounds × 1 local epoch |
| 客户端参与率 | 1.0 |
| 服务器聚合 | equal FedAvg，local 模式下聚合结果不回写客户端 |
| 模型选择 | 每个客户端最佳 validation round 对应的 test metric |
| Koopman 阶数 | $K=2$ |
| ODE | RK4，$T=1.6$，$\Delta t=0.1$ |
| 优化器 | persistent Adam，weight decay $10^{-4}$ |

Cora 各客户端只有 41–61 个训练节点，输入特征已做行归一化，平均行和为 1。
128 维 V0.3 有 332,295 个参数；64 维为 129,287 个参数。这个参数量与每客户端极少的
监督节点形成明显失衡，是分析过拟合时必须考虑的背景。

## 3. 完整结果

### 3.1 汇总

| 维度 | Input dropout | LR | $\lambda_{cl}$ | ACC mean ± std | Macro-F1 | 结论 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 0.0 | 0.003 | 0.1 | **72.55 ± 12.32** | 30.20 | 128 维基准 |
| 128 | 0.0 | 0.005 | 0.1 | 70.44 ± 16.54 | 28.11 | 更高 LR 明显恶化难客户端 |
| 128 | 0.0 | 0.003 | 0.01 | 71.81 ± 17.69 | 31.01 | 弱化 closure 没有改善 ACC |
| 64 | 0.3 | 0.003 | 0.1 | 72.54 ± 13.51 | 26.99 | 降维与轻 dropout 仍不足 |
| 64 | 0.5 | 0.003 | 0.1 | **74.12 ± 13.25** | 29.53 | 当前最好，但未达到门槛 |

这里的 ACC/F1 都按“每客户端最佳 validation round 配对 test”汇总，而不是挑选最后一轮
或只汇报容易客户端。

### 3.2 逐客户端 ACC

| Client | 128 基准 | 128 / LR 0.005 | 128 / $\lambda_{cl}=0.01$ | 64 / drop 0.3 | 64 / drop 0.5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 81.05 | 84.21 | 87.37 | 82.11 | 82.11 |
| 1 | 76.29 | 76.29 | 71.13 | 75.26 | 79.38 |
| 2 | 83.33 | 81.11 | 83.33 | 81.11 | 83.33 |
| 3 | 46.81 | 40.43 | 37.23 | 47.87 | 52.13 |
| 4 | 52.58 | 38.14 | 42.27 | 48.45 | 49.48 |
| 5 | 73.83 | 73.83 | 73.83 | 73.83 | 73.83 |
| 6 | 84.55 | 84.55 | 85.45 | 84.55 | 84.55 |
| 7 | 69.03 | 64.60 | 64.60 | 64.60 | 64.60 |
| 8 | 76.60 | 79.79 | 89.36 | 85.11 | 88.30 |
| 9 | 81.44 | 81.44 | 83.51 | 82.47 | 83.51 |

64 维、dropout 0.5 的收益主要来自 client 1、3、8；client 4 与 7 仍是明显瓶颈。
标准差仍超过 13 个百分点，说明平均值的提高没有解决客户端间性能失衡。

## 4. 曲线与现象分析

128 维基准在第 100 轮的平均训练损失已经降到约 0.023，但平均 validation ACC 只有
约 0.737。训练目标几乎被拟合完，而泛化性能停滞，属于明显过拟合。

三个结果进一步排除了表面配置解释：

1. **维度不是越大越好。** 128 维没有接近 80%；改回 64 维并加强 dropout 后只提升到
   74.12%。
2. **更高学习率不是解法。** LR 从 0.003 提高到 0.005 后，client 3/4 分别跌至
   40.43%/38.14%，总体下降 2.11 个百分点。
3. **closure 权重不是唯一瓶颈。** $\lambda_{cl}$ 从 0.1 降至 0.01 后总体没有提升，
   且客户端标准差扩大。这说明当前不足不能简单归因于 closure 压制任务损失。
4. **普通 dropout 只能缓解，不能修复 backbone。** 64 维 dropout 0.5 比 0.3 高
   1.59 个百分点，但距离约 80% 仍差 5.88 个百分点。

## 5. 暂停前的受控筛选：定位问题，但不升格为正式结果

为了判断瓶颈来自模型结构还是训练协议，我在同一 Cora 十客户端分区、seed 42、100
轮、每轮一次更新、相同最佳验证配对规则下做了短流程筛选。该筛选直接调用同一个
`EnergyKoopmanGraphBackbone`，但没有经过完整 `main.py` 的持久 worker 与服务器日志链，
所以只作为诊断依据。

| 配置 | ACC mean ± std | 观察 |
| --- | ---: | --- |
| 原 V0.3：64 维、K=2、dropout 0.5、相对 closure 0.1 | 74.12 ± 13.25% | 当前唯一正式 local 基线 |
| 线性 encoder、64 维、K=1、dropout 0.3、无 closure | 78.69 ± 11.21% | 降低 encoder 过拟合，仍未稳定过门 |
| 线性 encoder、128 维、K=1、dropout 0.3、无 closure | 79.95 ± 12.18% | 容量增加有帮助，但 closure 被移除 |
| 线性 encoder、128 维、K=1、dropout 0.3、raw closure 0.001 | 80.01 ± 11.02% | Energy-GCN 接收闭合梯度时出现能量幅值塌缩 |
| 线性 encoder、128 维、K=1、dropout 0.3、raw closure 0.01 + 50 轮 warm-up | 80.52 ± 11.36% | 最佳点主要位于 closure 生效前，不能算闭合模型 |

这些结果给出了三个可靠诊断：

1. **相对 closure 的尺度不稳定。** 初始 Energy-GCN 的向量场很小时，按
   `mean square / ‖V_energy‖²` 归一化会放大随机闭合梯度，直接压制任务学习。
2. **无约束的 raw closure 也不是答案。** 让 Energy-GCN 一起追逐闭合目标会把能量梯度
   推向近零，得到的是 observable collapse，而不是有意义的 graph flow。
3. **K=1 和较小 encoder 更有希望。** 当前 K=2 的多尺度项没有显示稳定收益；线性
   encoder、128 维状态和较弱动力学扰动接近本地门槛，但还缺少完整生命周期验证。

暂停前的困难客户端仍是 3、4、7。它们的图内同质性约为 0.60、0.61、0.80；同一分区
上的官方 A-DGN 纯 local 对照约为 80.57% 平均 ACC，说明 V0.3 的主要问题是自身的
动力学/优化耦合，而不只是数据切分。

## 6. 哪些结果明确不采用

以下历史结果不作为算法证据：

- 逐客户端运行 `run_v03_single_client.py` 得到的数字，因为它绕开了项目的主联邦入口；
- graph stem、Dirichlet diffusion 或 matched initialization 得到的较高数字，因为这些
  是调试过程中擅自加入、且不属于当前 V0.3 设计的结构；
- 任何只看单个客户端、最后一轮或多数类准确率的结果。

这些路径对应的代码均已从当前 V0.3 主干撤掉。

## 7. 当前判断与下一步门槛

!!! warning "暂不恢复全局模型覆盖"
    正式 V0.3 local 仍只有 74.12%，尚未证明 backbone 本身具备应有容量；直接 FedAvg
    也从未在这版实现上运行。之前注释的全局参数替换行保持不变，不能把服务器聚合日志
    误称为 FedAvg 结果。

以上诊断已经足够说明当前联合闭合方案存在表达与稳定性问题，V0.3 不再追加超参数
扫描。客户端的全局参数覆盖行继续保持注释，直到 V0.4 的 staged local gates 全部
通过。

## 8. 转入 V0.4：当前设计入口

!!! info "最新版 V0.4 已定稿"
    早期的 native trajectory / fixed snapshots handover 候选已经被取代。正式 V0.4
    不积分第二条 teacher/native trajectory，而采用任务锚定神经图向量场作为瞬时
    参考，以 stop-gradient Lie/JVP 闭合唯一的稳定 Graph-Koopman dynamics。完整公式、
    optimizer 隔离和实验门见 [Method V0.4 算法设计](../methodv04.md)。

V0.4 按以下顺序推进：

1. 先验证 observable identity、JVP、梯度隔离、$K=1$ one-tap 等价和耗散稳定性；
2. Lie 关闭时跑通 stable $K=1$ Koopman backbone；
3. 独立训练 probe 与 task-anchored field，确认 field 非零且 task cosine 为正；
4. 只在前三门通过后按轮次 warmup/ramp Lie direction loss；
5. 在相同十客户端、100 轮完整生命周期中验证约 80% local 门槛与更低方差；
6. 通过 local gate 后才恢复客户端 global-state replacement，运行正常 equal FedAvg。

这意味着 V0.3 的结论边界是“发现当前联合闭合方案的优化与尺度问题”，而不是宣称
Energy-Koopman 结构已经通过本地门槛。

## 9. 结果产物

| 实验 | 日志目录 |
| --- | --- |
| 128 基准 | `logs/Cora_disjoint/clients_10/20260910_030810_v03_local_nosync_128/` |
| 128 / LR 0.005 | `logs/Cora_disjoint/clients_10/20260910_031414_v03_local_lr005/` |
| 128 / $\lambda_{cl}=0.01$ | `logs/Cora_disjoint/clients_10/20260910_031414_v03_local_cl001/` |
| 64 / dropout 0.3 | `logs/Cora_disjoint/clients_10/20260910_032722_v03_local64_drop03/` |
| 64 / dropout 0.5 | `logs/Cora_disjoint/clients_10/20260910_032722_v03_local64_drop05/` |

每个完整目录包含解析后的 `config.json`、逐轮 `metrics.csv`、逐客户端
`client_best.csv` 和最终 `result.json`。
