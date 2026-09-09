---
title: 参考论文
description: >-
  面向低维动力学代理与异质图联邦学习的 Koopman 论文阅读库。
tags:
  - Koopman
  - 动力学系统
  - 图联邦学习
  - 变分表示
---

# Koopman 参考论文

这一组论文不是需要机械拼接的模块，而是围绕同一个问题提供七种互补的
技术视角：**如何找到有用的动力学坐标，如何从图状态提取主导动力学，如何压缩，
如何避免长程滚动漂移，以及如何在新环境中快速适配算子。**

```text
学习动力学坐标 → 图状态的主导动力学 → 潜空间演化 → 完整轨迹验证 → 条件变化下适配
```

当前项目把它们迁移到异质图联邦学习时，先把客户端看成私有图动力系统
\(\mathcal S_m=(G_m,\Phi_m,\mathcal O_m)\)，再问哪些轨迹响应可以作为可运输的
动力学知识。论文中的 \(K\)、编码器和解码器不能直接跨客户端平均；只有在公共
probe、功能坐标或显式 transport 规则下，它们才可能具有可比较的含义。

## 七篇论文的分工

<div class="reference-grid" markdown>

<div class="reference-card" markdown>

<p class="card-kicker">01 · COORDINATE LEARNING</p>

### [DeepKoopman](deepkoopman.md)

联合学习编码器、潜空间线性演化和解码器，把非线性系统变成可滚动的低维表示。

[阅读方法](deepkoopman.md){ .md-button }

</div>

<div class="reference-card" markdown>

<p class="card-kicker">02 · INFORMATION BOTTLENECK</p>

### [Balanced Neural ODEs](balanced-neural-odes.md)

用轨迹级变分信息约束识别必要动力学维度，讨论压缩和预测之间的平衡。

[阅读方法](balanced-neural-odes.md){ .md-button }

</div>

<div class="reference-card" markdown>

<p class="card-kicker">03 · GRAPH DYNAMICS</p>

### [DMD-GNN](dmd-gnn.md)

把 GNN 层间传播当作动力系统，用截断 DMD 模态构造低秩谱滤波和稠密图传播。

[阅读方法](dmd-gnn.md){ .md-button }

</div>

<div class="reference-card" markdown>

<p class="card-kicker">04 · COMPOSITIONAL DYNAMICS</p>

### [Compositional Koopman](compositional-koopman.md)

用对象中心图编码器和按关系共享的块矩阵，支持可变对象数的预测与模型控制。

[阅读方法](compositional-koopman.md){ .md-button }

</div>

<div class="reference-card" markdown>

<p class="card-kicker">05 · MULTISCALE TRAFFIC</p>

### [Micro-Macro Coupled Koopman](micro-macro-coupled-koopman.md)

在车辆中心动态图上耦合宏观流量 PDE 与微观车辆 Koopman 控制，实现无历史预测。

[阅读方法](micro-macro-coupled-koopman.md){ .md-button }

</div>

<div class="reference-card" markdown>

<p class="card-kicker">06 · ROLLING CORRECTION</p>

### [Course Correcting](course-correcting.md)

分析潜态长程漂移，用周期重编码将模型自己的预测拉回有效表示区域。

[阅读方法](course-correcting.md){ .md-button }

</div>

<div class="reference-card" markdown>

<p class="card-kicker">07 · OPERATOR ADAPTATION</p>

### [MetaKoopman](metakoopman.md)

用 MNIW 先验和闭式 Bayesian 更新，让潜空间算子能够适应新的动力学环境。

[阅读方法](metakoopman.md){ .md-button }

</div>

</div>

| 论文 | 主要回答的问题 | 对本项目的启发 |
| --- | --- | --- |
| [DeepKoopman](deepkoopman.md) | 如何联合学习坐标、线性演化和解码器？ | 以多初值、多步自由滚动共同塑造可预测坐标。 |
| [Balanced Neural ODEs](balanced-neural-odes.md) | 如何让潜空间真正压缩而不是只重建？ | 用轨迹级变分信息约束识别必要动力学维度。 |
| [DMD-GNN](dmd-gnn.md) | 如何从 GNN 状态估计主导图动力学？ | 用多列节点快照、截断 SVD 和 DMD 模态构造低秩谱传播。 |
| [Compositional Koopman](compositional-koopman.md) | 如何让 Koopman 模型适应可变对象数？ | 用对象中心嵌入和按关系共享的块矩阵降低识别成本。 |
| [Micro-Macro Coupled Koopman](micro-macro-coupled-koopman.md) | 如何统一微观车辆和宏观流量？ | 用车辆中心图 PDE、意图门控和有界 Koopman control 建立双向耦合。 |
| [Course Correcting Koopman](course-correcting.md) | 为什么潜态长程滚动会漂移？ | 将周期重编码作为漂移诊断和推理期纠偏对照。 |
| [MetaKoopman](metakoopman.md) | 算子如何适应分布变化并表达不确定性？ | 固定表示后，用小规模 Bayesian/ridge 更新替代粗糙参数平均。 |

## 面向图联邦的统一映射

| 论文对象 | 图联邦中的对应物 | 需要额外补上的问题 |
| --- | --- | --- |
| 状态 \(x_t\) | 节点状态 \(H_t^m\) | 变长图、节点置换和稀疏传播。 |
| 编码器 \(E\) | 图感知编码器 \(E_m(H_t^m,G_m)\) | 不同客户端的潜坐标如何运输和比较。 |
| 线性算子 \(K\) | 客户端动力学算子 \(K_m\) | 不能直接对不同坐标下的矩阵做 FedAvg。 |
| 输出 \(y_t\) | 冻结任务读出 \(Y_t^m=\mathcal O_m(H_t^m)\) | 迁移应以任务响应而非参数距离判定有效性。 |
| 支持集/历史片段 | 客户端本地轨迹或公共 probe 响应 | 如何在隐私和通信约束下形成可用统计量。 |

## 阅读边界

- 本页面主要记录动机、方法和可迁移技术，不复述论文实验结果。
- 这些论文都没有直接解决异质图之间的功能坐标对齐；这仍是本项目的研究问题。
- 低维 \(K\) 的参数量很小，不代表编码器、解码器、图缓存和完整节点解码成本很小。
- 大图的拉普拉斯、邻接矩阵和节点特征矩阵不做 SVD；图信息通过稀疏消息传递和
  小型潜空间线性代数处理。

当前本项目的实现起点见 [Method V0](../methodv0.md)，完整研究流程草稿见仓库根目录
的 `Koopman_Dynamics_Proxy_GFL_Pipeline.md`。
