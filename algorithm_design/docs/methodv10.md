---
title: FedDGN Method V1.0：冲突感知的联邦 Koopman 模式迁移
description: >-
  介绍 FedDGN V1.0 的本地 A-DGN 动力学、共享 ResKoop 观测字典、
  Functional Map 坐标对齐、服务端动力学冲突筛选，以及客户端被动 K* 合成与日志系统。
tags:
  - 图联邦学习
  - 图动力学
  - Koopman
  - Functional Map
  - 动力学冲突筛选
  - Method V1.0
---

# FedDGN Method V1.0：冲突感知的联邦 Koopman 模式迁移

## 1. 问题定义

考虑由 $M$ 个客户端构成的联邦图学习系统。客户端 $m$ 持有私有图

$$
\mathcal G_m=(X_m,A_m,Y_m),
$$

并学习节点隐状态的离散动力学。不同客户端在拓扑、特征和标签分布上的差异，会同时引起两类异质性：

1. **任务动力学异质性**：各客户端适合的状态演化规律并不完全相同；
2. **观测坐标异质性**：即使存在相近的传播机制，其 Koopman 表示仍可能具有不同的排列、相位和基坐标。

FedDGN V1.0 的目标不是构造一个统一的全局 Koopman 算子，而是在不共享原始图、节点轨迹和标签的条件下，识别对目标客户端**可靠、非冲突且具有本地需求**的外部动力学模式。

整体知识流为

$$
\boxed{
K_m
\longrightarrow
\text{共享坐标对齐}
\longrightarrow
\text{可靠模式提取}
\longrightarrow
\text{目标条件冲突筛选}
\longrightarrow
\text{本地需求估计}
\longrightarrow K_m^\star.
}
$$

其中 $K_m^\star$ 是客户端侧的个性化分析算子。当前协议将它保存为动力学诊断产物，不用于 A-DGN 的前向传播、任务损失、梯度更新或下一轮轨迹生成。任务反馈机制不属于本版本的算法定义。

## 2. 方法概览

一轮通信包含四个阶段：

```text
客户端本地任务学习
  └─ 持久化 A-DGN + 本地 Adam
             │
             ▼
客户端动力学辨识
  └─ detached 隐状态轨迹 → ResKoop 字典 → 本地 K_m
             │
             ▼
服务端模式迁移
  └─ 字典 FedAvg + Functional Map + 动力学冲突筛选
             │
             ▼
客户端被动分析
  └─ 私有残差需求 → 闭式 K_m* → 保存与诊断
```

协议边界如下：

| 对象 | 联邦操作 | 当前用途 |
|---|---|---|
| A-DGN 参数与任务 Adam | 不通信 | 每个客户端独立训练和推理 |
| ResKoop 可学习字典参数 | FedAvg | 提供较一致的观测坐标起点 |
| 本地 Koopman 算子 $K_m$ | 上传但不平均 | 服务端对齐、提取和筛选模式 |
| 私有观测快照 $Z_m,Z_m^+$ | 不上传 | 客户端计算动力学需求与 $K_m^\star$ |
| 个性化候选包 $\mathcal P_m$ | 服务端返回 | 客户端被动合成 $K_m^\star$ |
| $K_m^\star$ | 仅客户端保存 | 数值诊断，不参与任务训练 |

服务端不构建 mode graph，不执行 Louvain 或其他社区发现，也不产生全局模式库。服务端的核心操作是对每个目标客户端进行**动力学模式冲突筛选**。

## 3. 本地任务动力学

### 3.1 A-DGN 主干

客户端首先将节点特征编码为 $d_h$ 维隐状态：

$$
H_m^{(0)}=\operatorname{Linear}(X_m).
$$

随后使用共享权重的反对称图动力学进行 $L$ 次显式 Euler 更新：

$$
H_m^{(\ell+1)}
=H_m^{(\ell)}+\epsilon F_{\theta_m}
\left(H_m^{(\ell)},A_m\right),
$$

$$
F_{\theta_m}(H,A)
=\sigma\!\left[
(W_m-W_m^\top-\gamma I)H
+\Phi_{\mathrm{GCN},m}(H,A)+b_m
\right].
$$

默认配置为 $d_h=64$、$L=16$、$\epsilon=0.1$、$\gamma=0.1$ 和 $\sigma=\tanh$。所有传播步共享同一组动力学参数，最终分类头作用于 $H_m^{(L)}$。

### 3.2 本地个性化边界

每个客户端独立维护 encoder、A-DGN field、classifier 和持久化 Adam 状态。任务模型参数及其优化器状态均不上传，服务端也不广播任务模型。因此，任务目标为

$$
\min_{\theta_m}
\mathcal L_{\mathrm{task},m}(\theta_m;\mathcal G_m),
$$

而不是一个共享参数上的全局经验风险。联邦协作发生在 Koopman 观测与模式迁移层，不改变本地任务优化器的生命周期。

## 4. ResKoop 动力学辨识

### 4.1 私有轨迹与快照

任务更新后，客户端重新生成

$$
\mathcal T_m=
\left(H_m^{(0)},H_m^{(1)},\ldots,H_m^{(L)}\right).
$$

该轨迹在 `no_grad` 环境中产生，并在构造快照时显式 `detach`。相邻时间步和节点维被展平为

$$
\mathcal D_m=
\left\{
(h_{m,n}^{(\ell)},h_{m,n}^{(\ell+1)})
\right\}_{\ell,n}.
$$

实现最多等间隔保留 8192 对样本，避免大图上的辨识成本随节点数无界增长。原始轨迹和观测快照始终保存在客户端。

### 4.2 残差式观测字典

观测字典定义为

$$
\Psi_{\omega_m}(h)
=\left[1,\ h,\ \psi_{\omega_m}(h)\right].
$$

默认 $\psi_{\omega_m}$ 为

```text
Linear(64, 64) -> Tanh
Linear(64, 64) -> Tanh
Linear(64, 16)
```

因此观测维数为

$$
d_z=1+d_h+d_{\mathrm{aux}}=1+64+16=81.
$$

常数通道保证字典能够表示平凡不变量，原状态通道保证隐状态可以从观测中直接读取，可学习通道负责补充非线性特征。

### 4.3 EDMD 与算子约定

ResKoop 内部采用行样本矩阵 $Z_X,Z_Y\in\mathbb R^{B\times d_z}$，并求解

$$
Z_Y\approx Z_XK_m^{\mathrm{row}},
$$

$$
K_m^{\mathrm{row}}
=\left(Z_X^\top Z_X+\lambda I\right)^{-1}
Z_X^\top Z_Y,
$$

默认岭正则 $\lambda=10^{-3}$。服务端模式算法和客户端 $K^\star$ 合成统一使用列作用约定：

$$
Z_m=Z_X^\top,\qquad Z_m^+=Z_Y^\top,
$$

$$
Z_m^+\approx K_mZ_m,
\qquad
\boxed{K_m=(K_m^{\mathrm{row}})^\top.}
$$

普通转置用于行作用到列作用的约定转换；复数坐标变换中的逆映射则使用共轭转置。二者不能混淆。

### 4.4 字典学习与可靠性

对解析得到的 $K_m^{\mathrm{row}}$ 做特征分解：

$$
K_m^{\mathrm{row}}v_{m,i}
=\lambda_{m,i}v_{m,i}.
$$

逐模态归一化谱残差为

$$
\rho_{m,i}
=\frac{
\|Z_Yv_{m,i}-\lambda_{m,i}Z_Xv_{m,i}\|_2
}{
\|Z_Yv_{m,i}\|_2+\varepsilon
}.
$$

字典以平均平方谱残差训练；在每个 outer step 内冻结当前特征值和特征向量，仅更新 $\omega_m$，随后重新解析计算 $K_m$。默认每轮执行 1 个 outer step 和 4 个 inner steps。

客户端上传列作用 $K_m$、源特征值、逐模态残差及整体谱残差。由于行作用算子和列作用算子的右特征向量不同，服务端重新分解列作用算子，并用 Hungarian 特征值匹配将源残差关联到新的特征对，而不是复用原数组位置。

## 5. 共享观测字典

客户端只上传可学习字典 $\psi_{\omega_m}$ 的参数。服务端使用与配置一致的等权或样本量加权方式聚合：

$$
\omega_g^r=
\sum_{m\in\mathcal S_r}p_m\omega_m^r,
\qquad
\sum_m p_m=1.
$$

下一轮客户端先在旧私有坐标中完成上一轮候选包的被动 $K_m^\star$ 合成，然后安装 $\omega_g^r$。安装全局字典后重建普通 Koopman Adam：

$$
\omega_m^{r+1}\leftarrow\omega_g^r,
\qquad
\operatorname{AdamState}_{m}^{r+1}\leftarrow0.
$$

共享字典的作用是减小客户端之间的观测坐标漂移，但它不意味着各客户端的 $K_m$ 应当相同。每个 $K_m$ 仍由本地任务轨迹解析得到，且服务端不计算 $\sum_mp_mK_m$。

## 6. Functional Map 对齐

### 6.1 谱 medoid 参考

每轮服务端先计算客户端算子谱之间的归一化 Hungarian 距离：

$$
d_{\mathrm{spec}}(m,n)
=\frac{
\min_{\pi}\frac1{d_z}
\sum_i|\lambda_{m,i}-\lambda_{n,\pi(i)}|
}{
\frac12\left(
\frac1{d_z}\sum_i|\lambda_{m,i}|\right.
+\left.\frac1{d_z}\sum_i|\lambda_{n,i}|\right)+\varepsilon
}.
$$

参考客户端是谱空间的 medoid：

$$
m_{\mathrm{ref}}
=\arg\min_m\sum_{n\ne m}d_{\mathrm{spec}}(m,n).
$$

参考算子只定义本轮共享坐标系，不代表性能最好的客户端，也不是全局动力学模型。当前实现每轮重新选择参考。

### 6.2 酉 Functional Map

对于客户端 $m$，服务端求解

$$
\boxed{
\min_{C_m}\|C_mK_m-K_{\mathrm{ref}}C_m\|_F,
\qquad C_m^HC_m=I.
}
$$

求解流程包括：

1. 分解本地算子和参考算子；
2. 通过 Hungarian 匹配谱排列；
3. 构造 $V_{\mathrm{ref}}P\operatorname{pinv}(V_m)$；
4. 用 polar/SVD 投影到酉矩阵；
5. 在酉群切空间执行少量投影梯度细化。

对齐后的算子为

$$
\widetilde K_m=C_mK_mC_m^H.
$$

服务端同时记录交换残差

$$
R_{C,m}=
\frac{\|C_mK_m-K_{\mathrm{ref}}C_m\|_F}
{\sqrt{\|K_m\|_F^2+\|K_{\mathrm{ref}}\|_F^2}}
$$

和酉性误差

$$
R_{U,m}=
\frac{\|C_m^HC_m-I\|_F}{\sqrt{d_z}}.
$$

所有特征分解、伪逆和 SVD 均作用于默认 $81\times81$ 的观测算子，不作用于图邻接矩阵、图拉普拉斯或节点特征矩阵。

## 7. 可靠动力学模式

在共享坐标中分解

$$
\widetilde K_mU_{m,i}=U_{m,i}T_{m,i}.
$$

当前实现采用一维 block：

$$
U_{m,i}\in\mathbb C^{d_z\times1},
\qquad
T_{m,i}=[\lambda_{m,i}].
$$

候选模式经过以下过滤：

1. 移除最接近 $\lambda=1$ 的常数模态；
2. 移除非有限值和零范数特征向量；
3. 要求匹配后的谱残差不超过阈值；
4. 按谱残差升序、$|\lambda|$ 降序稳定排序；
5. 每客户端最多保留 `feddgn_candidate_top_modes` 个模式，默认 16。

逐模态残差衡量模式对**来源客户端**轨迹的解释质量。它是可靠性指标，而不是目标客户端上的迁移收益。

## 8. 服务端动力学冲突筛选

### 8.1 子空间重叠

对目标客户端 $m$ 的本地模式 $a$ 与外部候选 $c$，定义

$$
O_{a,c}=U_{m,a}^HU_c,
$$

$$
s_{a,c}
=\frac{\|O_{a,c}\|_F^2}{\min(r_a,r_c)}.
$$

$s_{a,c}$ 描述两个 invariant subspaces 的重叠程度。低重叠表示候选主要提供新方向，本身不构成冲突。

### 8.2 重叠方向上的演化分歧

若两个模式在重叠方向上兼容，应近似满足

$$
T_{m,a}O_{a,c}\approx O_{a,c}T_c.
$$

归一化分歧定义为

$$
d_{a,c}
=\frac{
\|T_{m,a}O_{a,c}-O_{a,c}T_c\|_F^2
}{
(\|T_{m,a}\|_2^2+\|T_c\|_2^2)
\|O_{a,c}\|_F^2+\varepsilon
}.
$$

于是单对模式的冲突分数为

$$
\chi_{a,c}=s_{a,c}d_{a,c}.
$$

候选 $c$ 对目标客户端 $m$ 的冲突使用最大 veto：

$$
\boxed{
\chi_{m\leftarrow c}=\max_a\chi_{a,c}.
}
$$

该定义区分三类关系：

| 关系 | 重叠 $s$ | 分歧 $d$ | 含义 |
|---|---:|---:|---|
| 共享 | 高 | 低 | 相同方向具有相容演化 |
| 互补 | 低 | 任意 | 候选主要提供本地未占据方向 |
| 冲突 | 高 | 高 | 相同方向要求不相容演化 |

这与按模式余弦相似度建图不同：相似度低的互补模式可以被保留，而占据相同方向但具有矛盾特征值的模式会被拒绝。

### 8.3 兼容性与通信预算

若

$$
\chi_{m\leftarrow c}>\tau_{\mathrm{conflict}},
$$

候选被硬筛除。其余候选获得平滑兼容性

$$
\boxed{
g_{m,c}^{\mathrm{compat}}
=\exp\!\left(
-\frac{\chi_{m\leftarrow c}}{\tau_{\mathrm{comp}}}
\right).
}
$$

候选首先按来源客户端分组，各组内部按冲突、可靠性和模式编号排序；随后以 round-robin 方式跨来源选择，直到达到目标客户端的通信预算。该策略避免单一来源凭借大量近重复模式占满候选包。

### 8.4 返回目标客户端坐标

共享空间候选 $U_c$ 被映射回目标客户端的私有观测坐标：

$$
\boxed{
U_{m\leftarrow c}=C_m^HU_c.
}
$$

服务端为每个目标客户端单独构造候选包

$$
\mathcal P_m=
\left\{
(U_{m\leftarrow c},T_c,\chi_{m\leftarrow c},
g_{m,c}^{\mathrm{compat}},\rho_c,\operatorname{source}_c)
\right\}.
$$

候选包严格携带目标客户端、来源轮次和参考客户端信息。它不是全局 $K$，也不会覆盖客户端的 $K_m$。

## 9. 客户端私有需求估计

客户端保留与 $K_m$ 同坐标的私有快照

$$
Z_m,Z_m^+\in\mathbb C^{d_z\times B}.
$$

本地 Koopman 预测残差为

$$
R_m=Z_m^+-K_mZ_m.
$$

对于回拉后的候选 $(U_{m\leftarrow c},T_c)$，先对 $U$ 做 QR 规范化，并同步变换 $T_c$。候选诱导的一阶算子方向为

$$
D_{m,c}
=-(K_mU_{m\leftarrow c}-U_{m\leftarrow c}T_c)
U_{m\leftarrow c}^H.
$$

其在私有快照上的一阶预测变化为 $D_{m,c}Z_m$。定义动力学需求分数

$$
\boxed{
N_{m,c}
=\operatorname{Re}
\langle R_m,D_{m,c}Z_m\rangle_F.
}
$$

对小正步长 $\beta$，有

$$
\|R_m-\beta D_{m,c}Z_m\|_F^2
=\|R_m\|_F^2
-2\beta N_{m,c}
+\beta^2\|D_{m,c}Z_m\|_F^2.
$$

因此 $N_{m,c}>0$ 表示该候选在一阶意义下可能减小本地预测残差；$N_{m,c}\le0$ 的候选不参与合成。需求门控为

$$
g_{m,c}^{\mathrm{need}}
=\frac{[N_{m,c}]_+}
{\max_j[N_{m,j}]_++\varepsilon},
$$

最终权重为

$$
\boxed{
\alpha_{m,c}
=g_{m,c}^{\mathrm{compat}}
g_{m,c}^{\mathrm{need}}.
}
$$

服务端 compatibility 回答“候选是否与目标动力学冲突”，客户端 necessity 回答“候选是否有机会修正我的私有残差”。二者承担不同职责。

## 10. 被动个性化算子 $K_m^\star$

给定候选集合及其权重，客户端求解带本地锚定的软约束问题：

$$
\boxed{
K_m^\star
=\arg\min_K
\|K-K_m\|_F^2
+\eta\sum_c\alpha_{m,c}
\|KU_{m\leftarrow c}-U_{m\leftarrow c}T_c\|_F^2.
}
$$

第一项限制个性化算子偏离本地辨识结果，第二项仅吸收非冲突且具有正需求的候选动力学。其正规方程为

$$
K_m^\star A_m=B_m,
$$

$$
A_m=I+\eta\sum_c\alpha_{m,c}
U_{m\leftarrow c}U_{m\leftarrow c}^H,
$$

$$
B_m=K_m+\eta\sum_c\alpha_{m,c}
U_{m\leftarrow c}T_cU_{m\leftarrow c}^H.
$$

因此

$$
\boxed{
K_m^\star=B_mA_m^{-1},
}
$$

实现通过线性方程求解获得结果，不显式计算逆矩阵。若候选为空或所有需求非正，则 $A_m=I$、$B_m=K_m$，自然得到 $K_m^\star=K_m$。

需要强调的是，$K_m^\star$ 当前只用于分析：

$$
\frac{\partial\mathcal L_{\mathrm{task}}}
{\partial K_m^\star}=0.
$$

它不修改 A-DGN 参数，不构造外力，不更新 ResKoop 字典，也不作为下一轮的本地 $K_m$。从 $K_m^\star$ 到任务动力学的耦合需要独立定义稳定性约束、状态空间映射和因果消融，因此留给后续方法版本。

## 11. 跨轮协议

### 11.1 第 1 轮

1. 客户端从共同初始化开始，但各自训练本地 A-DGN；
2. 客户端从 detached 轨迹拟合本地字典和 $K_m^1$；
3. 客户端上传字典参数、列作用 $K_m^1$ 和可靠性元数据；
4. 服务端聚合字典，选择谱 medoid，完成对齐与冲突筛选；
5. 服务端保存数值分析并发布每个客户端的候选包 $\mathcal P_m^1$。

第 1 轮没有匹配的上一轮候选，因此不计算 $K_m^\star$。

### 11.2 第 $r\ge2$ 轮

1. 客户端恢复本地任务模型、任务 Adam、上一轮私有 $K_m^{r-1},Z_m^{r-1},Z_m^{+,r-1}$；
2. 验证候选包目标身份、协议版本与 `source_round=r-1`；
3. 在上一轮私有坐标中计算并保存 $K_m^{\star,r-1}$；
4. 安装上一轮聚合字典并重置 Koopman Adam；
5. 独立训练本轮本地 A-DGN；
6. 生成新 detached 轨迹，拟合本轮字典和 $K_m^r$；
7. 上传本轮字典与 Koopman 数据，服务端生成 $\mathcal P_m^r$。

该顺序保证 $K_m^{r-1}$、私有快照和回拉候选处于同一观测坐标中。候选包只在下一轮被消费，所以最后一个服务端轮次生成的候选不影响已完成的任务训练。

## 12. 通信与隐私边界

| 对象 | 方向 | 典型尺寸 | 服务端操作 |
|---|---|---:|---|
| ResKoop 字典参数 | 双向 | 9,360 个 float32 | FedAvg |
| $K_m$ | 客户端→服务端 | $81\times81$ | 对齐和模式分解，不平均 |
| 特征值与逐模态残差 | 客户端→服务端 | 各 81 个 | 可靠性匹配与过滤 |
| 候选 $U,T$ 及诊断 | 服务端→客户端 | 最多 16 个模式 | 目标条件返回 |
| A-DGN 参数与任务 Adam | 不通信 | - | 服务端不可见 |
| 原始图、标签、隐状态轨迹 | 不通信 | - | 服务端不可见 |
| 私有 $Z_m,Z_m^+$ | 不通信 | 最多 $81\times1024$ | 客户端需求估计 |
| $K_m^\star$ | 不通信 | $81\times81$ | 客户端保存与可视化 |

该边界不等同于形式化隐私保证；上传的算子和谱仍可能包含关于本地动力学的统计信息。若需要差分隐私或安全聚合，应在当前协议之外单独定义。

## 13. 日志系统

### 13.1 任务指标

默认 `minimal` 日志不会展开模型参数。运行目录包含：

| 文件 | 内容 |
|---|---|
| `config.json` | 解析后的完整运行配置 |
| `metrics.csv` | 每轮客户端均值，包括 train/validation/test 指标及 Koopman 诊断 |
| `client_best.csv` | 每客户端最佳验证轮及其配对测试指标 |
| `result.json` | 最终的 per-client best-validation paired-test 汇总 |

`result.json` 的 ACC/AUC 不是最后一轮测试均值，而是对每个客户端先选择其最佳验证轮，再读取同轮测试指标，最后计算客户端均值和标准差。

当 `log_profile=diagnostic` 时，系统额外写入逐客户端指标。由于当前任务模型不通信，模型更新向量的跨客户端诊断不会在 `feddgn` 路径中生成。

### 13.2 Koopman 与候选数值日志

服务端在

```text
<run>/koopman/candidates/
```

写入：

| 文件 | 内容 |
|---|---|
| `latest.pt` | 本轮完整分析：raw/aligned $K$、Functional Maps、模式、冲突记录、候选包 |
| `latest.json` | 可读摘要：参考客户端、冲突分布、拒绝率、来源矩阵、对齐误差 |
| `server_rounds.csv` | 每轮参考、候选量、冲突统计和 Functional Map 质量 |

客户端在

```text
<run>/koopman/client_<id>/
```

保存：

| 文件 | 内容 |
|---|---|
| `latest_K.pt` | 按保存间隔记录当前本地列作用 $K$、特征值和模式残差 |
| `personalized/latest.pt` | 完整 $K$、$K^\star$、$\Delta K$、候选明细和统计 |
| `personalized/latest.json` | 可读的候选权重、需求、残差及可视化入口 |

关键客户端诊断包括：候选总数、正需求数、活跃候选数、平均 compatibility/need/$\alpha$、$\|K^\star-K\|_F/\|K\|_F$、前后私有快照预测残差、前后谱半径，以及候选约束残差。

### 13.3 耗时日志

`timing.csv` 每轮写一行，与日志 profile 无关。主要字段为：

| 字段组 | 测量范围 |
|---|---|
| `round_wall_seconds` | 完整通信轮 wall time |
| `client_parallel_wall_seconds` | 服务端等待并行客户端完成的 wall time |
| `task_mean/max_seconds` | 本地任务训练、验证、测试和最佳模型保存 |
| `reskoop_mean/max_seconds` | detached rollout、字典训练、$K$ 辨识、快照与数据包构造 |
| `kstar_mean/max_seconds` | 候选接收、私有需求、$K^\star$ 求解、保存及选择性绘图 |
| `kstar_compute_mean/max_seconds` | 仅 $K^\star$ 数值求解 |
| `kstar_visualization_mean/max_seconds` | 仅被触发轮次的客户端绘图 |
| `dictionary_receive_mean/max_seconds` | 安装全局字典并重建 Koopman Adam |
| `client_other_mean/max_seconds` | 客户端切换、数据加载及未归入上述阶段的工作 |
| `server_dictionary_fedavg_seconds` | 字典聚合 |
| `server_modes_build_seconds` | Functional Map、模式提取、冲突筛选和候选构造 |
| `server_modes_return_seconds` | 发布候选包与全局字典 |
| `server_numeric_artifacts_seconds` | 保存候选数值产物 |
| `server_metric_logging_seconds` | 写任务和诊断指标 |
| `server_checkpoint_seconds` | 保存服务端 checkpoint |
| `server_update_seconds` | 服务端从接收消息到完成本轮更新的总耗时 |

客户端的 mean/max 是并行 worker 上独立测得的持续时间，不能把各阶段均值相加后当作整轮 wall time。`kstar_seconds` 已包含 `kstar_compute_seconds` 和可能的 `kstar_visualization_seconds`，也不能重复相加。

## 14. 可视化与诊断面板

可视化必须沿着实际知识传递链组织，而不是只展示最终 ACC 或一个低维谱图。当前设计围绕六个诊断问题展开：

```text
Functional Map 对齐
        -> 动力学冲突筛选
        -> 候选漏斗
        -> compatibility × need
        -> K -> K* 的修改效果
        -> 跨轮知识流
```

这些图严格保持服务端与客户端的观测边界。服务端图只使用对齐算子、模式元数据和冲突记录；客户端图才使用私有 $Z_m,Z_m^+$、need 和 $K_m^\star$。不上传私有快照来换取可视化。

### 14.1 已实现的客户端 $K\rightarrow K^\star$ 页面

当前自动页面只比较**同一个客户端、同一个来源轮次、同一个私有观测坐标**中的 $K_m$ 与 $K_m^\star$。输出目录为

```text
<run>/koopman/client_<id>/personalized/figures/
  source_round_<r>/
    K_vs_Kstar_matrix.png
    summary.json
    index.html
  latest.html
```

默认在第一次能够计算 $K^\star$、每 `feddgn_kstar_visualize_every` 个目标训练轮（默认 10）以及最后一个可消费候选的轮次生成。

页面并排显示三张矩阵热图：

1. `log10 |K|`：本地解析算子的元素幅值；
2. `log10 |K*|`：软约束合成后的元素幅值；
3. `log10 |K* - K|`：候选模式在算子层面诱导的更新位置与强度。

$K$ 与 $K^\star$ 使用完全相同的色标，$\Delta K$ 使用独立色标。因此第三张图适合判断修改集中在哪些观测通道，但不能将其颜色深浅直接与前两张图比较。

`summary.json` 同时保留

$$
\|K\|_F,\quad \|\Delta K\|_F,\quad
\frac{\|\Delta K\|_F}{\|K\|_F},\quad
\rho(K),\quad \rho(K^\star),\quad
\frac{\|\operatorname{Im}(\Delta K)\|_F}{\|\Delta K\|_F}.
$$

它们分别用于判断修改幅度、谱稳定性变化和复数更新占比。矩阵热图只说明 $K^\star$ 的数值变化，不能证明任务性能提高；$K^\star$ 当前没有进入任务前向或梯度计算。

### 14.2 服务端图：Functional Map 对齐质量

#### 图 A：跨客户端残差对比

对每个客户端记录三个归一化残差：

$$
R_m^{\mathrm{id}},\qquad R_m^{\mathrm{spec}},\qquad
R_m^{\mathrm{final}},
$$

分别对应恒等映射、谱初始化和最终 Functional Map 的交换残差；同时记录酉性误差

$$
E_m^{\mathrm{unitary}}
=\frac{\|C_m^HC_m-I\|_F}{\sqrt{d_z}}.
$$

推荐使用 grouped bar 或 dumbbell，而不是只画最终值。正常情况下应有

$$
R_m^{\mathrm{final}}<R_m^{\mathrm{id}},
$$

并且大多数客户端满足 $R_m^{\mathrm{final}}\le R_m^{\mathrm{spec}}$。数据来自 `latest.pt` 的 `functional_map_reports` 和 `server_rounds.csv` 的聚合字段。

#### 图 B：mode correspondence heatmap

对目标客户端 $m$ 绘制

$$
M_{ij}=\left|v_{\mathrm{ref},i}^H C_m v_{m,j}\right|^2.
$$

清晰的置换对角结构表示 $C_m$ 建立了相对明确的 mode 对应；大面积均匀扩散表示一个本地 mode 被映射到多个参考 mode，此时后续 mode-level overlap 只能作弱解释。该图只对指定 target client 生成，避免批量产生大量 $d_z\times d_z$ 图片。

### 14.3 服务端图：Mode Conflict Map 与候选漏斗

#### 图 C：overlap--disagreement 冲突图

每个外部候选绘制点

$$
s_{m,c}=\text{overlap},\qquad d_{m,c}=\text{disagreement},
$$

并叠加当前硬冲突边界

$$
d=\frac{\tau_{\mathrm{conflict}}}{s}.
$$

横轴高、纵轴高的右上区域表示“同方向且演化不相容”，点被 veto；横轴低的点即使 disagreement 较大，通常也代表互补方向而不是冲突。颜色编码 `accepted/rejected`，marker 可编码 `source_client`，可靠性 $\rho$ 建议保留在 hover 或表格中，避免把较小残差误画成较小质量。

数据来自 `latest.pt` 的 `conflict_rows`：`overlap`、`disagreement`、`conflict`、`compatibility`、`reliability` 和 `rejected`。该图验证的是服务端“重叠方向上的动力学冲突”，不是客户端需求。

#### 图 D：Candidate Funnel

对每个 target client，将候选数量按以下阶段显示为水平 funnel bar：

| 阶段 | 数量 | 语义 |
|---|---:|---|
| 冲突评估 | `raw_external_count` | 通过来源可靠性筛选、进入目标条件比较的外部模式 |
| 非冲突 | `raw_external_count - conflict_rejected_count` | 通过动力学冲突 veto |
| 通信发送 | `candidate_count` | 经过来源分组、round-robin 和候选预算 |
| 正向需求 | `candidate_positive_need` | $N_{m,c}>0$ 的候选 |
| 实际吸收 | `candidate_active` | $\alpha_{m,c}>0$ 并进入 $A_m,B_m$ |

“通信发送”阶段减少不一定表示模式质量差，也可能只是预算截断；“正向需求”减少才表示候选没有沿一阶方向降低私有预测残差。

### 14.4 客户端图：Compatibility--Necessity Map

对每个收到的候选绘制

$$
x=g_{m,c}^{\mathrm{compat}},\qquad
y=g_{m,c}^{\mathrm{need}},\qquad
\alpha_{m,c}=xy.
$$

四个区域分别表示：

| 区域 | 解释 |
|---|---|
| 高 compatibility、高 need | 最有可能被吸收的候选 |
| 高 compatibility、低 need | 安全但对当前私有残差帮助小 |
| 低 compatibility、高 need | 目标有需求但服务端认为风险高 |
| 低 compatibility、低 need | 当前轮次可忽略 |

图中可用颜色表示 `source_client`，点大小表示 $\alpha$；原始 $N_{m,c}$ 另存为 hover 或附图，避免 need 归一化后掩盖绝对残差尺度。数据来自客户端 `personalized/latest.pt` 的 `candidate_rows`。

### 14.5 客户端图：$K\rightarrow K^\star$ 的真实修改效果

矩阵热图之外，页面应增加三个诊断 panel。

#### Panel A：私有轨迹残差 before/after

$$
R_{\mathrm{before}}=
\frac{\|Z^+-KZ\|_F}{\|Z^+\|_F},\qquad
R_{\mathrm{after}}=
\frac{\|Z^+-K^\star Z\|_F}{\|Z^+\|_F}.
$$

用 dumbbell 显示两者，并记录只用于诊断的

$$
G_{\mathrm{pred}}=
\frac{R_{\mathrm{before}}-R_{\mathrm{after}}}
{R_{\mathrm{before}}+\varepsilon}.
$$

$G_{\mathrm{pred}}>0$ 表示私有轨迹拟合改善，$G_{\mathrm{pred}}<0$ 表示合成后恶化。

#### Panel B：active mode constraint before/after

对 active candidate 绘制

$$
r_c^{\mathrm{before}}=
\frac{\|KU_c-U_cT_c\|_F}{\|U_cT_c\|_F},\qquad
r_c^{\mathrm{after}}=
\frac{\|K^\star U_c-U_cT_c\|_F}{\|U_cT_c\|_F}.
$$

候选约束变好而私有残差变差是一个需要单独报告的失效模式：它说明 $K^\star$ 更服从外部 mode，但外部约束的联合吸收没有改善目标客户端的本地动力学。

#### Panel C：数值稳定性摘要

显示 $\|\Delta K\|_F/\|K\|_F$、$\rho(K)$、$\rho(K^\star)$ 和正规方程的 `A_condition`。谱半径只作数值摘要，不再绘制复平面谱分布。

### 14.6 跨轮图：Round-wise Knowledge Flow

跨轮页面同时画出以下曲线，不将它们人为压缩为单一“知识饱和分数”：

| 曲线 | 指标 | 数据来源 |
|---|---|---|
| 对齐质量 | `mean_functional_map_residual`、`max_functional_map_orthogonality_error` | `server_rounds.csv` |
| 冲突动态 | `conflict_rejection_ratio`、median、q95 | `server_rounds.csv` |
| 候选生存 | sent、positive need、active | server summary + client history |
| 吸收强度 | `candidate_mean_alpha`、`candidate_delta_k_relative` | client history |
| 私有收益 | `candidate_prediction_residual_gain` | client history |

如果出现知识吸收与饱和，通常应观察到后期 `candidate_active`、平均 $\alpha$ 和相对 $\Delta K$ 下降，而早期 $G_{\mathrm{pred}}$ 可能为正并逐渐接近零。若这些量长期不变，说明外部动力学需求没有减弱，不能只用 ACC 曲线宣称已发生知识饱和。

### 14.7 推荐的六面板布局

对指定 target client 和 source round，网页可采用以下结构；矩阵热图作为页面下方的可展开区域：

```text
┌───────────────────────────────────────────────────────┐
│ Functional Map residual + mode correspondence        │
├──────────────────────────┬────────────────────────────┤
│ Mode Conflict Map        │ Candidate Funnel           │
├──────────────────────────┼────────────────────────────┤
│ Compatibility × Need     │ K -> K* residual effects   │
├──────────────────────────┴────────────────────────────┤
│ Round-wise Knowledge Flow                              │
└───────────────────────────────────────────────────────┘
```

这套布局将“服务端能不能给”和“客户端需不需要”分开呈现，避免把冲突筛选和需求门控混成一个黑盒分数。

### 14.8 数据契约、历史日志与触发策略

当前 `latest.pt` 和 `personalized/latest.pt` 已足以支持单轮诊断：

| 文件 | 可视化数据 |
|---|---|
| `koopman/candidates/latest.pt` | Functional Map、对齐算子、`conflict_rows`、候选包、来源矩阵 |
| `koopman/candidates/server_rounds.csv` | 每轮对齐质量、冲突统计、候选统计 |
| `koopman/client_<id>/personalized/latest.pt` | `candidate_rows`、$K$、$K^\star$、$\Delta K$、私有残差和条件数 |
| `koopman/client_<id>/personalized/latest.json` | 单轮可读摘要与页面索引 |

跨轮客户端图不能依赖会被覆盖的 `personalized/latest.pt`。应在每次成功计算 $K^\star$ 后追加 `personalized/history.csv`，至少包含 `source_round`、候选总数、正需求数、active 数、平均 compatibility/need/$\alpha$、相对 $\Delta K$、before/after 私有残差及 `candidate_prediction_residual_gain`。

数值日志每轮写入，绘图按间隔触发：第一次有效 $K^\star$、每 `feddgn_kstar_visualize_every` 轮、最后一个可消费候选的轮次，或显式 `force=True`。不绘图时也必须照常保存 CSV 和 JSON；绘图耗时单独计入 `kstar_visualization_seconds`。

### 14.9 解释边界

本版本不绘制或不推断：

- mode graph、Louvain 或社区结构；
- 服务端全局 $K$ 或跨客户端两两算子距离图；
- Koopman 复平面谱分布；
- 外力幅度、方向、轨迹 tracking；
- $K^\star$ 对任务 ACC 的因果提升。

候选约束残差下降不等于私有预测残差下降，私有预测残差下降也不等于任务性能提升。服务端图应从 `latest.pt`/`server_rounds.csv` 解读，客户端图应从同一私有坐标中的 `personalized/latest.pt` 解读。当前实现的核心模块边界仍为：`candidate_pipeline.py` 负责 Functional Map、模式提取和冲突筛选；`personalized_koopman.py` 负责 need gate、闭式 $K^\star$ 与诊断；`client.py` 负责跨轮时序；`kstar_visualize.py` 负责已实现的客户端矩阵页面。

## 15. 默认配置

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `n_clients` / `n_workers` | 10 / 10 | 客户端数与并行 worker 数 |
| `n_rnds` / `n_eps` | 100 / 1 | 通信轮数与本地 epoch 数 |
| `local_only` | `true` | 任务模型保持客户端本地 |
| `aggregation` | `equal` | 仅用于 ResKoop 字典 FedAvg |
| `feddgn_hidden_dim` | 64 | A-DGN 隐状态维数 |
| `feddgn_steps` | 16 | 共享权重 Euler 步数 |
| `feddgn_step_size` | 0.1 | Euler 步长 |
| `feddgn_gamma` | 0.1 | 反对称动力学阻尼 |
| `feddgn_koopman_aux_dim` | 16 | 可学习辅助观测维数 |
| `feddgn_koopman_ridge` | $10^{-3}$ | EDMD 岭正则 |
| `feddgn_koopman_max_snapshots` | 8192 | ResKoop 最大快照对数 |
| `feddgn_candidate_top_modes` | 16 | 每来源客户端可靠模式上限 |
| `feddgn_candidate_max_per_client` | 16 | 每目标客户端候选包上限 |
| `feddgn_candidate_mode_residual_max` | 1.0 | 模式可靠性阈值 |
| `feddgn_candidate_conflict_threshold` | 0.15 | 硬冲突 veto 阈值 |
| `feddgn_candidate_compat_temperature` | 0.1 | compatibility 温度 |
| `feddgn_candidate_eta` | 0.1 | $K^\star$ 软约束强度 |
| `feddgn_candidate_need_snapshots` | 1024 | 客户端需求估计快照上限 |
| `feddgn_functional_map_steps` | 8 | Functional Map 最大细化步数 |
| `feddgn_kstar_visualize_every` | 10 | 客户端热图间隔 |

## 16. 复杂度

设客户端 $m$ 有 $N_m$ 个节点，A-DGN 步数 $L$，最大 ResKoop 快照数 $B$，观测维数 $d_z$，每客户端可靠模式上限 $q$。

- A-DGN rollout 的主要开销是 $L$ 次图卷积；
- EDMD Gram 构造为 $O(Bd_z^2)$；
- 本地算子求解和谱分解为 $O(d_z^3)$；
- 每个 Functional Map 的小矩阵运算为 $O(d_z^3)$；
- 目标条件冲突比较约为 $O(M^2q^2d_z)$；
- 客户端 $K^\star$ 求解为 $O(d_z^3)$。

默认 $d_z=81$、$q=16$。服务端成本来自多个小算子的谱分解和目标条件模式比较，不涉及对大图矩阵做稠密分解。

## 17. 工程不变量

实现应始终满足：

1. `feddgn` 使用本地持久化任务模型与任务 Adam；
2. 客户端上传中不含 task model、task optimizer 或 O4 block-v；
3. Koopman 轨迹对 A-DGN 无梯度；
4. 全局字典只包含可学习字典参数，不包含 $K_m$ 或 $K_m^\star$；
5. 接收全局字典后重建 Koopman Adam；
6. 服务端不平均 $K_m$，不构建全局 $K$；
7. 所有服务端模式比较都在 Functional Map 对齐后的共享坐标中完成；
8. 候选来源客户端不能等于目标客户端；
9. 候选包与私有 $K,Z,Z^+$ 的来源轮次必须一致；
10. $K_m^\star$ 在安装新一轮全局字典之前计算；
11. $K_m^\star$ 不进入任务前向、任务损失、任务梯度或 ResKoop 拟合；
12. 所有 SVD、特征分解和闭式求解只作用于小型 Koopman 空间。

协议标识、轮次、目标身份、张量形状、聚合权重和有限性检查均采用显式失败策略，防止陈旧候选或坐标错配被静默使用。

## 18. 方法总结

FedDGN V1.0 将联邦 Koopman 知识迁移拆分为三个可检验的问题：

$$
\boxed{
\begin{aligned}
&\text{可比较性：Functional Map 将局部算子映射到共享坐标；}\\
&\text{安全性：服务端拒绝重叠方向上的不相容动力学；}\\
&\text{必要性：客户端仅吸收能够一阶修正私有残差的候选。}
\end{aligned}
}
$$

共享 ResKoop 字典减小观测坐标漂移，本地 $K_m$ 保留客户端特有动力学，冲突筛选替代无条件的相似度聚类，闭式 soft operator fitting 则生成以本地算子为锚的 $K_m^\star$。当前版本把 $K_m^\star$ 作为被动、可审计的动力学分析对象；其与任务网络之间不存在计算路径。
