---
title: FedDGN Method V1.0：共享观测字典与功能映射模式图
description: >-
  详细说明 FedDGN V1.0 的 A-DGN 任务主干、O4 聚合、ResKoopNet
  观测字典联邦平均、本地 Koopman 算子、Functional Map 对齐、模式图与个性化模式返回。
tags:
  - 图联邦学习
  - 图动力学
  - ResKoopNet
  - Functional Map
  - Koopman 模式图
  - Method V1.0
---

# FedDGN Method V1.0：共享观测字典与功能映射模式图

## 1. 页面定位

FedDGN V1.0 研究的是一个比“直接平均所有动力学对象”更谨慎的问题：在客户端图结构、特征分布和标签分布均不一致时，如何先把各客户端学到的动力学放到可比较的坐标系中，再发现可迁移的传播模式。

当前版本已经形成两条彼此隔离的联邦链路：

1. **任务链路**：A-DGN 网络参数执行 FedAvg，并使用 O4 `block_v` 聚合 AdamW 的分块二阶矩标量；
2. **动力学分析链路**：ResKoopNet 的可学习观测字典参数执行独立的 FedAvg；每个客户端的 Koopman 算子 $K_m$ 仍由本地轨迹解析计算；服务端用 Functional Map 对齐不同 $K$ 空间，再构建跨客户端模式图。

两条链路的当前边界非常重要：

!!! warning "V1.0 仍是被动分析版本"
    服务端返回的 Functional Map、全局模式库和个性化相关模式**尚未进入 A-DGN 的前向传播、损失函数或外力端口**。因此当前实现可以验证“模式是否可比较、社区是否有结构”，但不能把任务 ACC 的变化归因于模式图反馈。

当前代码对应关系如下：

| 功能 | 实现位置 | 当前状态 |
|---|---|---|
| A-DGN 主干与轨迹输出 | `codev0/models/feddgn_local/` | 已实现 |
| 任务参数 FedAvg 与 O4 `block_v` | `codev0/models/feddgn_avg/`、`codev0/modules/fed_optimizer.py` | 已实现 |
| ResKoopNet 与解析 EDMD | `codev0/models/feddgn/reskoop.py` | 已实现 |
| 观测字典参数 FedAvg | `codev0/models/feddgn/client.py`、`server.py` | 已实现 |
| Functional Map 对齐 | `codev0/models/feddgn/functional_map.py` | 已实现 |
| 模式图、Louvain 与模式库 | `codev0/models/feddgn/mode_graph.py` | 已实现 |
| 服务端模式反馈进入任务动力学 | 尚无 | 未实现 |

## 2. 核心问题：先区分两类异质性

设共有 $M$ 个客户端。客户端 $m$ 持有图

$$
\mathcal G_m=(X_m,A_m,Y_m),
$$

并学习图状态 $H_m(t)$ 的非线性演化：

$$
\frac{dH_m(t)}{dt}=F_{\theta_m}(H_m(t),A_m).
$$

客户端差异至少包含两层含义。

### 2.1 任务动力学异质性

不同的 $A_m$、$X_m$ 和训练样本会产生不同的局部梯度以及不同的有效向量场：

$$
F_{\theta_1}\neq F_{\theta_2}\neq\cdots\neq F_{\theta_M}.
$$

任务网络需要联邦协同，但普通 FedAvg 在非独立同分布场景中容易受到局部 Adam 状态不一致的影响。因此 V1.0 对任务参数使用 FedAvg，对优化器二阶矩使用 O4 分块标量协议。

### 2.2 Koopman 坐标异质性

即使两个客户端存在相似的传播机制，其可学习字典也可能给这个机制分配不同的坐标、排列、相位和尺度。于是两个局部算子 $K_m$ 的特征向量不能仅凭数组位置直接比较。

V1.0 用两层措施处理这个问题：

1. 对 ResKoopNet 的可学习字典参数做 FedAvg，为所有客户端提供共同的观测坐标起点；
2. 对本地更新后仍然存在的坐标偏差，用 Functional Map 显式对齐。

这两步并不重复。字典 FedAvg 减少坐标漂移，Functional Map 则处理本地适配、数据异质性和神经网络参数对称性造成的剩余偏差。

## 3. 记号

| 记号 | 含义 |
|---|---|
| $m$ | 客户端编号，$m\in\{1,\ldots,M\}$ |
| $r$ | 联邦通信轮次 |
| $\theta_m^r$ | 客户端 $m$ 在第 $r$ 轮的 A-DGN 任务参数 |
| $\omega_m^r$ | 客户端 $m$ 的 ResKoopNet 可学习字典参数 |
| $H_m^{(\ell)}$ | A-DGN 第 $\ell$ 个离散传播步的隐藏状态 |
| $\Psi_{\omega_m}$ | 客户端 $m$ 的 Koopman 观测字典 |
| $d_z$ | Koopman 观测维数，默认 $1+64+16=81$ |
| $K_m^r$ | 从客户端本轮本地轨迹解析得到的 Koopman 算子 |
| $C_m^r$ | 从客户端 $m$ 的 $K$ 空间到参考 $K$ 空间的 Functional Map |
| $v_{m,i}$、$\lambda_{m,i}$ | 第 $i$ 个 Koopman 特征向量与特征值 |
| $p_m$ | 聚合权重；默认等权，也可按训练样本数加权 |

## 4. 总体数据流

一轮训练可概括为：

```text
服务器广播
  ├─ 全局 A-DGN 参数 + O4 分块二阶矩
  ├─ 上一轮全局 ResKoop 字典参数
  └─ 上一轮 Functional Map / 模式图个性化视图
              │
              ▼
客户端本地执行
  ├─ 安装全局 A-DGN 参数，按 O4 初始化任务优化器
  ├─ 安装全局观测字典，重置 Koopman Adam
  ├─ 训练 A-DGN 任务网络
  ├─ 重新生成无外力、detach 的本地轨迹
  ├─ 微调观测字典并解析计算本地 K_m
  └─ 上传任务参数、block-v、字典参数、K_m 与候选模式
              │
              ▼
服务器聚合与分析
  ├─ FedAvg 聚合 A-DGN 参数
  ├─ O4 聚合任务网络的分块二阶矩标量
  ├─ FedAvg 聚合 ResKoop 字典参数
  ├─ 保留每个 K_m，不对 K 做平均
  ├─ Functional Map 对齐 K 空间
  ├─ 构建跨客户端模式图并运行 Louvain
  └─ 返回全局模式库与客户端个性化模式视图
```

## 5. 任务主干：16 步 A-DGN

### 5.1 编码、动力学和分类

客户端首先把输入特征编码到 $d_h=64$ 维隐藏空间：

$$
H_m^{(0)}=\operatorname{Linear}(X_m).
$$

随后使用共享权重的反对称图动力学进行 16 次显式 Euler 更新：

$$
H_m^{(\ell+1)}
=H_m^{(\ell)}+\epsilon F_{\theta_m}(H_m^{(\ell)},A_m),
\qquad \ell=0,\ldots,15,
$$

其中默认步长 $\epsilon=0.1$，向量场为

$$
F_{\theta}(H,A)
=\sigma\!\left[
\left(W-W^\top-\gamma I\right)H
+\Phi_{\mathrm{GCN}}(H,A)+b
\right].
$$

默认 $\gamma=0.1$、$\sigma=\tanh$。$W-W^\top$ 提供反对称自作用，$-\gamma I$ 是显式阻尼，$\Phi_{\mathrm{GCN}}$ 负责邻域耦合。16 个传播步共享同一组动力学参数，因此它不是 16 层互不共享参数的普通 GCN。

最后由线性分类头给出节点预测：

$$
\widehat Y_m=\operatorname{Linear}(H_m^{(16)}).
$$

### 5.2 外力端口保持关闭

代码保留了

$$
\dot H=F_{\mathrm{intrinsic}}(H,A)
+\beta F_{\mathrm{external}}(H,A)
$$

的接口，但当前训练和 Koopman 轨迹提取均明确使用

```text
external_force = None
force_scale = 0.0
```

因此当前模型只有内禀 A-DGN 动力学，不存在外力模块，也不存在模式库向本地动力学注入的隐式路径。

## 6. 任务网络的 FedAvg 与 O4 `block_v`

### 6.1 参数聚合

每轮客户端完成任务训练后上传 $\theta_m^r$。服务端使用

$$
\theta_g^r=\sum_{m\in\mathcal S_r}p_m\theta_m^r,
\qquad \sum_m p_m=1.
$$

默认配置 `aggregation=equal`，故 $p_m=1/|\mathcal S_r|$；若切换为样本量加权，则 $p_m$ 与本地训练样本数成正比。

### 6.2 O4 聚合的对象

任务网络默认使用 `feddgn_fed_optimizer=block_v`。设 AdamW 对参数张量 $q$ 的逐元素二阶矩为 $V_{m,q}$，客户端只上传一个标量：

$$
s_{m,q}=\operatorname{mean}(V_{m,q}).
$$

服务端按与模型参数相同的权重聚合：

$$
s_{g,q}=\sum_m p_m s_{m,q}.
$$

下一轮客户端为整个参数块 $q$ 初始化

$$
V_{m,q}\leftarrow s_{g,q}\mathbf 1.
$$

同时，一阶矩 $M_{m,q}$ 从零开始，局部一阶矩步数也重新从 0 计数；二阶矩的偏置修正步数则继承服务端累计的 `global_v_step`。因此 O4 不是“聚合完整 Adam”，而是：

- 聚合每个任务参数张量的一个二阶矩均值；
- 不上传逐元素二阶矩；
- 不聚合一阶矩；
- 不把 Koopman Adam 混入 O4。

默认 A-DGN 共有 7 个任务参数块，所以每个客户端每轮只为 O4 额外上传 7 个标量。

## 7. ResKoopNet：共享字典起点，本地解析算子

### 7.1 单向依赖与轨迹样本

任务更新结束后，客户端用更新后的 A-DGN 再运行一次 16 步、无外力轨迹：

$$
\mathcal T_m^r=
\left(H_m^{(0)},H_m^{(1)},\ldots,H_m^{(16)}\right).
$$

该轨迹在 `torch.no_grad()` 下产生，并在构造快照时再次 `detach`。所以依赖关系只有

$$
\theta_m^r\longrightarrow \mathcal T_m^r
\longrightarrow (\omega_m^r,K_m^r),
$$

不存在 Koopman 损失反向更新 A-DGN 的路径。

相邻时间步和节点维被展平成样本对：

$$
\mathcal D_m^r=
\left\{
(h_{m,n}^{(\ell)},h_{m,n}^{(\ell+1)})
\right\}_{\ell,n}.
$$

样本过多时，代码以确定性的等间隔方式最多保留 8192 对，避免大图产生不受控的 Koopman 拟合开销，也不消耗任务优化器的随机数流。

### 7.2 残差式观测字典

观测字典不是只输出神经网络特征，而是保留常数项和原状态：

$$
\boxed{
\Psi_{\omega}(h)=
\left[1,\ h,\ \psi_{\omega}(h)\right]
}
$$

默认维数为：

$$
d_z=1+d_h+d_{\mathrm{aux}}
=1+64+16=81.
$$

这也是当前 $K_m\in\mathbb R^{81\times81}$ 的来源。可学习部分 $\psi_\omega$ 是

```text
Linear(64, 64) -> Tanh
Linear(64, 64) -> Tanh
Linear(64, 16)
```

共 6 个参数张量、9,360 个标量参数。常数项和原始状态通道没有可学习参数，也不会被聚合。

### 7.3 EDMD 解析估计本地 $K_m$

令

$$
Z_X=\Psi_{\omega_m}(H_t),
\qquad
Z_Y=\Psi_{\omega_m}(H_{t+1}),
$$

实现采用行样本约定

$$
Z_Y\approx Z_XK_m.
$$

每个 outer step 都通过带岭正则的 EDMD 重新计算

$$
\boxed{
K_m=
\left(Z_X^\top Z_X+\lambda I\right)^{-1}
Z_X^\top Z_Y
}
$$

默认 $\lambda=10^{-3}$。线性方程求解失败时才退化为对这个小型正则 Gram 矩阵求伪逆。

### 7.4 字典训练目标

对当前解析得到的 $K_m$ 做特征分解：

$$
K_mV_m=V_m\Lambda_m.
$$

在一个 outer step 内固定 $\Lambda_m,V_m$，用归一化逐模态谱残差训练字典：

$$
r_{m,i}=
\frac{
\left\|Z_Yv_{m,i}-\lambda_{m,i}Z_Xv_{m,i}\right\|_2
}{
\left\|Z_Yv_{m,i}\right\|_2+\varepsilon
},
$$

$$
\mathcal L_{\mathrm{dict}}
=\frac{1}{d_z}\sum_i r_{m,i}^2.
$$

默认每轮执行 1 个 outer step、4 个 inner step，inner batch size 为 1024。结束后用更新后的字典再次解析计算最终 $K_m$、特征值、特征向量和状态空间模式。

## 8. 新增：观测字典的独立 FedAvg

### 8.1 聚合什么

客户端上传且服务端平均的只有 $\psi_{\omega_m}$ 的 `state_dict`：

$$
\boxed{
\omega_g^r=\sum_{m\in\mathcal S_r}p_m\omega_m^r
}
$$

聚合权重与任务模型 FedAvg 保持一致。默认等权时，任务参数和字典参数都是 10 个客户端的算术平均。

### 8.2 明确不聚合什么

以下对象不属于字典 FedAvg：

- 本地解析算子 $K_m^r$；
- Koopman 特征值与特征向量；
- Koopman 状态空间模式；
- Koopman Adam 的一阶矩、二阶矩和步数；
- Functional Map $C_m^r$；
- 模式图的社区标签。

$K_m^r$ 会上传给服务端用于对齐和比较，但服务端不会计算

$$
K_g=\sum_m p_mK_m,
$$

也不会把任何所谓的“全局 $K$”覆盖到客户端。原因是 $K_m$ 同时依赖本地轨迹和本地更新后的字典坐标；在未对齐坐标中直接平均不仅缺少物理含义，还会抹掉需要分析的动力学异质性。

### 8.3 接收后为什么重置 Koopman Adam

客户端在下一轮开始时，把本地字典参数从 $\omega_m^r$ 替换成服务端平均 $\omega_g^r$。旧 Adam 状态是在旧参数轨迹上积累的，尤其一阶矩包含客户端特有的更新方向。继续沿用会使“全局参数点”和“本地优化器历史”错配。

因此实现采用：

$$
\omega_m^{r+1}\leftarrow\omega_g^r,
\qquad
\operatorname{AdamState}_m^{r+1}\leftarrow 0.
$$

这里是完整重建普通 Adam，而不是套用任务链路的 O4 `block_v`。两套优化器状态协议必须分开理解：

| 分支 | 参数接收 | 优化器状态处理 |
|---|---|---|
| A-DGN 任务网络 | FedAvg 全局参数 | O4：聚合 block-v，重置一阶矩 |
| ResKoop 可学习字典 | FedAvg 全局字典 | 重置整个 Koopman Adam |
| 解析 $K_m$ | 不接收、不平均 | 无优化器状态 |

### 8.4 轮次一致性保护

字典消息使用协议标识 `reskoop_dictionary_fedavg_v1`，并携带 `source_round`。客户端在第 $r+1$ 轮只接受来源恰好为第 $r$ 轮的字典；缺失、协议不匹配或轮次陈旧都会直接报错，而不是静默沿用旧参数。

日志还记录：

- 是否收到全局字典；
- 是否重置 Koopman Adam；
- 覆盖前本地字典到全局字典的相对距离；
- 覆盖后的相对距离；
- 字典来源轮次。

这些字段使“字典确实被聚合并安装”成为可验证事实，而不是仅凭代码路径推测。

## 9. 本地 Koopman 模式筛选

### 9.1 两类“模式”不要混淆

实现同时保存：

1. **观测空间特征向量** $v_{m,i}\in\mathbb C^{81}$：用于 Functional Map 对齐与模式图建边；
2. **状态空间 Koopman mode** $\xi_{m,i}\in\mathbb C^{64}$：由特征向量逆变换得到，用于解释原隐藏状态的重构方向。

当前模式图真正比较的是对齐后的观测空间特征向量，而不是未经对齐的 64 维状态 mode。

### 9.2 去除常数模态并按残差选择

由于字典含固定常数观测量，通常存在一个最接近 $\lambda=1$ 的平凡模态。客户端先移除该模态，再过滤非有限值和零范数模态，然后按归一化谱残差从小到大排序；$|\lambda|$ 只作为稳定的次级排序键。

默认每个客户端上传最多 16 个非平凡候选模式：

$$
\mathcal M_m^r=
\{(v_{m,i},\xi_{m,i},\lambda_{m,i},r_{m,i})\}_{i=1}^{16}.
$$

## 10. Functional Map：把局部 $K$ 空间映射到共享参考空间

### 10.1 固定参考算子

第一次构图时，服务端选择谱残差最小的客户端作为参考：

$$
s=\arg\min_m R_m,
$$

并保存当轮的 $K_s$。V1.0 后续轮次固定这个参考客户端、参考算子和参考轮次，不会每轮重选。这样模式社区随轮次的变化不会混入“参考坐标系也在移动”的额外因素。

这是一项有意的设计约束，不代表固定参考一定是最终最优方案；参考空间的迭代更新属于后续版本。

### 10.2 对齐目标

对每个客户端求一个复数酉映射

$$
C_m:\mathcal Z_m\rightarrow\mathcal Z_s,
$$

使动力学尽量满足交换关系：

$$
\boxed{C_mK_m\approx K_sC_m},
$$

并约束

$$
C_m^HC_m=I.
$$

归一化交换残差定义为

$$
R_C=
\frac{\|C_mK_m-K_sC_m\|_F}
{\sqrt{\|K_m\|_F^2+\|K_s\|_F^2}}.
$$

酉性误差为

$$
R_U=
\frac{\|C_m^HC_m-I\|_F}{\sqrt{d_z}}.
$$

约束 $C_m$ 为酉矩阵可以防止退化的零映射，并尽量保留向量内积、范数和后续余弦相似度。

### 10.3 求解流程

当前求解器只作用在默认 $81\times81$ 的小算子上：

1. 分别分解 $K_m$ 和 $K_s$；
2. 根据特征值距离，用 Hungarian 匹配建立谱排列 $P$；
3. 构造无约束初值 $V_sP\operatorname{pinv}(V_m)$；
4. 通过 polar/SVD 投影到最近酉矩阵；
5. 在单位酉群切空间上做最多 8 次投影梯度细化，并用回溯保证残差不增；
6. 在恒等映射与谱匹配初值中选择交换残差较低者作为细化起点。

参考客户端使用严格恒等映射 $C_s=I$。

!!! info "不会对大图做 SVD"
    Functional Map 的 SVD、特征分解和伪逆仅作用于 $d_z\times d_z$ 的小型 Koopman 算子或映射。实现不会对图拉普拉斯、邻接矩阵或节点特征矩阵执行稠密 SVD。

### 10.4 模式运输

局部观测特征向量被运输到参考空间：

$$
\widetilde v_{m,i}=C_mv_{m,i}.
$$

只有 $\widetilde v_{m,i}$ 参与跨客户端相似度计算。原始 $v_{m,i}$、状态模式 $\xi_{m,i}$、特征值和残差仍作为元数据保存，便于解释和回查。

## 11. 仅用空间相似度构建模式图

### 11.1 唯一边权

根据奥卡姆剃刀原则，当前图只保留 Functional Map 对齐后的空间方向相似度：

$$
\boxed{
S_{ij}=
\left|
\frac{\widetilde v_i^H\widetilde v_j}
{\|\widetilde v_i\|_2\|\widetilde v_j\|_2}
\right|
}
$$

绝对值使它对复相位和实数符号翻转不敏感。特征值距离、残差可靠性和时间衰减不进入边权；它们只作为诊断信息。

### 11.2 跨客户端、客户端均衡的稀疏化

直接对全部模式做全连接会让拥有大量相似内部模式的客户端主导社区。当前稀疏化策略为：

1. 对源模式 $i$，完全排除同一客户端的模式；
2. 在每个其他客户端中先保留相似度最高的一个候选；
3. 再从这些“每客户端一个”的候选中取前 $k$ 个，默认 $k=5$；
4. 对所有有向选择取并集并对称化。

因此图中每条边都跨客户端，且一个外国客户端不能用多个近重复模式占满某个节点的邻居名额。

### 11.3 Louvain 自动发现社区数

服务端在加权稀疏图上执行确定性 Louvain：

$$
\{\mathcal C_1,\ldots,\mathcal C_K\}
=\operatorname{Louvain}(G_{\mathrm{mode}}).
$$

默认 `resolution=1.0`，随机种子固定，因此轮次间差异主要来自模式图本身，而不是社区算法随机重启。社区数量 $K$ 由图结构发现，不使用预设聚类数，也不依赖 eigengap。

## 12. 聚类质量：不能只看社区数量

发现多个社区不等于社区真的分得开。V1.0 同时在**完整相似度几何**和**稀疏模式图**上报告质量指标。

### 12.1 完整几何指标

令距离 $D_{ij}=1-S_{ij}$。报告：

- `within_similarity`：同社区模式的平均完整空间相似度；
- `between_similarity`：不同社区模式的平均完整空间相似度；
- `similarity_gap`：二者之差，越大越好；
- `within_distance` 和 `between_distance`；
- `between_within_distance_ratio`：

$$
\rho_D=
\frac{\overline D_{\mathrm{between}}}
{\overline D_{\mathrm{within}}}.
$$

理想情况下 $\rho_D>1$，且越大说明簇间距离相对簇内距离越明显。

还报告全空间 Silhouette：

$$
s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}},
$$

其中 $a(i)$ 是样本到本社区的平均距离，$b(i)$ 是到最近其他社区的平均距离。接近 1 较好，接近 0 表示社区在完整几何中严重重叠，负值说明可能分错。

### 12.2 稀疏图指标

- `modularity`：Louvain 所优化的加权模块度；
- `mean_conductance`：社区平均传导率，越低通常越好。

不能只用 modularity 证明聚类有效，因为 Louvain 本身就在优化它。若 modularity 很高、但 Silhouette 约为 0 且 $\rho_D\approx1$，更可能是 kNN 稀疏化制造了看似清楚的图分区，而原始对齐空间并没有强分离。

## 13. 全局模式库与个性化返回

### 13.1 社区代表模式

对社区 $\mathcal C_k$，先把成员模式归一化，选择与其他成员总相似度最大的模式作为 anchor。随后校正每个复数模式相对 anchor 的相位，再求均值并重新归一化：

$$
\bar v_k=
\operatorname{Normalize}
\left(
\frac{1}{|\mathcal C_k|}
\sum_{i\in\mathcal C_k}
e^{-j\phi_i}\widetilde v_i
\right).
$$

代表特征值取社区成员特征值的算术平均：

$$
\bar\lambda_k=
\frac{1}{|\mathcal C_k|}
\sum_{i\in\mathcal C_k}\lambda_i.
$$

由此形成

$$
\mathcal B=
\{(\bar v_k,\bar\lambda_k,|\mathcal C_k|)\}_{k=1}^{K}.
$$

### 13.2 客户端个性化视图

客户端 $m$ 收到：

- 自己的 Functional Map $C_m$ 及其残差、酉性误差；
- 完整的全局模式库 $\mathcal B$；
- 自己上传的候选模式；
- 与自己社区重合的外国客户端模式，按与本地模式的最大空间相似度排序，最多返回 16 个；
- 全局模式图的公开摘要。

“个性化”目前只表示返回内容按客户端社区关系筛选，并不表示这些模式已经作用到本地任务网络。

## 14. 一轮完整协议

### 14.1 第 1 轮

1. 服务端广播初始 A-DGN 参数和空的 O4 二阶矩；
2. 所有客户端从同一个确定性初始化的 ResKoop 字典开始；
3. 客户端训练任务网络，生成 detach 轨迹，微调本地字典，解析 $K_m^1$；
4. 客户端上传任务参数、block-v、字典参数和模式分析包；
5. 服务端分别聚合任务参数、block-v 和字典参数；
6. 服务端选定并固定参考 $K_s$，求 Functional Maps，构图和返回模式视图。

### 14.2 第 $r\ge2$ 轮

1. 客户端从私有 checkpoint 恢复上轮状态；
2. 安装服务端 A-DGN 参数，并按 O4 重建任务优化器；
3. 安装上一轮全局字典，验证协议和轮次，重置 Koopman Adam；
4. 读取上一轮匹配的个性化模式视图，但不用于任务计算；
5. 执行本轮任务训练和本地 ResKoop 拟合；
6. 解析新的本地 $K_m^r$ 并上传；
7. 服务端重复三类聚合与 Functional Map 模式分析。

这里有一个关键时间顺序：客户端收到的是“上一轮聚合结果”，而客户端上传的是“本轮本地更新结果”。代码用严格的 `source_round` 和模式图 `round` 检查防止两者错位。

## 15. 通信对象总表

| 对象 | 方向 | 服务端操作 | 下一轮是否覆盖客户端 | 是否影响任务前向 |
|---|---|---|---|---|
| A-DGN 参数 $\theta_m$ | 双向 | FedAvg | 是 | 是 |
| 任务 block-v 标量 | 双向 | 按参数块加权平均 | 是，用于初始化任务二阶矩 | 是，通过优化步骤 |
| 字典参数 $\omega_m$ | 双向 | 独立 FedAvg | 是，并重置 Koopman Adam | 否 |
| 本地 $K_m$ | 客户端到服务端 | 保留、对齐，不平均 | 否 | 否 |
| 特征值/特征向量/状态模式 | 客户端到服务端 | 筛选、构图、聚类 | 不直接覆盖 | 否 |
| Functional Map $C_m$ | 服务端到客户端 | 个性化返回 | 仅保存分析视图 | 否 |
| 全局模式库 $\mathcal B$ | 服务端到客户端 | 社区内相位对齐平均 | 仅保存分析视图 | 否 |

默认配置下，字典参数为 9,360 个 float32，原始参数体积约 37.4 KB；$K_m$ 为 $81\times81$ float32，约 26.2 KB。客户端还上传最多 16 个复数特征向量、状态模式和少量元数据。实际通信统计以日志中的 `upload_bytes`、`koopman_dictionary_bytes` 和 `koopman_upload_bytes` 为准。

## 16. 保存结果与可视化

### 16.1 客户端 Koopman 文件

每个客户端的最新分析保存在：

```text
<run>/koopman/client_<id>/
```

包含：

- `K.pt`：本地解析 Koopman 算子；
- `dictionary.pt`：本轮本地更新后的字典；
- `eigenvalues.pt`、`eigenvectors.pt`；
- `residual.pt`：逐模态归一化谱残差；
- `modes.pt`：64 维状态空间模式；
- `metadata.json`：拟合、谱半径、条件数、有效秩等诊断；
- `personalized_modes.pt/json`：最近一次接收的服务端个性化视图。

### 16.2 服务端模式图文件

最新模式图位于：

```text
<run>/koopman/mode_graph/latest.pt
<run>/koopman/mode_graph/latest.json
```

默认每 10 轮以及最终轮归档一次：

```text
<run>/koopman/mode_graph/round_0010/
<run>/koopman/mode_graph/round_0020/
...
```

服务端 checkpoint 额外保存全局字典、字典来源轮次、字典聚合摘要、固定参考 $K_s$ 和 O4 状态，支持审计和续跑。

### 16.3 模式图仪表盘

模式图只在归档轮次渲染，默认不是每轮渲染。当前仪表盘包含 6 张图：

1. `global_mode_graph.png`：模式网络、社区与客户端来源；
2. `mode_similarity_matrix.png`：完整对齐空间相似度；
3. `client_community_matrix.png`：客户端—社区计数；
4. `client_community_distribution.png`：每个客户端的社区比例；
5. `functional_map_quality.png`：交换残差与酉性误差；
6. `community_evolution.png`：归档轮次上的社区变化。

`community_spectrum.png` 已废弃并会在重新生成仪表盘时清理。原因是当前边权不使用特征值；把 Louvain 社区画在复平面谱图上，容易让读者误以为社区是由频率或衰减率定义的。客户端级总览中的 `koopman_spectrum.png` 仍然保留，因为它只用于查看本地算子谱，不承担模式图社区解释。

## 17. 默认配置

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `n_clients` / `n_workers` | 10 / 10 | 客户端数与并行 worker 数 |
| `n_rnds` / `n_eps` | 100 / 1 | 联邦轮数与每轮本地 epoch |
| `aggregation` | `equal` | 任务参数、block-v、字典均使用等权聚合 |
| `feddgn_hidden_dim` | 64 | A-DGN 隐藏维数 |
| `feddgn_steps` | 16 | 共享权重 Euler 传播步数 |
| `feddgn_step_size` | 0.1 | Euler 步长 |
| `feddgn_gamma` | 0.1 | 反对称动力学阻尼 |
| `feddgn_fed_optimizer` | `block_v` | 任务网络使用 O4 |
| `feddgn_koopman_aux_dim` | 16 | 可学习观测维数，最终 $d_z=81$ |
| `feddgn_koopman_ridge` | $10^{-3}$ | EDMD 岭正则 |
| `feddgn_koopman_outer_steps` | 1 | 每轮字典 outer step |
| `feddgn_koopman_inner_steps` | 4 | 每个 outer step 的字典梯度步数 |
| `feddgn_koopman_lr` | $10^{-3}$ | Koopman 字典 Adam 学习率 |
| `feddgn_mode_graph_top_modes` | 16 | 每客户端上传候选模式上限 |
| `feddgn_mode_graph_neighbors` | 5 | 每个模式保留的跨客户端邻居上限 |
| `feddgn_functional_map_steps` | 8 | Functional Map 最大细化步数 |
| `feddgn_mode_graph_resolution` | 1.0 | Louvain resolution |
| `feddgn_mode_graph_save_every` | 10 | 模式图归档和可视化间隔 |

## 18. 复杂度与显存边界

设本地节点数为 $N_m$、A-DGN 步数为 $L=16$、观测维数为 $d_z=81$、保留快照数为 $B\le8192$。

- A-DGN 的主要开销来自 16 次图卷积和为 Koopman 保存的 17 个隐藏状态；
- EDMD 构造 Gram 矩阵约为 $O(Bd_z^2)$；
- 本地 $K_m$ 求解和特征分解约为 $O(d_z^3)$；
- 服务端每个 Functional Map 的小矩阵分解约为 $O(d_z^3)$；
- 模式图最多有 $16M$ 个节点，完整相似度矩阵约为 $O((16M)^2d_z)$。

显存峰值仍主要受 A-DGN 多步轨迹、图卷积中间量和多 worker 并发影响，而不是 9,360 参数的 ResKoop MLP。Koopman 分支通过 detach 阻止保留任务反向图，并用 `max_snapshots` 限制字典拟合样本数。

## 19. 工程不变量与失败保护

实现应始终满足以下不变量：

1. `force_rms=0` 且 `force_intrinsic_ratio=0`；
2. Koopman 轨迹对 A-DGN 无梯度；
3. 全局 dictionary 只包含 `network.*` 参数，不包含 `operator` 或 `K`；
4. 第 2 轮及以后，字典覆盖后的相对差应为数值 0；
5. 字典接收后 Koopman Adam 必须重建；
6. O4 只管理 A-DGN 的 block-v，不管理 Koopman Adam；
7. 参考 $K_s$ 一经首轮选定，在本次运行中保持不变；
8. 模式图边必须跨客户端，边权只由对齐空间余弦给出；
9. 返回模式视图的 `task_feedback` 必须为 `false`；
10. 所有 SVD/特征分解仅作用于小型 Koopman 空间，不作用于大图矩阵。

任何协议轮次不匹配、张量键或形状不一致、聚合权重非法、非有限字典参数或非有限模式都会显式抛错，避免生成表面完整但语义错误的结果。

## 20. 2026-09-15 两轮联调验证

以下是工程 smoke test，不是正式精度结论。设置为 Cora、10 客户端、10 workers、2 轮、每轮 1 epoch，并关闭绘图以专门验证通信生命周期。

验证结果：

- O4 checkpoint：`optimizer_state_mode=fedadamw_v_block`；
- `global_v_step=2`，任务网络共有 7 个 block-v 标量；
- 全局字典来源轮次为 2，含 6 个张量、9,360 个参数；
- 全局字典键全部为 `network.*`，不含 $K$ 或 operator；
- 第 2 轮 10/10 客户端均记录 `dictionary_received=1`；
- 第 2 轮 10/10 客户端均记录 `optimizer_reset=1`；
- 覆盖前相对差范围约为 0.0325 到 0.0352，证明客户端字典在本地训练后确实分化；
- 覆盖后相对差全部为 0，证明全局字典确实安装成功；
- Functional Map 模式图仍正常生成，共 160 个节点、567 条边、10 个社区；
- 模式图继续记录 `task_feedback=false`。

该 smoke 的聚类质量为：Silhouette 约 0.077、簇间/簇内距离比约 1.159、modularity 约 0.649。它说明稀疏图社区较明显，但完整对齐几何中的分离度仍然偏弱，正好证明不能只凭“发现了 10 个社区”判断模式聚类已经可靠。

两轮 paired test ACC 均值约为 0.652，仅用于证明训练流程跑通；轮数太少，不能与 100 轮正式实验比较，也不能据此判断 dictionary FedAvg 对任务精度的作用。

## 21. 必做消融与判定标准

下一阶段至少应比较：

| 变体 | 字典 | Functional Map | 目的 |
|---|---|---|---|
| A | 客户端私有 | 开启 | 原始坐标自由度下的基线 |
| B | FedAvg + 接收后 reset Adam | 开启 | 当前 V1.0 |
| C | FedAvg + 接收后 reset Adam | 关闭或恒等映射 | 判断共享字典是否已足以对齐 |
| D | FedAvg + 保留旧 Adam | 开启 | 验证参数点—优化器状态错配的影响 |

建议同时报告：

- 每客户端最佳验证轮对应的测试 ACC，而不是最后一轮 ACC；
- Functional Map 相对恒等映射带来的残差改进；
- $R_C$ 和 $R_U$ 的均值、最大值与轮次曲线；
- 字典覆盖前相对距离；
- Silhouette、距离比、similarity gap、modularity 和 conductance；
- 社区数、社区大小分布与客户端覆盖度；
- 通信量和每轮 wall time。

若 dictionary FedAvg 有效，合理预期是：覆盖前距离稳定非零、覆盖后为 0；在不破坏本地 $K_m$ 拟合的情况下，Functional Map 更接近恒等映射或交换残差下降；完整空间的 Silhouette、距离比和 similarity gap 改善。不能只以社区数量增加作为成功标准。

## 22. 当前限制与下一步

### 22.1 当前限制

1. 模式分支被动，尚不能直接提高任务 ACC；
2. 神经字典参数即使做 FedAvg，仍可能存在隐藏单元排列等参数对称性，参数平均不等于严格的函数平均；
3. 参考 $K_s$ 固定在首轮最优客户端，长期训练后可能逐渐过时；
4. $K_m$ 来自任务隐藏轨迹，它描述的是当前模型传播过程，不自动等价于可迁移、因果有效的任务知识；
5. 两轮 smoke 中 Silhouette 仍低，100 轮行为尚需正式验证；
6. 模式库当前在观测参考空间中构建，尚未设计安全、可解释的状态空间回注机制。

### 22.2 推荐的后续顺序

1. 先完成 100 轮、11 数据集的 V1.0 稳定性和消融，确认共享字典是否真实改善 K 空间可比性；
2. 检查 Functional Map 是否随训练逐渐接近恒等映射，以及固定参考是否老化；
3. 只在模式质量指标通过门槛后，设计从全局模式库到本地状态空间的受控映射；
4. 最初反馈应使用可关闭、可限幅、可测量的弱正则或门控残差项；
5. 最终再考虑启用外力端口，并与无反馈 V1.0 做严格因果消融。

不建议在当前阶段直接平均 $K_m$，也不建议仅凭局部轨迹拟合残差低就宣称获得了可迁移动力学。真正的判据应是：跨客户端方向保持、对齐残差下降、社区几何分离改善，以及后续受控反馈对最佳验证轮 paired test 指标产生可复现增益。

## 23. 客户端与服务端伪代码

### 23.1 客户端

```text
ClientRound(m, r):
    restore private checkpoint of logical client m

    theta_m <- receive global task parameters
    task_optimizer <- O4(global block-v, fresh first moment)

    if r > 1:
        validate dictionary protocol and source_round == r - 1
        omega_m <- receive global dictionary
        koopman_optimizer <- fresh Adam

    train A-DGN for local epochs with external force disabled

    T_m <- detached 16-step A-DGN trajectory
    update omega_m using spectral residual loss
    K_m <- ridge_EDMD(Psi_omega(T_m))
    modes_m <- eigendecompose_and_select(K_m)

    upload theta_m, block_v_m, omega_m, K_m, modes_m
    save private checkpoint
```

### 23.2 服务端

```text
ServerRound(r):
    theta_g <- weighted_average({theta_m})
    block_v_g <- weighted_average({block_v_m})
    omega_g <- weighted_average({omega_m})

    do not average {K_m}

    if first graph round:
        choose fixed reference K_s by minimum spectral residual

    for each client m:
        C_m <- solve min ||C_m K_m - K_s C_m||, C_m^H C_m = I
        aligned_modes_m <- C_m modes_m

    G_mode <- cross-client balanced kNN(aligned modes)
    communities <- Louvain(G_mode)
    quality <- full-geometry metrics + sparse-graph metrics
    global_bank <- phase_aligned_community_representatives(communities)

    broadcast theta_g, block_v_g, omega_g
    return passive personalized mode views
```

## 24. 小结

FedDGN V1.0 的核心不是把所有动力学量都平均，而是建立清晰的职责边界：

$$
\boxed{
\begin{aligned}
&\text{A-DGN 参数：FedAvg，任务优化器用 O4 block-v；}\\
&\text{ResKoop 字典：独立 FedAvg，接收后重置 Adam；}\\
&\text{本地 }K_m\text{：解析计算、上传比较，但不平均；}\\
&\text{跨客户端模式：Functional Map 对齐后再建图；}\\
&\text{模式反馈：当前仅保存和可视化，不进入任务动力学。}
\end{aligned}
}
$$

共享字典减少坐标漂移，Functional Map 处理剩余坐标偏差，本地 $K_m$ 保留客户端特有动力学，模式图负责发现跨客户端结构。这一版本首先把“可比较、可审计、可消融”的分析基础搭好，再为后续真正的个性化动力学知识流提供可靠入口。
