---
title: KoopmanGraph v0.15.0
description: >-
  从源码拆解 KoopmanGraph 的 GNN encoder—线性图 Koopman operator—GNN decoder 架构，并界定其连续 graph generator 与 Lie/JVP 机制向 Method V0.4 的迁移方式。
tags:
  - Research software
  - Koopman autoencoder
  - Graph neural network
  - Networked dynamics
  - Backbone migration
---

# KoopmanGraph v0.15.0：图感知 Koopman 架构及其 V0.4 迁移边界

**作者**: Travis Kessler  
**本页审计版本**: v0.15.0，发布元数据显示为 2026-08-19  
**项目定位**: 面向网络动力学预测的 PyTorch Geometric 研究软件  
**项目主页**: [GitHub](https://github.com/tjkessler/KoopmanGraph) · [文档](https://koopmangraph.readthedocs.io/) · [Zenodo DOI](https://doi.org/10.5281/zenodo.21926723)  
**本地源码**: `KoopmanGraph-0.15.0/`

!!! info "证据边界"
    本页结论来自本地 v0.15.0 的 `paper.md`、文档和实际源码，而不是只看 README。
    这是软件架构审计，不是 Cora 实验复现。该项目解决的是图快照预测，没有给出
    我们当前 10-client Cora 节点分类协议下的结果，因此“可迁移”不等于“已经证明
    可以达到 80%”。

!!! warning "V0.4 最终取舍已经更新"
    本页最初提出的“两层 GCN encoder + graph Koopman factor”是性能诊断候选，现在只
    保留为 B1–B3 基线/消融。最新版 [Method V0.4](../methodv04.md) 的主模型使用
    node-wise encoder，将图拓扑集中到任务锚定 reference field 和稳定 Graph-Koopman
    dynamics；真正迁移自 KoopmanGraph 的是连续 generator 语义、self/neighbor 图
    算子与 Lie consistency 的 JVP 机制，而不是默认 GCN autoencoder 全套结构。

## 一句话总结

KoopmanGraph 把每一步预测明确拆成：**用 GNN 将当前图状态提升到逐节点潜空间，
用共享的小矩阵因子在线性潜空间推进，再用匹配的 GNN 解码器恢复下一时刻图状态**。
它最值得当前 V0.4 借鉴的不是庞大的功能集合，而是：

1. continuous graph Koopman step 使用节点数无关的 self/neighbor 通道矩阵；
2. 连续 generator 与离散 propagator 在语义和接口上明确区分；
3. Lie consistency 用 JVP 计算 $J_\Psi F$，无需显式 Jacobian 或未来快照；
4. 编码、线性推进与输出模块边界清楚，适合拆出最小可验证实现。

## 1. 先讲清楚：它的 Backbone 到底是什么

### 1.1 总体数据流

对时刻 $t$ 的图快照

$$
G_t=(X_t,E_t),\qquad X_t\in\mathbb R^{N\times F},
$$

一次标准 forward 是：

<div class="research-flow">
<div class="flow-step"><span>01</span><strong>图快照</strong><small>$X_t,E_t$</small></div>
<div class="flow-arrow" aria-hidden="true">→</div>
<div class="flow-step"><span>02</span><strong>GNN Encoder</strong><small>$Z_t=E_\theta(X_t,E_t)$</small></div>
<div class="flow-arrow" aria-hidden="true">→</div>
<div class="flow-step"><span>03</span><strong>线性潜在推进</strong><small>$Z_{t+1}=\mathcal K_G Z_t$</small></div>
<div class="flow-arrow" aria-hidden="true">→</div>
<div class="flow-step"><span>04</span><strong>GNN Decoder</strong><small>$\widehat X_{t+1}=D_\psi(Z_{t+1},E_{t+1})$</small></div>
</div>

对应源码入口是 `GraphKoopmanModel.forward`：先调用 `encode`，再调用
`_advance_latent`，最后把推进后的 latent 交给 decoder。它不是把整张图展平成
$NF$ 维向量，而是始终保留

$$
Z_t\in\mathbb R^{N\times d},
$$

让小型 $d\times d$ 通道矩阵与稀疏图传播组合。

### 1.2 默认 GNN Encoder：两层 GCN lifting

README quickstart 使用：

```python
encoder = GNNEncoder(in_channels, 64, 64)
```

`GNNEncoder` 默认 `num_layers=2`、激活为 ReLU。实际结构为：

$$
H_1=\operatorname{ReLU}
\left(
S_{\mathrm{gcn}}X_tW_0+b_0
\right),
$$

$$
Z_t=S_{\mathrm{gcn}}H_1W_1+b_1,
$$

其中

$$
S_{\mathrm{gcn}}
=\widetilde D^{-1/2}(A+I)\widetilde D^{-1/2}.
$$

源码通过 PyG `GCNConv` 实现，默认自动加入自环并带 bias，最后一层**不加激活**；
所有中间层后加指定激活，末层直接产生 $N\times d$ latent。库还提供 GAT、
GraphSAGE、双向随机游走 DiffConv 和 Graph Transformer encoder，但这些都是替代项，
不是 quickstart 默认。

这意味着默认 encoder 已经完成两轮局部图消息传递。图结构不是在动力学末端才
间接影响表示，而是在 $X_t\rightarrow Z_t$ 的第一阶段就参与特征去噪和同类聚合。

### 1.3 必须区分两种 Koopman operator

这是阅读该项目时最容易混淆的地方。

#### 默认 `koopman=None`：逐节点共享、但推进本身不看图

默认会构造 `KoopmanOperator`：

$$
Z_{t+1}=Z_tK^\top,
\qquad K\in\mathbb R^{d\times d}.
$$

同一个 $K$ 独立作用到每个节点。图结构只存在于 GNN encoder 和 decoder，latent
advance 本身不使用 `edge_index`。因此不能把 README 的默认 quickstart 直接描述成
“图耦合 Koopman 动力学”。

#### `koopman="graph"`：self/neighbor 分解的线性图推进

真正与当前 V0.3 高度相关的是 `GraphKoopmanOperator`。当 `filter_degree=1` 且使用
对称归一化邻接时，它执行：

$$
\boxed{
Z_{t+1}
=
Z_tK_{\mathrm{self}}^\top
+
(\widehat A Z_t)K_{\mathrm{nbr}}^\top
}
$$

等价的整体算子为：

$$
K_{\mathrm{eff}}
=
I_N\otimes K_{\mathrm{self}}
+
\widehat A\otimes K_{\mathrm{nbr}}.
$$

两个因子都只有 $d\times d$，不依赖客户端节点数 $N$。`filter_degree=P>1` 时继续
加入

$$
\sum_{k=2}^{P}
(\widehat A^kZ_t)K_k^\top.
$$

源码还支持 random-walk、dual-random-walk 和连续 generator。连续 graph 版本为

$$
L_{\mathrm{eff}}
=I_N\otimes L_{\mathrm{self}}
+\widehat A\otimes L_{\mathrm{nbr}},
\qquad
Z(t+\Delta t)=\exp(\Delta t L_{\mathrm{eff}})Z(t).
$$

但它的 dense 路径需要对 $Nd\times Nd$ 矩阵做指数，不能直接搬到 Cora 客户端；
`block_diagonal` 快捷路径又会忽略 neighbor coupling。对我们的稀疏图，保留因子形式，
用稀疏 matmul 加 RK4/离散步推进更合理。

### 1.4 一个很重要的稳定起点：identity self + near-zero neighbor

`GraphKoopmanOperator` 默认使用 `identity_noise` 初始化：

$$
K_{\mathrm{self}}\approx I+\varepsilon,
\qquad
K_{\mathrm{nbr}}\approx 0.
$$

因此初始模型接近“保留 encoder 表示”，训练再逐渐学习图耦合修正。这和随机初始化
整个动力学后让任务损失从头寻找可用流相比，优化条件明显更友好。它不是稳定性证明，
但提供了一个不会在训练开始就破坏表示的 residual-like 起点。

### 1.5 GNN Decoder：镜像图解码，而不是普通 MLP

quickstart 的 decoder 是：

```python
decoder = GNNDecoder(64, 64, out_channels)
```

同样默认两层 GCN：

$$
R_1=\operatorname{ReLU}(S_{\mathrm{gcn}}Z_{t+1}W_{d,0}+b_{d,0}),
$$

$$
\widehat X_{t+1}=S_{\mathrm{gcn}}R_1W_{d,1}+b_{d,1}.
$$

最后一层也不加激活。原项目把它用于下一时刻物理量重建；迁移到节点分类时，
`out_channels` 可以改成类别数 $C$，此时它是 graph-aware classifier，而不再应称为
物理状态 decoder。

### 1.6 训练目标：默认其实只有一步重建

`GraphKoopmanModel.fit` 端到端联合训练 encoder、operator 和 decoder。默认
`LossWeights` 为：

$$
w_{\mathrm{rec}}=1,
\qquad
w_{\mathrm{forward}}=w_{\mathrm{backward}}=w_{\mathrm{rollout}}=0.
$$

完整可选目标为：

$$
\mathcal L
=w_r\mathcal L_{\mathrm{rec}}
+w_f\mathcal L_{\mathrm{forward}}
+w_b\mathcal L_{\mathrm{backward}}
+w_{roll}\mathcal L_{\mathrm{rollout}}
+w_{eig}\mathcal L_{\mathrm{eig}}
+\cdots.
$$

因此库“支持 rollout/consistency”不等于默认训练已经启用它们。v0.15.0 的闭式
identification 集成也主要面向 discrete dense per-node $K$；graph/continuous operator
不能直接套用其 `fit(..., identification=...)` 路径。

### 1.7 Lie consistency：用 JVP 对齐连续向量场

`src/koopman_graph/losses/consistency.py::LieConsistencyLoss` 提供了不依赖下一时刻
快照的连续闭合接口。给定 physical state $x$、已知向量场 $f(x)$、observable
$\phi(x)$ 和 continuous generator $L$，它计算

$$
\phi(x),\ J_\phi(x)f(x)
=\operatorname{jvp}(\phi,x,f(x)),
$$

再比较

$$
J_\phi(x)f(x)
\quad\text{与}\quad
L(\phi(x))\phi(x).
$$

实际源码通过 `torch.autograd.functional.jvp` 完成这一计算，不物化完整 Jacobian；
原实现使用 MSE，并假设 `dynamics_fn` 给出的场已经有物理或数据依据。V0.4 迁移的是
这条 **Lie derivative + JVP** 数值路径，但不能直接照搬监督语义：我们的场是学习
得到的，所以必须先被任务方向独立锚定，在 Lie phase 对场 stop-gradient，并先用
direction cosine 而不是 relative MSE。

## 2. 把 KoopmanGraph 的网络真正算一遍

下面用一个三节点玩具图把张量和矩阵全部展开。它不是训练结果，而是用于回答三个
容易混淆的问题：GNN 在哪里混合邻居，Koopman 矩阵乘在哪一维，以及 graph operator
为何不需要存储一个庞大的 $Nd\times Nd$ 参数矩阵。

### 2.1 同一张图有两种归一化矩阵

取无向路径 $v_1-v_2-v_3$，在 `edge_index` 中把每条无向边写成两个方向：

$$
A=
\begin{bmatrix}
0&1&0\\
1&0&1\\
0&1&0
\end{bmatrix}.
$$

默认 `GNNEncoder`/`GNNDecoder` 内的 PyG `GCNConv` 自动加入自环，因此 GCN 使用

$$
S_{\mathrm{gcn}}
=\widetilde D^{-1/2}(A+I)\widetilde D^{-1/2}
=
\begin{bmatrix}
0.5000&0.4082&0\\
0.4082&0.3333&0.4082\\
0&0.4082&0.5000
\end{bmatrix}.
$$

`GraphKoopmanOperator(adjacency="symmetric")` 调用的是项目自己的稀疏归一化函数，
不会额外加入自环；self effect 已由 $K_{\mathrm{self}}$ 单独承担。因此它使用

$$
\widehat A=D^{-1/2}AD^{-1/2}
=
\begin{bmatrix}
0&0.7071&0\\
0.7071&0&0.7071\\
0&0.7071&0
\end{bmatrix}.
$$

这两个矩阵不能混写：$S_{\mathrm{gcn}}$ 服务于 GCN 层，$\widehat A$ 服务于 graph
Koopman 的 neighbor channel。只有原始 `edge_index` 本身显式含自环时，后者才会把
这些自环计入图聚合。

### 2.2 算例 A：两层 GCN Encoder 如何得到逐节点 latent

为了能手算，把 quickstart 的 64 维隐藏层缩成 2 维。令

$$
X=
\begin{bmatrix}
1&0\\
0&2\\
1&-1
\end{bmatrix},\qquad
W_0=W_1=I_2,\qquad b_0=b_1=0.
$$

第一层先把本节点和一跳邻居加权，再做 ReLU：

$$
H_1=\operatorname{ReLU}(S_{\mathrm{gcn}}X)
=
\begin{bmatrix}
0.5000&0.8165\\
0.8165&0.2584\\
0.5000&0.3165
\end{bmatrix}.
$$

第二层再次聚合，但末层没有激活：

$$
Z=S_{\mathrm{gcn}}H_1
=
\begin{bmatrix}
0.5833&0.5137\\
0.6804&0.5487\\
0.5833&0.2637
\end{bmatrix}.
$$

例如 $v_1$ 的第一层表示包含 $v_1$ 自身和 $v_2$，第二层再通过 $v_2$ 间接接收
$v_3$，所以两层默认 encoder 的理论感受野达到两跳。真实 quickstart 只是把通道数
放大：

$$
(N,3)\xrightarrow{\mathrm{GCNConv}(3,64)}(N,64)
\xrightarrow{\mathrm{GCNConv}(64,64)}(N,64).
$$

节点轴 $N$ 一直保留，GCN 权重只在通道轴上学习。

### 2.3 算例 B：默认逐节点 Koopman 为什么不混合节点

先暂时使用一组便于观察的 latent，而不是上一节 encoder 的输出：

$$
Z=
\begin{bmatrix}
1&0\\
0&1\\
2&-1
\end{bmatrix},\qquad
K=
\begin{bmatrix}
1&0.2\\
0&0.5
\end{bmatrix}.
$$

默认 `koopman=None` 构造 per-node operator，代码执行 `Z @ K.T`：

$$
Z^+_{\mathrm{pernode}}=ZK^\top
=
\begin{bmatrix}
1.0000&0\\
0.2000&0.5000\\
1.8000&-0.5000
\end{bmatrix}.
$$

以列向量写，第 $i$ 个节点是 $z_i^+=Kz_i$。所有节点共享同一个 $K$，但任一行的
结果都不含其他节点的 $z_j$；所以参数共享不等于图耦合。

### 2.4 算例 C：degree-1 Graph Koopman 如何分开算 self 与 neighbor

现在显式选择 `koopman="graph"`，继续使用上面的 $Z$ 和 $K_{\mathrm{self}}=K$，并取

$$
K_{\mathrm{nbr}}=
\begin{bmatrix}
0&0.5\\
-0.25&0
\end{bmatrix}.
$$

第一步只做稀疏邻居聚合：

$$
\widehat AZ=
\begin{bmatrix}
0&0.7071\\
2.1213&-0.7071\\
0&0.7071
\end{bmatrix}.
$$

第二步分别用两个通道矩阵变换 self 与 neighbor：

$$
ZK_{\mathrm{self}}^\top=
\begin{bmatrix}
1.0000&0\\
0.2000&0.5000\\
1.8000&-0.5000
\end{bmatrix},
$$

$$
(\widehat AZ)K_{\mathrm{nbr}}^\top=
\begin{bmatrix}
0.3536&0\\
-0.3536&-0.5303\\
0.3536&0
\end{bmatrix}.
$$

相加得到

$$
\boxed{
Z^+_{\mathrm{graph}}=
\begin{bmatrix}
1.3536&0\\
-0.1536&-0.0303\\
2.1536&-0.5000
\end{bmatrix}
}.
$$

以中间节点为例，等价的列向量公式是

$$
z_2^+
=K_{\mathrm{self}}z_2
+\frac{1}{\sqrt2}K_{\mathrm{nbr}}z_1
+\frac{1}{\sqrt2}K_{\mathrm{nbr}}z_3.
$$

于是可以很明确地读出：$K_{\mathrm{self}}$ 学节点自身 latent 如何变化，
$K_{\mathrm{nbr}}$ 学“归一化邻居摘要”如何影响该节点。图决定**谁给谁传信息以及权重**，
两个 $d\times d$ 矩阵决定**信息到达后如何在通道间变换**。

### 2.5 算例 D：看似是 $6\times6$ 算子，实际只学两个 $2\times2$ 因子

按节点逐行展平

$$
\operatorname{vec}_{\mathrm{node}}(Z)
=[1,0,\;0,1,\;2,-1]^\top,
$$

上面的 graph step 等价于乘

$$
K_{\mathrm{eff}}
=I_3\otimes K_{\mathrm{self}}
+\widehat A\otimes K_{\mathrm{nbr}}
=
\begin{bmatrix}
1&0.2&0&0.3536&0&0\\
0&0.5&-0.1768&0&0&0\\
0&0.3536&1&0.2&0&0.3536\\
-0.1768&0&0&0.5&-0.1768&0\\
0&0&0&0.3536&1&0.2\\
0&0&-0.1768&0&0&0.5
\end{bmatrix}.
$$

这里 $K_{\mathrm{eff}}$ 有 $6\times6=36$ 个位置，但 dense 参数化真正学习的只有
$K_{\mathrm{self}}$ 和 $K_{\mathrm{nbr}}$ 的 $2\times4=8$ 个标量。源码 forward 也不
显式构造 $K_{\mathrm{eff}}$，而是按

```text
self = Z @ K_self.T
neighbor_state = sparse_normalized_adjacency_matvec(edge_index, Z)
neighbor = neighbor_state @ K_nbr.T
Z_next = self + neighbor
```

执行。节点数增大主要增加稀疏聚合的计算量，不增加这两个通道矩阵的参数量。

### 2.6 算例 E：`filter_degree=2` 到底多算了什么

在同一个三节点路径上，二跳矩阵是

$$
\widehat A^2=
\begin{bmatrix}
0.5&0&0.5\\
0&1&0\\
0.5&0&0.5
\end{bmatrix}.
$$

若额外因子 $K_2=0.1I_2$，degree-2 operator 计算

$$
Z^+=ZK_{\mathrm{self}}^\top
+(\widehat AZ)K_{\mathrm{nbr}}^\top
+(\widehat A^2Z)K_2^\top.
$$

新增的二跳项为

$$
(\widehat A^2Z)K_2^\top=
\begin{bmatrix}
0.15&-0.05\\
0&0.10\\
0.15&-0.05
\end{bmatrix},
$$

所以结果变为

$$
Z^+_{P=2}=
\begin{bmatrix}
1.5036&-0.0500\\
-0.1536&0.0697\\
2.3036&-0.5500
\end{bmatrix}.
$$

$v_1$ 因二跳路径接收到 $v_3$，但同时也经 $v_1\to v_2\to v_1$ 收回自身信息。
一般 degree $P$ 使用 $P+1$ 个 $d\times d$ 因子，参数量是 $(P+1)d^2$；它不是把
$A^P$ 当作新的可训练邻接矩阵。

### 2.7 算例 F：完整 encoder → graph operator → decoder

现在把算例 A 的 encoder 输出接入算例 C 的 graph operator，再使用两层 identity-weight
GCN decoder。完整数据流为

$$
X\xrightarrow{\mathrm{GCN+ReLU+GCN}}Z
\xrightarrow{K_{\mathrm{self}},K_{\mathrm{nbr}}}Z^+
\xrightarrow{\mathrm{GCN+ReLU+GCN}}\widehat X^+.
$$

在相同玩具权重下，operator 和 decoder 的逐步数值是

$$
Z^+=
\begin{bmatrix}
0.8801&0.1366\\
1.0650&0.0681\\
0.8301&0.0116
\end{bmatrix},
$$

$$
R_1=\operatorname{ReLU}(S_{\mathrm{gcn}}Z^+)=
\begin{bmatrix}
0.8748&0.0961\\
1.0532&0.0832\\
0.8498&0.0336
\end{bmatrix},
$$

$$
\widehat X^+=S_{\mathrm{gcn}}R_1=
\begin{bmatrix}
0.8674&0.0820\\
1.0552&0.0807\\
0.8549&0.0508
\end{bmatrix}.
$$

所有中间量都是 $(3,2)$；一般形式则是

$$
(N,F_{\mathrm{in}})\to(N,d)\to(N,d)\to(N,F_{\mathrm{out}}).
$$

一次默认两层 encoder + degree-1 graph step + 两层 decoder，理论上可沿图传播至多
$2+1+2=5$ 跳；默认 per-node operator 没有中间的一跳，至多是 $2+2=4$ 跳。
这是结构上的最大感受野，不表示所有远端节点具有相同影响，也不保证更深一定更好。

### 2.8 三种具体网络配置应怎样读

| 配置 | 张量路径 | 图在哪里进入 | operator 参数量 |
| --- | --- | --- | ---: |
| README 默认预测器 | $(N,3)\to(N,64)\to(N,64)\to(N,3)$ | 两层 encoder + 两层 decoder；per-node $K$ 不看边 | $64^2=4{,}096$ |
| degree-1 graph 预测器 | $(N,3)\to(N,64)\to(N,64)\to(N,3)$ | encoder、graph $K$、decoder | $2\times64^2=8{,}192$ |
| degree-2 graph 预测器 | $(N,3)\to(N,64)\to(N,64)\to(N,3)$ | 再增加 $\widehat A^2Z$ 通道 | $3\times64^2=12{,}288$ |

对应源码中的构造参数如下。函数每次新建 encoder/decoder，避免三个模型意外共享权重：

```python
def make_model(koopman=None, degree=1):
    return GraphKoopmanModel(
        encoder=GNNEncoder(3, 64, 64),
        decoder=GNNDecoder(64, 64, 3),
        latent_dim=64,
        time_step=0.1,
        koopman=koopman,
        koopman_filter_degree=degree,
    )

per_node_model = make_model()             # 默认逐节点 K
graph_degree1_model = make_model("graph", 1)
graph_degree2_model = make_model("graph", 2)
```

若迁移成 Cora 节点分类 backbone，可写成

$$
(N,1433)\xrightarrow{\mathrm{2\mbox{-}layer\ GCN}}(N,64)
\xrightarrow{\mathrm{graph\ }K}(N,64)
\xrightarrow{\mathrm{Linear}}(N,7).
$$

在带 bias 的 dense 实现中，两层 GCN encoder 有

$$
(1433\times64+64)+(64\times64+64)=95{,}936
$$

个参数；degree-1 graph operator 有 $8{,}192$ 个矩阵参数，Linear 分类头有
$64\times7+7=455$ 个参数，总计 $104{,}583$。如果使用原项目的两层 GCN decoder
作为分类头，decoder 为 $64\to64\to7$，则增加 $4{,}615$ 个参数，整个网络为
$108{,}743$ 个参数。

!!! note "本地数值 smoke 的边界"
    本页用已安装的 PyTorch 1.13.1 与 PyG 2.6.1 执行了上述 GCNConv 和源码同式
    graph matvec。GCNConv 与显式 $S_{\mathrm{gcn}}$ 计算的最大绝对误差为
    $1.19\times10^{-7}$；factorized graph step 与显式公式、与 $K_{\mathrm{eff}}$
    乘法的误差均为 $0$。本机环境是 Python 3.9，而 v0.15.0 声明 Python
    $\ge 3.10$、Torch $\ge 2.8$，所以直接导入完整包会在 Python 3.10 类型语法处
    失败。本次没有改包、安装新环境或进行训练，因此这些数字只验证 forward 算法，
    不构成预测性能结果。

## 3. 为什么它对当前 V0.3 Backbone 有吸引力

当前 V0.3 的实际路径是：

$$
X
\xrightarrow{\mathrm{MLP}}
H_0
\xrightarrow{-\nabla_H\mathcal E(H_0,G)}
V_0
\xrightarrow{\text{second-order Koopman Graph ODE}}
H(T)
\xrightarrow{\mathrm{Linear}}
\widehat Y.
$$

其连续场为：

$$
\dot H=V,
$$

$$
\dot V
=
\sum_{k=0}^{2}
T_k(P)
\left(HB_k+VC_k\right).
$$

两者的关键差异如下。

| 维度 | 当前 V0.3 | KoopmanGraph 可迁移骨架 | 可能影响 |
| --- | --- | --- | --- |
| 初始编码 | 两层 node-wise MLP，不看图 | 两层 GCN，直接看图 | Cora 的同质邻居信息更早进入表示 |
| 图信息入口 | Energy-GCN 梯度与后续 $\dot V$ | encoder、graph operator、可选 decoder | 当前任务梯度的有效图路径更长、更间接 |
| 动力学阶数 | $[H,V]$ 二阶连续系统 | $Z$ 的一阶离散/连续线性推进 | 一阶路径更短，先验证更容易 |
| 初始化 | $B_k,C_k\sim\mathcal N(0,0.01)$ | self 近 identity、neighbor 近 zero | 初始 latent 不易被随机动力学破坏 |
| 输出头 | 单层 Linear | 可用图感知 decoder/classifier | 分类前可再做一次受控邻居聚合 |
| 主要任务 | 静态节点分类 | 时序图状态预测 | 训练目标不能原样照搬 |

当前最好 local Cora 只有 $74.12\%$，且客户端间差异很大。源码结构表明一个合理
怀疑是：模型必须先从 node-wise MLP 状态出发，再通过能量梯度产生 $V_0$，最后让
任务梯度穿过 16 个 RK4 积分步和二阶耦合才得到图传播收益。KoopmanGraph 骨架则
先得到可用 GCN 表示，再学习接近恒等的线性图修正。这是值得优先验证的优化假设，
不是已经由该软件的 forecasting benchmark 证明的结论。

## 4. 早期 GCN Backbone 候选：保留为对照，不作为正式 V0.4

### 4.1 V0.3-KG 诊断性分类 Backbone

小数据集仍使用 $d=64$。建议第一版写成：

$$
H_0
=
\operatorname{GCNEnc}_{2\text{-layer}}(X,P)
\in\mathbb R^{N\times64},
$$

$$
Z_0=\operatorname{LayerNorm}(H_0),
$$

$$
\boxed{
Z_{s+1}
=
Z_sK_0^\top
+
(PZ_s)K_1^\top
}
$$

以及

$$
\widehat Y
=
\operatorname{Head}(Z_S,P).
$$

其中：

- $K_0=I+\epsilon A_0$，保证初始 self path 近恒等；
- $K_1=\epsilon A_1$，让图耦合从小修正开始；
- 第一轮只用 `filter_degree=1`、$S=1$ 或 $2$，避免一次引入高阶传播和长滚动；
- `Head` 先使用 Linear，确认 operator 本身有效后，再测试一层或两层 GCN classifier；
- encoder、operator、head 都在现有 `models/v03/model.py` 内本地实现，不从
  `KoopmanGraph-0.15.0` 运行时 import，保持 V0.3 自包含和现有联邦架构不变。

这不是“在 MLP encoder 前再加一个 graph stem”，而是明确把 observable encoder
从 $E(X)$ 改成 $E(X,P)$。它会修改当前 V0.3 的设计边界，因此应作为显式的
**V0.3 backbone revision** 记录，而不能伪装成无语义影响的实现优化。

### 4.2 先不要把 Energy 二阶系统一起搬过去

为了回答性能问题，第一阶段应先隔离变量：

```text
GCN encoder → graph Koopman factor → classifier
```

先不使用 Energy-GCN、$V_0$、$[H,V]$ 和 ODE solver。原因不是否定 energy
observable，而是当前还不知道性能损失来自 encoder、能量梯度、二阶动力学、
closure loss 还是它们的组合。一次全部混合无法定位因果。

若简化 backbone 通过约 80% local gate，再恢复能量机制：

$$
V_0=-\nabla_H\mathcal E(H_0,P),
\qquad
\widetilde Z_0=[H_0,\alpha V_0],
$$

并用 block-structured self/neighbor 因子推进 $\widetilde Z$。此时需要比较：

- 只有 $H_0$；
- $[H_0,V_0]$ 但没有 closure；
- $[H_0,V_0]$ 加 closure；
- 当前二阶连续 ODE。

只有 energy 分支在 matched setting 下带来验证/测试收益，才保留为最终方法贡献。

### 4.3 离散版本优先，连续版本作为后续等价化

KoopmanGraph 的 graph operator 天然给出一个稳定的离散对照。若后续必须回到连续
生成元，可将其参数化为：

$$
K_0=I+\Delta t A_0,
\qquad
K_1=\Delta t A_1,
$$

对应一阶 Euler 近似：

$$
\dot Z=ZA_0^\top+(PZ)A_1^\top.
$$

我们应继续使用稀疏 $PZ$ 和现有 ODE 求解器，不复制 v0.15.0 的 dense
$Nd\times Nd$ matrix exponential。这样既保留 operator 的节点数无关因子，又不会
在客户端图规模上产生不可接受的矩阵指数。

### 4.4 与现有联邦流程的接口

迁移只改 `models/v03/model.py` 内的 backbone，不改变：

- `main.py → server → worker → client` 的正常生命周期；
- 客户端完整模型上传；
- server 的 equal FedAvg；
- local debug 的定义。

本地 gate 阶段仍只注释 `client.py::on_receive_message` 中 global-to-local state
replacement。通过 gate 后恢复该行，再跑正常 100-round FedAvg。不要改用
KoopmanGraph 自带的 federated helper，因为当前工程必须继续与 `Fedrated` 架构对齐。

## 5. 最小消融矩阵

所有实验固定 Cora disjoint、10 clients、seed 42、64 维、100 rounds、1 local epoch，
只改变 backbone：

| 编号 | Encoder | Latent step | Head | 回答的问题 |
| --- | --- | --- | --- | --- |
| B0 | 当前 MLP | 当前 Energy + K=2 ODE | Linear | 现有基线，74.12% best valid local |
| B1 | 2-layer GCN | 无 | Linear | 单纯强 GCN backbone 能到哪里 |
| B2 | 2-layer GCN | per-node $K$ | Linear | 线性通道推进是否损伤/改善表示 |
| B3 | 2-layer GCN | graph $K_0,K_1$，degree 1 | Linear | self/neighbor Koopman factor 的净贡献 |
| B4 | 2-layer GCN | graph degree 2 | Linear | 高阶图基是否必要 |
| B5 | 2-layer GCN | 最佳 graph operator | GCN classifier | graph-aware 输出头是否继续增益 |
| B6 | 最佳方案 | 加回 $V_0$/closure | matched head | Energy observable 是否真正有价值 |

至少同时记录 mean/std、Macro-F1、逐客户端 ACC、最佳验证轮次、参数量、上传字节、
训练耗时，以及 hard clients 3/4/7 的变化。若 B1 已接近 80% 而 B3 明显下降，问题在
operator；若 B1 仍低，则应先检查数据划分、训练预算和 GCN baseline，而不是继续
增加 Koopman 复杂度。

## 6. 哪些内容不应直接迁移

- **不要直接复制整个库。** v0.15.0 包含大量 forecasting、control、heterogeneous、
  UQ 和拓扑扩展，会破坏当前 V0.3 的可读性与依赖边界。
- **不要把默认 per-node $K$ 写成 graph dynamics。** 必须显式使用 self/neighbor
  因子才有图耦合 latent advance。
- **不要照搬 reconstruction-only loss。** Cora 是带 mask 的节点分类，应以 CE 为
  主任务；rollout/closure 只能作为有独立诊断支持的辅助目标。
- **不要声称静态层索引天然等于真实时间。** 若把网络深度当 propagation time，
  必须按内部计算轨迹表述。
- **不要在 Cora 上构造 dense $Nd\times Nd$ exponential。** 保留 Kronecker/稀疏
  factorized matvec。
- **不要因模型拥有小型 $d\times d$ 因子就直接跨客户端平均其语义。** encoder 坐标
  仍可能有 similarity ambiguity；正常 FedAvg 可以先作为工程基线，operator-only
  federation 仍需坐标对齐证据。

## 7. 源码审计锚点

| 结论 | v0.15.0 源码位置 |
| --- | --- |
| 一次 forward 是 encode → advance → decode | `src/koopman_graph/model/estimator.py` |
| GCN encoder/decoder 默认两层、只在中间层后激活 | `src/koopman_graph/nn/gnn.py`, `encoder.py`, `decoder.py` |
| 默认 `koopman=None` 解析为 `pernode` | `src/koopman_graph/model/factory.py::parse_koopman_arg` |
| graph operator 的 self/neighbor/多跳公式 | `src/koopman_graph/operators/graph.py::GraphKoopmanOperator` |
| continuous graph generator 与 dense exponential 边界 | `src/koopman_graph/operators/continuous_graph.py` |
| Lie derivative 通过 JVP 计算、不构造完整 Jacobian | `src/koopman_graph/losses/consistency.py::LieConsistencyLoss` |
| reconstruction-only 默认及可选 consistency/rollout | `src/koopman_graph/training/history.py`, `training/objectives.py` |
| 闭式 identification 的适用范围 | `docs/source/identification.rst` |

## 结论

KoopmanGraph 给当前项目最可靠的启发是：**保留逐节点 latent，通过
self/neighbor 因子定义连续图生成元，并用 JVP 检查 observable 的 Lie 导数与生成元
RHS 是否一致。** 正式 V0.4 在此基础上加入 task-anchored neural field、identity-
preserving observable 和 Bernstein dissipative parameterization，以适配没有真实时间
快照的静态节点分类。B1–B3 的 GCN 路线继续作为强任务基线与结构消融，不再覆盖
V0.4 的主设计。
