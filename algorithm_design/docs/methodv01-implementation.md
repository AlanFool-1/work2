---
title: Method V0.1：当前代码实现
description: >-
  以 codev0 当前源码为准，说明节点编码、线性图传播、损失、FedAvg 生命周期、纯本地对照与日志产物。
tags:
  - Method V0.1
  - 代码实现
  - 线性图动力学
---

# Method V0.1：当前代码实现

版本：2026-09-08 UTC · 本页描述**已经运行过的代码**，不再是实现占位稿。

!!! warning "当前定位"
    `v01_linear` 是一个可运行的单阶段、节点级、低维线性图传播 backbone。它通过了单测、单图训练、10 客户端纯本地训练和 Cora 100 轮 FedAvg。当前传播在数学上属于带残差和共享参数的线性 GNN；它还没有加入跨客户端动力学对齐、函数空间聚合或个性化机制，因此不能仅凭“动力学”命名主张超越普通 GCN。

[算法设计](methodv01.md){ .md-button .md-button--primary }
[查看真实实验](experiments/methodv01.md){ .md-button }

## 1. 实际执行链

```text
本地节点特征 X、边 edge_index
  → 加自环并构造对称归一化图算子 P
  → 拼接 [X, PX, log(1 + degree)]
  → 两层 MLP 编码为 Z0 [N, r]
  → 共享 A0、A1 执行 L 次线性 Euler 更新
  → ZL [N, r]
  → 线性任务读出
  → train_mask 上的 CE / BCE
```

默认训练不需要 A-DGN 教师、教师 checkpoint、离线轨迹、VAE、KL、图级池化、静态节点条件、响应蒸馏或第二个优化器。可选输入解码器存在，但默认关闭。

## 2. 源码边界

| 文件 | 当前职责 |
| --- | --- |
| `codev0/models/v01/model.py` | 图归一化、节点编码、$A_0/A_1$ 传播、读出和可选解码器 |
| `codev0/models/v01/client.py` | 本地损失、Adam 更新、有限值检查、动力学与显存诊断 |
| `codev0/models/v01/server.py` | 构建 V0.1 全局模型并复用 S0 的 FedAvg 生命周期 |
| `codev0/models/v01/logger.py` | 参数量、通信量、潜态范数和生成元谱范数日志 |
| `codev0/main.py` | 按 `--model v01_linear` 注册 client/server，并解析数据集维度 |
| `codev0/configs/v01_linear_gnn.json` | V0.1 默认配置 |
| `codev0/scripts/run_v01_single_client.py` | 单个分区的 Gate 0 / 纯本地训练 |
| `codev0/scripts/run_v01_local_cora_all_clients.sh` | Cora 10 客户端完全独立的 N1 对照 |
| `codev0/tests/test_v01_model.py` | 形状、图算子、置换等变、传播线性、零步和梯度测试 |
| `codev0/tests/test_v01_logger.py` | 动力学、成本与累计通信日志测试 |

`s0_ode` 仍是 CLI 默认模型。V0.1 只有显式传入 `--model v01_linear` 才会启用；官方 A-DGN 类本身没有被改写。

## 3. 输入图算子与节点编码

### 3.1 当前归一化的准确含义

`normalize_graph()` 直接调用 PyG `gcn_norm`，配置为：

```python
improved=False
add_self_loops=True
flow="source_to_target"
```

对无权无向图，对应

$$
\widetilde A=A+I,
\qquad
P=\widetilde D^{-1/2}\widetilde A\widetilde D^{-1/2}.
$$

`normalized_aggregate()` 没有构造稠密邻接矩阵。对 COO 边 $u\to v$，它执行

$$
[PX]_v=\sum_{u:(u,v)\in\widetilde E}\widehat w_{uv}X_u.
$$

若数据带 `edge_weight`，归一化和聚合都会使用它。度特征则在加入自环**之前**，按原始边的加权入度计算：

$$
d_v=\sum_{u:(u,v)\in E}w_{uv},
\qquad q_v=\log(1+d_v).
$$

### 3.2 编码器

设 $X\in\mathbb R^{N\times d_x}$，当前 `NodeEncoder` 为：

$$
U=[X,PX,\log(1+d)]\in\mathbb R^{N\times(2d_x+1)},
$$

$$
Z^{(0)}=
\operatorname{Linear}_{64\to r}
\left(
\operatorname{SiLU}
\left(
\operatorname{Linear}_{2d_x+1\to64}(U)
\right)
\right).
$$

这里的下标表示默认 `encoder_width=64`；潜维默认 $r=32$。节点轴始终保留，没有 V0 的整图 attention pooling。

## 4. 线性图动力学的真实形式

自身生成元由一个无约束参数 $Q_0$ 构造：

$$
A_0=Q_0-Q_0^\top-\gamma I,
\qquad \gamma=0.1.
$$

邻域生成元 $A_1\in\mathbb R^{r\times r}$ 当前完全无约束。每一步执行：

$$
Z^{(\ell+1)}
=Z^{(\ell)}+\eta
\left(
Z^{(\ell)}A_0+PZ^{(\ell)}A_1
\right),
$$

其中默认 $L=16,\eta=0.1$，全部步骤共享同一组 $A_0,A_1$。

训练调用 `forward_with_aux()`，返回：

- `logits`: $[N,C]$；
- `reconstruction`: 默认 `None`；
- `latent_trajectory`: $[L+1,N,r]$。

验证和测试调用普通 `forward()`，只保留最终潜态，不额外堆叠整条轨迹。

### 4.1 它与 GCN 的边界

令

$$
B_0=I+\eta A_0,
\qquad B_1=\eta A_1,
$$

则一步传播就是

$$
Z^{(\ell+1)}=Z^{(\ell)}B_0+PZ^{(\ell)}B_1.
$$

对固定图重复展开后，可写成

$$
Z^{(L)}=\sum_{k=0}^{L}P^kZ^{(0)}C_{L,k},
$$

其中 $C_{L,k}$ 是由共享 $B_0,B_1$ 递推得到的通道矩阵。因此当前核心传播是一个带残差、共享权重和特殊自身矩阵参数化的多阶线性图滤波器。

!!! danger "不能夸大的结论"
    当前实现没有逐层非线性、真实时间序列辨识、Koopman 闭合损失或跨客户端算子对齐。$A_0$ 的反对称耗散形式是一种稳定性偏置，但 $A_1$ 无约束，所以它也不保证完整图算子稳定。更准确的名称是 **Koopman-inspired tied linear graph-dynamics backbone**；在加入新的联邦动力学机制前，它与线性残差 GCN/SGC 的差异主要是参数化和解释方式。

## 5. 任务头与损失

最终读出是

$$
\widehat Y=Z^{(L)}W_R+b_R.
$$

多分类数据集在 `train_mask` 上使用交叉熵；Minesweeper、Tolokers、Questions 使用单 logit BCE。默认目标只有：

$$
\mathcal L=\mathcal L_{task}.
$$

当 `reconstruction_weight > 0` 时才构造两层解码器，并从 $Z^{(0)}$ 重建原始输入：

$$
\widehat X=D_\psi(Z^{(0)}),
\qquad
\mathcal L=\mathcal L_{task}+\lambda_{rec}\operatorname{MSE}(\widehat X,X).
$$

代码在反向前检查总损失是否有限，反向后逐参数检查梯度是否包含 NaN/Inf；当前没有梯度裁剪。

Gate 0 已发现：Cora/CiteSeer 的稀疏特征使普通逐元素 MSE 接近“预测全零”的捷径，因此这个重建头虽然实现正确，但目前没有证据表明它能防止表示坍塌。正式 M1 继续使用 `reconstruction_weight=0`。

## 6. FedAvg 生命周期

V0.1 复用 S0 已有的多进程联邦框架。每轮实际顺序为：

```text
服务端发布全局 state_dict
  → 客户端加载自己的持久 Adam 一阶/二阶矩
  → 全局参数覆盖客户端模型参数
  → 在本地整图执行 n_eps 次更新
  → validation / test 前向
  → validation 提升时保存同轮 paired test
  → 上传完整模型 state_dict 和 train_size
  → 服务端等权或样本量加权 FedAvg
```

首版聚合对象是整个耦合参数块：

$$
\Theta=(E_\phi,Q_0,A_1,R_\omega[,D_\psi]).
$$

默认等权聚合：

$$
\Theta^{(r+1)}=\frac1M\sum_{m=1}^M\Theta_m^{(r,+)}.
$$

传入 `--aggregation weighted` 时才按 `train_mask` 节点数加权。节点特征、边、节点潜态和 Adam 状态都不会上传。

!!! note "最新 Cora 运行的覆盖参数"
    JSON 默认仍是 `n_eps=2`。最新 Cora 对照按用户指定显式传入 `--n-eps 1`，运行 100 轮、10 客户端全参与、等权 FedAvg。命令行显式值优先于 JSON。

### 6.1 完全本地 N1 不是 FedAvg 的特殊模式

当前 `main.py` 没有 `aggregation=none`。N1 使用 `run_v01_single_client.py` 分别启动 10 次训练：每个客户端从 seed 42 的相同初始化开始，连续运行 100 个本地 epoch，不接收广播、不上传参数，也不共享 Adam 状态，最后仍按本地最佳验证 epoch 配对测试。

这条路径用于回答“backbone 本身能否本地拟合”，不能称为联邦学习结果。

## 7. 参数、模型大小与通信

默认配置：

| CLI 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--latent-dim` | 32 | 节点潜维 |
| `--encoder-width` | 64 | 输入 MLP 宽度 |
| `--linear-steps` | 16 | 共享线性传播次数 |
| `--linear-step-size` | 0.1 | Euler 步长 |
| `--linear-gamma` | 0.1 | $A_0$ 耗散系数 |
| `--reconstruction-weight` | 0 | 输入重建权重；0 时不构造 decoder |
| `--lr` | 0.015 | Adam 学习率 |
| `--weight-decay` | $10^{-4}$ | Adam weight decay |
| `--n-rnds` | 100 | 默认通信轮数 |
| `--n-eps` | 2 | JSON 默认本地 epoch；可由 CLI 覆盖 |

在 $r=32$、关闭重建时：

| 数据集 | 可训练参数 | 单个 state_dict | 10 客户端单轮单方向通信 |
| --- | ---: | ---: | ---: |
| Cora | 187,911 | 751,644 B | 7,516,440 B |
| CiteSeer | 478,438 | 1,913,752 B | 19,137,520 B |

Cora 100 轮全参与运行记录了 751,644,000 B 上传和同量下载，总双向通信 1,503,288,000 B。这里不包括进程、数据加载或 checkpoint 文件开销。

## 8. 日志与 checkpoint

通用运行目录写出：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 合并 CLI 与 JSON 后的真实配置 |
| `metrics.csv` | 每轮平均 train/val/test loss、ACC/AUC、paired local-best test 和轮耗时 |
| `client_best.csv` | 每客户端最佳验证轮及同轮 test/F1 |
| `result.json` | 最终客户端配对结果的均值、标准差和原始数组 |
| `model_cost.json` | 参数量、state_dict 字节、潜维、步数和 decoder 开关 |
| `dynamics.csv` | $\|Z^{(0:L)}\|_F$、最大增长/相对增量、$\|A_0\|_2$、$\|A_1\|_2$、客户端耗时、峰值显存和累计通信 |

客户端 checkpoint 保存模型、持久 Adam 状态和 best-validation 记录；服务端 checkpoint 保存轮数、全局模型和聚合类型。运行结束后只清理 `*_current_state.pt`，保留各客户端 `*_state.pt` 最佳状态。

## 9. 已验证命令

### Cora：100 轮 FedAvg，1 个本地 epoch

```bash
/root/anaconda3/envs/torch/bin/python codev0/main.py \
  --model v01_linear --dataset Cora --gpu 0,1 \
  --seed 42 --n-workers 10 --n-clients 10 --frac 1 \
  --n-rnds 100 --n-eps 1 --latent-dim 32 \
  --reconstruction-weight 0 \
  --run-tag federated_m1_seed42_100r_1ep
```

### Cora：10 客户端完全本地

```bash
bash codev0/scripts/run_v01_local_cora_all_clients.sh
```

### 测试

```bash
cd /opt/data/private/xzc/work2
PYTHONPATH=codev0 /root/anaconda3/envs/torch/bin/python \
  -m unittest discover -s codev0/tests -p 'test_*.py' -v

PYTHONPATH=codev0 /root/anaconda3/envs/torch/bin/python \
  codev0/tests/validate_s0.py
```

当前共 15 项 unittest 通过；额外的 S0 验证确认官方源码加载、共享 Euler 场、不同客户端拓扑与 Adam 状态均正常。真实数值结果和当前失败判断见[实验分析](experiments/methodv01.md)。
