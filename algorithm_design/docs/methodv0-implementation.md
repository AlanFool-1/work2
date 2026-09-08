---
title: Method V0：代码实现与完整执行流程
description: >-
  逐步对应 codev0 的真实代码，说明官方 A-DGN 训练、轨迹采集、图变分编码、固定线性潜态滚动、联合损失和验证选型。
tags:
  - Method V0
  - 代码实现
  - Koopman
  - A-DGN
  - 图动力学
---

# Method V0：代码实现与完整执行流程

版本：2026-09-08 UTC · 本页以 `codev0/` 当前代码为准。

[返回算法设计](methodv0.md){ .md-button }
[查看实验结果](experiments/index.md){ .md-button .md-button--primary }

本页回答一个具体问题：**从一份客户端图数据开始，代码怎样得到冻结的 A-DGN，怎样生成教师轨迹，又怎样训练和选择一个低维线性动力学代理？**

当前实现包含两个严格分开的训练阶段。第一阶段训练官方 A-DGN/FedAvg；第二阶段加载一个客户端的最佳验证 checkpoint，把它冻结为教师，再单独训练本地代理。代理不会反向修改 A-DGN，也不会参与 FedAvg。

<figure markdown="span">
  ![Method V0 的联邦教师训练与本地 Koopman 代理完整框架](assets/methodv0-framework.png){ width="100%" }
  <figcaption>阶段一训练并评估真实 A-DGN；阶段二选择一个客户端 checkpoint，在本地拟合动力学代理。绿色旁路是随时间和扰动轨迹固定的节点条件。</figcaption>
</figure>

## 读图：两个容易混淆的问题

### 静态条件是什么，为什么需要它

动态潜态 $z_t\in\mathbb R^r$ 是整张图共享的一个向量。解码时，它会被广播给所有节点。如果共享解码器只读取 $z_t$，每个节点收到的输入完全相同，因而只能产生相同的节点状态：

$$
\widehat H_{t,i}=D_\psi(z_t)
\quad\Longrightarrow\quad
\widehat H_{t,i}=\widehat H_{t,j}\ \text{for all }i,j.
$$

这无法恢复一张图中不同节点的表示。因此代码从固定基准初态 $\bar H_0$ 和稀疏图 $G$ 为每个节点构造静态条件：

$$
C_i=f_\omega\!\left(
\bar H_{0,i},
[\mathcal A_G(\bar H_0)]_i,
\log(1+m_i)
\right),
\qquad
C_{G,i}=[\mathcal A_G(C)]_i.
$$

实际解码变成：

$$
\widehat H_{t,i}=D_\psi(z_t,C_i,C_{G,i}).
$$

这里可以把两类信息分开理解：

- $z_t$ 负责“整张图现在演化到了哪里”，随时间和扰动轨迹变化。
- $(C_i,C_{G,i})$ 负责“这是哪个节点、它处于什么局部结构和参考特征环境”，对所有时刻和轨迹共用。

静态条件还有两个作用。第一，它随节点重编号一起置换，使共享解码器保持节点置换等变；第二，它避免把完整参考状态直接复制到每条轨迹的动态通道中，让 32 维 $z_t$ 专门承载变化。

“静态”指条件的**输入基准不随时间和扰动轨迹改变**，不表示条件网络从训练开始就被冻结。训练期间 $f_\omega$ 会和代理一起更新，每一步都从同一个 $\bar H_0,G$ 重新计算 $C$；代理训练完成后才缓存最终条件。

它也可能成为捷径：解码器可能主要依赖 $C$ 重现共性轨迹，而忽略不同扰动之间的细小差异。这正是当前主轨迹误差较低、response NRMSE 却接近 1 时需要重点检查的机制；潜态打乱、潜态固定和 response-only 对照尚待完成。

### 阶段二是本地训练，ACC 属于哪个阶段

**当前网页中报告的官方 ACC/AUC 属于阶段一。** 它是 10 个客户端各自在最佳验证轮对应的测试指标，再在客户端之间求均值和标准差。它不是阶段二重新训练代理后得到的分类准确率。

阶段二确实完全在单客户端本地执行：当前实验选择 client 0 的最佳验证 checkpoint，冻结其 A-DGN 和 readout，只训练 $f_\omega,E_\phi,K,D_\psi$。代理不上传服务端，也不进行第二次 FedAvg。

| 阶段 | 是否使用真实标签训练 | 当前报告指标 | 指标含义 |
| --- | --- | --- | --- |
| 阶段一：A-DGN/FedAvg | 是 | ACC 或 AUC | 相对数据集真实标签的任务性能；按每客户端最佳验证轮配对测试后汇总 |
| 阶段二：本地代理 | 否 | state/logit/response NRMSE | 代理相对冻结 A-DGN 教师轨迹的数值保真度 |
| 阶段二：本地代理 | 否 | argmax agreement | 代理预测类别与冻结教师预测类别的一致率，不等于相对真实标签的 ACC |

当前阶段二代码没有单独输出代理的 ground-truth ACC。即使以后补充这个指标，也应同时报告教师 ACC、代理 ACC 和二者 agreement，避免把“模仿教师一致”误写成“真实任务准确”。

```text
阶段 A：客户端图分区
  → 官方 A-DGN 本地训练
  → 全模型 FedAvg
  → 按客户端最佳验证轮保存 checkpoint

阶段 B：加载一个客户端 checkpoint 并冻结
  → 生成基准/扰动完整轨迹
  → 图条件 + 图变分编码器
  → 固定线性 K 多步自由滚动
  → 节点解码 + 冻结 readout
  → 验证选型与测试指标
```

## 1. 阶段 A：训练真实动力学教师

### 1.1 官方 A-DGN 的实际更新

`codev0/models/s0/model.py` 直接加载官方 `GraphAntiSymmetricNN`，没有在 `codev0` 中重写主干。对客户端图 $G=(V,E)$，输入特征先映射为隐藏状态：

$$
H_0 = XW_{\mathrm{emb}}^\top+b_{\mathrm{emb}},
\qquad H_0\in\mathbb R^{n\times d}.
$$

官方卷积层在全部传播步共享同一组参数。令 $W$ 为自作用矩阵，$\gamma$ 为阻尼，$\operatorname{GCN}_G$ 为官方 `GCNConv`，则第 $t$ 次固定 Euler 更新是：

$$
W_{\mathrm{asym}}=W-W^\top-\gamma I,
$$

$$
H_{t+1}
=H_t+\varepsilon\tanh\!\left(
H_tW_{\mathrm{asym}}^\top+
\operatorname{GCN}_G(H_t)+b
\right).
$$

默认配置中 $d=64$、传播次数 $T=16$、$\varepsilon=0.1$、$\gamma=0.1$。最终节点 logits 为：

$$
Y_T=\mathcal O(H_T)=H_TW_{\mathrm{out}}^\top+b_{\mathrm{out}}.
$$

`encode_native_dynamics_states(data)` 使用 forward hook 捕获官方实现每次 `GCNConv` 的输入和卷积模块的最终输出，从而得到**真实执行产生的**：

$$
(H_0,H_1,\ldots,H_T)\in\mathbb R^{(T+1)\times n\times d}.
$$

它没有复制或近似官方更新；测试还会验证 `readout(H_T)` 与原生 `model(data)` 完全一致。

### 1.2 FedAvg 生命周期

默认有 10 个客户端和 10 个常驻 worker。每轮开始时，服务端广播全局参数；客户端加载自己的持久 Adam 状态，再用全局参数覆盖模型权重，执行本地训练。这样一来，模型参数参与 FedAvg，而每个客户端的 Adam 一阶、二阶矩保持本地连续。

默认等权 FedAvg 为：

$$
\theta^{(r+1)}=\frac{1}{M}\sum_{m=1}^{M}\theta_m^{(r,+)}.
$$

若显式选择样本量加权，则代码改用：

$$
\theta^{(r+1)}=
\sum_{m=1}^{M}
\frac{N_m}{\sum_jN_j}\theta_m^{(r,+)}.
$$

每个客户端在每轮本地训练后同时计算验证和测试指标，但只有验证指标用于更新最佳 checkpoint：

$$
r_m^*=\arg\max_r\operatorname{Metric}_{\mathrm{val},m}^{(r)},
\qquad
\operatorname{Report}_m=\operatorname{Metric}_{\mathrm{test},m}^{(r_m^*)}.
$$

因此最终报告的是“最佳验证轮对应的测试指标”，不是在测试集上选轮。

```text
初始化服务端全局模型 θ⁰
为每个客户端保存独立 Adam 状态和 best-validation 记录

for communication round r = 0 ... R-1:
    服务端广播 θʳ
    for selected client m in parallel:
        载入客户端 m 的 Adam 状态
        用 θʳ 覆盖本地模型参数
        在该客户端整图训练 n_eps 个 epoch
        计算 validation 和 paired test 指标
        若 validation 提升，保存 m_state.pt
        上传本地模型参数 θ_m^(r,+)
        保存客户端 Adam 状态
    服务端对全部上传参数执行 FedAvg
```

## 2. 阶段 B：冻结教师并生成完整轨迹

### 2.1 加载边界

代理入口是 `codev0/scripts/run_koopman_proxy_v0.py`。当前只为 Cora 和 CiteSeer 声明了输入维度与类别数。若未传入 `--checkpoint`，脚本会在对应数据集的 checkpoint 目录中查找客户端的 `*_state.pt`，并按文件修改时间选择最新文件。

加载后执行：

```text
model.eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
```

这会冻结 embedding、A-DGN 和 readout。后续教师轨迹在 `torch.no_grad()` 中产生。代理损失再次调用 readout 时，readout 参数仍不更新，但其对输入状态的梯度存在，所以任务损失能够回传到代理解码器。

### 2.2 基准轨迹与扰动轨迹

原始特征 $X$ 产生唯一的基准轨迹：

$$
\mathcal B^0=(H_{0:T}^0,Y_{0:T}^0).
$$

第 $\rho$ 条训练样本使用逐元素乘性扰动：

$$
\alpha_\rho\sim\mathcal U(0.01,0.05),
\qquad
\epsilon_\rho\sim\mathcal N(0,I),
$$

$$
X^\rho=X\odot(1+\alpha_\rho\epsilon_\rho).
$$

由于扰动与 $X$ 相乘，原本为零的特征仍然为零。图、A-DGN 参数和传播方程保持不变，每个初值经过同一冻结系统得到：

$$
\mathcal B^\rho=(H_{0:T}^\rho,Y_{0:T}^\rho).
$$

默认 `--seed 11` 时，代码实际给 train/validation/test 轨迹生成器使用的种子分别是：

$$
11+101=112,\qquad 11+202=213,\qquad 11+303=314.
$$

默认轨迹数量与形状为：

| Split | 轨迹数 | 状态张量 | Logit 张量 |
| --- | ---: | --- | --- |
| train | 128 | $[128,T+1,n,d]$ | $[128,T+1,n,c_y]$ |
| validation | 32 | $[32,T+1,n,d]$ | $[32,T+1,n,c_y]$ |
| test | 64 | $[64,T+1,n,d]$ | $[64,T+1,n,c_y]$ |

三组轨迹按初始条件独立生成，测试轨迹不参与训练或 checkpoint 选择。标签也不进入代理训练；代理只拟合冻结教师的状态与 logits。

```text
载入客户端图 G 和最佳验证 checkpoint
冻结 A-DGN 教师

base_states, base_logits = teacher_rollout(X, G)

for split, count, split_seed in [(train,128,112),
                                  (val,32,213),
                                  (test,64,314)]:
    for ρ = 1 ... count:
        采样 α 和逐元素噪声 ε
        Xρ = X * (1 + αε)
        states[ρ], logits[ρ] = teacher_rollout(Xρ, G)
    保存完整轨迹缓存
```

## 3. 稀疏图条件：给解码器提供节点身份

代理不构造稠密邻接矩阵。`make_edge_cache()` 给原图加入自环，并建立归一化稀疏消息权重。对边 $u\to v$，令：

$$
d_v=\sum_{u:(u,v)\in\widetilde E}w_{uv},
\qquad
\widehat w_{uv}=\frac{w_{uv}}{\sqrt{d_ud_v}},
$$

其中 $\widetilde E$ 包含自环。代码中的聚合操作是：

$$
[\mathcal A_G(Q)]_v
=\sum_{u:(u,v)\in\widetilde E}\widehat w_{uv}Q_u.
$$

固定参考状态取基准轨迹的初态：

$$
\bar H_0=H_0^0.
$$

实际实现中的第三个条件量不是原始整数度，而是归一化边权的绝对质量：

$$
m_v=\sum_{u:(u,v)\in\widetilde E}|\widehat w_{uv}|.
$$

节点条件由一个两层 MLP 产生：

$$
C=\operatorname{MLP}_{\omega}
\left([\bar H_0,\mathcal A_G(\bar H_0),\log(1+m)]\right)
\in\mathbb R^{n\times c},
\qquad c=16.
$$

随后再计算邻域条件 $C_G=\mathcal A_G(C)$。$C$ 和 $C_G$ 对所有轨迹和时刻共用。条件网络本身是可训练的，所以训练时每一步都会根据固定 $\bar H_0$ 重新计算条件；训练结束后才适合缓存最终条件。

## 4. 图变分编码器：整张图压成一个潜态

设当前节点状态为 $H_t\in\mathbb R^{n\times d}$。编码器先将动态状态与静态条件拼接：

$$
U_t^{(0)}=\operatorname{SiLU}
\left(W_0[H_t,C]+b_0\right)
\in\mathbb R^{n\times w}.
$$

再将本节点表示和一次稀疏邻域聚合拼接：

$$
U_t=\operatorname{SiLU}
\left(W_1[U_t^{(0)},\mathcal A_G(U_t^{(0)})]+b_1\right),
\qquad w=64.
$$

代码维护 $p=8$ 个学习查询 $q_j\in\mathbb R^w$，每个查询在节点维执行注意力汇聚：

$$
a_{tji}=
\frac{\exp(q_j^\top U_{ti}/\sqrt w)}
{\sum_{v=1}^{n}\exp(q_j^\top U_{tv}/\sqrt w)},
\qquad
v_{tj}=\sum_i a_{tji}U_{ti}.
$$

拼接全部查询输出 $v_t=[v_{t1};\ldots;v_{tp}]$，通过两个线性头得到对角高斯后验：

$$
\mu_t=W_\mu v_t+b_\mu,
\qquad
\log\sigma_t^2=\operatorname{clip}
(W_\sigma v_t+b_\sigma,-8,4),
$$

$$
q_\phi(z_t\mid H_t,G,C)
=\mathcal N(\mu_t,\operatorname{diag}(\sigma_t^2)).
$$

训练时使用重参数化采样：

$$
z_t=\mu_t+\exp\!\left(\tfrac12\log\sigma_t^2\right)\odot\xi,
\qquad \xi\sim\mathcal N(0,I_r).
$$

这里 $r=32$ 是**整张图共享的动态潜态维度**。共享节点网络、稀疏聚合和节点维 softmax 汇聚保证了联合节点重编号下的编码不变性；它不意味着不同客户端的潜坐标已经对齐。

## 5. 固定线性算子与节点解码器

### 5.1 代码中的真实初始化和滚动

代理包含一个无结构限制的参数矩阵 $A_z\in\mathbb R^{r\times r}$。代码使用全零初始化：

$$
A_z^{(0)}=0,
\qquad
K^{(0)}=I_r.
$$

训练过程中始终按下式构造离散算子：

$$
K=I_r+hA_z,
\qquad h=0.1.
$$

代码把潜态存成行向量，因此一次 `current @ operator.T` 对应数学形式：

$$
z_{s+1}=Kz_s.
$$

同一个窗口只在起点采样一次，随后重复使用同一个 $K$：

$$
\widehat z_{a+s}=K^sz_a,
\qquad s=0,1,\ldots,L.
$$

中途没有重新编码，也没有使用真实未来状态修正潜态。

### 5.2 解码与冻结读出

对每个节点 $i$，代码把全图潜态广播到该节点，并与两个静态条件拼接：

$$
\widehat H_{a+s,i}
=D_\psi([\widehat z_{a+s},C_i,C_{G,i}]),
$$

其中解码器结构为：

$$
(r+2c)\rightarrow64\xrightarrow{\mathrm{SiLU}}d.
$$

预测 logits 直接使用冻结教师的线性 readout：

$$
\widehat Y_{a+s}=\mathcal O(\widehat H_{a+s}).
$$

## 6. 训练损失：代码究竟比较哪些量

### 6.1 训练集固定尺度

全部代理损失使用训练轨迹计算出的三个全局标量尺度：

$$
s_H=\operatorname{mean}\!\left((H-\bar H)^2\right),
\qquad
s_Y=\operatorname{mean}\!\left((Y-\bar Y)^2\right),
$$

$$
s_R=\operatorname{mean}\!\left((Y^\rho-Y^0)^2\right).
$$

三个尺度都截断到至少 $10^{-8}$，验证集和测试集不会重新估计尺度。

### 6.2 联合训练窗口

每次更新先抽取一批轨迹，再随机选择窗口起点 $a$ 和当前跨度 $L$。编码器会编码批内每条轨迹的全部 $T+1$ 个时刻，但自由滚动和主要损失只使用窗口 $a:a+L$。

扰动轨迹和复制后的基准轨迹分别编码。对第 $b$ 对轨迹，代码为两个起点后验复用同一个噪声 $\xi_b$：

$$
z_a^{\rho,b}=\mu_a^{\rho,b}+\sigma_a^{\rho,b}\odot\xi_b,
\qquad
z_a^{0,b}=\mu_a^{0}+\sigma_a^{0}\odot\xi_b.
$$

这降低了成对响应差中的采样方差。两条轨迹随后独立地由同一个 $K$ 自由滚动。

实现中的六项损失是：

$$
\mathcal L_{\mathrm{rec}}
=\frac{\operatorname{MSE}(\widehat H_a,H_a)}{s_H},
$$

$$
\mathcal L_{\mathrm{pred}}
=\frac{1}{s_H}
\operatorname{MSE}(\widehat H_{a+1:a+L},H_{a+1:a+L}),
$$

$$
\mathcal L_{\mathrm{lin}}
=\operatorname{MSE}
(\widehat z_{a+1:a+L},\mu_{a+1:a+L}),
$$

$$
\mathcal L_{\mathrm{out}}
=\frac{1}{s_Y}
\operatorname{MSE}(\widehat Y_{a:a+L},Y_{a:a+L}),
$$

$$
\Delta Y_{a+s}=Y_{a+s}^{\rho}-Y_{a+s}^{0},
\qquad
\Delta\widehat Y_{a+s}=\widehat Y_{a+s}^{\rho}-\widehat Y_{a+s}^{0},
$$

$$
\mathcal L_{\mathrm{resp}}
=\frac{1}{s_R}
\operatorname{MSE}
(\Delta\widehat Y_{a:a+L},\Delta Y_{a:a+L}),
$$

$$
\mathcal L_{\mathrm{KL}}
=\frac12\operatorname{mean}_{\rho,t,j}
\left(
\mu_{\rho tj}^2+\exp(\log\sigma_{\rho tj}^2)
-\log\sigma_{\rho tj}^2-1
\right).
$$

需要注意两个实现口径：

1. `linear` 项比较的是**采样起点的自由滚动潜态**和真实未来状态的后验均值，不是单独比较 $K^s\mu_a$ 与 $\mu_{a+s}$。
2. `output` 和 `response` 项包含窗口起点 $s=0$；`pred` 项只包含未来 $s=1,\ldots,L$。

总损失为：

$$
\mathcal L=
\mathcal L_{\mathrm{rec}}
+\mathcal L_{\mathrm{pred}}
+0.1\mathcal L_{\mathrm{lin}}
+\mathcal L_{\mathrm{out}}
+\mathcal L_{\mathrm{resp}}
+10^{-4}\mathcal L_{\mathrm{KL}}.
$$

### 6.3 前 200 步预热

预热分支仍会编码整条轨迹，并对所有真实时刻计算 KL；但状态和输出只重建第一个真实时刻 $H_0,Y_0$：

$$
\mathcal L_{\mathrm{warmup}}
=\frac{\operatorname{MSE}(D(z_0),H_0)}{s_H}
+\frac{\operatorname{MSE}(\mathcal O(D(z_0)),Y_0)}{s_Y}
+10^{-4}\mathcal L_{\mathrm{KL}}.
$$

预热时 `A_z` 不参与计算，因此不会得到梯度。

## 7. 3200 次更新怎样执行

默认参数为 Adam、学习率 $10^{-3}$、批大小 4、梯度范数上限 1。跨度课程由 `select_horizon()` 根据当前更新编号计算：

| 更新编号 | 次数 | 实际训练分支 | 自由滚动跨度 |
| --- | ---: | --- | ---: |
| 1–200 | 200 | 编解码预热 | 不使用 $K$ |
| 201–950 | 750 | 完整联合损失 | 1 |
| 951–1700 | 750 | 完整联合损失 | 4 |
| 1701–2450 | 750 | 完整联合损失 | 8 |
| 2451–3200 | 750 | 完整联合损失 | 16 |

每次联合更新的实际伪代码为：

```text
输入：训练轨迹库、复制到批大小的基准轨迹、固定参考 H̄₀、稀疏图缓存

随机抽取至多 4 条扰动轨迹
根据当前 step 选择跨度 L
随机采样起点 a ∈ {0,...,T-L}

C, C_G = condition_net(H̄₀, G)
μ[0:T], logvar[0:T] = encoder(H[0:T], C, G)
μ⁰[0:T], logvar⁰[0:T] = encoder(H⁰[0:T], C, G)

ξ = standard_normal(batch, latent_dim)
z_a  = sample(μ_a,  logvar_a,  ξ)
z⁰_a = sample(μ⁰_a, logvar⁰_a, ξ)

K = I + 0.1 * A_z
z_rollout  = [z_a,  Kz_a,  ..., K^L z_a]
z⁰_rollout = [z⁰_a, Kz⁰_a, ..., K^L z⁰_a]

H_hat  = decoder(z_rollout,  C, C_G)
H⁰_hat = decoder(z⁰_rollout, C, C_G)
Y_hat  = frozen_readout(H_hat)
Y⁰_hat = frozen_readout(H⁰_hat)

计算 rec + pred + 0.1*linear + output + response + 1e-4*KL
反向传播到 condition_net、encoder、A_z、decoder
裁剪总梯度范数到 1
Adam 更新一次
```

## 8. 验证选型与测试指标

### 8.1 确定性自由滚动

验证和测试不从后验采样，而是从真实初态的后验均值出发：

$$
\widehat H_s=D_\psi(K^s\mu_0,C,C_G),
\qquad s\in\{1,4,8,16\}.
$$

基准轨迹也从自己的 $\mu_0^0$ 确定性滚动。此时不存在采样噪声。

### 8.2 四类指标

状态增量 NRMSE 使用整个 split 的平方和：

$$
e_H(s)=
\sqrt{
\frac{\sum_\rho\|\widehat H_s^\rho-H_s^\rho\|_F^2}
{\sum_\rho\|H_s^\rho-H_0^\rho\|_F^2+10^{-8}}
}.
$$

Logit 增量 NRMSE 同理：

$$
e_Y(s)=
\sqrt{
\frac{\sum_\rho\|\widehat Y_s^\rho-Y_s^\rho\|_F^2}
{\sum_\rho\|Y_s^\rho-Y_0^\rho\|_F^2+10^{-8}}
}.
$$

成对响应定义为相对未扰动基准的差：

$$
R_s^\rho=Y_s^\rho-Y_s^0,
\qquad
\widehat R_s^\rho=\widehat Y_s^\rho-\widehat Y_s^0,
$$

$$
e_R(s)=
\sqrt{
\frac{\sum_\rho\|\widehat R_s^\rho-R_s^\rho\|_F^2}
{\sum_\rho\|R_s^\rho\|_F^2+10^{-8}}
}.
$$

此外还报告冻结教师与代理在节点类别上的一致率：

$$
\operatorname{Agreement}(s)=
\frac{1}{Nn}\sum_{\rho,i}
\mathbf 1\!\left[
\arg\max \widehat Y_{s,i}^\rho
=\arg\max Y_{s,i}^\rho
\right].
$$

### 8.3 checkpoint 怎样选择

每 100 次更新计算一次完整验证集合的四个时距。验证分数是三类 NRMSE 的直接求和：

$$
S_{\mathrm{val}}=
\sum_{s\in\{1,4,8,16\}}
\big(e_H(s)+e_Y(s)+e_R(s)\big).
$$

只有当前训练课程已经进入完整 16 步阶段时，模型才有资格更新最佳 checkpoint。训练结束后恢复最低 $S_{\mathrm{val}}$ 对应的代理权重，再一次性计算 train、validation 和 test 指标。测试集不参与选型。

```text
每 100 次更新：
    proxy.eval()
    从每条 validation 轨迹的 μ₀ 开始自由滚动
    计算 horizon = 1, 4, 8, 16 的 state/logit/response NRMSE
    val_score = 所有 horizon 的三类 NRMSE 之和
    若训练课程已到 horizon=16 且 val_score 更低：
        保存内存中的 best_state

训练结束：
    恢复 best_state
    保存 proxy.pt
    对 train/validation/test 分别做确定性评估
```

## 9. 模块、张量和产物对应关系

| 代码位置 | 负责内容 | 关键输出 |
| --- | --- | --- |
| `models/s0/model.py` | 加载官方 A-DGN，捕获原生传播状态 | $H_{0:T}$、冻结 readout |
| `models/s0/client.py` | 本地训练、持久 Adam、最佳验证 checkpoint | `client_id_state.pt` |
| `models/s0/server.py` | 全模型 FedAvg 与结构化日志 | 全局模型、轮次记录 |
| `proxy/graph_koopman_v0.py` | 图条件、编码器、$K$、解码器和损失 | 本地代理 $\mathcal P_m$ |
| `scripts/run_koopman_proxy_v0.py` | 轨迹生成、课程训练、验证和导出 | `proxy.pt`、JSON、曲线图 |

默认 $d=w=64,c=16,p=8,r=32$ 时，代理共有 65,488 个可训练参数，其中 $A_z$ 只有 1,024 个参数。完整预测仍需图编码和每节点解码，不能只根据 $K$ 的大小判断端到端成本。

每次代理运行输出：

| 文件 | 内容 |
| --- | --- |
| `proxy.pt` | 验证集选择后的代理参数 |
| `training.jsonl` | 每次更新的跨度与六项损失 |
| `metrics.json` | 三个 split、四个时距的全部指标 |
| `metadata.json` | checkpoint/partition 哈希、形状、参数和运行配置 |
| `training_curves.png` | 训练损失曲线 |
| `horizon_metrics.png` | 测试集多时距误差 |

## 10. 当前实现边界

- 代理脚本目前只支持 Cora 和 CiteSeer；官方基线脚本支持 11 个数据集。
- 一个代理只对应“一个客户端图 + 一个冻结 checkpoint + 一段扰动范围”。主干或图变化后需要重新识别。
- 当前没有跨客户端潜坐标对齐，也没有上传、聚合或迁移代理参数。
- 轨迹缓存当前按 split 文件名与首维数量复用，没有把 checkpoint 哈希、种子和全部配置编码进缓存键；复用旧输出目录时需要核对 `metadata.json`。
- 若未显式指定 checkpoint，代码按修改时间选择最新的客户端最佳验证文件；正式复现实验应固定路径并核对 SHA-256。
- 当前实验显示长时距主轨迹误差较低，但扰动响应 NRMSE 接近 1。它说明代理主要捕获了共性传播轨迹，尚未满足动力学知识迁移的前置条件。

当前数值、失败分析和后续对照见[实验分析](experiments/index.md)。
