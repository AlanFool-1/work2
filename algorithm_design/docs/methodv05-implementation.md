---
title: V0.4+ Personalized Operator Flow：从概念到代码的实现设计
description: >-
  基于 V0.4 的明确 reference dynamics → Koopman generator 骨架，
  将 Personalized Dynamics Knowledge Flow 落实为可实现的 server/client 代码流程。
  当前重点不重新设计 nonlinear dynamics，而是实现：operator extraction、operator geometry、
  dynamics identifiability、directional transferable flow、personalized target、equilibrium diagnostics。
tags:
  - Graph Federated Learning
  - Koopman
  - Operator Manifold
  - Personalized Aggregation
  - Knowledge Flow
  - Code Design
---

# V0.4+ Personalized Operator Flow：从概念到代码的实现设计

版本：2026-09-11

> **目标**
>
> 本文档把上一版“Personalized Dynamics Knowledge Flow”的概念设计推进到可编码实现。
>
> 当前不解决：
>
> $$F_m(H,A)$$
>
> 到底采用哪一种 nonlinear graph dynamics。
>
> 暂时继续复用 V0.4 已有的 reference dynamics / Lie closure 接口，只假设客户端能够得到：
>
> $$\boxed{
> J_\Psi(H)F_m(H,A)
> \approx
> \sum_{b=1}^{B}
> X_{m,b}(H,A)K_{m,b}^{\top}
> }$$
>
> 本文档只解决第二条主线：
>
> $$\boxed{
> \text{客户端如何从 }K_m
> \text{ 出发，在动力学空间中实现个性化知识流？}
> }$$

---

# 0. 当前版本的工程边界

第一版必须保持简单。

保留 V0.4：

```text
shared encoder
observable / Koopman coordinate
reference dynamics interface
Lie / differential fitting
graph spectral basis
stable generator parameterization
ODE forward
classifier
shared-model federated optimizer
```

新增：

```text
operator extraction
operator geometry
local dynamics-identifiability statistics
server-side directional operator flow
personalized operator target
operator reinjection
flow/equilibrium diagnostics
```

暂时不做：

```text
new nonlinear dynamics
learned knowledge-potential network
client clustering
functional map
pairwise neural attention
full trajectory upload
raw embedding upload
task-logit upload
complex meta-learning
```

---

# 1. 整体训练框架

完整 round 分成两条独立通道：

```text
                     ┌───────────────────────┐
                     │       SERVER          │
                     │                       │
                     │ shared model channel  │
                     │ operator flow channel │
                     └──────────┬────────────┘
                                │
           ┌────────────────────┴────────────────────┐
           │                                         │
           ▼                                         ▼
  shared encoder / classifier               personalized K target
           │                                         │
           └──────────────────┬──────────────────────┘
                              ▼
                       Client m local model
                              │
                     V0.4 local training
                              │
                    reference dynamics F_m
                              │
                       Lie / operator fit
                              │
                     local operator K_m
                              │
                compact operator statistics
                              │
                              ▼
                           upload
```

服务器每轮做两件完全不同的事情：

1. **共享 representation 聚合**
2. **个性化 dynamics knowledge routing**

禁止把两者重新混回一个 `state_dict()` 全量 FedAvg。

---

# 2. 第一原则：服务器操作的是“实际 K”，不是 raw parameter

V0.4 稳定 generator 采用：

$$
\boxed{
K
=
\Omega-R
}
$$

其中：

$$
\Omega=A-A^\top,
$$

$$
R=CC^\top+\gamma I.
$$

因此代码中真正的模型参数可能是：

```python
A: [r, r]
C: [r, r]
```

但服务器不能直接聚合：

```text
A
C
```

因为：

$$
\overline C\,\overline C^\top
\neq
\overline{CC^\top}.
$$

所以 server/client 通信对象统一规定为：

$$
\boxed{
K_{m,b}
}
$$

即真实 generator matrix。

---

# 3. OperatorBundle：客户端动力学知识的统一数据结构

不要在 federation 代码中写死：

```text
K0
K1
```

定义通用 operator bundle。

```python
@dataclass
class OperatorBlock:
    name: str
    K: torch.Tensor              # [r, r]
    excitation_gram: torch.Tensor # [r, r]
    fit_residual: float
    num_samples: int
    stability_margin: float

@dataclass
class OperatorBundle:
    client_id: int
    round_id: int
    blocks: dict[str, OperatorBlock]
```

例如 V0.4 K=1 Bernstein：

```python
blocks = {
    "band0": OperatorBlock(...),
    "band1": OperatorBlock(...),
}
```

如果未来换成：

```text
self / neighbor
```

只需要：

```python
blocks = {
    "self": ...,
    "neighbor": ...,
}
```

server-side flow 算法不改。

---

# 4. 每个 operator block 的结构分解

对真实 generator：

$$
K\in\mathbb R^{r\times r},
$$

定义：

$$
\boxed{
\Omega(K)
=
\frac12(K-K^\top)
}
$$

$$
\boxed{
R(K)
=
-\frac12(K+K^\top)
}
$$

理想稳定情况下：

$$
R\succ0.
$$

于是：

$$
\boxed{
K=\Omega-R.
}
$$

代码：

```python
def decompose_generator(K):
    omega = 0.5 * (K - K.T)
    R = -0.5 * (K + K.T)
    return omega, R
```

---

# 5. 第一版不要直接上 AIRM，先采用 Log-Euclidean SPD geometry

理论上可以使用 affine-invariant Riemannian metric，但第一版代码建议使用：

$$
\boxed{
S=\log R
}
$$

将：

$$
R\in\mathbb S_{++}^r
$$

映射到对称矩阵空间。

因此 operator 的几何坐标写成：

$$
\boxed{
\chi(K)
=
(\Omega,S),
\qquad
S=\log R.
}
$$

在这个坐标中：

- $\Omega$ 位于 skew-symmetric linear space；
- $S$ 位于 symmetric linear space；
- SPD 约束通过 $R=\exp S$ 自动保证。

这比直接对 SPD 做普通平均更合理，又比 AIRM 简单稳定。

---

# 6. SPD 数值工具

实现统一工具：

```python
def symmetrize(M):
    return 0.5 * (M + M.T)

def skew_symmetrize(M):
    return 0.5 * (M - M.T)

def spd_eigh(R, eps):
    R = symmetrize(R)
    eigvals, eigvecs = torch.linalg.eigh(R)
    eigvals = eigvals.clamp_min(eps)
    return eigvals, eigvecs

def spd_log(R, eps=1e-6):
    vals, vecs = spd_eigh(R, eps)
    return vecs @ torch.diag(torch.log(vals)) @ vecs.T

def spd_exp(S):
    S = symmetrize(S)
    vals, vecs = torch.linalg.eigh(S)
    return vecs @ torch.diag(torch.exp(vals)) @ vecs.T
```

所有 server geometry 使用：

```text
float64
```

最后发送回 client 前再转回模型 dtype。

---

# 7. OperatorCoordinate

```python
@dataclass
class OperatorCoordinate:
    omega: torch.Tensor  # skew [r, r]
    log_R: torch.Tensor  # symmetric [r, r]
```

转换：

```python
def operator_to_coordinate(K, eps=1e-6):
    omega, R = decompose_generator(K)
    return OperatorCoordinate(
        omega=skew_symmetrize(omega),
        log_R=spd_log(R, eps),
    )

def coordinate_to_operator(coord):
    omega = skew_symmetrize(coord.omega)
    R = spd_exp(coord.log_R)
    return omega - R
```

---

# 8. 为什么第一版使用这种 geometry

这一步不是为了“用流形包装算法”。

它有三个直接工程好处。

第一：

$$
R=\exp(S)\succ0
$$

自动满足耗散部分合法。

第二：

个性化知识注入后可以保证 operator 仍然处于合法 dynamics space。

第三：

我们可以分别控制：

$$
\Omega
$$

与：

$$
R
$$

上的知识流。

后续实验可以直接回答：

```text
是 transport knowledge 更可迁移？
还是 dissipation knowledge 更可迁移？
```

---

# 9. 本地客户端必须额外计算“operator identification confidence”

如果只上传：

$$
K_m,
$$

服务器只知道客户端学到了什么，却不知道：

> 哪些方向客户端掌握得可靠，哪些方向客户端其实没见够数据。

所以必须上传一个小型 identifiability statistic。

关键是这个统计量应该来自 V0.4 的 Lie / differential fitting 本身。

---

# 10. 从 Lie regression 构造 Excitation Gram

假设 V0.4 某个 operator block $b$ 的输入为：

$$
X_{m,b}(t)
\in
\mathbb R^{N_m\times r}.
$$

例如 Bernstein basis：

$$
X_{m,b}(t)
=
B_{m,b}(A_m)\Psi(H_m(t)).
$$

那么定义：

$$
\boxed{
Q_{m,b}
=
\frac{1}{SN_m}
\sum_{s=1}^{S}
X_{m,b}(t_s)^\top
X_{m,b}(t_s).
}
$$

其中：

$$
Q_{m,b}\in\mathbb S_+^r.
$$

代码：

```python
def excitation_gram(block_states):
    # block_states: list[[N, r]]
    Q = 0.0
    count = 0

    for X in block_states:
        Q = Q + X.T @ X
        count += X.shape[0]

    return Q / max(count, 1)
```

---

# 11. Q 的物理含义

$Q_{m,b}$ 不是普通 embedding covariance。

它来自：

$$
X_{m,b}
=
\text{graph basis response}
\times
\text{current latent state}.
$$

所以它同时受：

$$
\boxed{
\text{feature state}
+
\text{local topology}
}
$$

影响。

若某个方向 $u$：

$$
u^\top Q_{m,b}u
$$

很大，说明客户端在该 operator input direction 上有充分 excitation。

若：

$$
u^\top Q_{m,b}u
\approx0,
$$

说明客户端很难从本地数据可靠识别该方向的 dynamics。

这正好给：

$$
\boxed{
\text{missing dynamics}
}
$$

提供一个可实现的定义。

---

# 12. Local Confidence Matrix

将：

$$
Q_{m,b}
$$

映射为 $[0,1]$ confidence：

$$
\boxed{
C_{m,b}
=
Q_{m,b}
\left(
Q_{m,b}+\rho I
\right)^{-1}.
}
$$

如果：

$$
Q=U\Lambda U^\top,
$$

则：

$$
C
=
U
\operatorname{diag}
\left(
\frac{\lambda_i}{\lambda_i+\rho}
\right)
U^\top.
$$

因此：

$$
0\preceq C\preceq I.
$$

实现：

```python
def confidence_from_gram(Q, rho):
    vals, vecs = torch.linalg.eigh(symmetrize(Q))
    vals = vals.clamp_min(0.0)

    conf = vals / (vals + rho)

    return (
        vecs
        @ torch.diag(conf)
        @ vecs.T
    )
```

---

# 13. Missing Matrix

定义：

$$
\boxed{
U_{m,b}
=
I-C_{m,b}.
}
$$

解释：

$$
C_{m,b}
$$

表示：

> 本地已经被充分 excitation / 识别的 dynamics directions。

$$
U_{m,b}
$$

表示：

> 本地较弱、潜在需要外部知识补充的 directions。

这比直接：

```text
K_m 与 K_n 距离大
```

更接近“缺失知识”。

---

# 14. Donor 还必须满足“自己的 dynamics fit 是可信的”

高 excitation 不代表 operator 一定拟合正确。

客户端同时上传 V0.4 已有的 Lie / field residual：

$$
e_{m}
=
\frac{
\|Y_m-\widehat Y_m\|_F^2
}{
\|Y_m\|_F^2+\epsilon
}.
$$

定义 donor scalar reliability：

$$
\boxed{
q_m
=
\exp
\left(
-\frac{e_m}{\tau_e}
\right).
}
$$

第一版 $\tau_e$ 可以取：

```text
median residual across clients
```

避免手工指定绝对量纲。

---

# 15. Directional Transfer Mask

对 receiver $m$ 和 donor $n$：

$$
U_m=I-C_m
$$

表示 receiver missing directions，

$$
C_n
$$

表示 donor confident directions。

定义：

$$
\boxed{
M_{m\leftarrow n}
=
U_m^{1/2}
C_n
U_m^{1/2}.
}
$$

它满足：

$$
M_{m\leftarrow n}\succeq0.
$$

物理含义：

> 只有“m 缺失”且“n 掌握”的方向会得到较大权重。

注意：

$$
M_{m\leftarrow n}
\neq
M_{n\leftarrow m}
$$

一般自然成立。

因此方向性不是人为加进去的，而是由：

$$
\boxed{
\text{receiver deficiency}
\times
\text{donor confidence}
}
$$

自然产生。

---

# 16. PSD square root

```python
def psd_sqrt(M, eps=0.0):
    vals, vecs = torch.linalg.eigh(symmetrize(M))
    vals = vals.clamp_min(eps)
    return (
        vecs
        @ torch.diag(torch.sqrt(vals))
        @ vecs.T
    )
```

---

# 17. Operator difference 不直接用 K_n-K_m

首先转换到 dynamics coordinates：

$$
\chi(K_m)
=
(\Omega_m,S_m),
$$

$$
\chi(K_n)
=
(\Omega_n,S_n).
$$

定义：

$$
\Delta\Omega_{mn}
=
\Omega_n-\Omega_m,
$$

$$
\Delta S_{mn}
=
S_n-S_m.
$$

这是 Log-Euclidean product geometry 下的 tangent direction。

---

# 18. Missing-mode projection

令：

$$
P_{mn}
=
M_{m\leftarrow n}^{1/2}.
$$

对 transport part：

$$
\boxed{
V^{\Omega}_{m\leftarrow n}
=
\operatorname{skew}
\left(
P_{mn}
\Delta\Omega_{mn}
P_{mn}
\right).
}
$$

对 dissipation part：

$$
\boxed{
V^{R}_{m\leftarrow n}
=
\operatorname{sym}
\left(
P_{mn}
\Delta S_{mn}
P_{mn}
\right).
}
$$

因为：

- congruence 保留 mode selection；
- `skew()` 保证仍在 $\mathfrak{so}(r)$；
- `sym()` 保证仍在 symmetric tangent space。

---

# 19. Pairwise Directional Knowledge Flow

第一版定义：

$$
\boxed{
F_{m\leftarrow n}^{\Omega}
=
q_n
V^{\Omega}_{m\leftarrow n}
}
$$

$$
\boxed{
F_{m\leftarrow n}^{R}
=
q_n
V^{R}_{m\leftarrow n}.
}
$$

这里没有任何：

$$
\exp(-d(K_m,K_n))
$$

形式的 similarity weight。

流动只依赖：

1. receiver 哪些 dynamics directions 缺失；
2. donor 哪些 dynamics directions 可靠；
3. donor 自己的 dynamics fitting 是否可信；
4. 两者在这些方向上是否确实存在 operator difference。

---

# 20. 多 donor 聚合

对 receiver $m$：

$$
\boxed{
V_m^\Omega
=
\frac{
\sum_{n\neq m}
F_{m\leftarrow n}^\Omega
}{
\sum_{n\neq m}q_n+\epsilon
}
}
$$

$$
\boxed{
V_m^R
=
\frac{
\sum_{n\neq m}
F_{m\leftarrow n}^R
}{
\sum_{n\neq m}q_n+\epsilon
}.
}
$$

这是一个 personalized tangent vector。

不同客户端得到：

$$
V_1\neq V_2\neq\cdots\neq V_M.
$$

服务器不是返回一个 global $K_*$，而是每个客户端一个：

$$
\boxed{
V_m.
}
$$

---

# 21. 为什么这种 aggregation 仍不是 similarity aggregation

虽然公式中存在：

$$
K_n-K_m,
$$

但是否发生 transfer 并不由 difference magnitude 决定。

如果 client $m$ 已经充分掌握某个 direction：

$$
U_m\approx0,
$$

那么即使：

$$
\|K_n-K_m\|
$$

很大，

仍有：

$$
F_{m\leftarrow n}\approx0.
$$

如果 donor $n$ 自己不可靠：

$$
q_n\approx0,
$$

也不会发生 transfer。

所以：

$$
\boxed{
\text{difference}
\neq
\text{knowledge}.
}
$$

difference 只是知识流方向；

identifiability / confidence 才决定它是否值得流动。

---

# 22. Knowledge Flow Norm

定义每个 block：

$$
\phi_{m,b}^2
=
\|V_{m,b}^\Omega\|_F^2
+
\lambda_R
\|V_{m,b}^{R}\|_F^2.
$$

全客户端：

$$
\boxed{
\Phi_m
=
\sqrt{
\sum_b
\phi_{m,b}^2
}.
}
$$

这就是代码中真正应该记录的：

```text
federated knowledge-flow magnitude
```

而不是：

```text
distance to global K
```

---

# 23. Equilibrium

定义：

$$
\boxed{
\Phi_m<\varepsilon_{\rm flow}
}
$$

时：

```python
active_flow[m] = False
```

客户端本轮不接收 operator injection。

如果连续：

```text
P patience rounds
```

都满足：

$$
\Phi_m<\varepsilon_{\rm flow},
$$

则标记：

```text
client m reached dynamics equilibrium
```

重要：

此时不要求：

$$
K_m=K_n.
$$

只要求：

$$
\boxed{
\text{没有显著可迁移动力学知识继续流入 m}.
}
$$

---

# 24. 不要再设计手工 alpha 衰减

第一版只需要一个固定 global step：

$$
\eta_{\rm flow}.
$$

因为：

$$
K_m^{new}
=
K_m
+
\eta_{\rm flow}V_m.
$$

当知识逐渐被吸收：

$$
\|V_m\|\downarrow,
$$

所以实际更新：

$$
\|\Delta K_m\|
=
\eta_{\rm flow}\|V_m\|
$$

自然下降。

无需：

```text
alpha round schedule
```

---

# 25. Personalized Target Construction

在 coordinate space：

$$
\boxed{
\Omega_{m,\mathrm{tar}}
=
\Omega_m
+
\eta_{\rm flow}
V_m^\Omega
}
$$

$$
\boxed{
S_{m,\mathrm{tar}}
=
S_m
+
\eta_{\rm flow}
V_m^R.
}
$$

然后：

$$
R_{m,\mathrm{tar}}
=
\exp(S_{m,\mathrm{tar}}).
$$

最终：

$$
\boxed{
K_{m,\mathrm{tar}}
=
\Omega_{m,\mathrm{tar}}
-
R_{m,\mathrm{tar}}.
}
$$

所以每个客户端收到不同的：

$$
K_{m,\mathrm{tar}}.
$$

---

# 26. 如何把 K_target 重新写回 V0.4 的 A/C 参数

这是从概念到代码最重要的一步。

V0.4：

$$
K=A-A^\top-CC^\top-\gamma I.
$$

给定 server 返回的：

$$
K_{\rm tar},
$$

先分解：

$$
\Omega_{\rm tar}
=
\frac12(K_{\rm tar}-K_{\rm tar}^\top),
$$

$$
R_{\rm tar}
=
-\frac12(K_{\rm tar}+K_{\rm tar}^\top).
$$

希望：

$$
A-A^\top
=
\Omega_{\rm tar}.
$$

直接取：

$$
\boxed{
A_{\rm tar}
=
\frac12
\Omega_{\rm tar}.
}
$$

因为：

$$
A_{\rm tar}-A_{\rm tar}^\top
=
\Omega_{\rm tar}.
$$

再令：

$$
CC^\top+\gamma I
=
R_{\rm tar}.
$$

所以：

$$
\boxed{
CC^\top
=
R_{\rm tar}-\gamma I.
}
$$

如果右边 SPD：

$$
\boxed{
C_{\rm tar}
=
\operatorname{chol}
(
R_{\rm tar}-\gamma I
).
}
$$

---

# 27. set_from_operator_()

在 generator class 中新增：

```python
@torch.no_grad()
def set_from_operator_(self, K_target):
    omega = 0.5 * (K_target - K_target.T)
    R = -0.5 * (K_target + K_target.T)

    # floor dissipation eigenvalues above gamma
    vals, vecs = torch.linalg.eigh(0.5 * (R + R.T))
    vals = vals.clamp_min(self.gamma + self.factor_eps)
    R = vecs @ torch.diag(vals) @ vecs.T

    B = R - self.gamma * torch.eye(
        R.shape[0],
        device=R.device,
        dtype=R.dtype,
    )

    B = 0.5 * (B + B.T)

    # add small jitter for Cholesky
    jitter = self.factor_eps * torch.eye(
        B.shape[0],
        device=B.device,
        dtype=B.dtype,
    )

    C = torch.linalg.cholesky(B + jitter)

    self.A.copy_(0.5 * omega)
    self.C.copy_(C)
```

---

# 28. round start 的 operator injection

客户端 round 开始：

```python
load_shared_global_state(...)
load_personalized_operator_target(...)
```

流程：

```python
for block_name, K_target in message.operator_targets.items():
    model.generator[block_name].set_from_operator_(K_target)
```

然后才进行本地 V0.4 training。

---

# 29. Operator optimizer state 必须独立处理

server 注入：

$$
K_{m,\mathrm{tar}}
$$

后，raw：

$$
A,C
$$

发生外部跳变。

因此不能保留旧的 Adam moments。

否则重新出现：

```text
parameter changed externally
but optimizer moments still belong to old point
```

的问题。

所以：

$$
\boxed{
A,C\text{ 的 optimizer state 每轮 operator injection 后 reset。}
}
$$

shared encoder/classifier 仍然可以继续当前验证过的 federated optimizer protocol。

必须把 optimizer parameter groups 分开。

---

# 30. 推荐 optimizer 分组

```python
shared_params = [
    encoder,
    observable_if_shared,
    classifier,
]

operator_params = [
    generator.block0.A,
    generator.block0.C,
    generator.block1.A,
    generator.block1.C,
]
```

推荐：

```text
shared params:
    current validated FedAdamW-v protocol

operator params:
    fresh AdamW / SGD every round
```

第一版优先：

```text
operator optimizer = fresh AdamW
```

因为 V0.4 local 已经证明 adaptive optimization 更容易建立 Lie alignment。

---

# 31. 共同坐标问题必须在代码中显式处理

直接比较：

$$
K_m
$$

隐含一个重要假设：

> 所有客户端的 Koopman coordinate 是一致的。

如果 encoder / observable 在每个客户端本地大幅漂移，则：

$$
K_1,K_2
$$

不一定能直接比较。

第一版禁止马上引入 functional map。

先用最简单的两阶段训练验证。

---

# 32. Phase A：Coordinate Warmup

前：

$$
R_{\rm warm}
$$

轮：

```text
normal V0.4 training
operator flow OFF
```

训练：

```text
encoder
observable
classifier
local generators
```

目标：

建立可用共同 representation。

例如：

```text
R_warm = 20
```

---

# 33. Phase B：Freeze Koopman Coordinate

从：

$$
r=R_{\rm warm}+1
$$

开始：

```text
freeze encoder
freeze observable
```

至少先做第一版验证。

此时：

$$
\Psi
$$

在所有客户端完全相同。

所以：

$$
K_m
$$

具有明确公共坐标语义。

继续训练：

```text
generator
classifier (optional)
reference dynamics parameters (depending on V0.4 design)
```

并打开：

```text
personalized operator flow
```

---

# 34. 为什么第一版一定要 freeze

这不是最终方法必须 freeze。

这是一个**科学验证 gate**：

> 如果固定共同 Koopman coordinate 后，operator flow 仍然无效，那么问题在 aggregation 本身；
>
> 如果 freeze 后明显有效，而 unfreeze 后失败，那么关键问题就是 coordinate drift。

这样再决定是否需要：

```text
functional map
parallel transport
slow-moving global observable
```

而不是一开始就增加复杂模块。

---

# 35. Local statistics extraction 时机

每轮 local training 完成后，使用最终模型做一个：

```text
no_grad diagnostics / calibration pass
```

收集：

```text
K block matrices
basis-response excitation Grams
Lie residual
sample count
stability margin
```

不从训练过程随意取某一个 minibatch。

对于 transductive graph：

```text
use all train nodes for fitting statistics
```

不要用 test labels。

---

# 36. 多时间点统计

建议从 ODE trajectory 中取：

$$
t\in\{0,T/2,T\}
$$

三个时间点。

对每个 block：

$$
Q_{m,b}
=
\frac13
\sum_t
Q_{m,b}(t).
$$

这样 identifiability 描述的是：

$$
\boxed{
\text{finite-horizon dynamics excitation}
}
$$

而不是只有 $t=0$。

---

# 37. ClientPacket

建议定义：

```python
@dataclass
class DynamicsClientPacket:
    client_id: int
    round_id: int

    operator_blocks: dict[str, torch.Tensor]
    excitation_grams: dict[str, torch.Tensor]

    lie_residual: float
    stability_margin: dict[str, float]

    num_train_nodes: int
    local_val_acc: float | None = None
```

第一版不要上传：

```text
node embeddings
node logits
trajectory tensors
labels
raw adjacency
```

---

# 38. ServerResult

```python
@dataclass
class PersonalizedFlowResult:
    client_id: int
    round_id: int

    target_operators: dict[str, torch.Tensor]

    flow_norm: float
    block_flow_norms: dict[str, float]

    active: bool
    donor_strengths: dict[int, float]

    transport_flow_norm: float
    dissipation_flow_norm: float
```

---

# 39. DynamicsKnowledgeRouter

主类：

```python
class DynamicsKnowledgeRouter:
    def __init__(
        self,
        rho_conf,
        flow_step,
        flow_eps,
        fit_temperature,
        manifold_eps,
    ):
        ...

    def route(
        self,
        packets: list[DynamicsClientPacket],
    ) -> dict[int, PersonalizedFlowResult]:
        ...
```

内部流程：

```text
1. validate packets
2. convert each K to operator coordinates
3. build confidence matrices from Q
4. build receiver missing matrices
5. build donor reliability
6. compute pairwise directional tangent flows
7. sum flows per receiver
8. check equilibrium
9. construct personalized target K
10. return per-client result
```

---

# 40. route() 伪代码

```python
def route(self, packets):

    cache = {}

    for p in packets:
        cache[p.client_id] = {}

        for block in p.operator_blocks:
            K = p.operator_blocks[block]
            Q = p.excitation_grams[block]

            coord = operator_to_coordinate(K)
            C = confidence_from_gram(Q, self.rho_conf)
            U = I - C

            cache[p.client_id][block] = {
                "K": K,
                "coord": coord,
                "confidence": C,
                "missing": U,
            }

    reliability = {
        p.client_id:
            fit_reliability(
                p.lie_residual,
                self.fit_temperature,
            )
        for p in packets
    }

    outputs = {}

    for recv in packets:

        recv_id = recv.client_id

        block_targets = {}
        block_norms = {}

        for block in recv.operator_blocks:

            base = cache[recv_id][block]
            V_omega = zeros_like(base["coord"].omega)
            V_logR = zeros_like(base["coord"].log_R)

            denom = 0.0

            for donor in packets:

                if donor.client_id == recv_id:
                    continue

                donor_id = donor.client_id

                flow = pairwise_flow(
                    receiver=base,
                    donor=cache[donor_id][block],
                    donor_reliability=reliability[donor_id],
                )

                V_omega += flow.omega
                V_logR += flow.log_R
                denom += reliability[donor_id]

            if denom > 0:
                V_omega /= denom
                V_logR /= denom

            flow_norm = product_tangent_norm(
                V_omega,
                V_logR,
            )

            if flow_norm <= self.flow_eps:
                target = base["K"]
            else:
                target_coord = OperatorCoordinate(
                    omega=(
                        base["coord"].omega
                        + self.flow_step * V_omega
                    ),
                    log_R=(
                        base["coord"].log_R
                        + self.flow_step * V_logR
                    ),
                )

                target = coordinate_to_operator(
                    target_coord
                )

            block_targets[block] = target
            block_norms[block] = flow_norm

        outputs[recv_id] = build_result(
            block_targets,
            block_norms,
        )

    return outputs
```

---

# 41. pairwise_flow()

```python
def pairwise_flow(
    receiver,
    donor,
    donor_reliability,
):
    U_m = receiver["missing"]
    C_n = donor["confidence"]

    U_half = psd_sqrt(U_m)

    M = (
        U_half
        @ C_n
        @ U_half
    )

    P = psd_sqrt(M)

    d_omega = (
        donor["coord"].omega
        - receiver["coord"].omega
    )

    d_logR = (
        donor["coord"].log_R
        - receiver["coord"].log_R
    )

    V_omega = skew_symmetrize(
        P @ d_omega @ P
    )

    V_logR = symmetrize(
        P @ d_logR @ P
    )

    return OperatorCoordinate(
        omega=donor_reliability * V_omega,
        log_R=donor_reliability * V_logR,
    )
```

---

# 42. 第一版不要加入 task similarity

暂时不要使用：

```text
validation gradient similarity
class distribution similarity
label overlap
classifier cosine
```

否则很难判断效果究竟来自：

```text
dynamics geometry
```

还是：

```text
task similarity heuristic
```

第一版 knowledge flow 必须只由：

$$
\boxed{
\text{dynamics identifiability}
+
\text{operator structure}
}
$$

驱动。

后续如果需要 task utility，再作为单独扩展。

---

# 43. Flow response metric 第一版用于 diagnostics，不进入 routing 权重

实现：

$$
d_{\rm flow}^2(K_i,K_j)
=
\frac1S
\sum_{s=1}^{S}
\|
e^{t_sK_i}
-
e^{t_sK_j}
\|_F^2.
$$

代码：

```python
def flow_distance(K1, K2, times):
    dist = 0.0

    for t in times:
        E1 = torch.matrix_exp(t * K1)
        E2 = torch.matrix_exp(t * K2)
        dist += torch.sum((E1 - E2) ** 2)

    return torch.sqrt(
        dist / len(times)
    )
```

推荐：

```text
times = [0.25, 0.5, 1.0]
```

先作为：

- analysis；
- pairwise heatmap；
- equilibrium validation；
- operator-space sanity check。

暂时不要：

```python
weight = exp(-flow_distance)
```

---

# 44. 为什么 flow distance 暂时不参与 teacher selection

因为：

$$
d_{\rm flow}
$$

仍然只是：

$$
\boxed{
\text{difference}
}
$$

而不是：

$$
\boxed{
\text{transferability}.
}
$$

它可以验证：

> operator knowledge flow 以后 dynamics behavior 是否真的靠近。

但不能直接回答：

> 是否应该吸收。

这个职责目前由：

$$
U_m
\quad+\quad
C_n
\quad+\quad
q_n
$$

承担。

---

# 45. 推荐目录结构

不要破坏 V0.4 backbone。

新增：

```text
codev0/
├── models/
│   └── v04/
│       ├── ...
│       ├── operator_adapter.py
│       └── operator_stats.py
│
├── federation/
│   └── operator_flow/
│       ├── __init__.py
│       ├── types.py
│       ├── geometry.py
│       ├── confidence.py
│       ├── pairwise_flow.py
│       ├── router.py
│       └── diagnostics.py
│
└── tests/
    ├── test_operator_geometry.py
    ├── test_operator_reparameterization.py
    ├── test_directionality.py
    ├── test_equilibrium.py
    └── test_operator_router.py
```

---

# 46. `operator_adapter.py`

负责模型参数与真实 K 之间转换：

```python
class GeneratorOperatorAdapter:

    @staticmethod
    def extract(generator) -> torch.Tensor:
        return generator.actual_matrix()

    @staticmethod
    def inject_(generator, K_target):
        generator.set_from_operator_(K_target)
```

所有 federation 代码禁止访问：

```text
A
C
```

只调用 adapter。

---

# 47. `operator_stats.py`

提供：

```python
class OperatorStatisticsCollector:
    def collect(
        self,
        model,
        graph,
        trajectory,
    ) -> DynamicsClientPacket:
        ...
```

职责：

- 收集真实 $K$；
- 收集 block responses；
- 计算 $Q_{m,b}$；
- 计算 Lie residual；
- 计算 stability margin；
- 打包上传。

---

# 48. `geometry.py`

只放纯数学工具：

```text
decompose_generator
spd_log
spd_exp
psd_sqrt
operator_to_coordinate
coordinate_to_operator
product_tangent_norm
flow_distance
```

不接触：

```text
client id
round
model
optimizer
```

这样方便单元测试。

---

# 49. `confidence.py`

只负责：

```text
Q → confidence C
C → missing U
Lie residual → donor reliability q
```

不要把 aggregation 逻辑塞进去。

---

# 50. `pairwise_flow.py`

只实现：

$$
(m,n)
\rightarrow
F_{m\leftarrow n}.
$$

接口：

```python
def compute_pairwise_flow(
    receiver_coord,
    donor_coord,
    receiver_missing,
    donor_confidence,
    donor_reliability,
):
    ...
```

---

# 51. `router.py`

只做：

```text
all pairwise flow
→ sum per receiver
→ equilibrium check
→ personalized target
```

它是 server-side 主算法。

---

# 52. Shared model 与 operator flow 的 round 顺序

推荐严格使用：

```text
Round r start

1. server broadcasts global shared representation
2. server sends client-specific K_m^target
3. client loads shared representation
4. client injects K_m^target
5. reset operator optimizer state
6. V0.4 local training
7. evaluate local task
8. calibration/statistics pass
9. upload shared update
10. upload DynamicsClientPacket

Server

11. aggregate shared representation
12. run DynamicsKnowledgeRouter
13. store personalized targets for round r+1
```

不要：

```text
先 local train
→ server target
→ 同一轮再 local train一次
```

第一版保持每 round 一个 local training stage。

---

# 53. Warmup / Flow 阶段

配置：

```yaml
operator_flow:
  enabled: true
  start_round: 21
  freeze_coordinate_at_start: true
```

流程：

```text
round 1-20:
    V0.4 warmup
    collect operator diagnostics
    no operator injection

round 21+:
    freeze shared Koopman coordinate
    enable personalized operator flow
```

---

# 54. 第一版建议超参数

```yaml
operator_flow:
  start_round: 21

  rho_conf: 0.05
  fit_temperature: auto_median

  flow_step: 0.2
  flow_eps: 1.0e-3
  equilibrium_patience: 5

  spd_eps: 1.0e-6

  flow_times:
    - 0.25
    - 0.5
    - 1.0

  freeze_encoder: true
  freeze_observable: true

  operator_optimizer_reset_each_round: true
```

注意：

这些是第一轮 engineering defaults，不是论文最终超参数。

---

# 55. rho_conf 不应直接使用固定绝对尺度

因为：

$$
Q
$$

受 latent scale 影响。

更稳妥做：

$$
\widetilde Q
=
\frac{Q}
{\operatorname{tr}(Q)/r+\epsilon}.
$$

然后：

$$
C=
\widetilde Q
(
\widetilde Q+\rho I
)^{-1}.
$$

这样推荐：

```text
rho_conf = 0.1
```

具有跨客户端更一致的含义。

---

# 56. Lie residual 同样需要归一化

必须使用：

$$
\boxed{
e_m
=
\frac{
\|Y-\widehat Y\|_F
}{
\|Y\|_F+\epsilon
}
}
$$

而不是 raw residual。

否则不同客户端：

```text
node count
field scale
latent norm
```

都会影响 donor reliability。

---

# 57. 必须记录的 diagnostics

每轮至少记录：

### 每客户端

```text
flow_norm
active_flow
equilibrium_counter
lie_residual
operator_stability_margin
mean_confidence
mean_missingness
num_active_donors
```

### transport / dissipation 分开

```text
omega_flow_norm
logR_flow_norm
```

### 每 block

```text
band0_flow_norm
band1_flow_norm
```

### pairwise

```text
pair_flow[m, n]
flow_distance[m, n]
directional_asymmetry[m, n]
```

---

# 58. Directional Asymmetry

定义：

$$
\boxed{
A_{mn}
=
\frac{
\|F_{m\leftarrow n}\|
-
\|F_{n\leftarrow m}\|
}{
\|F_{m\leftarrow n}\|
+
\|F_{n\leftarrow m}\|
+\epsilon
}.
}
$$

如果：

$$
A_{mn}\neq0,
$$

说明知识流确实具有方向性。

这是后续论文很重要的可视化。

---

# 59. Equilibrium Ratio

定义：

$$
\boxed{
R_{\rm eq}^r
=
\frac{
\#\{m:\Phi_m^r<\varepsilon\}
}{
M
}.
}
$$

期望：

$$
R_{\rm eq}^r
$$

随着 federation 增大。

这比：

```text
all K become close
```

更符合我们的理论目标。

---

# 60. Individuality Retention

定义 operator spread：

$$
S_K^r
=
\frac{2}{M(M-1)}
\sum_{m<n}
d_{\rm flow}(K_m,K_n).
$$

期望：

```text
flow norm ↓
```

但：

```text
operator spread 不应强制 → 0
```

即：

$$
\boxed{
\Phi_m\rightarrow0
\quad
\text{while}
\quad
S_K>0.
}
$$

这就是最核心的 equilibrium evidence。

---

# 61. Transfer Before/After Diagnostic

对客户端收到的 target：

$$
K_m^{tar}
$$

在本地数据上，不训练，分别运行：

```text
original K_m
target K_m^tar
```

比较：

$$
\Delta L_m^{instant}
=
L_m(K_m)
-
L_m(K_m^{tar}).
$$

这里只做 diagnostics，不用于第一版 routing。

如果：

$$
\Delta L_m^{instant}>0,
$$

说明 server 注入方向在当前任务上即时正迁移。

这可以后续作为 learned potential 的监督信号。

---

# 62. 第一阶段实验 Gate

不要直接和所有 baseline 比。

先只做内部科学验证。

## Gate A：Geometry correctness

验证：

```text
K → coordinate → K_reconstructed
```

误差：

$$
<10^{-6}
$$

float64。

---

## Gate B：Reparameterization correctness

验证：

```text
K_target
→ A,C
→ generator.actual_matrix()
```

满足：

$$
\frac{
\|K_{\rm recovered}-K_{\rm target}\|_F
}{
\|K_{\rm target}\|_F
}
<10^{-5}.
$$

---

## Gate C：Directionality

构造 synthetic：

```text
client A:
    direction u highly excited

client B:
    direction u missing
```

必须得到：

$$
\|F_{B\leftarrow A}\|
>
\|F_{A\leftarrow B}\|.
$$

---

## Gate D：No-transfer when receiver already knows

令：

$$
C_m\approx I.
$$

应有：

$$
U_m\approx0
$$

从而：

$$
F_{m\leftarrow n}\approx0.
$$

---

## Gate E：No-transfer from unreliable donor

令：

$$
e_n\gg1.
$$

则：

$$
q_n\approx0.
$$

必须：

$$
F_{m\leftarrow n}\approx0.
$$

---

## Gate F：Equilibrium without consensus

synthetic 设计：

```text
clients retain different confident subspaces
```

最终要求：

$$
F_m^{fed}\approx0
$$

但：

$$
K_m\neq K_n.
$$

这是最重要的算法单测之一。

---

# 63. 第一批真实实验

在 Cora 10 clients 上只比较：

```text
B0: V0.4 without operator flow
B1: one global operator mean
B2: personalized operator flow
```

不需要先 sweep 大量超参数。

主要看：

```text
task ACC
Macro-F1
worst client
flow norm
equilibrium ratio
operator spread
directional asymmetry
instant transfer gain
```

---

# 64. 第二批 Synthetic 必须重新发挥作用

Synthetic 的价值不只是 ACC。

需要人为控制：

```text
feature shift
structure shift
mixed shift
```

然后看：

1. $Q_{m,b}$ 是否随 feature/structure shift 改变；
2. $K_m$ 是否形成不同 dynamics；
3. directional flow 是否从“掌握某 mode”的 client 流向“缺失该 mode”的 client；
4. 吸收后 flow 是否减弱；
5. 最终是否保留不同 operator。

这会比 Cora 更直观地验证 storyline。

---

# 65. 一个需要特别警惕的问题：Q 不是“知识量”的最终定义

当前：

$$
Q_{m,b}
$$

只是第一版 **identifiability proxy**。

它回答：

> 本地 trajectory 有没有充分 excitation 某个 operator direction？

它并不能保证：

> 这个方向对 task 一定有益。

所以第一版必须明确称为：

```text
Dynamics Identifiability-Aware Flow
```

而不是直接宣称：

```text
Optimal Knowledge Potential
```

如果第一版有效，再扩展 task utility。

---

# 66. 后续升级路径

如果第一版验证有效，下一步再考虑：

### V2：Task utility

引入：

$$
\Delta L_{m\leftarrow n}
$$

或 local validation directional derivative。

### V3：AIRM geometry

把 log-Euclidean SPD 改为 affine-invariant geometry。

### V4：Coordinate drift

如果 freeze/unfreeze 差异明显，再引入 functional alignment。

### V5：Learned potential

最后才考虑神经网络学习：

$$
\Gamma_{m\leftarrow n}.
$$

顺序不能反。

---

# 67. 当前核心算法伪代码

```python
# ==========================================
# CLIENT m
# ==========================================

def client_round(
    shared_state,
    personalized_targets,
):
    load_shared_state(shared_state)

    if personalized_targets is not None:
        for block, K_target in personalized_targets.items():
            generator[block].set_from_operator_(K_target)

        reset_operator_optimizer_state()

    train_v04_locally()

    packet = collect_dynamics_packet()

    shared_update = export_shared_update()

    return shared_update, packet


# ==========================================
# SERVER
# ==========================================

def server_round(client_outputs):

    # A. normal shared representation update
    shared_state = aggregate_shared_updates(
        [x.shared_update for x in client_outputs]
    )

    # B. personalized dynamics flow
    packets = [
        x.dynamics_packet
        for x in client_outputs
    ]

    targets = dynamics_router.route(packets)

    return shared_state, targets
```

---

# 68. Router 的核心不是“找最近邻”

核心代码路径必须是：

```text
operator K
    ↓
dynamics decomposition
    ↓
(Ω, log R)
    │
    ├───────────────┐
    │               │
receiver Q        donor Q
    │               │
missing U        confidence C
    └──────┬────────┘
           ↓
 directional transfer mask
           ↓
project operator tangent
           ↓
sum complementary flows
           ↓
personalized tangent V_m
           ↓
K_m^target
```

如果最后实现变成：

```python
similarity = cosine(Km, Kn)
weight = softmax(similarity)
target = weighted_average(Kn)
```

说明方法已经跑偏。

---

# 69. 当前版本的正式数学骨架

客户端上传：

$$
\boxed{
\left(
K_{m,b},
Q_{m,b},
e_m
\right)_{b=1}^{B}.
}
$$

confidence：

$$
\boxed{
C_{m,b}
=
\widetilde Q_{m,b}
(
\widetilde Q_{m,b}+\rho I
)^{-1}.
}
$$

missing：

$$
\boxed{
U_{m,b}=I-C_{m,b}.
}
$$

directional mask：

$$
\boxed{
M_{m\leftarrow n,b}
=
U_{m,b}^{1/2}
C_{n,b}
U_{m,b}^{1/2}.
}
$$

operator geometry：

$$
\boxed{
K_{m,b}
\leftrightarrow
(
\Omega_{m,b},
S_{m,b}
),
\quad
S_{m,b}=\log R_{m,b}.
}
$$

pairwise flow：

$$
\boxed{
F_{m\leftarrow n,b}^{\Omega}
=
q_n
\operatorname{skew}
\left[
M_{mn,b}^{1/2}
(
\Omega_{n,b}-\Omega_{m,b}
)
M_{mn,b}^{1/2}
\right]
}
$$

$$
\boxed{
F_{m\leftarrow n,b}^{R}
=
q_n
\operatorname{sym}
\left[
M_{mn,b}^{1/2}
(
S_{n,b}-S_{m,b}
)
M_{mn,b}^{1/2}
\right].
}
$$

client flow：

$$
\boxed{
V_{m,b}
=
\operatorname{Agg}_{n\neq m}
F_{m\leftarrow n,b}.
}
$$

personalized target：

$$
\boxed{
\Omega_{m,b}^{tar}
=
\Omega_{m,b}
+
\eta V_{m,b}^{\Omega}
}
$$

$$
\boxed{
S_{m,b}^{tar}
=
S_{m,b}
+
\eta V_{m,b}^{R}
}
$$

$$
\boxed{
K_{m,b}^{tar}
=
\Omega_{m,b}^{tar}
-
\exp(S_{m,b}^{tar}).
}
$$

equilibrium：

$$
\boxed{
\sum_b
\|V_{m,b}\|^2
<
\varepsilon_{\rm flow}^2.
}
$$

---

# 70. 后续代码设计必须守住的四条线

## 线 1：Dynamics line

$$
F_m
\rightarrow
K_m.
$$

任何 operator 都必须能回答：

> 它在逼近哪个 dynamics？

---

## 线 2：Geometry line

$$
K_m
\rightarrow
(\Omega_m,R_m).
$$

不能把 operator 当普通参数向量。

---

## 线 3：Knowledge line

$$
Q_m
\rightarrow
C_m/U_m
\rightarrow
F_{m\leftarrow n}.
$$

不能把 difference/similarity 直接等价成 knowledge。

---

## 线 4：Equilibrium line

$$
F_m^{fed}\rightarrow0,
$$

而不是：

$$
K_m\rightarrow K_*.
$$

---

# 71. 现阶段最重要的工程目标

第一版代码不需要追求最终 SOTA。

它首先必须证明四件事：

$$
\boxed{
1.\;K_m\text{ 可以稳定提取和重参数化}
}
$$

$$
\boxed{
2.\;\text{knowledge flow 确实具有方向性}
}
$$

$$
\boxed{
3.\;\text{每个 client 得到不同 operator target}
}
$$

$$
\boxed{
4.\;\text{flow 可以趋近 0，而 operator 不必趋同}
}
$$

只要这四点成立，后续再研究 nonlinear dynamics 与更精确的 knowledge potential，整个方法主轴就不会跑偏。
