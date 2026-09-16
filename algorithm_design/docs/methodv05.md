---
title: Personalized Dynamics Knowledge Flow：V0.4+ 总体思想与后续方法设计约束
description: >-
  固定后续图联邦方法设计主轴：将客户端视为不同局部图动力系统，以低维 Koopman/线性
  动力学算子承载可交换知识；服务器不构造唯一全局算子，而是在动力学算子空间中为每个
  客户端形成个性化、方向性的知识流。随着互补动力学知识被吸收，外部驱动力自然衰减，
  最终达到“无显著可迁移知识”的异质性平衡，而非完全同步。
tags:
  - Graph Federated Learning
  - Graph Dynamics
  - Koopman
  - Operator Manifold
  - Personalized Aggregation
  - Knowledge Flow
  - Heterogeneity Equilibrium
---

# Personalized Dynamics Knowledge Flow：V0.4+ 总体思想与后续方法设计约束

版本：2026-09-11

> **本文档不是某个具体 V0.x 的实现说明，而是后续方法设计的思想约束。**
>
> 后续任何模块、距离、聚合策略或理论目标，都应围绕同一个核心问题：
>
> $$\boxed{\text{如何让互补动力学知识在异质客户端之间定向流动，并在知识吸收后自然停止？}}$$

---

# 1. 核心研究问题

图联邦通常把异质性拆成 feature heterogeneity 与 structure heterogeneity，但图学习真正执行的是：

$$
\boxed{\text{node states evolving under graph propagation}}
$$

特征决定初始状态：

$$
H_m(0)=E_\theta(X_m),
$$

结构决定状态如何相互作用：

$$
\dot H_m=F_m(H_m,A_m).
$$

因此真正要描述的是：

$$
\boxed{\text{feature states evolve under local topology}}
$$

形成的客户端特定 graph dynamics。

主线固定为：

$$
\boxed{
\text{Feature + Structure}
\rightarrow
\text{Graph Dynamics}
\rightarrow
\text{Dynamics Knowledge}
\rightarrow
\text{Personalized Federated Knowledge Flow}.
}
$$

---

# 2. 为什么必须使用 ODE / Dynamics

ODE 不是为了替代 GCN 层，而是为了把 feature 与 structure 的交互统一进 vector field：

$$
\boxed{
\frac{dH_m(t)}{dt}=F_m(H_m(t),A_m).
}
$$

于是客户端异质性统一表现为：

$$
\boxed{
F_1\neq F_2\neq\cdots\neq F_M.
}
$$

因此客户端应被视为不同的局部动力子系统，而不是仅仅拥有不同的数据分布。

---

# 3. 为什么需要低维动力学算子

真实 nonlinear graph dynamics $F_m$ 通常高维、非线性、难以直接通信和聚合。

因此需要低维 dynamics carrier：

$$
\boxed{K_m.}
$$

其作用不是普通参数压缩，而是描述客户端局部 evolution law。

若 observable 为：

$$
Z=\Psi(H),
$$

则理想关系：

$$
\boxed{
J_\Psi(H)F_m(H,A)
\approx
K_m\Psi(H).
}
$$

因此：

$$
\boxed{
K_m=\text{client }m\text{ 的 compact dynamics knowledge carrier}.
}
$$

---

# 4. K 不是普通矩阵

对连续系统：

$$
\dot z=Kz,
$$

真正的动力学对象是：

$$
\boxed{\mathcal T_K(t)=e^{tK}.}
$$

因此两个客户端的动力学差异不应只用：

$$
\|K_i-K_j\|_F.
$$

更自然的是比较有限时间演化：

$$
\boxed{
d_{\rm flow}^2(K_i,K_j)
=
\int_0^T
w(t)
\left\|
e^{tK_i}-e^{tK_j}
\right\|_F^2dt.
}
$$

核心原则：

> **比较 operator 时，应优先比较它们产生的动态行为，而不是矩阵元素本身。**

---

# 5. K 的动力学结构

连续 generator 可写成：

$$
\boxed{K=\Omega-R}
$$

其中：

$$
\Omega=\frac{K-K^\top}{2},
\qquad
\Omega^\top=-\Omega,
$$

$$
R=-\frac{K+K^\top}{2}.
$$

若 $R\succ0$，则：

$$
\frac{d}{dt}\frac12\|z\|^2=-z^\top Rz.
$$

因此：

$$
\boxed{
\Omega:\text{ transport / rotation / conservative component}
}
$$

$$
\boxed{
R:\text{ dissipation / diffusion component}.
}
$$

动力学异质性因此可以分成：

$$
\boxed{\text{how information moves}}
$$

和

$$
\boxed{\text{how information dissipates}}.
$$

---

# 6. Dynamics Manifold

一个自然的合法 operator space 为：

$$
\boxed{
\mathcal M_{\rm dyn}
=
\mathfrak{so}(r)
\times
\mathbb S_{++}^{r}.
}
$$

其中：

- $\Omega\in\mathfrak{so}(r)$；
- $R\in\mathbb S_{++}^{r}$。

若客户端存在多个 generator，例如 self / neighbor：

$$
K_{m,s},\quad K_{m,n},
$$

则：

$$
\boxed{
\mathcal M_{\rm client}
=
\mathcal M_{\rm dyn}
\times
\mathcal M_{\rm dyn}.
}
$$

后续应明确区分：

$$
\boxed{\text{manifold}:\text{规定合法 dynamics}}
$$

与

$$
\boxed{\text{flow metric}:\text{衡量 dynamics knowledge gap}.}
$$

---

# 7. 不能以唯一 global K 为最终目标

服务器可以计算 barycenter $K_*$ 作为 baseline/reference，但正式方法不能要求：

$$
K_m\rightarrow K_*.
$$

否则只是把 FedAvg 从参数空间搬到 operator space。

正式方法应输出：

$$
\boxed{
K_{1,\mathrm{tar}},\ldots,K_{M,\mathrm{tar}}
}
$$

且一般：

$$
\boxed{
K_{m,\mathrm{tar}}\neq K_{n,\mathrm{tar}}.
}
$$

服务器的角色应从“平均器”变成：

$$
\boxed{\text{dynamics knowledge field builder / router}.}
$$

---

# 8. Personalized target 不能退化成 similarity-weighted average

禁止把正式方法写成：

$$
K_{m,\mathrm{tar}}=\sum_nw_{mn}K_n,
$$

再令：

$$
w_{mn}\propto e^{-d(K_m,K_n)}.
$$

这种设计仍然只是 similarity aggregation。

真正的问题是：

$$
\boxed{
\text{client }n\text{ 是否拥有 client }m\text{ 当前缺失的 dynamics mode？}
}
$$

---

# 9. Directional Transferable Knowledge

定义从 client $n$ 流向 client $m$ 的知识作用：

$$
\boxed{
F_{m\leftarrow n}
=
\Gamma_{m\leftarrow n}
\operatorname{Log}_{K_m}(K_n).
}
$$

其中真正需要设计的是：

$$
\boxed{\Gamma_{m\leftarrow n}.}
$$

它表示：

> client $n$ 的 dynamics 对 client $m$ 当前是否存在可迁移、互补、可靠的知识。

必须允许：

$$
\boxed{
\Gamma_{m\leftarrow n}\neq\Gamma_{n\leftarrow m}.
}
$$

knowledge flow 本质上是方向性的。

---

# 10. Missing Dynamics Projector

不能把整个：

$$
K_n-K_m
$$

直接传给 client $m$。

需要定义：

$$
\boxed{
\mathcal P_m^{\rm miss}
:
T_{K_m}\mathcal M
\rightarrow
T_{K_m}\mathcal M
}
$$

只保留：

> client $m$ 当前缺失、而 client $n$ 可靠掌握的 dynamics components。

于是：

$$
\boxed{
\Delta K_{m\leftarrow n}^{\rm transferable}
=
\mathcal P_m^{\rm miss}
\left[
\operatorname{Log}_{K_m}(K_n)
\right].
}
$$

个性化目标可抽象为：

$$
\boxed{
K_{m,\mathrm{tar}}
=
\operatorname{Exp}_{K_m}
\left(
\sum_{n\neq m}
\Gamma_{m\leftarrow n}
\mathcal P_m^{\rm miss}
[
\operatorname{Log}_{K_m}(K_n)
]
\right).
}
$$

其含义是：

$$
\boxed{
\text{personalized target}
=
\text{own dynamics}
+
\text{complementary external dynamics}.
}
$$

---

# 11. 联邦过程本身也是一个动力系统

引入联邦知识交换时间 $\tau$。

要区分：

$$
t:\text{ node-state evolution time},
$$

$$
\tau:\text{ federated knowledge evolution time}.
$$

外层 operator dynamics：

$$
\boxed{
\frac{dK_m}{d\tau}
=
F_m^{\rm local}
+
F_m^{\rm fed}.
}
$$

其中：

$$
F_m^{\rm local}
=
-\operatorname{grad}U_m(K_m)
$$

保持本地 individuality；

$$
F_m^{\rm fed}
=
\sum_{n\neq m}
F_{m\leftarrow n}
$$

负责吸收外部 knowledge。

因此整个方法可以理解为：

$$
\boxed{
\text{Dynamics of states}
+
\text{Dynamics of dynamics}.
}
$$

---

# 12. Knowledge Potential 不是普通距离

距离：

$$
d(K_m,K_n)
$$

只能说明：

> 两个 operator 差多少。

Knowledge potential 应回答：

> 这部分差异里，有多少是 client $m$ 可吸收、值得吸收的知识。

因此：

$$
\boxed{
\Phi_{m\leftarrow n}
=
\Phi(
K_m,
K_n,
\text{confidence},
\text{missing modes},
\text{task utility}
).
}
$$

不能直接令：

$$
\Phi=d.
$$

特别要避免：

$$
\boxed{
\text{large distance}
\Rightarrow
\text{large transferable knowledge}
}
$$

这种错误等价。

---

# 13. Heterogeneity Equilibrium

最终 equilibrium 不能定义成：

$$
K_m=K_n
$$

或：

$$
K_m=K_*.
$$

正确停止条件是：

$$
\boxed{
\|F_m^{\rm fed}\|
\le\varepsilon,
\qquad
\forall m.
}
$$

即：

> 对每个客户端，外部 dynamics field 中已经没有显著值得继续吸收的知识。

最终完全允许：

$$
\boxed{
K_m^\star\neq K_n^\star.
}
$$

但：

$$
\boxed{
F_m^{\rm fed}\rightarrow0.
}
$$

这就是：

$$
\boxed{
\text{heterogeneous equilibrium rather than complete synchronization}.
}
$$

---

# 14. 为什么 knowledge flow 会自然停止

定义剩余可迁移知识：

$$
\Delta_m^{\rm know}
=
\|F_m^{\rm fed}\|.
$$

随着客户端逐渐吸收互补 dynamics modes：

$$
\mathcal P_m^{\rm miss}
$$

提取到的有效分量会减少：

$$
\Delta_m^{\rm know}\downarrow.
$$

最终：

$$
\Delta_m^{\rm know}<\varepsilon
$$

则：

$$
\boxed{
F_m^{\rm fed}=0.
}
$$

因此 federated dependence 不是人工 schedule：

```text
alpha: 0.5 → 0.4 → 0.3 ...
```

而是：

$$
\boxed{
\text{knowledge potential naturally vanishes as missing knowledge is absorbed}.
}
$$

---

# 15. Local individuality 必须通过本地势能保留

如果只有外部知识流，客户端仍可能被拉向公共区域。

因此定义 local fidelity：

$$
\boxed{
U_m(K)
=
\frac{\lambda_m}{2}
d_{\rm dyn}^2
(
K,
\widehat K_m
).
}
$$

其中：

$$
\widehat K_m
$$

是当前本地数据识别出的 dynamics operator。

总能量：

$$
\boxed{
\mathcal E_m(K)
=
U_m(K)
+
\Phi_m^{\rm fed}(K).
}
$$

外层 flow：

$$
\boxed{
\frac{dK_m}{d\tau}
=
-
\operatorname{grad}
\mathcal E_m(K_m).
}
$$

最终 equilibrium 是：

$$
\boxed{
\operatorname{grad}U_m
+
\operatorname{grad}\Phi_m^{\rm fed}
=
0,
}
$$

而不是 operator consensus。

---

# 16. 为什么共享 dynamics operator 能缓解异质性

假设真实客户端 dynamics：

$$
K_m=K_c+\Delta_m,
$$

其中：

- $K_c$：transferable common dynamics；
- $\Delta_m$：client-specific individuality。

有限本地数据只能得到：

$$
\widehat K_m=K_m+\xi_m,
$$

其中：

$$
\xi_m
$$

是 estimation noise。

完全 Local：

$$
\boxed{\text{保留 individuality，但 estimation variance 高。}}
$$

完全 Global：

$$
\boxed{\text{降低 variance，但引入 heterogeneity bias。}}
$$

Personalized dynamics flow 的目标是：

$$
\boxed{
\text{variance reduction}
+
\text{personalized bias preservation}.
}
$$

---

# 17. 最简单的 bias–variance 解释

简化为：

$$
\widetilde K_m
=
(1-\eta_m)\widehat K_m
+
\eta_mK_c.
$$

有：

$$
\widetilde K_m-K_m
=
(1-\eta_m)\xi_m-\eta_m\Delta_m.
$$

因此：

$$
\boxed{
\mathbb E
\|\widetilde K_m-K_m\|^2
=
(1-\eta_m)^2\sigma_m^2
+
\eta_m^2
\|\Delta_m\|^2.
}
$$

解释：

- 本地 operator uncertainty 大 → 更需要外部知识；
- local individuality 大 → 更应减少同步；
- 随着本地知识变完整、$\sigma_m^2$ 下降 → 对 federation 的依赖应减弱。

这为后续 adaptive knowledge-flow strength 提供了理论动机。

---

# 18. 目标不是消灭异质性，而是利用异质性

不要表述为：

> reduce all client heterogeneity.

更准确：

$$
\boxed{
\text{extract transferable knowledge from heterogeneous dynamics}.
}
$$

抽象分解：

$$
\boxed{
K_n-K_m
=
\Delta_{n\rightarrow m}^{\rm transferable}
+
\Delta_{mn}^{\rm individuality}.
}
$$

联邦只传：

$$
\Delta_{n\rightarrow m}^{\rm transferable}.
$$

保留：

$$
\Delta_{mn}^{\rm individuality}.
$$

因此：

> **异质性不仅是困难，也是 complementary knowledge 的来源。**

---

# 19. 后续方法的三个核心数学对象

## 19.1 Dynamics Manifold

$$
\boxed{\mathcal M_{\rm dyn}}
$$

回答：

> 一个合法、稳定的 dynamics operator 应位于什么空间？

---

## 19.2 Directional Knowledge Potential

$$
\boxed{\Gamma_{m\leftarrow n}}
$$

回答：

> client $n$ 对 client $m$ 到底有多少可迁移 dynamics knowledge？

必须：

- directional；
- uncertainty-aware；
- mode-aware；
- task/dynamics-fidelity-aware。

---

## 19.3 Personalized Operator Flow

$$
\boxed{
\frac{dK_m}{d\tau}
=
F_m^{\rm local}
+
F_m^{\rm fed}.
}
$$

回答：

> client $m$ 的 dynamics knowledge 在联邦训练中如何演化？

终点：

$$
\boxed{
F_m^{\rm fed}\rightarrow0.
}
$$

而不是：

$$
K_m\rightarrow K_*.
$$

---

# 20. 当前最完整的高层公式

后续可以围绕：

$$
\boxed{
\frac{dK_m}{d\tau}
=
-
\operatorname{grad}U_m(K_m)
+
\sum_{n\neq m}
\Gamma_{m\leftarrow n}
\,
\mathcal P_m^{\rm miss}
\left[
\operatorname{Log}_{K_m}(K_n)
\right].
}
$$

其中：

- $U_m$：local dynamics fidelity；
- $\Gamma_{m\leftarrow n}$：directional transferable knowledge potential；
- $\mathcal P_m^{\rm miss}$：missing-dynamics projector；
- $\operatorname{Log}_{K_m}(K_n)$：operator-space knowledge direction。

这是当前后续方法设计的主数学骨架。

---

# 21. 离散联邦实现骨架

工程不需要真的积分 $\tau$。

先算：

$$
V_m^r
=
\sum_{n\neq m}
\Gamma_{m\leftarrow n}^r
\mathcal P_m^{\rm miss}
\left[
\operatorname{Log}_{K_m^r}(K_n^r)
\right].
$$

然后：

$$
\boxed{
K_m^{r+1,0}
=
\operatorname{Exp}_{K_m^r}
\left(
\eta_m^rV_m^r
\right).
}
$$

若：

$$
\|V_m^r\|\le\varepsilon,
$$

则：

$$
\boxed{
K_m^{r+1,0}=K_m^r.
}
$$

表示本轮不再需要 federated dynamics injection。

---

# 22. 服务器的最终角色

服务器不应是：

$$
\boxed{\text{global model constructor}.}
$$

而是：

$$
\boxed{\text{dynamics knowledge router}.}
$$

其职责：

```text
collect local operators
        ↓
analyze dynamics geometry
        ↓
identify complementary modes
        ↓
estimate directional transfer potential
        ↓
construct personalized operator flow
        ↓
send client-specific dynamics updates
```

---

# 23. 后续设计的禁止项

## 禁止 1：唯一 global operator 作为正式目标

$$
K_m\rightarrow K_*.
$$

只能作为 baseline。

## 禁止 2：只用 Frobenius/cosine similarity 定义聚合权重

这不是 dynamics knowledge flow。

## 禁止 3：把大距离解释成大知识量

差异大不代表可迁移价值大。

## 禁止 4：假设 knowledge transfer 对称

一般：

$$
\Gamma_{m\leftarrow n}\neq\Gamma_{n\leftarrow m}.
$$

## 禁止 5：把 equilibrium 定义成 consensus

正确的是：

$$
\|F_m^{\rm fed}\|\rightarrow0.
$$

## 禁止 6：为了 ACC 增加与 operator knowledge 无关的复杂模块

除非它明确改善：

- operator identification；
- operator geometry；
- missing-mode detection；
- transferable potential；
- personalized knowledge flow。

---

# 24. 后续研究优先级

### P1：Nonlinear Local Dynamics

先明确：

$$
F_m(H,A)
$$

是什么。

### P2：Dynamics Manifold

确定：

$$
K_m
$$

的稳定性结构与几何。

### P3：Dynamics Distance

定义真正反映 evolution behavior 的：

$$
d_{\rm dyn}.
$$

### P4：Directional Knowledge Potential

设计：

$$
\Gamma_{m\leftarrow n}.
$$

### P5：Missing Dynamics

设计：

$$
\mathcal P_m^{\rm miss}.
$$

### P6：Equilibrium

理论和实验验证：

$$
F_m^{\rm fed}\rightarrow0.
$$

---

# 25. 后续实验必须验证的现象

不能只报告 ACC。

至少要验证：

1. $K_m$ 确实描述 local dynamics；
2. 客户端之间存在可解释 dynamics heterogeneity；
3. transfer 具有方向性；
4. partial/mode-wise transfer 优于 full operator replacement；
5. $\|F_m^{\rm fed}\|$ 随训练下降；
6. 最终 $K_m^\star\neq K_n^\star$ 仍可成立；
7. 但 $\|F_m^{\rm fed}\|\approx0$；
8. 随知识饱和，外部依赖与通信需求下降。

---

# 26. 推荐论文 Storyline

## Motivation

Feature and structural heterogeneity jointly shape how node representations evolve over local graphs. Treating them independently misses their interaction during propagation.

## Local dynamics

Each client is modeled as a graph dynamical subsystem:

$$
\dot H_m=F_m(H_m,A_m).
$$

## Compact dynamics knowledge

A low-dimensional Koopman-style generator $K_m$ summarizes the client-specific evolution law.

## Operator geometry

Rather than treating $K_m$ as an ordinary matrix, we model it on a structured dynamics manifold and compare clients by their induced evolution flows.

## Personalized knowledge flow

The server does not force all clients toward one global operator. It identifies directional complementary dynamics and constructs a personalized operator flow for each client.

## Equilibrium

As useful external dynamics are absorbed:

$$
F_m^{\rm fed}\rightarrow0.
$$

The system reaches a heterogeneous equilibrium while preserving client individuality.

---

# 27. 一句话总纲

$$
\boxed{
\text{We do not synchronize heterogeneous clients;}
\quad
\text{we let complementary dynamics knowledge flow until no useful transfer remains.}
}
$$

中文：

> **我们不是把异质客户端同步成同一个模型，而是让互补的动力学知识在客户端之间定向流动，直到不存在显著可迁移知识，从而形成保留个性的异质性平衡。**

---

# 28. 后续方法设计必须回答的五个问题

任何新模块加入前必须回答：

1. 它刻画的是哪一种 dynamics knowledge？
2. 它为什么位于当前 operator geometry 中？
3. 它解决的是“差异”还是“可迁移知识”？
4. 它是否保持 knowledge flow 的方向性和个性化？
5. 它是否支持最终 $F_m^{\rm fed}\rightarrow0$ 的 equilibrium？

如果回答不了：

> **不要加入主方法。**

---

# 29. 最终研究定位

本方向不是：

```text
graph FedAvg + ODE
```

也不是：

```text
Koopman + similarity aggregation
```

而是：

$$
\boxed{
\textbf{Personalized Dynamics Knowledge Flow for Graph Federated Learning}
}
$$

核心对象：

$$
\boxed{K_m}
$$

核心空间：

$$
\boxed{\mathcal M_{\rm dyn}}
$$

核心驱动力：

$$
\boxed{
\Gamma_{m\leftarrow n}
\mathcal P_m^{\rm miss}
[
\operatorname{Log}_{K_m}(K_n)
]
}
$$

核心终点：

$$
\boxed{
\|F_m^{\rm fed}\|\rightarrow0
\quad
\text{while}
\quad
K_m^\star\neq K_n^\star.
}
$$

这就是后续所有具体算法设计应围绕的主轴。
