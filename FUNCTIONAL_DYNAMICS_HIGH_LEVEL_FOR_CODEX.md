# Functional-Dynamics GFL：高层设计说明（Codex 快速上手版）

## 1. 这项工作要解决什么

当前任务是**异质图联邦学习**。不同客户端不仅节点特征分布不同，图结构、同配度、类别组成也不同，因此每个客户端中的节点表示会形成不同的传播轨迹。

对客户端 $m$，把 A-DGN 写成：

$$
H_m(0)=H_m^0,
$$

$$
\dot H_m(t)=F_m(H_m(t),\mathcal G_m).
$$

其中：

- $H_m^0$ 体现 feature-side initial condition；
- $\mathcal G_m$ / $P_m$ 改变本地 graph vector field；
- 二者共同决定客户端自己的 representation trajectory。

传统 FedAvg 只做：

$$
\theta^{r+1}
=
\sum_m p_m\theta_m^r,
$$

它平均的是模型参数，但没有显式利用：

> **不同客户端到底学到了怎样的传播规律，以及这些传播规律如何跨异质图系统迁移。**

本项目的核心目标不是替代 FedAvg，而是在 FedAvg 之外增加一条**低维动力学知识通道**。

---

# 2. 核心思想：Extract → Translate → Share → Complement

整个算法只需要记住四个词：

$$
\boxed{
\text{Extract}
\rightarrow
\text{Translate}
\rightarrow
\text{Share}
\rightarrow
\text{Complement}
}
$$

### Extract：提取本地传播规律

从 client $m$ 的真实 A-DGN trajectory：

$$
H_m^0
\rightarrow
H_m^1
\rightarrow
\cdots
\rightarrow
H_m^K
$$

提取一个低维 generator：

$$
A_m\in\mathbb R^{r\times r}.
$$

直观上：

> $A_m$ 是“client $m$ 的节点表示通常怎么演化”的低维摘要。

### Translate：把不同客户端的动力学翻译到公共坐标

不同客户端的低维 functional coordinates 不一致，因此不能直接比较：

$$
A_m-A_n.
$$

客户端学习 Functional Map：

$$
C_m:\mathcal F_m\rightarrow\mathcal F_*,
$$

把本地动力学 transport 到 canonical functional space：

$$
\widetilde A_m
=
C_mA_mC_m^{-1}.
$$

Functional Map 的核心目标不是静态 embedding 对齐，而是保持动力学：

$$
\boxed{
A_*C_m
\approx
C_mA_m.
}
$$

### Share：服务器只聚合“翻译后”的传播规律

服务器不聚合原始 $A_m$，而只聚合已经进入公共坐标的：

$$
\widetilde A_m.
$$

因此新增联邦知识是：

$$
\boxed{
\text{传播规律，而不是节点表示。}
}
$$

### Complement：公共传播经验再反哺本地

服务器得到公共动力学 prototype 后，client $m$ 将其 pull back：

$$
A_{*\rightarrow m}
=
C_m^{-1}A_*C_m.
$$

它不替代 native A-DGN，而只形成一个低秩、稳定受控的额外动力学输入。

所以总体思想是：

$$
\boxed{
\text{保留 native graph dynamics}
+
\text{补充可迁移的 federated dynamics}.
}
$$

---

# 3. 本地动力学是怎么提取的

## 3.1 不做 Laplacian 特征分解

大图上不依赖 Laplacian eigenvectors。

每轮本地训练结束后，关闭 federated correction，额外做一次 `no_grad` 的 native A-DGN forward，采集纯本地轨迹：

$$
H_m^0,\ldots,H_m^K.
$$

取固定时间切片，例如：

$$
0,\quad K/4,\quad K/2,\quad K.
$$

使用共享随机 probe：

$$
R\in\mathbb R^{d\times q}
$$

构造 trajectory snapshots：

$$
X_m^{snap}
=
[
H_m^0R,\,
H_m^{K/4}R,\,
H_m^{K/2}R,\,
H_m^KR
].
$$

对这个窄矩阵做 thin QR：

$$
\boxed{
Q_m
=
\operatorname{orth}(X_m^{snap})
}
$$

得到：

$$
Q_m\in\mathbb R^{n_m\times r}.
$$

$Q_m$ 是由真实传播轨迹诱导出的低维 functional basis。

## 3.2 在这个空间里拟合低维 generator

投影：

$$
Z_m^k
=
Q_m^\top H_m^k.
$$

假设局部宏观 dynamics 可以近似写成：

$$
\dot Z_m
\approx
A_mZ_m.
$$

通过带 Tikhonov / ridge 正则的最小二乘拟合 $A_m$。

重点：

- $A_m$ 不是网络参数；
- 它是从本轮真实传播轨迹估计出来的动力学摘要；
- 必须持续记录 `DynFitErr`，确认这个低维表示确实能解释本地 trajectory。

---

# 4. Functional Map 在这里具体做什么

客户端还需要一个 trajectory descriptor：

$$
B_m
=
Q_m^\top X_m^{snap}.
$$

它不要求类别一一对应，主要用于给 functional coordinates 建立对应关系。

客户端拿到服务器广播的公共参照：

$$
(A_*^{(c)},B_*^{(c)})
$$

后，求自己的 Functional Map $C_m$。

核心 objective 是：

$$
\boxed{
\mathcal L_{FM}
=
\lambda_{desc}
\|C_mB_m-B_*^{(c)}\|_F^2
+
\lambda_{dyn}
\|A_*^{(c)}C_m-C_mA_m\|_F^2
+
\text{regularization}.
}
$$

两项分别表示：

- descriptor correspondence：解决“坐标怎么对应”；
- dynamics intertwining：解决“这种坐标变换是否保持传播规律”。

### 当前实现的重要变化

Synthetic 上正交 Functional Map 可以工作得比较稳定，但 Cora 表明真实客户端的 dynamics 并不只是“同一种动力学的正交旋转”。

因此当前 Cora 版本允许**正则化的非正交 $C_m$**，并约束：

- singular values；
- condition number；
- near-orthogonality；
- ridge stability。

目标是允许 moderate scaling / deformation，但不允许 map 退化或病态。

---

# 5. 为什么服务器现在维护多个 dynamics prototype

早期版本只有一个：

$$
A_*.
$$

但 Cora 上 dynamics intertwining residual 很大，说明真实客户端不能都被一个公共 generator 良好解释。

因此当前设计允许：

$$
\boxed{
\{
(A_*^{(1)},B_*^{(1)}),
\ldots,
(A_*^{(K_p)},B_*^{(K_p)})
\}
}
$$

多个 dynamical prototypes。

直观上：

> 联邦系统中可能存在多个典型传播 regime，而不是一个万能 global dynamics。

### Bootstrap

第一次没有 prototype 时，根据 trajectory descriptor 的 Procrustes distance 对客户端做初始 K-medoids，建立若干 prototype。

### 后续 round

client 自己比较本地：

$$
(A_m,B_m)
$$

与各 prototype 的 Functional Map objective，选择最兼容的 prototype：

$$
c_m.
$$

为避免每轮反复跳 cluster，保留 hysteresis / switch margin。

服务器随后只在同一 prototype 内聚合 transport 后的 dynamics。

---

# 6. 服务器每轮实际做什么

服务器同时维护两套全局知识：

$$
\boxed{
\theta^r
}
$$

以及：

$$
\boxed{
\mathcal P^r
=
\{
A_*^{(c),r},
B_*^{(c),r}
\}_{c=1}^{K_p}.
}
$$

### 模型参数通道

完全保留原 FedAvg：

$$
\theta^{r+1}
=
\sum_m p_m\theta_m^r.
$$

### 动力学通道

client $m$ 先在本地完成 Functional Map，再上传公共坐标中的：

$$
\widetilde A_m
=
C_mA_mC_m^{-1},
$$

$$
\widetilde B_m
=
C_mB_m,
$$

以及 prototype ID：

$$
c_m.
$$

服务器按 $c_m$ 分桶，对每个 prototype 内的客户端做聚合：

$$
\bar A_c
=
\sum_{m\in c}p_{m|c}\widetilde A_m,
$$

$$
\bar B_c
=
\sum_{m\in c}p_{m|c}\widetilde B_m.
$$

然后 EMA 更新：

$$
A_*^{(c)}
\leftarrow
\rho_AA_*^{(c)}
+
(1-\rho_A)\bar A_c,
$$

$$
B_*^{(c)}
\leftarrow
\rho_BB_*^{(c)}
+
(1-\rho_B)\bar B_c.
$$

服务器下一轮广播：

$$
\theta^{r+1}
$$

和完整 prototype bank：

$$
\mathcal P^{r+1}.
$$

---

# 7. 当前 Cora 还做了一个关键处理：共享“shape”，保留本地“speed”

不同客户端 generator 的 norm 可能差很多。

如果直接共享完整 $A_m$，会把其他客户端的 propagation speed 也强行传过来。

因此当前 Cora 版本先计算：

$$
s_m
=
\|A_m\|_F,
$$

$$
\bar A_m
=
\frac{A_m}{s_m}.
$$

Functional Map 和 server prototype 主要在：

$$
\bar A_m
$$

上工作。

也就是：

$$
\boxed{
\text{server 共享“怎样传播”，client 保留“传播多快”。}
}
$$

公共 prototype 拉回 client 后，再乘回 client 自己的 $s_m$。

这个设计是为了减少真实客户端之间时间尺度差异造成的错误迁移。

---

# 8. 公共动力学如何进入下一轮本地 A-DGN

client $m$ 选中 prototype $c_m$ 后：

$$
A_{*\rightarrow m}
=
C_m^{-1}
A_*^{(c_m)}
C_m.
$$

结合本地 speed，构造动力学补充量。

但这个 correction 不直接注入，而先做稳定化。

把 correction 分成：

$$
\Delta A
=
\Delta A_{skew}
+
\Delta A_{sym}.
$$

保留反对称部分，对 symmetric part 只保留非正特征值，得到耗散版本：

$$
\Delta A^{diss}.
$$

再做 spectral norm clipping，得到：

$$
\boxed{
\Delta A_m^{safe}.
}
$$

最后 A-DGN 的每个 Euler step 变成：

$$
\boxed{
H_m^{k+1}
=
H_m^k
+
\Delta t
\left[
F_m(H_m^k,\mathcal G_m)
+
\beta
Q_m
\Delta A_m^{safe}
Q_m^\top
H_m^k
\right].
}
$$

其中：

- 第一项：原生 A-DGN；
- 第二项：低秩 federated dynamics correction。

因此整个方法不会替换 native graph dynamics。

---

# 9. Cached-Basis 时序

当前实现采用上一轮缓存，避免 forward 与 basis extraction 的循环依赖。

### Round $r$ 开始

client 使用上一轮缓存：

$$
Q_m^{r-1},
\quad
A_m^{r-1},
\quad
C_m^{r-1}
$$

和服务器最新 prototype bank，计算本轮：

$$
\Delta A_m^{safe,r}.
$$

### Local training

使用：

$$
\text{native field}
+
\text{cached federated correction}.
$$

### Round 末

关闭 correction，做一次 no-grad native forward：

$$
\text{trajectory}
\rightarrow
Q_m^r
\rightarrow
A_m^r
\rightarrow
B_m^r
\rightarrow
C_m^r.
$$

然后 transport 并上传：

$$
\widetilde A_m^r,
\qquad
\widetilde B_m^r.
$$

第 0 轮没有 cache，因此不注入 federated dynamics。

---

# 10. 当前方法最重要的工程边界

Codex 修改代码时必须遵守以下约束：

1. **FedAvg backbone 不改。**  
   动力学通道是新增旁路，不替代原始参数通信。

2. **不上传节点级表示。**  
   $H_m,Q_m,Z_m,A_m,C_m$ 均应保持本地。

3. **Functional Map 必须在 client 端求。**  
   server 只处理已经 transport 到 canonical space 的低维对象。

4. **大图不做 Laplacian eigendecomposition。**  
   functional basis 来自 trajectory snapshots + thin QR。

5. **本地 generator 必须来自 native trajectory。**  
   calibration forward 关闭 federated correction，避免循环自证。

6. **correction 必须可关闭。**  
   `beta=0` 或关闭 functional-dynamics 后应严格退化到原 FedAvg + A-DGN。

7. **稳定层不能删。**  
   correction 需要 dissipative projection + spectral norm clipping。

8. **隐私表述保持克制。**  
   当前代码是 privacy-compatible / simulated secure aggregation 设计，不是正式 DP 或密码学隐私证明。

---

# 11. 当前实验告诉了我们什么

### Synthetic

目前 trajectory basis 和 Functional Map 对齐较稳定：

- descriptor residual 低；
- dynamics intertwining residual 低；
- basis overlap 接近 1；
- federated injection 后期逐渐变成小修正。

说明：

$$
\boxed{
\text{Extract → Translate → Share → Complement}
}
$$

至少在受控异质性下可以形成稳定闭环。

### Cora

Cora 上：

- local dynamics fitting 本身并不差；
- basis 也比较稳定；
- 但跨客户端 dynamics intertwining 明显更困难；
- correction 长期需要强 clipping。

这说明真实图中的异质性不只是 coordinate mismatch，而存在：

$$
\boxed{
\text{intrinsic dynamical heterogeneity}.
}
$$

因此才引入：

- multiple dynamics prototypes；
- regularized non-orthogonal Functional Map；
- generator shape/speed decoupling。

这些不是额外故事，而是在解决同一个问题：

> **真实客户端不能被强行解释成同一个公共动力学的简单坐标变换。**

---

# 12. Codex 从哪里入手

从代码结构上，优先定位以下逻辑，而不是先重构整个工程：

```text
1. A-DGN forward / trajectory recording
2. trajectory → Q_m
3. Q_m + trajectory → A_m
4. descriptor B_m
5. Functional Map C_m
6. local prototype selection
7. transport:
       A_tilde = C A C^{-1}
       B_tilde = C B
8. server:
       model FedAvg
       dynamics prototype aggregation + EMA
9. client:
       prototype pullback
       safe correction
10. A-DGN injection
```

如果需要改算法，优先保证这条主闭环不被破坏：

$$
\boxed{
\text{Native trajectory}
\rightarrow
\text{Local dynamics}
\rightarrow
\text{Functional transport}
\rightarrow
\text{Federated prototype}
\rightarrow
\text{Safe local complement}
\rightarrow
\text{Next trajectory}.
}
$$

---

# 13. 一句话理解整个项目

> **每个客户端是一个异质的图表示动力系统。我们从真实传播轨迹中提取低维演化规律，用 Functional Map 把不同客户端的动力学翻译到公共坐标，在服务器上形成若干公共传播模式，再把兼容的模式翻译回本地，以稳定的低秩附加动力补充 native A-DGN。**

最核心的区别是：

$$
\boxed{
\text{FedAvg 共享参数；本方法额外共享“经过翻译后的传播规律”。}
}
$$
