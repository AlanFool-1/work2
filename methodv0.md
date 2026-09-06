# Method v0.2：异质图联邦学习中的有限互补动力学知识吸收

## 当前优化骨架：无 FedAvg 的 Functional Action Flow（R012）

候选方法不上传、平均或广播模型参数。客户端 \(i\) 始终保留本地参数 \(\theta_i\)，服务器只维护 Functional Map 网络并路由低维 action pairs。FedAvg 只作为对照方法；本文后续历史章节中出现的 FedAvg 生命周期不属于 R012。

对可靠的有向边 \(j\to i\)，Functional Map 运输来源作用方程并产生交换缺陷 \(E_{j\to i}\)。其任务相容边电导为

\[
g_{j\to i}
=m_{j\to i}[\chi_{j\to i}]_+
\mathbf 1[\|E_{j\to i}^{\mathrm{heldout}}\|>\varepsilon_E],
\]

对应的知识流不是参数差 \(\theta_j-\theta_i\)，而是作用缺陷在接收方参数空间中的下降力

\[
I_{j\to i}
=-g_{j\to i}\nabla_{\theta_i}
\frac12\|E_{j\to i}(\theta_i)\|^2.
\]

客户端按

\[
\boxed{
\theta_i^{r+1}
=\theta_i^r
-\eta_i\nabla\mathcal L_i^{\mathrm{task}}
+\eta_i\sum_j I_{j\to i}
}
\]

更新。来源 action sketch 在一次吸收微步内冻结，随后随来源本地学习周期性刷新。该结构没有全局模型或参数凸组合；跨客户端作用只在方程可运输、未掌握且与本地任务相容时开启。方程学会后 \(E\to0\)，任务不相容时 \(g=0\)，当前模型不可实现时 \(\nabla_{\theta_i}\|E\|^2=0\)，三种情形都会使 \(I_{j\to i}\to0\)。系统的平衡是有效作用流归零，不是客户端参数达成共识。

## 当前骨架：Functional Map 运输下的动力学约束补全（R011）

当前交换对象是一组低维有限时域作用方程，而不是正交响应模态。客户端 \(j\) 在本地功能空间中测量

\[
Z_j\mapsto Y_j^\tau=\mathcal P_j^\tau(Z_j).
\]

Functional Map \(C_{j\to i}\) 把整条作用关系运输到客户端 \(i\)：

\[
C_{j\to i}Z_j
\mapsto
C_{j\to i}Y_j^\tau.
\]

接收方用真实本地 ODE-GNN 执行左侧输入。如果

\[
\boxed{
E_{j\to i}^\tau
=
\mathcal P_i^\tau(C_{j\to i}Z_j)
-
C_{j\to i}\mathcal P_j^\tau(Z_j)
\ne0,
}
\]

则“先运输后传播”与“先传播后运输”不交换。只有当映射可信、该方程可以由本地反向传播学会、且更新与本地任务相容时，它才构成客户端 \(i\) 的互补知识。

多客户端协作是作用方程的累积与满足，不生成全局模型、prototype 或平均 operator。接收方通过普通 task loss 加 action-pair distillation 学习这些方程；不在线构造完整响应 Jacobian、不做正交缺失子空间，也不每轮运行 local-only counterfactual。当所有可信外部方程已经满足或不再对本地任务有益时，知识流停止。

Functional Map 在这里是跨客户端函数空间的运输边，并通过 map-network cycle consistency 约束；它不再只是把本地 generator 对齐到 canonical prototype 的辅助工具。为了避免循环验证，第一版用 descriptor/cycle split 估计映射，在独立 action probes 和 horizons 上测交换缺陷。

完整规范、理论边界、代码复用关系和最小实验见 .collab/FUNCTIONAL_INTERTWINING_DYNAMICS.md。以下 R010 正交响应子空间方案已经降为历史候选。

## 历史 R010：低秩响应算子与正交缺失模态（已被 R011 修正）

本方法尚不能直接进入边际增益或完整联邦实验。必须先证明所交换的低维对象能够表示本地系统在共同干预下的有限时域动力学，并能识别“来源客户端拥有、接收客户端缺失”的响应模态。旧的自治 ridge generator、与其匹配的离散 \(P=I+hA\)、prototype、重心或参数均值都没有通过该门槛。

R010 当时的首选候选是低秩有限时域因果响应算子

\[
\delta y_i=\mathcal R_i u_i+O(\|u_i\|^2),
\qquad
\mathcal R_i\approx U_i\Sigma_iV_i^\top,
\]

其中输入轴由公共端口和施加时刻定义，输出轴由公共任务观测和观测时刻定义。它描述真实 ODE-GNN 如何把干预传播为任务相关响应，而不是用一个自治线性模型重建自己的隐状态轨迹。

跨客户端组合采用接收方条件化的子空间补全：将来源响应模态投影到接收方已有算子子空间的正交补，合并非冗余创新，再通过接收方的本地可实现性与相对 local-only 的边际任务增益筛选。它不构造公共重心，也不要求所有客户端最后一致。随着新模态被吸收，接收方创新残差下降；所有可实现且有益的创新耗尽时，知识流停止。

完整定义、候选比较、新颖性边界和淘汰实验见 `.collab/LOW_DIMENSIONAL_DYNAMICS_OPERATOR_SELECTION.md`。在已知互补模态与冻结真实 checkpoint 上通过留出响应保真、缺失模态恢复和接收方实际增益三项门槛之前，本文后续优化方程均应视为**条件于代理选择成立**，不得直接实现为 production 算法。

## Agent 快速迁移入口：边际增益耗尽原则（2026-09-06）

### 学习过程前提

联邦学习的目标是让每个客户端借助协作成为更充分的本地学习者。固定任务、客户端集合、模型容量和通信接口后，客户端 $i$ 能从其他客户端吸收的任务相关互补知识具有有限价值。训练初期，本地知识缺口较大，外部知识带来的边际增益较高；随着互补知识被吸收，剩余可学习增益逐渐下降；当任何外部可实现方向都不能比等预算的本地学习带来额外收益时，客户端达到联邦自足，跨客户端知识流应停止。

严格地说，知识池会随其他客户端的学习而变化，因此“知识总量固定”是一个阶段性有界假设，不是所有非凸联邦训练中的无条件事实。算法每轮重新估计剩余增益，不预设一条随轮数衰减的流量曲线。

核心链条修正为：

\[
\boxed{
\text{客户端异质性}
\rightarrow
\text{互补知识供给}
\rightarrow
\text{本地可学习增益}
\rightarrow
\text{动力学知识吸收}
\rightarrow
\text{剩余增益衰减}
\rightarrow
\text{联邦自足平衡}
}
\]

最终状态不要求客户端响应一致。仅当可迁移且对本地任务有益的动力学差异已经被吸收、不可实现或不再提供额外收益时，知识流归零；任务必要差异可以保留。

### 外部边际增益作为离线机制 estimand

R011 不在每轮显式求解响应 Jacobian 或运行 local-only 分叉。以下 \(G_i\) 保留为早/中/晚 checkpoint 的离线机制审计量，用来检验联邦作用方程相对等预算本地学习是否真的提供额外价值。

令 \(\mathcal V_i^{\mathrm{loc}}\) 是客户端用相同计算预算能够产生的本地更新集合，\(\mathcal V_i^{\mathrm{fed}}(x_{-i})\) 在其中加入由外部 action-pair distillation 产生的本地更新方向。显然

\[
\mathcal V_i^{\mathrm{loc}}
\subseteq
\mathcal V_i^{\mathrm{fed}}(x_{-i}).
\]

在当前冻结快照上，用训练内部 guard 数据构造局部任务模型

\[
q_i(v)=g_i^Tv+\frac12v^TH_iv+\frac{\lambda}{2}\|v\|^2.
\]

定义客户端当前可从联邦协作获得的预测边际增益：

\[
\boxed{
G_i
=
\min_{v\in\mathcal V_i^{\mathrm{loc}}}q_i(v)
-
\min_{v\in\mathcal V_i^{\mathrm{fed}}(x_{-i})}q_i(v)
\ge0.
}
\]

$G_i$ 衡量外部知识相对等预算本地学习的额外价值。响应距离大但 $G_i=0$ 时不应交换；响应距离已经很小但仍存在可靠正增益时仍可交换。由此，旧 v0 的响应方差 $\mathcal E_{\mathrm{het}}$ 降为描述性诊断，不再单独决定流量。

候选联邦更新 $v_i^{\mathrm{fed}}$ 必须与等预算本地更新 $v_i^{\mathrm{loc}}$ 在同一冻结快照上真实回放。实现后的边际增益定义为

\[
\widehat G_i
=
L_i^{\mathrm{guard}}(\theta_i+\alpha v_i^{\mathrm{loc}})
-
L_i^{\mathrm{guard}}(\theta_i+\alpha v_i^{\mathrm{fed}}).
\]

在线时使用作用交换缺陷、方程梯度与任务梯度的一阶相容性以及本地 guard；只有离线审计才计算 \(\widehat G_i\)。外部流量由尚未满足且任务相容的作用方程决定，而不是由固定 injection coefficient 或轮数衰减表决定。

### Functional Map 运输的作用方程提供什么

客户端 \(j\) 在本地功能空间中测量少量有限时域 action pairs：

\[
Z_j\mapsto Y_j^\tau=\mathcal P_j^\tau(Z_j).
\]

Functional Map 把输入与演化结果一起运输到接收方。接收方在自己的真实 ODE-GNN 中执行 \(C_{j\to i}Z_j\)，比较

\[
\mathcal P_i^\tau(C_{j\to i}Z_j)
\quad\text{与}\quad
C_{j\to i}\mathcal P_j^\tau(Z_j).
\]

互补性由一条外部方程是否尚未满足、能否通过普通本地反向传播学会、以及是否与本地任务一致来定义。因果响应核可以生成 action pairs 或作为离线诊断，但不再被 SVD 成正交“缺失模态”。

这形成三重筛选：

1. **可运输**：descriptor/cycle 估计的 Functional Map 在留出数据上可靠；
2. **可学习**：普通本地更新能降低留出作用交换缺陷；
3. **有增益**：更新与本地任务梯度/guard 相容，并在离线审计中优于等预算 local-only。

### 平衡和有限增益的理论表述

联邦自足平衡定义为

\[
\boxed{
G_i(\theta_i^*,x_{-i}^*)\le\varepsilon_G,
\qquad \forall i.
}
\]

它表示没有客户端还能从当前外部知识池得到显著的额外可学习收益。若每个被接受的外部更新都带来非负实际任务下降 $\delta_i^r$，本地目标有下界，并且接受规则保证

\[
L_i(\theta_i^r)-L_i(\theta_i^{r+1})\ge\delta_i^r\ge0,
\]

则

\[
\sum_{r=0}^{\infty}\delta_i^r
\le
L_i(\theta_i^0)-L_i^{\inf}<\infty,
\qquad
\delta_i^r\rightarrow0.
\]

这给出了“外部可学习知识总价值有限，因此边际增益最终耗尽”的最小数学依据。它是条件性结论，不等价于非凸参数收敛或测试精度单调上升。

### 对昨晚 v0 的关键修正

知识可以被复制，发送方不会因为接收方学会而失去自身知识。因此，$\sum_i p_iJ_i=0$ 不能解释为知识质量守恒。它可以在需要时作为响应坐标去漂移、通信预算或数值稳定约束保留，但不再是流动停止的根本原因。流停止的判据改为 $G_i\rightarrow0$。

当前优化不再以“把所有客户端响应方差降到最低”为最终目标，而以“耗尽每个客户端仍可从外部获得的正边际任务增益”为目标。最终异质性应表述为：

> 可迁移且任务有益的动力学知识已被充分吸收；剩余差异在当前模型、任务和通信约束下不再具有可学习价值。

### 下一步实验

先执行已知 Functional Map 的 action-constraint pilot。合成系统显式包含已掌握、可迁移缺失和任务有害三类作用。比较 ground-truth、descriptor/cycle learned、wrong 和 identity maps；映射拟合与作用缺陷评估必须使用不同 probes/horizons。

接收方通过普通 action-pair distillation 学习外部方程。报告 map/cycle error、留出交换缺陷、缺陷下降、任务变化、梯度相容性、通信量和时间，并与旧 canonical prototype pullback、R010 正交模态方法及 matched local-only 审计对照。只有正确运输的可迁移缺失方程同时降低缺陷与任务损失，而重复、有害、wrong-map 和 shuffled 方程不能复现时，才进入冻结真实图 checkpoint。

## 昨晚 v0 的守恒响应流规格（历史推导，受以上原则修正）

### 文档与状态

- 本文件是**当前方法设计的规范入口**；`.collab/NEXT_TASK.md` 仍是当前执行人、阶段和下一动作的规范入口。
- 当前状态是 `NEEDS_RESEARCH`。算法骨架已经收敛，但因果响应核的任务语义、覆盖率、本地可实现性和有限步误差还没有通过研究门槛，因此尚未批准接入完整训练。
- 本节吸收了 `.collab/DYNAMICAL_RESPONSE_KERNEL_RESEARCH.md` 中晚于本文主体的修正。若本节与后文 2.2、10.1、10.5、10.8 的旧表述冲突，以本节为准。
- `.collab/DYNAMICAL_RESPONSE_KERNEL_RESEARCH.md`、`.collab/DYNAMICAL_RESPONSE_CHECKS.md` 和 `.collab/MULTISYSTEM_DYNAMICS_LITERATURE.md` 是推导、数值证据和文献边界，不是第二套算法规格。

### 一句话方法

> 冻结每个客户端的真实 ODE-GNN，用具有共同动力学语义的端口测量有限时间因果响应；只在各客户端真实模型能够实现、任务约束允许且全局一阶守恒的响应方向上交换知识，再用实际 rollout 检查响应变化、守恒漂移和任务行为。

主链条为：

\[
\boxed{
\text{真实图动力学}
\rightarrow
\text{端口--时间因果响应核}
\rightarrow
\text{联合可实现守恒流}
\rightarrow
\text{本地向量场更新}
\rightarrow
\text{实际响应与任务复测}
}
\]

它不使用客户端聚类、prototype、anchor、EMA target 或预设 shared/private 分支作为主机制。加权均值只作为响应能量的代数中心和完全可实现情况下的退化对照，不是服务器发布的公共动力学目标。

### 当前交换对象

客户端 $m$ 的原生离散传播写成

\[
h_{m,\ell+1}=\Phi_{m,\ell}(h_{m,\ell},u_{m,\ell}),
\qquad y_{m,t}=\mathcal O_m(h_{m,t}),
\]

并要求零端口严格恢复原生模型：

\[
\Phi_{m,\ell}(h,0)=\Phi^{\mathrm{native}}_{m,\ell}(h).
\]

固定端口类别、施加时刻 $s$、观测时刻 $t$ 和输出通道后，交换对象是实际 Euler/ODE 传播的一阶因果响应：

\[
\mathcal K_{m,t,s}
=
\left.
\frac{\partial\mathcal O_m(h_{m,t})}{\partial u_{m,s}}
\right|_{u=0},
\qquad
\mathcal K_{m,t,s}=0\quad(t\le s).
\]

初始端口族为初值幅度、邻居耦合和局部衰减。将具有相同端口、时间和观测语义的核块按交换前固定的尺度白化并向量化：

\[
x_m=\operatorname{whiten}\!\left(\operatorname{vec}
\{\mathcal K_{m,t,s}\}_{\text{port},s,t,\text{output}}\right).
\]

核使用 JVP/VJP 沿真实稀疏图传播计算，不构造稠密节点 Jacobian，不对大图矩阵做 SVD。旧的 $r\times r$ ridge generator、Functional Map 和直接 probe 轨迹只保留为诊断、运输备选或等预算对照。连续 $A$ 与满足 $P=I+hA$ 的 identity-centered 离散 $P$ 在匹配目标和正则化时是同一模型，不能把二者当作表达能力不同的候选。

### 联合可实现守恒流

固定一次交换快照，令

\[
\bar x=\sum_m p_mx_m,
\qquad e_m=x_m-\bar x,
\qquad D_m=D_{\theta_m}x_m,
\]

其中 $D_m$ 描述客户端原生向量场参数更新 $v_m$ 能造成的真实核变化。交换方向通过一个联合问题求解：

\[
\begin{aligned}
\min_{\{v_m\}}\quad
&\sum_m p_me_m^TD_mv_m
+\frac{1}{2\eta}\sum_mp_m\|v_m\|^2\\
\text{s.t.}\quad
&\sum_mp_mD_mv_m=0,\\
&g_m^Tv_m\le0,\qquad \|v_m\|\le b_m.
\end{aligned}
\]

第一条约束保证响应的一阶加权总量守恒；后两条分别近似本地任务可容许性和有限线性化范围。令 $J_m=D_mv_m$，则响应异质性能量

\[
\mathcal E=\frac12\sum_mp_m\|x_m-\bar x\|^2
\]

在理想连续交换中满足一阶非增。无任务约束时，令 $M_m=D_mD_m^T$，可由

\[
\left(\sum_mp_mM_m\right)\mu
=\sum_mp_mM_me_m,
\qquad
J_m=-\eta M_m(e_m-\mu)
\]

得到

\[
\sum_mp_mJ_m=0,
\qquad
\frac{d\mathcal E}{d\tau}
=-\eta\sum_mp_m\|D_m^T(e_m-\mu)\|^2\le0.
\]

如果客户端之间不存在共同可实现的守恒下降方向，最优流允许为零，系统保留非共识差异。若显式采用客户端边通信，必须在同一个联合问题中加入反对称边流 $j_{mn}=-j_{nm}$；不能先在摘要空间求均值流，再让客户端独立裁剪或回写。

### 一轮算法

1. 在 global broadcast 后、local training 前冻结客户端模型、图、readout、端口协议和响应尺度，并保存 native snapshot。
2. 验证所有端口在零幅度时逐步等价于原生 forward。
3. 客户端用 JVP 计算带端口、$s$、$t$ 和输出语义的因果核，形成 $x_m$；节点状态、节点基和参数 Jacobian 留在本地。
4. 检查未来到过去响应为零、AD/中心差分一致性、节点重排不变性、非线性余项、端口覆盖和任务观测支持。不通过的响应块不进入交换。
5. 客户端提供 $D_mv$ 与 $D_m^Tw$ 的算子接口或小规模 oracle 所需统计量；协调器联合求解守恒、任务可容许且有界的更新方向。
6. 所有客户端使用同一个试探缩放 $\alpha$ 更新本地向量场参数；不重建 persistent Adam 状态。
7. 在原冻结快照上重新 rollout，分别测量理想线性守恒、求解误差、实际有限步守恒漂移、实现残差、响应能量和本地 guard 指标。失败时整体缩步或拒绝该交换。
8. 接受交换后进入原有 local training、FedAvg 和 paired best-validation 评估；交换、本地学习、FedAvg/broadcast 和传感器刷新造成的变化分别记账。

### 固定边界、开放选择与下一步

已经固定：真实有限时间动力学响应是候选知识对象；端口和时间索引不能丢；交换必须由本地可实现方向决定；守恒只声明于冻结交换阶段；真实有限步误差必须复测。

仍然开放：任务观测 $\mathcal O_m$ 的充分语义、真实数据缺类协议、向量场参数方向库、核压缩方式，以及是否需要 NCF 式条件场或 MP-NODE 式增广消息。最近讨论的“最大公共动力学因子/sheaf global section”尚未被采用进 v0，不能由接手 agent 自行当成已定算法。

下一步只有一个：在冻结 synthetic 模型上完成响应识别和联合可实现性验证。推进门槛为 zero-port 等价、未来到过去核为零、float64 AD/中心差分相对误差目标小于 $10^{-6}$、有效幅度内非线性响应相对误差中位数不超过 `0.10` 且 90 分位不超过 `0.25`、线性实现相对残差不超过 `0.10`、归一化理想守恒残差不超过 $10^{-8}$，并观察到有限步漂移随缩步呈二阶下降。A/B/C 门槛通过后，才允许至少 3 个种子的短 synthetic FedAvg 接入实验。

接手 agent 启动时依次读取本节、`.collab/NEXT_TASK.md` 和本文相关细节；不得直接启动 11 数据集训练，也不得继续调 prototype、injection strength 或 correction clipping。

## 0. 文档定位

本文档给出当前工作的第一版方法机制。它先固定问题、知识对象、比较规则、守恒量、异质性能量和可实现的交换过程，再决定具体代码模块。Functional Map、低秩基、生成器拟合和局部注入都是实现工具，不能替代方法本身的因果解释。

本文档的中心命题是：

> 异质图客户端携带不同但可能互补的动力学知识。跨客户端协作应当把这些知识运输到可比较的功能坐标中，并沿动力学异质性势差进行守恒流动；流动只消除任务有害的差异，在本地动力学约束下达到最小异质性平衡态。

因此，整篇工作的高层链条固定为：

\[
\boxed{
\text{异质图动力系统}
\rightarrow
\text{动力学知识运输}
\rightarrow
\text{守恒知识流}
\rightarrow
\text{异质性能量耗散}
\rightarrow
\text{可容许的动力学平衡态}
}
\]

这里的贡献不是把普通参数平均重新写成 ODE，也不是先规定客户端必须趋同。关键问题是：什么动力学信息值得交换，交换过程为什么守恒，什么差异是有害的，以及交换如何真正改变客户端的图动力学。

## 1. 客户端是私有图动力系统

客户端 \(m\) 持有私有图 \(\mathcal G_m\)、初始表示 \(H_m^0\) 和本地向量场 \(F_m\)。将其写为：

\[
\mathcal S_m=(H_m^0,F_m;\mathcal G_m),
\qquad
\frac{dH_m(t)}{dt}=F_m(H_m(t),\mathcal G_m).
\]

特征异质性主要改变 \(H_m^0\)，结构异质性主要改变 \(F_m\) 及其图耦合项；两者会在有限时间表示轨迹中共同作用。因此，参数差异只是结果，不能直接作为要交换的知识定义。

联邦过程有两个时间尺度：

\[
t=\text{本地节点表示传播时间},
\qquad
\tau=\text{客户端之间动力学知识交换时间}.
\]

本地动力学沿 \(t\) 演化，联邦交换沿 \(\tau\) 改变各客户端下一轮可实现的动力学状态。守恒和耗散的理论陈述首先针对冻结本地学习的交换阶段；本地训练产生的知识源项必须单独记录。

## 2. 动力学知识是什么

### 2.1 私有动力学状态

客户端的私有动力系统仍抽象为：

\[
\Xi_m=(H_m^0,F_m;\mathcal G_m).
\]

它描述客户端从什么状态开始、在什么图上、按照什么向量场传播。低维生成器 \(A_m\) 可以作为压缩和诊断工具，但不再被默认定义为联邦交换的知识对象。原因是一个自治线性算子无法自动表达投影误差、非线性、时间变化和有限时间图响应。

### 2.2 公共 probe 下的有限时间传播响应

构造一个在交换前固定的公共 probe 集合：

\[
\mathcal P=\{P_q^c\}_{q=1}^{Q},
\qquad
P_q^c\in\mathcal F_c.
\]

客户端通过运输规则把公共 probe 拉回本地：

\[
H_{m,q}^{0}=\mathcal T_m^{-1}(P_q^c),
\qquad
H_{m,q}^{\ell+1}=\Phi_{m,\Delta t}(H_{m,q}^{\ell};\mathcal G_m),
\quad \ell=0,\ldots,L-1.
\]

其中 \(\Phi_{m,\Delta t}\) 是客户端真实采用的有限步图动力学传播。用客户端的排列不变观测算子 \(\mathcal O_m\) 把每个时刻的节点状态汇总，再运输回公共坐标：

\[
R_{m,q}^{\ell}
=
\mathcal T_m\!\left(\mathcal O_m(H_{m,q}^{\ell})\right),
\qquad
x_m=\operatorname{vec}\!\left(
\{R_{m,q}^{\ell}-R_{m,q}^{0}\}_{q=1,\ell=1}^{Q,L}
\right).
\]

因此，\(x_m\) 直接表示“同一个公共 probe 在客户端 \(m\) 的图动力系统中经过有限时间后如何传播”，而不是把初始状态摘要和生成器参数人工拼接。\(\mathcal O_m\) 可以使用节点均值、二阶矩或低秩功能系数，但必须在所有客户端使用相同的统计语义。初始 v0 使用固定随机 probe、节点均值和有限时间差分响应；不使用标签构造 probe，也不把客户端 ID 放入 \(x_m\)。

\(A_m\) 只用于诊断：比较它对 \(R_{m,q}^{\ell}\) 的重构误差。如果生成器误差很大，公共响应状态仍然可以直接计算；只有需要进一步压缩通信时，才比较 \(A_m\)、离散传播算子 \(P_m\) 或神经响应编码器。

### 2.3 为什么这种知识值得流动

知识流动的理由不是“平均后参数更接近”，而是同一类任务状态在不同客户端上经历了不同的传播规律。若某些差异使相同的任务相关状态在客户端之间产生不一致、不可迁移或不稳定的响应，它们会阻碍知识复用；若差异只反映标签覆盖、图结构或局部任务所需的特性，强行消除会损害本地任务。

因此，待交换对象必须同时满足三项检查：

1. 相同公共 probe 在所有客户端具有相同的输入语义；
2. 响应在公共功能坐标中可比较，且运输和观测误差可测；
3. 流量能通过本地模型更新改变后续 probe 响应，并由独立任务指标检验。

## 3. Functional Map 是知识运输规则

不同客户端的节点集合、图 Laplacian、基函数和表示坐标通常不同。即使 \(\Xi_m\) 与 \(\Xi_n\) 描述相似的动力学，也不能直接做减法。因此引入客户端到公共功能空间的运输：

\[
\mathcal T_m:\mathcal F_m\rightarrow\mathcal F_c,
\qquad
\widetilde\Xi_m=\mathcal T_m(\Xi_m).
\]

在当前代码体系中，Functional Map 是实现 \(\mathcal T_m\) 的候选工具。它的高层含义是：

\[
\boxed{\text{Functional Map 是动力学知识跨图流动的坐标运输规则。}}
\]

它不是一个独立的对齐模块，也不是为了让模型看起来相似。运输必须有以下可验证条件：

- **坐标条件**：同一公共探针在各客户端运输后具有一致语义；
- **可逆性条件**：在被交换的低维子空间上，运输和拉回误差受控；
- **规范条件**：符号、旋转、尺度等 gauge 差异被固定或显式计入；
- **隐私条件**：服务器只接收所需的聚合统计或受保护摘要，不接收节点级状态和轨迹基。

运输残差必须进入误差项，而不能被默认为零。若运输误差大于预设阈值，客户端应减少或拒绝该方向的知识流动。

## 4. 守恒动力学知识流

### 4.1 守恒量和守恒阶段

令 \(x_m\) 是运输后、经过知识观测的客户端状态，令 \(p_m\) 是固定的联邦权重。交换阶段定义净流入 \(J_m\)：

\[
\frac{dx_m}{d\tau}=J_m.
\]

第一版的守恒量是交换阶段的加权一阶矩：

\[
K(x)=\sum_m p_m x_m.
\]

要求：

\[
\boxed{\sum_m p_mJ_m=0}
\quad\Longrightarrow\quad
\frac{dK}{d\tau}=0.
\]

它表示交换过程只重新分配公共坐标中的动力学知识，不凭空制造一个未经客户端贡献的全局动力学状态。它不表示任务准确率、所有信息量或本地物理能量都守恒。

守恒只针对冻结交换阶段。完整训练更准确地写成：

\[
\frac{dx_m}{d\tau}=S_m^{\mathrm{learn}}+J_m,
\qquad
\frac{dK}{d\tau}=\sum_m p_mS_m^{\mathrm{learn}}.
\]

其中 \(S_m^{\mathrm{learn}}\) 是本地学习引入的新知识，\(J_m\) 是客户端之间的交换流。实验必须分别报告这两项，不能把整个训练过程的变化宣称为守恒。

### 4.2 异质性势能

最初的可分析势能是公共坐标中的响应差异：

\[
\mathcal E_{\mathrm{het}}(x_1,\ldots,x_M)
=
\frac12\sum_m p_m\|x_m-\bar x\|_G^2,
\qquad
\bar x=\sum_m p_mx_m,
\]

其中 \(G\succeq0\) 是由探针重要性、响应尺度和估计置信度决定的度量。等价的成对形式为：

\[
\mathcal E_{\mathrm{het}}
=
\frac12\sum_{m<n}p_mp_n\|x_m-x_n\|_G^2.
\]

这个能量不是“所有客户端必须相同”的价值判断。它只测量被 \(\mathcal Q\) 选择出来的、对公共任务响应有意义的差异；本地必要差异通过可容许约束保留。若 \(\mathcal Q\) 退化为无语义的参数向量，能量就退化成普通 consensus objective，方法主张失效。

### 4.3 势差驱动的理想流

在固定坐标、固定权重、无约束的理想情况下，采用：

\[
\boxed{
J_m=\kappa(\bar x-x_m),
\qquad \kappa>0.
}
\]

它满足：

\[
\sum_mp_mJ_m=0,
\]

并且：

\[
\frac{d\mathcal E_{\mathrm{het}}}{d\tau}
=-2\kappa\mathcal E_{\mathrm{het}}\le0
\]

在这里，\(\bar x-x_m\) 不是服务器发布的 global dynamics，而是客户端因公共动力学状态不平衡而产生的净知识流入。该方程用于给出守恒和耗散的最小理论基线；它本身不构成完整算法，也不作为普通均值共识的重新命名。

## 5. 本地可实现性决定最终平衡态

### 5.1 可容许（admissible）集合

客户端不能任意接收公共动力学状态。对每个客户端定义本地可实现集合：

\[
\mathcal A_m=\left\{
x:\begin{array}{l}
\text{有限时间图流可实现；}\\
\text{本地验证任务损失不超过容许增量；}\\
\text{表示和梯度行为满足有限时间约束；}\\
\text{运输及实现残差低于阈值}
\end{array}
\right\}.
\]

这些约束由客户端本地数据和本地轨迹评估，服务器不需要获得节点级数据。它们不是事后为保留个性化而设置的任意盒约束，而是模型是否能在本地任务中继续工作、以及流是否真的能被实现的可测条件。

### 5.2 受约束的交换流

独立地把每个客户端的流裁剪到 \(\mathcal A_m\) 会破坏守恒。因此，实际流应以成对交换为基本单位。令 \(j_{mn}\) 是从客户端 \(n\) 到客户端 \(m\) 的加权流，要求：

\[
j_{mn}=-j_{nm},
\qquad
p_mJ_m=\sum_{n\ne m}j_{mn}.
\]

于是守恒自动成立：

\[
\sum_mp_mJ_m=\sum_m\sum_{n\ne m}j_{mn}=0.
\]

在每个交换阶段，流方向应从降低 \(\mathcal E_{\mathrm{het}}\) 的可行方向中选择：

\[
\begin{aligned}
\{j_{mn}\} = \arg\min_{\{j_{mn}\}}
&\quad \left\langle \nabla_x\mathcal E_{\mathrm{het}},J\right\rangle
 +\frac{1}{2\kappa}\sum_{m<n}\|j_{mn}\|_{R_{mn}}^2 \\
\text{s.t.}
&\quad j_{mn}=-j_{nm},\\
&\quad J_m\in T_{\mathcal A_m}(x_m),\\
&\quad \|j_{mn}\|\le b_{mn}.
\end{aligned}
\]

其中 \(T_{\mathcal A_m}(x_m)\) 是本地可容许方向的切锥，\(R_{mn}\) 控制交换代价，\(b_{mn}\) 是通信、估计和实现能力给出的流上限。若存在严格下降的可行方向，则：

\[
\frac{d\mathcal E_{\mathrm{het}}}{d\tau}<0;
\]

若约束阻止进一步下降，最优流可以为零。这时系统到达的是受约束的平衡，而不是失败或被迫完全共识。

### 5.3 平衡态的含义

目标状态写成：

\[
\boxed{
\{x_m^\star\}
=
\arg\min_{x_m\in\mathcal A_m,;K(x)=K_0}
\mathcal E_{\mathrm{het}}(x_1,\ldots,x_M).
}
\]

一般情况下不要求 \(x_1^\star=\cdots=x_M^\star\)。标签覆盖、图结构、局部传播稳定性和任务必要响应会使部分差异无法运输或不应运输。个性化由本地可实现性和任务约束自然保留，不需要预先手工拆出 shared/private branch。

理论上，固定坐标的无约束流可证明能量指数下降；受约束流在可行集凸、度量固定、交换问题有解且估计误差受控时，可证明能量非增并收敛到一阶驻点。一般非线性训练中不能仅凭这条能量曲线宣称全局最优或准确率必然提升。

## 6. 从公共知识流回本地图动力学

公共流 \(J_m\) 必须真正改变客户端系统。定义本地实现算子 \(\mathcal R_m\)：

\[
U_m=\mathcal R_m(J_m),
\qquad
\frac{dH_m}{dt}
=F_m(H_m,\mathcal G_m)+U_m(H_m).
\]

实现不是把一个 global operator 原样复制给客户端。它可以修改初始状态、低秩向量场、有限时间流组合，或对本地参数施加可验证的动力学更新。必须测量实现后的响应：

\[
\mathcal Q\!\left(\mathcal T_m(\Xi_m^{\mathrm{eff}})\right)
=x_m+\Delta\tau J_m+\varepsilon_m^{\mathrm{real}}.
\]

\(\varepsilon_m^{\mathrm{real}}\) 是实现残差；它决定公共流是否真的进入本地系统。现有低秩 injection 可以作为 \(\mathcal R_m\) 的候选实现，但其 correction-only dissipative projection 不是高层机制要求。原生 A-DGN 的反对称结构仍是官方基线属性；若改用一般向量场，必须作为独立 backbone variant 评估。

## 7. 一轮联邦过程

一轮过程按以下因果顺序执行：

1. 客户端用当前本地图动力学在有限时间窗内观测 \(\Xi_m\) 和 \(x_m\)，同时记录拟合、运输和任务响应误差。
2. 客户端上传可聚合的公共坐标摘要；服务器通过模拟 secure aggregation 得到 \(K_0=\sum_mp_mx_m\) 及计算势差所需的聚合量。
3. 服务器和客户端依据当前 \(x_m\)、可容许性信息和成对约束计算净流 \(J_m\)，保证 \(\sum_mp_mJ_m=0\)。服务器发送的是与本地状态相关的净流规则或统计量，不是统一的 global dynamics。
4. 客户端将 \(J_m\) 拉回本地并生成 \(U_m\)，在下一段本地表示传播和训练中使用。
5. 客户端重新测量有效动力学 \(x_m^{\mathrm{eff}}\)、实现残差和本地任务行为；这一步用于验证流动是否落地，而不是用来回填本轮目标。
6. 本地学习引入的变化作为 \(S_m^{\mathrm{learn}}\) 单独记录，下一轮再进行新的运输和交换。

这种顺序避免了循环定义：不能用已经注入的有效轨迹定义目标，再用该目标证明注入有效。必须保存 exchange 前的 native snapshot、exchange 计算出的流、exchange 后的 effective snapshot 和本地学习源项。

## 8. 可证伪的研究预测

### P1：冻结交换守恒

在固定 \(p_m\)、运输坐标和观测算子的交换微步内，应有：

\[
\left\|\sum_mp_mJ_m\right\|
\]

接近数值精度；若不满足，说明流实现或局部约束破坏了守恒。

### P2：公共响应异质性耗散

在固定目标和可行方向存在时，\(\mathcal E_{\mathrm{het}}\) 应在 exchange 时间上下降；带有估计误差和本地学习源项时，下降应符合可解释的扰动界，而不是只展示一条训练准确率曲线。

### P3：有效流比原始摘要更重要

若 \(\mathcal E_{\mathrm{het}}\) 下降但 \(\|\varepsilon_m^{\mathrm{real}}\|\) 很大，客户端后续轨迹不会稳定地改善。这种结果应判为运输或实现失败，不能算作知识迁移成功。

### P4：收益依赖异质性类型

在 feature-only、structure-only 和 joint heterogeneity 的受控数据上，公共响应能量、实现残差和任务指标应呈现不同关系。若所有场景都只表现为一般正则化收益，动力学知识流的因果解释不成立。

### P5：约束会产生非共识平衡

当客户端具有任务必要或本地稳定性差异时，能量下降应停止在非零残差；强行取消可容许性约束会降低摘要方差，却损害本地任务行为。这是区别于无约束 consensus 的关键实验。

## 9. 诊断和对照

每轮至少记录：

- 守恒残差 \(\|\sum_mp_mJ_m\|\)；
- 交换前后 \(\mathcal E_{\mathrm{het}}\) 及其 pairwise 分解；
- 每个客户端的流量、可行性状态和零流原因；
- Functional Map 的运输残差、条件数和 gauge 对齐误差；
- 动力学拟合误差、实现残差和有效轨迹响应变化；
- 本地任务损失、验证指标、梯度范数和有限时间传播稳定性；
- 本地学习源项与交换流对 \(K\) 变化的分别贡献。

最小对照集合为：

1. official A-DGN + FedAvg，验证模块关闭等价性；
2. 仅观测和运输、无流动，分离计算路径影响；
3. local-only/no-transfer，验证收益是否来自跨客户端交换；
4. unconstrained flow，观察完全共识对本地任务的影响；
5. constrained flow，验证非共识可容许平衡态；
6. shuffled/wrong transport，验证坐标运输和知识方向是否具有因果作用；
7. feature-only、structure-only、joint synthetic regimes，分离初始状态和向量场异质性。

所有方法使用相同 FedAvg 生命周期、persistent Adam、客户端数量、通信轮数和 paired best-validation test evaluation。结果需同时报告任务指标和上述机制指标。

## 10. v0 的具体可运行化方案

前面的抽象在 v0 中收敛为一个最小闭环，目标是先验证机制是否成立，而不是一次性解决所有任务语义问题。

### 10.1 知识状态与公共 probe 响应

v0 不再把 \(z_m^0\) 和 \(A_m\) 展平拼接成交换状态。交换前固定一个公共 probe 集合：

\[
\mathcal P=\{P_q^c\}_{q=1}^{Q},
\qquad P_q^c\in\mathbb R^{r\times d}.
\]

其中 \(r\) 是客户端功能基维数，\(d\) 是隐藏表示维数。probe 由固定随机种子生成并用 QR 做小矩阵正交化，不使用标签、客户端 ID 或大图谱分解。客户端通过 Functional Map 拉回 probe：

\[
H_{m,q}^{0}=Q_mC_m^{-1}P_q^c,
\qquad
H_{m,q}^{\ell+1}=\Phi_{m,\Delta t}(H_{m,q}^{\ell};\mathcal G_m).
\]

对每个时间点，用本地功能观测 \(\mathcal O_m\) 得到系数并运输回公共坐标：

\[
R_{m,q}^{\ell}
=C_m\mathcal O_m(H_{m,q}^{\ell}),
\qquad
x_m=\operatorname{whiten}\!\left[
\operatorname{vec}(R_{m,q}^{\ell}-R_{m,q}^{0})
\right]_{q=1,\ell=1}^{Q,L}.
\]

所以 \(x_m\) 直接表示“相同公共 probe 在客户端图动力系统中经过有限时间后的传播响应”。它不需要先假设存在一个能准确解释全部轨迹的自治线性生成器。\(\mathcal O_m\) 的第一版取固定的低秩功能系数和节点均值；后续可加入二阶矩，但必须在所有客户端保持相同统计语义。

现有 `encode_native_dynamics_states`、`build_trajectory_basis` 和 Functional Map 只负责生成和运输响应；`fit_ridge_generator` 降为压缩诊断。若生成器拟合误差大，仍可直接使用 probe 响应，只要响应本身的测量、运输和本地实现误差受控。

### 10.2 运输与任务相关性

Functional Map 负责把局部 probe 响应运输到公共坐标。v0 用以下误差判断知识是否值得交换：

\[
e_m^{\mathrm{transport}}
=
\frac{\|\widehat{R}_{m,q}^{\ell}-R_{m,q}^{\ell}\|}{
\|R_{m,q}^{\ell}\|+\epsilon},
\qquad
e_m^{\mathrm{coverage}}
=
\frac{\|H_{m,q}^{\ell}-Q_mQ_m^\top H_{m,q}^{\ell}\|}{
\|H_{m,q}^{\ell}\|+\epsilon}.
\]

仅当 probe 响应的运输误差和功能子空间覆盖误差低于阈值时，客户端才进入本轮可交换集合。公共能量测量的是相同 probe 在不同图动力系统中的有限时间响应差异；本地验证损失、梯度范数和轨迹幅度则作为任务与动力学可容许性约束。这样“有害”有两个可观察含义：公共 probe 上的传播不一致，以及该不一致被实现后造成不可接受的本地行为变化。

### 10.3 守恒边流

对进入交换集合的客户端建立完全图。对任意客户端对 \((m,n)\)，定义从 \(n\) 流向 \(m\) 的加权知识流：

\[
j_{mn}
=
\kappa g_{mn}p_mp_n(x_n-x_m),
\qquad
j_{nm}=-j_{mn},
\qquad
g_{mn}=g_{nm}\in[0,1].
\]

净流入满足：

\[
p_mJ_m=\sum_{n\ne m}j_{mn}.
\]

v0 的 \(g_{mn}\) 是双方可行性的对称门控：客户端先报告本地可容许半径和误差置信度，只有双方都能接受该方向的有限步更新时边才开放；任一端任务损失超限、运输误差超限或实现误差超限时，令 \(g_{mn}=0\)。对称门控是守恒的关键，不能把每个客户端的流单独裁剪。

当所有边开放时，\(J_m=\kappa(\bar x-x_m)\)。当部分边关闭时，系统变成带可行性边的动力学扩散，而不是向一个服务器 target 做独立 proximal 更新。

### 10.4 v0 的离散交换步

交换步长为 \(\delta\tau\)，客户端先保存 native snapshot，再执行：

\[
x_m^{+}=x_m+\delta\tau J_m.
\]

为保证一阶离散守恒，直接使用反对称边流：

\[
\sum_mp_m(x_m^{+}-x_m)
=
\delta\tau\sum_m p_mJ_m=0
\]

在无门控、公共坐标固定且 \(0<\kappa\delta\tau\le1\) 时：

\[
\mathcal E^{+}=(1-\kappa\delta\tau)^2\mathcal E.
\]

部分边开放时，能量仍满足一阶下降条件：

\[
\mathcal E^{+}-\mathcal E
=
-\frac{\delta\tau\kappa}{2}
\sum_{m,n}g_{mn}p_mp_n\|x_m-x_n\|^2
 +O(\delta\tau^2).
\]

离散步长必须通过本地实现后的响应误差校验；若 \(\|\varepsilon_m^{\mathrm{real}}\|\) 超限，则回收该边的下一步门控。这里的回收是机制诊断，不把低秩 Euler 范数误称为完整非线性稳定性证明。

### 10.5 本地实现算子

因为 \(x_m\) 是 probe 响应，公共流不能直接被解释为 \(\Delta A_m\)。客户端用低秩场参数 \(\theta_m\) 表示可施加的本地修正，并在当前 native 动力学附近计算响应 Jacobian：

\[
B_m
=
\left.\frac{\partial x_m(\theta)}{\partial\theta}\right|_{\theta=0}.
\]

固定 \(B_m\) 后，用一个小型凸二次问题实现目标流：

\[
\theta_m^*
=
\arg\min_{\theta}
\frac12\|B_m\theta-\delta\tau J_m\|_2^2
 +\frac{\lambda_\theta}{2}\|\theta\|_2^2
\quad
\text{s.t.}\quad
\|\theta\|_2\le b_m,
\quad
\widehat{\Delta\mathcal L}_m(\theta)\le\rho_m.
\]

其中 \(\widehat{\Delta\mathcal L}_m\) 是本地验证损失的线性或二次近似，\(b_m\) 是有限时间修正预算，\(\rho_m\) 是任务容许增量。\(\theta_m^*\) 再映射为低秩图场修正并进行真实 probe rollout，得到实现残差：

\[
\varepsilon_m^{\mathrm{real}}
=
x_m(\theta_m^*)-x_m-\delta\tau J_m.
\]

这样，生成器不是被强行当成知识，而只是低秩本地修正的一种参数化；若 \(B_m\) 的列空间无法实现公共流，残差会明确暴露这一点。官方 A-DGN backbone、FedAvg 生命周期、persistent Adam 和 paired best-validation 评估保持不变。当前 correction-only dissipative projection 只作为对照变量，不是 v0 的核心约束。

### 10.6 最小实现接口

下一轮实现只需增加一个 opt-in 的 `ConservativeKnowledgeFlow` 组件，包含：

1. `encode_probe_response`: 用相同公共 probe 运行有限时间图动力学，生成 \(x_m\) 和覆盖/运输误差；
2. `aggregate_flux`: 在模拟 secure aggregation 边界计算加权守恒流；
3. `admissibility_gate`: 用双方本地校验结果生成对称 \(g_{mn}\)；
4. `linearize_realization`: 计算响应对本地低秩修正的 Jacobian，并解凸二次实现问题；
5. `measure_effective_response`: 重新测量交换后的 \(x_m^+\) 和实现残差。

实现必须保留 `--enable-functional-dynamics` 关闭时的 native-backbone 等价性，并记录 exchange 前后的 native/effective snapshot。v0 默认只使用一个公共状态，不使用 prototypes、cluster switching、EMA target 或单边 correction clipping。

### 10.7 超参数、聚合选择与数值约束

v0 的第一轮实验固定一组小而可解释的超参数：

| 参数 | 初始值 | 作用 |
| --- | ---: | --- |
| \(\delta\tau\) | `0.25` | 一次知识交换的离散步长 |
| \(\kappa\) | `0.4` | 动力学势差到知识流的耦合强度 |
| 轨迹窗口 | `16` 个 Euler 状态 | 观测有限时间传播响应 |
| probe 数量 \(Q\) | `4` | 覆盖公共功能方向 |
| 低秩维数 \(r\) | `8` | 公共 probe 的功能维数，按隐藏维度和快照数取上限 |
| 响应时间权重 | `1.0` | 各时间点响应的初始公共度量 |
| 实现正则 \(\lambda_\theta\) | `1e-3` | 本地响应实现问题的条件化强度 |
| probe 运输误差阈值 | `0.25` | 超过后该客户端不参与交换 |
| 实现残差阈值 | `0.25` | 超过后关闭相关交换边 |
| 本地验证损失容许增量 | `5%` | 双方可容许性门控的任务约束 |
| 最大流范数 | `0.1` | 防止单轮交换超过有限时间线性化范围 |

这些值是初始实验设置，不是普适最优超参数。实验必须报告 \(\kappa\)、\(\delta\tau\)、\(r\)、误差阈值和门控比例；只报告最终准确率无法解释机制。

FedAvg 的样本数加权只用于官方模型参数的生命周期，不能自动继承为动力学知识的聚合权重。动力学状态的 \(p_m\) 应先解释为联邦系统的质量或测度权重；第一轮合成实验固定 \(p_m=1/M\)，再与按训练样本数加权、按轨迹置信度加权进行对照。若加权后公共能量下降只是由大客户端支配，不能称为异质性得到普遍改善。

v0 的知识交换目标使用固定门控下的凸二次目标，而不是简单加权平均：

\[
\begin{aligned}
\min_{\{j_{mn}\}}\quad
&\frac{1}{2}\sum_{m<n}
\frac{\|j_{mn}\|_2^2}{\kappa g_{mn}p_mp_n+\epsilon}
+\left\langle\nabla_x\mathcal E_{\mathrm{het}},J\right\rangle \\
\text{s.t.}\quad
&j_{mn}=-j_{nm},
\quad \|j_{mn}\|_2\le b_{mn},
\quad g_{mn}=0\Rightarrow j_{mn}=0,
\quad J_m\in T_{\mathcal A_m}(x_m).
\end{aligned}
\]

在 \(G\succeq0\)、门控在本轮开始前固定、切锥用线性约束近似时，这个子问题是凸的；反对称边变量使守恒成为约束结构的一部分。完全图上的加权均值只在所有 \(g_{mn}=1\)、没有流量上限和没有本地约束时作为这个目标的闭式特例。第一版可以在 10 客户端模拟环境中显式求解这个小问题，随后再研究怎样压缩成 secure aggregation 可交换的统计量。

数值实现遵循以下硬约束：

- 禁止对大图拉普拉斯矩阵、邻接矩阵或节点级特征矩阵做 SVD；不通过谱分解构造大图公共坐标。
- 轨迹基只在低维快照矩阵上使用带确定性 gauge 的 QR；生成器使用 ridge 线性方程求解。
- 若 Functional Map 需要条件控制，优先使用 QR、正则化线性 solve 或小维对称特征值检查；任何小维 SVD 只能用于明确的数值诊断，并记录矩阵维度和计算位置。
- 大图传播沿稀疏边执行，公共知识只上传低维摘要、误差和聚合所需统计量。
- 图拉普拉斯若用于诊断，只计算稀疏乘法、度统计、局部随机探针响应或小维迭代量；不计算其完整特征分解。

当前旧 Functional Dynamics 原型中的 Procrustes 和正则化 Functional Map 仍调用低维 SVD。它们不能直接作为 v0 的数值规范；在实现 v0 时应先替换为 QR/正则化 solve 版本，或明确限制在 \(r\le8\) 的小矩阵上并单独报告，不得把这套操作扩展到大图。

### 10.8 低维算子可用性门槛

当前 ridge 生成器的 `dyn_fit_error` 只表示训练窗口内的相对一步速度残差，不表示完整非线性图动力学已经被低维线性系统解释。已有 11 个数据集的稳态中位数约为 `0.276--0.727`，且没有和任务指标变化形成稳定对应关系。因此，在误差没有通过独立时间段检验前，不能把 \(A_m\) 直接作为可迁移知识，也不能仅用 Functional Map objective 下降来证明它可用。

R010 将公共因果响应块组成有限时域输入—输出算子

\[
\mathcal R_m\in\mathbb R^{qT_y\times pT_u},
\qquad
\mathcal R_m^{(r)}=U_m\Sigma_mV_m^\top,
\]

R010 曾把该低秩算子作为首选候选。未压缩响应是保真上界，截断 SVD 只作用于由公共端口/时间与公共观测/时间定义的小矩阵。只有当响应近似由时间差决定时，才进一步采用 block-Hankel 结构。D022/R011 已取消以正交响应模态定义互补性的方案。

自治连续生成元 \(A_m\) 和满足 \(P_m=I+\Delta t A_m\) 的 identity-centered 离散传播子，在匹配数据、正则化和误差尺度下代数等价，不能作为两个不同表达能力的候选。R010 的有效竞争者改为：

1. 自治生成元 \(A_m\)，作为旧基线；
2. 带偏置的仿射传播子；
3. 使用共同观测字典的 delay-AR/Koopman；
4. 低秩有限时域因果响应算子；
5. 未压缩响应上界。

仿射基线为

\[
\min_{P,b}
\sum_t\|z_{t+1}-Pz_t-b\|_2^2
 +\lambda_P\|P-I\|_F^2+\lambda_b\|b\|_2^2.
\]

压缩模型的选择不能依据拟合窗口误差。每个客户端按时间块划分拟合段和验证段，至少记录一步验证误差、\(h\) 步 rollout 误差、原始 probe 响应统计量误差和本地任务变化。只有当压缩误差低于预设实验阈值（初始候选为一步 `0.15`、多步 `0.25`）且实现残差低于 `0.25` 时，才允许用压缩算子替代原始 probe 响应进入公共知识流；否则保留原始响应或只参与诊断。阈值是可报告的实验超参数，不是理论常数。

上述固定一步/多步阈值是旧 R007 的历史方案，已被 R010 的匹配代理选择门槛取代。当前每个候选必须使用相同端口、观测、秩和通信预算，并报告留出端口/时间响应、有效秩、bootstrap 子空间稳定性、已知独有模态恢复、接收方响应实现残差和相对 matched local-only 的实际任务增益。完整淘汰规则见当前算子选择规范。

小型神经网络可以作为后续压缩对照：

\[
z_{t+1}=z_t+\Delta t\,f_\theta(z_t),
\qquad
\min_\theta\sum_t\|z_{t+1}-z_t-\Delta t f_\theta(z_t)\|_2^2.
\]

它能表达非线性，但目标是非凸的，容易在短轨迹上过拟合，潜在坐标和参数也没有天然的跨客户端可比性。v0 不交换神经网络参数；只有当所有固定低维候选都不能通过留出响应门槛、而神经模型稳定通过时，才把它作为“固定线性响应秩不足”的证据，并另行研究可比较的神经算子接口。

## 11. 初步理论结果与周报可用结论

在固定 \(p_m\)、公共坐标、观测算子和对称门控下，v0 已得到以下可直接写入周报的理论结果：

1. **交换守恒**：反对称边流逐项抵消，因此 \(\sum_mp_mJ_m=0\)，交换阶段的公共动力学知识一阶矩保持不变。
2. **理想耗散**：完全图、所有边开放时，\(J_m=\kappa(\bar x-x_m)\)，并有 \(d\mathcal E/d\tau=-2\kappa\mathcal E\)；离散交换在 \(0<\kappa\delta\tau\le1\) 时按 \((1-\kappa\delta\tau)^2\) 收缩。
3. **受约束耗散**：对称非负门控不会破坏一阶能量下降；能量只在不存在可行下降边、或估计/实现误差主导时停止下降，因此终态可以保留非零客户端差异。
4. **因果验证闭环**：只有同时观测到守恒残差小、公共响应能量下降、实现残差受控和本地任务不恶化，才能把结果解释为动力学知识流；单独的准确率提升或摘要方差下降都不够。

这给出了一个初步但可运行的研究方案：理论上可证明的部分是交换阶段的守恒和理想/一阶耗散，经验上需要在 feature-only、structure-only、joint 三种合成异质性上验证运输、实现和任务行为。

### 11.1 Toy exchange 的初步结果

为先验证流机制本身，使用 10 个客户端、6 维固定动力学状态、均匀 \(p_m\)、\(\delta\tau=0.25\)、\(\kappa=0.4\) 做 20 个交换步。该 toy 实验只执行公共坐标中的边流，不涉及大图、拉普拉斯特征分解或模型训练。

| 交互约束 | 初始能量 | 20 步后能量 | 能量比例 | 单调下降 | 最大守恒残差 |
| --- | ---: | ---: | ---: | --- | ---: |
| 完全连通 | 4.69634122 | 0.06941607 | 0.01478088 | 20/20 | \(4.25\times10^{-17}\) |
| 两个可容许客户端组 | 4.69634122 | 3.28964478 | 0.70046971 | 20/20 | \(7.49\times10^{-17}\) |

完全连通结果与理论比例 \((1-\kappa\delta\tau)^{40}=0.9^{40}\) 一致。两组门控结果说明：能量可以持续下降，同时因为跨组交换被本地可容许性关闭而保留非零差异；这提供了“约束产生非共识平衡态”的第一条数值证据。该结果只验证交换层机制，不能代替图任务准确率实验。

## 12. 实现边界和接受标准

当前阶段只批准 v0 的最小 opt-in 实现，不批准扩大 multi-prototype、调大 injection 或修改 A-DGN 主干。后续实现必须使用本文的单公共状态、对称边流、凸二次流目标和本地回写，并说明具体 \(\mathcal Q\)、运输误差定义、可行性近似和 \(\mathcal R_m\) 如何对应本文抽象对象。

进入实现阶段至少需要满足：

- 能写出交换阶段的守恒证明和离散守恒误差定义；
- 能写出理想流的能量下降推导，并说明约束和估计误差造成的剩余项；
- 能从本地模型测量实现残差，而不是假设 \(\mathcal T_m^{-1}\) 存在；
- 能定义任务保持约束且不使用交换后指标循环构造目标；
- 能在合成 feature、structure、joint 三种场景上区分跨客户端流动、一般正则化和错误运输；
- 模块关闭时复现官方 A-DGN + FedAvg，且不上传节点级状态或轨迹基。

## 13. 当前结论

Method v0 的研究对象是“运输后的任务相关动力学响应”，机制是“满足成对反对称约束的知识流”，目标是“在本地可实现性约束下耗散有害响应差异”。普通均值、prototype、proximal residual 和 correction clipping 都只能作为该机制的实现或对照。

最先需要验证的不是准确率是否上涨，而是以下闭环是否成立：

\[
\boxed{
\text{可比较知识}
\rightarrow
\text{守恒净流}
\rightarrow
\text{本地可实现更新}
\rightarrow
\text{有效响应变化}
\rightarrow
\text{任务相关异质性下降}
}
\]
