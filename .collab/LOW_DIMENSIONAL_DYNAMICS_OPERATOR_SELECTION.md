# 低维本地动力学代理/算子选择

日期：2026-09-06 UTC

状态：NEEDS_RESEARCH。本文给出当前首选候选和淘汰实验，不代表该算子已经被实验证实。

## 1. 当前真正的问题

客户端之间不应交换一个任意的低维参数向量，而应交换一个能够回答下列问题的动力学对象：

> 在相同类型的有限时域干预下，本地系统会把哪些输入方向传播成哪些任务相关输出变化？

因此，代理必须同时满足四项要求：

1. **动力学保真**：能预测未参与拟合的端口、时刻和轨迹响应，而不只重建训练窗口；
2. **跨客户端可比**：即使图的节点数和节点身份不同，输入端口、输出观测和时间坐标仍具有共同语义；
3. **互补性可定义**：能区分“对方拥有而本地缺失的响应模态”与普通数值距离；
4. **本地可吸收**：检测到的外部模态能映射为接收方模型可实现、且优于等预算本地学习的更新。

当前的低秩自治生成元只部分满足第一项，而且依赖本地轨迹基坐标。直接平均生成元、奇异向量或其参数，会重新落入坐标对齐和模型融合问题。

## 2. 首选对象：低秩有限时域因果响应算子

对客户端 \(i\)，固定一个小型公共端口族和任务观测族。令

\[
u_i=(u_{i,0},\ldots,u_{i,T_u-1})\in\mathbb R^{pT_u}
\]

表示初值、邻居耦合或局部衰减等端口的微小干预序列，令

\[
\delta y_i=(\delta y_{i,1},\ldots,\delta y_{i,T_y})
\in\mathbb R^{qT_y}
\]

表示任务相关观测的响应。沿客户端真实 ODE-GNN 轨迹线性化后定义

\[
\boxed{
\delta y_i = \mathcal R_i u_i + O(\|u_i\|^2),
\qquad
\mathcal R_i\in\mathbb R^{qT_y\times pT_u}.
}
\]

其块元素就是因果核

\[
[\mathcal R_i]_{t,s}=\mathcal K_{i,t,s}
=\left.\frac{\partial \mathcal O_i(h_{i,t})}
{\partial u_{i,s}}\right|_{u=0},
\qquad \mathcal K_{i,t,s}=0\quad(t\le s).
\]

该矩阵是一个有限时域输入—输出算子，而不是本地隐状态参数。它的行和列分别由共同的观测—时间语义与端口—时间语义确定，因此不要求跨图节点对应。

用截断 SVD 得到通信对象

\[
\mathcal R_i^{(r)}=U_i\Sigma_iV_i^\top,
\]

只上传 \(r(qT_y+pT_u+1)\) 个数及测量元数据。比较和组合必须使用重构后的算子或其子空间投影，不能直接比较带符号/旋转不唯一性的 \(U_i,V_i\) 列。

### 与 Hankel 算子的关系

当局部线性化近似时不变、核主要由时间差 \(t-s\) 决定时，可以把 Markov 响应块重排为 block-Hankel 矩阵，并用其低秩结构估计有效系统阶数。当前非线性 ODE-GNN 沿轨迹的变分系统通常是时变的，所以在实验确认“近似滞后平稳”之前，规范名称应是**有限时域因果响应算子**；Hankel 只是可检验的压缩特例，不能先验强加。

## 3. “互补”是接收方缺失的响应模态

将第 \(k\) 个奇异模态写成 Frobenius 算子空间中的单位原子

\[
b_{ik}=\operatorname{vec}(u_{ik}v_{ik}^{\top}),
\qquad
B_i=[b_{i1},\ldots,b_{ir}],
\qquad
P_i=B_iB_i^\top.
\]

来自客户端 \(j\) 的第 \(k\) 个模态，对接收方 \(i\) 的原始创新为

\[
\boxed{
n_{j\to i,k}
=\sigma_{jk}(I-P_i)b_{jk}.
}
\]

该量具有方向性：同一模态对客户端 \(i\) 可能是新知识，对客户端 \(l\) 可能已经冗余。两个响应算子距离很大，也可能因为差异落在不相关或不可实现方向而没有可学习价值。

实际发送分数还必须乘上两道门：

\[
s_{j\to i,k}
=\|n_{j\to i,k}\|
\cdot \rho_{i,k}^{\mathrm{realizable}}
\cdot \widehat g_{i,k}^{\mathrm{task}},
\]

其中 \(\rho_{i,k}^{\mathrm{realizable}}\) 衡量该算子变化在接收方响应 Jacobian 像空间中的可实现程度，\(\widehat g_{i,k}^{\mathrm{task}}\) 衡量它相对等预算 local-only 更新的预测边际任务增益。

## 4. 客户端交互、服务器组合与本地吸收

一次交换采用接收方条件化的“子空间补全”：

1. 每个客户端测量并上传低秩 \(\mathcal R_i^{(r)}\) 及留出误差、有效秩和量化尺度；
2. 服务器把各来源模态投影到接收方已有响应子空间的正交补；
3. 对多个来源的创新做增量 QR/SVD，删除重复方向，在固定字节预算内保留分数最高的正交创新；
4. 接收方求解局部响应匹配问题

\[
\min_{\Delta\theta_i}
\left\|D_i\Delta\theta_i-N_i\right\|_F^2
+\lambda\|\Delta\theta_i\|^2,
\]

并用同快照、同计算量、同更新范数的 local-only 分支验证实际增益；
5. 只有留出响应、可实现性和任务 guard 同时通过时才吸收该创新。

服务器组合的是不同客户端提供的**非冗余能力方向的并集**。随着客户端吸收某个模态，该模态进入其本地响应子空间，后续 \((I-P_i)b_{jk}\) 自动缩小。若所有接收方条件化、任务相关且可实现的创新分数均低于阈值，知识流停止。这个停止条件直接对应“外部可学习知识已经耗尽”。

## 5. 候选代理的当前排序

| 候选 | 优点 | 决定性问题 | 当前定位 |
| --- | --- | --- | --- |
| 低秩自治生成元 \(A_i\) | 最小、可用指数映射组合 | 当前残差高；闭包假设弱；本地基坐标不天然可比 | 必须保留的旧基线 |
| 仿射生成元/传播子 | 可吸收常数漂移 | 只比 \(A_i\) 多一个偏置；仍无记忆和输入—输出语义 | 轻量基线 |
| 延迟嵌入 AR/Koopman 算子 | 用记忆缓解非闭包 | 字典和坐标仍需共享；有限维不变子空间不一定存在；参数平均无意义 | 主要竞争者 |
| 低秩有限时域因果响应算子 | 直接描述干预如何传播；语义可共享；互补模态可定义 | 端口/传感器覆盖率、局部线性范围、有效秩和测量成本待验证 | **当前首选** |
| 学习式非线性潜在算子/条件向量场 | 表达力最强 | 潜在坐标不可辨识，跨客户端参数难比较，训练和通信更重 | 仅在低秩响应失败后考虑 |

连续生成元 \(A\) 与匹配目标下的 \(P=I+hA\) 是同一模型，不能作为两个表达能力不同的候选。延迟 AR/Koopman 必须使用与响应算子相同的公共观测和字节预算；否则比较会混入坐标与容量差异。

## 6. 第一个有判别力的实验

实验目的不是证明最终算法有效，而是选择能够表示“缺失动力学能力”的代理。

### Stage A：已知互补模态的可辨识性

构造稳定的受控图动力系统，使客户端具有共同模态和明确的客户端独有模态。端口、观测和时间坐标一致，但节点数、图结构和初始分布可以不同。按相同秩 \(r\in\{2,4,8\}\) 和相同通信字节比较：

- 当前自治 \(A_i\)；
- 仿射 \(A_i\)；
- delay-AR/Koopman，\(k\in\{2,4\}\)；
- 低秩因果响应算子；
- 未压缩响应核上界。

训练端口和时间块只用于拟合；留出端口组合、留出未来时间块和新初值只用于评估。主要指标是留出响应 NRMSE、bootstrap 子空间主角稳定性、有效秩、已知独有模态的 precision/recall，以及把正确创新加入接收方后对缺失响应的恢复率。

### Stage B：冻结真实 A-DGN checkpoint 的外部有效性

在 feature_shift、structure_homophily 和 mixed 三种合成异质性中选早/中/晚 checkpoint。先测量所有候选的留出响应误差；然后对每个接收方从同一 checkpoint 分叉：

- matched local-only；
- true source innovation；
- source/time shuffled innovation；
- redundant-source innovation；
- raw response upper bound；
- no-flow。

报告预测创新分数与实际额外任务增益的关联、响应实现残差、接受率、通信字节和运行时间。官方 validation/test 不参与流量选择。

### 淘汰规则

满足任一项就停止推进对应代理：

1. 留出响应误差没有稳定优于静态响应或旧生成元基线；
2. 有效秩在小预算内不稳定，bootstrap 子空间变化与客户端差异同量级；
3. 在已知模态实验中不能区分“新模态”和“重复模态”；
4. 创新分数不能比普通算子距离更好地预测接收方的真实额外收益；
5. source/time shuffle 与真实来源产生相同效果。

低秩响应算子只有在同时通过留出保真、互补模态恢复和接收方增益三项门槛后，才进入完整联邦训练。若未压缩响应上界本身不能产生增益，应停止代理压缩研究，回到端口/观测语义或“是否存在可迁移动力学知识”这一更上游的问题。

## 7. 文献边界与新颖性约束

低阶 LTI 系统的 Markov 参数和 block-Hankel 低秩结构已有成熟系统辨识依据；DMD/DMDc、子空间辨识和 Koopman 方法也已经研究从快照或输入—输出数据识别低维动力学。2024 年已有工作用 future-past Hankel 协方差区分多时间序列的 shared/private dynamics。因此，本项目不能把“用 Hankel 找共享动力学”本身作为新颖点。

若实验支持，可能成立的项目贡献是：在无节点对应、私有异质图联邦学习中，把真实 ODE-GNN 的端口—时间—任务响应压缩为低秩算子；以**接收方已有算子子空间的创新**定义互补知识；通过本地可实现性和相对 local-only 的边际增益决定吸收与停止。该贡献仍需文献检索和实验验证，当前不得写成已证实的新颖性结论。

参考：

- Sun et al., *Finite Sample System Identification: Optimal Rates and the Role of Regularization*, L4DC 2020: <https://proceedings.mlr.press/v120/sun20a.html>
- Proctor et al., *Dynamic Mode Decomposition with Control*, SIAM J. Applied Dynamical Systems 2016: <https://epubs.siam.org/doi/10.1137/15M1013857>
- Brunton et al., *Koopman Invariant Subspaces and Finite Linear Representations of Nonlinear Dynamical Systems*: <https://arxiv.org/abs/1510.03007>
- Modi et al., *Spectral Learning of Shared Dynamics Between Generalized-Linear Processes*, NeurIPS 2024: <https://papers.nips.cc/paper/2024/file/a267f6a1ea2bc9e0ec15ed4af7d5ae3f-Paper-Conference.pdf>
