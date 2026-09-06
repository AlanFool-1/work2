# Functional Map 下的动力学作用交换与约束补全

日期：2026-09-06 UTC

状态：NEEDS_RESEARCH。本文修正“正交缺失模态”方案，并作为 R011 的当前设计。

## 1. 核心修正

“来源算子的奇异模态不在接收方子空间中”只表示几何不重合，不能推出接收方缺少一种可学习能力。它仍然是一种 shared/private 子空间分解，而且在预先使用共同响应坐标时，Functional Map 没有承担实际作用。

当前设计改为：

> 客户端交换的不是一个正交方向，而是一条经过 Functional Map 运输的有限时域动力学作用方程。互补知识是接收方当前尚不能满足、但能够学习且对本地任务有益的方程。

这把“别人会、我不会”改写成一个可检验命题：对同一个已运输的函数或扰动，来源客户端给出的未来作用结果，接收方当前动力学不能复现；经过本地学习后，接收方能够复现，并改善或至少不损害本地任务。

## 2. Functional Map 是跨客户端运输，不是对齐到公共原型

客户端 \(i\) 在本地图上建立小型功能空间

\[
\mathcal F_i=\operatorname{span}(Q_i),\qquad
Q_i\in\mathbb R^{n_i\times r}.
\]

客户端 \(j\) 到 \(i\) 的 Functional Map

\[
C_{j\to i}:\mathcal F_j\rightarrow\mathcal F_i,
\qquad
C_{j\to i}\in\mathbb R^{r_i\times r_j},
\]

运输的是函数、扰动或观测的低维系数。它不把所有客户端变换到一个 prototype，也不要求节点对应。

对传播时长 \(\tau\)，令 \(\mathcal P_i^\tau\) 表示客户端真实 ODE-GNN 在 \(\mathcal F_i\) 上诱导的有限时域作用。理想的可运输动力学满足交换关系

\[
\boxed{
\mathcal P_i^\tau C_{j\to i}
=C_{j\to i}\mathcal P_j^\tau.
}
\]

两条路径的意义是：

1. 在来源客户端传播，再把结果运输到接收方；
2. 先把输入函数运输到接收方，再由接收方自己的动力学传播。

若两条路径一致，接收方已经能够执行该来源作用；若不一致，则存在动力学作用交换缺陷。

Functional Map 的原始框架允许用算子交换性形成线性约束；Functional Map Network 进一步用环一致性校正多对象之间的成对映射。本项目的推断是把“交换缺陷”从映射正则项改造成联邦知识传递的对象。该用途需要单独验证，不能由形状匹配文献直接推出。

## 3. 不强行拟合完整算子：交换有限时域作用草图

选择 \(Q\) 个低维探针系数

\[
Z_j=[z_{j1},\ldots,z_{jQ}]\in\mathbb R^{r_j\times Q}.
\]

客户端 \(j\) 在若干传播时长 \(\tau\in\mathcal T\) 上测量

\[
Y_j^\tau
=\mathcal P_j^\tau(Z_j)
\in\mathbb R^{r_j\times Q}.
\]

本地上传对象是小型作用草图

\[
\mathcal S_j
=\{Z_j\mapsto Y_j^\tau:\tau\in\mathcal T\},
\]

而不是生成元参数、完整响应 Jacobian 或奇异向量。对扰动响应，可以把 \(Z_j\) 理解为低维端口扰动，把 \(Y_j^\tau\) 定义为扰动前后轨迹差的低维投影。

运输到客户端 \(i\) 后，来源方程变为

\[
X_{j\to i}=C_{j\to i}Z_j,\qquad
T_{j\to i}^\tau=C_{j\to i}Y_j^\tau.
\]

接收方用真实本地模型计算

\[
\widehat T_{j\to i}^\tau
=\mathcal P_i^\tau(X_{j\to i}),
\]

并得到作用交换缺陷

\[
\boxed{
E_{j\to i}^\tau
=\widehat T_{j\to i}^\tau-T_{j\to i}^\tau.
}
\]

若局部作用可由矩阵 \(K_i^\tau\) 表示，则

\[
E_{j\to i}^\tau
=(K_i^\tau C_{j\to i}
-C_{j\to i}K_j^\tau)Z_j,
\]

即标准的 intertwining/commutation defect。实际实现优先直接比较有限时域 action pairs，不要求单一线性生成元在所有状态上闭合。

## 4. 互补知识是“新增约束”，不是正交补

来源方程

\[
\Gamma_{j\to i}^{q,\tau}:
\quad
\mathcal P_i^\tau(C_{j\to i}z_{jq})
=C_{j\to i}\mathcal P_j^\tau(z_{jq})
\]

对客户端 \(i\) 构成互补知识，当且仅当：

1. **运输可信**：Functional Map 在未参与求解的 descriptor 与环一致性检查上可靠；
2. **当前未满足**：留出作用交换缺陷显著非零；
3. **本地可学习**：用普通反向传播更新后，留出缺陷确实下降；
4. **任务有益**：该更新与本地任务下降方向相容，且 guard 指标不恶化。

因此，“互补”是一条尚未掌握的动力学经验，而不是两个模型的集合差或子空间正交分量。概念上，接收方的可行模型集合被新增方程逐步约束：

\[
\Theta_i^{r+1}
=\Theta_i^r
\cap
\{\theta:\ell_{\Gamma_{j\to i}}(\theta)\le\varepsilon_\Gamma\}.
\]

知识吸收后，同一方程的交换缺陷下降到阈值内，后续不再产生流量。不同来源若提供等价方程，只会重复已有约束，而不会因为方向略有不同被误判为新知识。

## 5. 多客户端交互不需要模型聚合

服务器维护稀疏的 Functional Map Network。边 \(j\to i\) 携带

\[
(C_{j\to i},Z_j,\{Y_j^\tau\}_{\tau\in\mathcal T})
\]

及映射置信度，不生成全局 operator、prototype 或 barycenter。多个来源给接收方提供的是一组作用方程：

\[
\mathcal C_i
=\bigcup_{j\in\mathcal N(i)}
\{\Gamma_{j\to i}^{q,\tau}\}.
\]

接收方使用标准训练目标

\[
\mathcal L_i
=\mathcal L_i^{\mathrm{task}}
+\lambda_{\mathrm{dyn}}
\sum_{\Gamma\in\mathcal B_i}
\rho\!\left(\ell_\Gamma(\theta_i)\right),
\]

其中 \(\mathcal B_i\) 是本轮仍未满足且任务相容的小型方程批次，\(\rho\) 是稳健损失。普通反向传播同时完成“本地实现”和参数更新，不显式构造 \(D_{\theta_i}\operatorname{vec}(\mathcal R_i)\)。

可用任务梯度与方程梯度的一阶内积作为廉价在线门控：

\[
\chi_{j\to i}
=\frac{
\langle\nabla\mathcal L_i^{\mathrm{task}},
\nabla\ell_{\Gamma_{j\to i}}\rangle
}{
\|\nabla\mathcal L_i^{\mathrm{task}}\|
\|\nabla\ell_{\Gamma_{j\to i}}\|+\epsilon
}.
\]

当 \(\chi_{j\to i}>0\) 时，沿联合负梯度的小步更新在一阶上与本地任务下降相容。真实 matched local-only 分叉只在早/中/晚 checkpoint 做机制审计，不进入每轮算法。

## 6. 知识流与停止条件

定义本轮有效知识流为被吸收方程在留出探针上的缺陷下降：

\[
J_{j\to i}^r
=
\big[
\|E_{j\to i}^{r,\mathrm{before}}\|_F^2
-
\|E_{j\to i}^{r,\mathrm{after}}\|_F^2
\big]_+.
\]

客户端达到联邦自足，当所有可信运输下的外部方程都满足

\[
\|E_{j\to i}^{\mathrm{heldout}}\|_F
\le\varepsilon_E
\quad\text{或}\quad
\chi_{j\to i}\le0.
\]

前者表示该作用已经学会，后者表示剩余差异对本地任务没有可吸收价值。最终不要求 \(K_i=K_j\)，只要求在可运输、被探针覆盖且任务有益的函数上不存在新的有效方程。

在线性固定映射的理想化情形，边损失

\[
\mathcal E_{ij}
=\frac12\|K_iC_{j\to i}-C_{j\to i}K_j\|_F^2
\]

对 \(K_i\) 做梯度流时满足

\[
\frac{d\mathcal E_{ij}}{d\tau}
=-
\|(K_iC_{j\to i}-C_{j\to i}K_j)C_{j\to i}^{\top}\|_F^2
\le0.
\]

这只是固定 \(K_j,C_{j\to i}\) 下的边缺陷耗散恒等式。它不证明非线性训练收敛，也不证明所有客户端应当共轭。

## 7. feature、structure 与 joint heterogeneity 的统一

- **feature heterogeneity** 改变被激发的初始函数和局部轨迹覆盖。Functional Map 运输来源客户端提供但接收方本地数据没有充分激发的作用方程；
- **structure heterogeneity** 改变 \(\mathcal P_i^\tau\)，直接体现在“先运输后传播”与“先传播后运输”的差异；
- **joint heterogeneity** 同时改变输入覆盖和传播作用，仍由同一交换图测量。

必要的本地结构差异不会因为数值不同就被消除。只有能够可靠运输、可以由接收方学会、并与本地任务一致的方程才参与更新。

## 8. 相比 R010 的工程减法

R011 删除四个在线步骤：

- 不构造完整因果响应 Jacobian；
- 不对响应矩阵做奇异模态分解；
- 不为每个接收方计算正交缺失子空间；
- 不在每轮运行 matched local-only counterfactual。

保留的在线对象只有低维 basis/descriptor、\(r\times r\) Functional Map、少量 \(r\times Q\) 有限时域 action pairs，以及普通训练梯度。Functional Map 和作用草图可以每 \(K\) 个联邦轮次校准一次，其他轮次复用。

## 9. 与现有代码的关系

现有代码已经包含以下可复用部分：

- trajectory basis 和 normalized descriptor；
- descriptor Procrustes 与 regularized Functional Map；
- 动力学交换残差 \(A_iC-CA_j\) 的求解结构；
- Functional Map 的条件数、descriptor error 和 dynamics error 诊断。

需要改变的是它们的角色：

- 删除“映射到 canonical prototype 后做加权平均”的主路径；
- Functional Map 改为客户端对之间的运输边，并加入网络环一致性；
- 当前 solver 中用同一 dynamics term 拟合映射再报告 commutation error 会产生循环验证。第一版用 descriptor/cycle split 拟合 \(C\)，在独立 action probes/horizons 上测缺陷；
- 原来的 commutation residual 不再只是 map regularizer，而是来源作用方程是否被接收方掌握的核心观测；
- 本地更新通过 action-pair distillation 回到真实 ODE-GNN，不再把低维矩阵差直接注入 hidden field。

## 10. 最小判别实验

第一项实验只验证 Functional Map 运输和方程补全，不实现完整联邦训练：

1. 构造已知跨图函数对应 \(C^\star_{j\to i}\) 的小型合成系统，并人为设置共享、可迁移独有和任务有害三类作用；
2. 比较 ground-truth map、descriptor-only learned map、wrong map、identity/no-map；
3. 在 fit probes 上估计 map，在 held-out probes 与 held-out horizons 上计算交换缺陷；
4. 用普通 action-pair distillation 更新接收方，检查可迁移独有作用的缺陷和任务误差是否同时下降；
5. 检查重复方程是否自然产生近零新流，任务有害方程是否被梯度/guard 拒绝；
6. 与旧 canonical-prototype pullback 和 R010 orthogonal-mode transfer 做等预算对照。

支持 R011 需要同时看到：

- learned map 接近 ground-truth map，并通过 cycle consistency；
- 正确 map 下的缺陷能区分已掌握、可迁移缺失和不可迁移作用；
- action-pair distillation 能把缺陷下降落实为任务改善；
- wrong/no-map、source/time shuffle 不能复现结果；
- 运行和通信成本明显低于完整响应 Jacobian 方案。

若 descriptor-only map 本身不可辨识，或者正确 map 下来源方程仍不能改善接收方，则停止算法扩展。此时问题在跨客户端语义对应或知识可迁移性，不应再增加聚合结构。

## 11. 文献边界

Functional Map 将函数间对应表示成低维线性映射，descriptor preservation、算子交换和映射组合是其已有性质；Functional Map Network 的环一致性也已有成熟定义。因此，不能把这些数学工具本身作为贡献。

可能的新颖点必须落在：将异质图客户端的有限时域神经动力学表示为可运输 action constraints；把 Functional Map 的交换缺陷解释为接收方尚未掌握的动力学作用；通过本地任务相容的约束补全实现吸收和自然停止。该表述仍需完整文献检索和实验证据。

参考：

- Ovsjanikov et al., *Functional Maps: A Flexible Representation of Maps Between Shapes*: <https://www.research.autodesk.com/publications/functional-maps-a-flexible-representation-of-maps-between-shapes/>
- Huang, Wang and Guibas, *Functional Map Networks for Analyzing and Exploring Large Shape Collections*: <https://doi.org/10.1145/2601097.2601111>
- Sun et al., *Spatially and Spectrally Consistent Deep Functional Maps*, ICCV 2023: <https://openaccess.thecvf.com/content/ICCV2023/html/Sun_Spatially_and_Spectrally_Consistent_Deep_Functional_Maps_ICCV_2023_paper.html>
