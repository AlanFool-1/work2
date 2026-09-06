# 周报：有限互补知识吸收与本地动力学算子选择

日期：2026-09-06 UTC

状态：学习过程思想已形成；算法骨架修正为无 FedAvg 的 Functional Action Flow，Functional Map 运输动力学作用方程，尚待受控实验验证。

## 1. 本周核心思想

联邦学习并非一次性的模型融合，其目标是让每个客户端借助协作提升自身学习能力。固定任务、客户端集合、模型容量与通信接口后，其他客户端能够提供的任务相关互补知识具有有限价值。训练初期，本地知识缺口较大，跨客户端协作能够产生较高的信息增益；随着互补知识被吸收，客户端对外部知识的依赖逐渐下降；当任何外部可实现知识都不能比等预算本地学习带来额外收益时，知识流自然停止，系统达到联邦自足平衡。

一句话概括为：

> 异质性提供互补知识，知识缺口产生外部可学习增益；客户端持续吸收有益且可实现的动力学知识，剩余边际增益逐步耗尽，最终知识流归零。

\[
\boxed{
\text{异质性}
\rightarrow
\text{互补知识供给}
\rightarrow
\text{本地知识缺口}
\rightarrow
\text{外部边际增益}
\rightarrow
\text{知识吸收}
\rightarrow
\text{增益耗尽平衡}
}
\]

这里的平衡不要求客户端完全相同。更准确的表述是：可迁移且对本地任务有益的动力学知识已被充分吸收，剩余差异在当前任务、模型和通信约束下不再具有可学习价值。

## 2. 为什么需要修改昨晚的 Method v0

昨晚 v0 主要以响应异质性能量下降和反对称守恒流解释收敛。新的学习过程视角揭示了两个缺口。

第一，响应不同不等于值得学习。某个客户端可能与其他客户端差异很大，但这些差异无法由本地模型实现，或者不会改善本地任务；此时继续降低响应距离会造成负迁移。

第二，知识不是会从发送方耗尽的物理质量。客户端把知识传给别人后仍然保留自己的知识，因此反对称流可以用于抑制公共坐标漂移或限制更新，但不能作为联邦学习的根本守恒定律。知识流停止的根本原因应当是接收方已经没有新的正边际收益。

因此，当前设计把主目标从“最小化响应差异”改为“逐步耗尽外部可学习增益”。响应差异仍用于发现候选互补方向，因果响应核仍用于表达动力学知识，本地可实现性仍用于过滤无效方向。

## 3. 初步优化形式

令 $\mathcal V_i^{\mathrm{loc}}$ 表示客户端 $i$ 在固定计算和更新预算下的本地可行方向，$\mathcal V_i^{\mathrm{fed}}(x_{-i})$ 加入由其他客户端动力学响应产生、并能通过本地模型实现的方向。使用当前冻结快照上的局部任务模型

\[
q_i(v)=g_i^Tv+\frac12v^TH_iv+\frac{\lambda}{2}\|v\|^2.
\]

定义外部知识的预测边际增益

\[
G_i
=
\min_{v\in\mathcal V_i^{\mathrm{loc}}}q_i(v)
-
\min_{v\in\mathcal V_i^{\mathrm{fed}}(x_{-i})}q_i(v)
\ge0.
\]

候选方向在真实模型上回放，并与同一快照、同一计算量和同一更新范数的 local-only 更新比较：

\[
\widehat G_i
=
L_i^{\mathrm{guard}}(\theta_i+\alpha v_i^{\mathrm{loc}})
-
L_i^{\mathrm{guard}}(\theta_i+\alpha v_i^{\mathrm{fed}}).
\]

仅在 $\widehat G_i$ 显著为正、响应实现误差受控且本地任务 guard 通过时接受外部更新。联邦平衡定义为

\[
G_i(\theta_i^*,x_{-i}^*)\le\varepsilon_G,
\qquad\forall i.
\]

若每次接受的外部更新都保证非负实际下降 $\delta_i^r$，且任务损失有下界，则

\[
\sum_r\delta_i^r<\infty,
\qquad
\delta_i^r\rightarrow0.
\]

这为“有限互补知识被逐步吸收，边际收益最终耗尽”提供了条件性的数学表达。它不等价于证明非凸模型参数收敛或测试精度单调上升。

## 4. 动力学知识用什么低维对象表示

正交响应模态只能表示两个低维对象不重合，不能说明接收方缺少一种可学习能力；如果所有客户端已经被放入公共响应坐标，Functional Map 也失去了实际作用。因此当前不再把互补知识定义为“来源奇异向量在接收方正交补中的分量”。

新的低维对象是一组有限时域作用方程。客户端 \(j\) 在自己的低维功能空间中测量

\[
Z_j\mapsto Y_j^\tau=\mathcal P_j^\tau(Z_j).
\]

Functional Map \(C_{j\to i}\) 把输入与来源演化结果一起搬到接收方。接收方用自己的真实 ODE-GNN 执行运输后的输入，比较

\[
E_{j\to i}^\tau
=
\mathcal P_i^\tau(C_{j\to i}Z_j)
-
C_{j\to i}\mathcal P_j^\tau(Z_j).
\]

它衡量“先运输后传播”与“先传播后运输”是否交换。若映射可信、该方程当前未被满足、普通本地反向传播可以学会它、并且更新与本地任务相容，这条方程才是接收方缺失的互补知识。

多个客户端提供的是作用方程集合，而不是待平均的模型或 operator。候选方法不再执行 FedAvg：客户端参数始终留在本地，服务器只维护 Functional Map 网络并路由 action pairs。接收方把小型 action-pair distillation loss 加入本地任务训练，逐步满足仍有价值的外部方程。

对边 \(j\to i\)，定义任务相容的作用流

\[
I_{j\to i}
=-g_{j\to i}\nabla_{\theta_i}
\frac12\|E_{j\to i}(\theta_i)\|^2,
\]

其中 \(g_{j\to i}\) 由映射可信度、留出缺陷和任务梯度相容性控制。客户端沿本地任务梯度与所有入流之和更新。方程被学会后 \(E\to0\)，任务冲突时 \(g=0\)，不可实现时缺陷梯度为零，因此跨客户端作用自然停止。最终平衡要求有效作用流归零，不要求参数一致，也不依赖全局模型。

这一修正也明显减轻了在线计算：不构造完整响应 Jacobian，不做响应 SVD 和接收方正交子空间搜索，也不在每轮运行 local-only counterfactual。Functional Map 与少量 action pairs 可以隔若干联邦轮次更新；matched local-only 只用于早/中/晚 checkpoint 的离线机制审计。

## 5. 已有实验与初步分析

### 5.1 旧 Functional Dynamics 的任务结果

已完成 11 个真实数据集的单种子实验。7 个数据集的点估计为正，4 个为负；描述性平均变化为 `+0.3366` 个百分点，中位数为 `+0.0302` 个百分点，仅 4 个数据集提升超过 `+0.5`。由于混合使用 ACC/AUC 且只有一个种子，这些结果不能构成稳定提升的统计结论。

旧实现的有效注入仅占 native field 的约 `0.57%–1.32%`，10/11 数据集的 spectral clipping 持续生效，所有数据集的 prototype switching 都为零。因此该实验主要验证了工程稳定性，没有验证外部知识被逐步吸收。

### 5.2 回顾性流量分析：降为旁证

对每个完整数据集运行，去除未启用注入的启动轮次，将有效轮次的前 20% 与后 20% 比较；先在客户端和窗口内取中位数，再对 11 个数据集汇总。

| 旧方法代理量 | 下降的数据集数 | 数据集级中位 early | 数据集级中位 late | late/early 中位比 |
| --- | ---: | ---: | ---: | ---: |
| raw correction norm | 11/11 | 10.176 | 4.349 | 0.427 |
| injected/native ratio | 10/11 | 1.203% | 0.896% | 0.730 |
| post-projection safe norm | 1/11 | 1.250 | 1.492 | 1.189 |
| cluster distance | 6/11 | 0.735 | 0.663 | 0.972 |

观察：未经安全投影的候选修正以及相对注入强度大多随训练下降。Roman-empire 的注入比例下降最明显，late/early 为 `0.415`；Photo 基本不下降，为 `1.007`。

这一分析不能指导算子选择，也不能证明知识吸收。raw correction 下降可能来自参数尺度、固定 prototype、Functional Map 或优化收敛；post-projection safe norm 在 10/11 数据集没有下降。当前日志没有同快照 local-only 反事实，无法判断流量下降是知识已被吸收、控制器失效还是任务已经不再可学。因此只保留数值记录，不再作为下一步实验依据。

### 5.3 因果响应核的微型检查

已有 12 节点随机模型 CPU float64 检查得到：zero-port forward 误差为零；因果核 AD 与中心差分相对误差为 `4.025e-11`；未来干预对过去输出的影响为零；扰动幅度减半时非线性余项约缩小四倍。独立 toy 中，理想线性守恒残差为 `6.360e-17`，且正交可达方向会在仍有响应差异时产生零更新。

这些结果支持因果响应可测、有限步误差具有局部二阶规律以及本地可实现性会决定停止流动。它们尚未证明响应与任务相关，也没有证明真实外部知识增益会随训练耗尽。

## 6. 当前判断

本周证据支持把问题从“表示距离”进一步改写为“作用方程是否可运输和可学习”，但 R011 尚未得到实验支持：

- 当前自治 ridge generator 的稳态拟合残差约为 `0.276–0.727`，尚无留出时间或端口响应证据，不能继续默认代表本地动力学；
- 匹配目标下的连续 \(A\) 与 \(P=I+hA\) 代数等价，单纯换成离散传播子不增加表达能力；
- 因果响应核已经通过 zero-port、因果性和 AD/有限差分一致性检查，说明它可被可靠测量；
- 现有 Functional Map solver 已包含算子交换残差，但目前只用于对齐 canonical generator；新设计把它提升为知识缺陷，并必须用独立 probes/horizons 防止循环验证；
- 因果响应微型检查说明有限时域作用可以被可靠测量，但尚未证明 map 语义、方程补全或任务收益。

因此，当前优先级是验证 Functional Map 能否把一条来源动力学作用关系正确搬到接收方，以及普通 action-pair distillation 能否让接收方真正学会它。R010 的响应 SVD 和正交缺失模态不再作为主机制。

## 7. 下一步最小实验

第一阶段构造具有已知跨客户端 Functional Map 的稳定受控系统，并设置三类作用：已经掌握、可迁移但接收方缺失、对接收方任务有害。比较：

1. ground-truth Functional Map；
2. descriptor/cycle learned map；
3. wrong map；
4. identity/no-map；
5. 旧 canonical prototype pullback 与 R010 正交模态传输；
6. 等更新预算的参数平均，仅作为 FedAvg 型对照。

映射只使用 descriptor 与 cycle split 估计，作用交换缺陷使用未参与映射拟合的 probes/horizons 评估。接收方随后用普通 action-pair distillation 学习外部方程，记录 map/cycle error、留出缺陷下降、任务变化、梯度相容性、通信量与运行时间。

只有 learned map 能在留出数据上接近 ground truth、正确运输的可迁移方程同时降低缺陷和任务损失、重复与有害方程自然停止、wrong/no-map、source/time shuffle 和参数平均不能复现这种选择性时，才进入冻结的 `feature_shift`、`structure_homophily` 和 `mixed` A-DGN checkpoint。

若 descriptor/cycle 信息不能辨识有效 Functional Map，或者 ground-truth map 下的来源方程仍不能改善接收方任务，则停止扩展算法，返回跨客户端语义对应或知识可迁移性问题。

## 8. 复现信息

- 当前设计起点提交：`f7b8670`；R010 与 R011 修正文档由本周后续提交记录。工作树仍有两个此前存在的文档删除。
- 11 数据集结果与诊断：`.collab/EXPERIMENTS.md` 的 E003。
- 因果响应数值检查：`.collab/EXPERIMENTS.md` 的 E013 与 `.collab/DYNAMICAL_RESPONSE_CHECKS.md`。
- 当前方法规范：`methodv0.md`。
- 当前 R011 规范：`.collab/FUNCTIONAL_INTERTWINING_DYNAMICS.md`。
- 历史 R010 算子选择：`.collab/LOW_DIMENSIONAL_DYNAMICS_OPERATOR_SELECTION.md`。
- 当前执行 handoff：`.collab/NEXT_TASK.md`，R011，`NEEDS_RESEARCH`。
- 回顾性流量分析命令：`python3 backbone_functional_dynamics_stable/scripts/analyze_flow_decay.py`。
- 本周没有启动新 GPU 训练，没有修改 production source、checkpoint 或数据。
