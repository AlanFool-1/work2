---
title: Method V0.2：联合轨迹监督的 Graph Koopman Backbone
description: 用非线性参考动力学监督节点级 E–K–D，并将解码器接入真实分类路径。
---

# Method V0.2：联合轨迹监督的 Graph Koopman Backbone

版本：2026-09-08 UTC · 首版代码与本地诊断已完成。

!!! warning "当前结论"
    V0.2 已让解码器参与分类，并加入可独立检验的参考状态转移与多步监督。首轮 Cora 单分区试验尚未超过 V0.1 / GCN；它是一个可验证的候选 backbone，尚不是已证实有效的改进方法。

[查看本地实验](experiments/methodv02.md){ .md-button .md-button--primary }
[回看 V0.1 实现](methodv01-implementation.md){ .md-button }

## 1. 两条路径，共同从头训练

```text
X、P、degree → Stem → H0
                      ├→ 非线性图传播 H1 … HT → native readout → 训练标签 CE
                      │             │
                      │             └→ detach → 状态重建 / 多步预测 / 潜态一致性目标
                      │
                      └→ Encoder → Z0 → 固定线性图算子滚动 → ZT → Decoder → readout → 训练标签 CE
```

第二条路径就是部署路径。推理不运行非线性参考传播，也不读取任何真实未来状态。Decoder 为共享节点 MLP，不接收原始输入、节点 ID、传播步数或参考未来状态。默认每节点隐状态为 64 维，潜态为 32 维。

### 参考状态

$$
H_0=\operatorname{Stem}([X,PX,\log(1+d)]),\qquad
H_{t+1}=H_t+\eta\tanh(H_tW_s+PH_tW_n+b).
$$

它使用独立分类头进行监督。参考转移参数只通过自身的分类任务学习；辅助损失中的 $H_t$ 及编码器输入均截断梯度。共享 Stem 仍接收两条分类路径的梯度，因此训练中的参考系统会变化。截断梯度阻断了直接通过移动目标降低辅助误差的路径，但不保证参考系统永不退化，也不等于已识别出一个固定物理系统。

### 编解码与预测

$$
Z_t=E([H_t,PH_t]),\qquad
\widehat Z_{t+s}=K_G^sZ_t,\qquad
\widehat H_{t+s}=D(\widehat Z_{t+s}),\qquad
\widehat Y=R(\widehat H_T).
$$

这里 $K_G$ 是固定图上一次 RK4 推进所定义的线性映射。图算子、模型参数与步长固定时，它对节点潜状态保持线性；它不是只作用在单个节点上的一个任意 $r\times r$ 矩阵。

## 2. 约束完整的图生成元

$$
\dot Z=ZA_0+PZA_1,
\qquad A_0=S_0-D_0-D_1,\quad A_1=S_1+D_1,
$$

$$
S_j=Q_j-Q_j^\top,\qquad
D_0=B_0B_0^\top+\gamma I,\qquad D_1=B_1B_1^\top.
$$

对非负权无向图，对称归一化 $P$ 的特征值 $\lambda\in[-1,1]$。对应图频率上的生成元为

$$
A(\lambda)=S_0+\lambda S_1-D_0-(1-\lambda)D_1,
$$

其对称部分非正。代码再对 $A_0,A_1$ 共同乘一个正缩放系数，使 $\|A_0\|_F+\|A_1\|_F\le4$；该操作保留上述耗散性质。默认 $\eta=0.1$、16 步；RK4 子步数由配置上界确定。

这是连续时间的耗散保证。RK4 的离散误差另外通过小图精确矩阵指数对照与实际范数诊断检查。代码拒绝非对称图和负权边，不自动改写输入图。这个约束限制了潜演化的可选范围，不能单独据此主张表达能力更强。

## 3. 实際优化的五项损失

$$
\mathcal L=\mathcal L_{\mathrm{CE,proxy}}+\mathcal L_{\mathrm{CE,native}}
+\mathcal L_{\mathrm{rec}}+\mathcal L_{\mathrm{pred}}+0.1\mathcal L_{\mathrm{lin}}.
$$

| 项目 | 预测与目标 |
| --- | --- |
| proxy CE | $R(D(K_G^TE(H_0)))$ 与训练标签 |
| native CE | 参考动力学的独立读出与训练标签 |
| rec | 所有真实时刻的 $D(E(H_t))$ 与截断梯度的 $H_t$ |
| pred | 所有有效起点的 $D(K_G^sE(H_t))$ 与截断梯度的 $H_{t+s}$ |
| lin | $K_G^sE(H_t)$ 与截断梯度的 $E(H_{t+s})$ |

预测跨度为 $\{1,4,8,T\}$，去重并限制在轨迹长度内。每条预测链只编码起点，后续完全使用自己的预测。NMSE 用目标能量归一化，分母截断梯度并设置数值下界。分类只读取 `train_mask`；重建可以使用当前 transductive 分区的全部无标签特征。

选模只依赖验证 ACC/AUC，同分时保留较早 checkpoint。测试标签在训练和选模完成后才使用。另报告相对真实状态增量的预测误差、潜态有效秩、一次保留扰动的响应误差，避免仅凭总体重建 NMSE 判断动力学成立。

## 4. 对四篇论文实际借用了什么

| 来源 | V0.2 当前实现 | 边界 |
| --- | --- | --- |
| [DeepKoopman](references/deepkoopman.md) | 真实状态重建、原状态空间多步预测、潜态一致性 | 没有照搬状态相关 `omega_net`；本版保留固定算子 |
| [Balanced Neural ODEs](references/balanced-neural-odes.md) | 连续潜动力学、全轨迹解码与容量诊断的思路 | 耗散图生成元是本项目设计；有效秩不是 B-NODE 的 KL 活跃维度 |
| [Course Correcting](references/course-correcting.md) | 可选每若干步 $Z\leftarrow E(D(Z))$ | 默认关闭；辅助线性窗口始终不纠偏；已做推理期对照 |
| [MetaKoopman](references/metakoopman.md) | 尚未接入后验适配 | 需要先验证转移保真及共享坐标 |

前三篇对应的实现依据详见本地 `GitHub/REFERENCE_CODE_MAPPING.md`。其中 Course Correcting 在已有参考检查中没有找到官方仓库，只依据已整理的论文机制；没有把第三方实现当作官方代码。

## 5. 代码入口与已验证范围

| 文件 | 作用 |
| --- | --- |
| `codev0/models/v02/model.py` | 参考轨迹、图生成元、E/K/D、全起点自由滚动 |
| `codev0/models/v02/training.py` | 本地与联邦共用目标、梯度检查、轨迹诊断 |
| `codev0/models/v02/client.py` / `server.py` / `logger.py` | 注册完整模型 FedAvg，保留客户端 Adam 状态 |
| `codev0/scripts/run_v02_single_client.py` | 本地训练及七种对照 |
| `codev0/scripts/summarize_v02_pilot.py` | 核对协议并汇总结果 |
| `codev0/configs/v02_koopman_gnn.json` | Cora/CiteSeer：100 轮、local epoch 1 |

```bash
/root/anaconda3/envs/torch/bin/python codev0/scripts/run_v02_single_client.py \
  --dataset Cora --client-id 3 --epochs 100 --variant joint --device cuda:0
```

已检查分类梯度经过 Decoder、参考分支不进入推理、辅助目标梯度隔离、所有起点自由滚动、图置换等变、生成元耗散、RK4 数值精度和客户端训练/日志接口。完整 V0.2 联邦实验尚未运行。

V0.2 的纯线性潜路径仍可写成非线性编解码器之间的受约束图滤波。它增加了可检验的轨迹学习目标和参与任务的解码器，但尚不能证明超越一般非线性 GNN 的表达能力；当前更准确的表述是“学习图传播深度上的 Koopman 表示”。
