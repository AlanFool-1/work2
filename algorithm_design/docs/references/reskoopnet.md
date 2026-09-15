---
title: ResKoopNet
description: >-
  从论文思路与本地 PyTorch 源码拆解 ResKoopNet 的可学习 observable dictionary、
  闭式 Koopman 识别和 spectral residual，并界定它向联邦图动力学迁移时的坐标与实现边界。
tags:
  - Koopman operator
  - Spectral residual
  - Dictionary learning
  - EDMD
  - Source audit
---

# ResKoopNet：用 spectral residual 学习 Koopman 表示

**论文**: *ResKoopNet: Learning Koopman Representations for Complex Dynamics with Spectral Residuals*  
**作者**: Yuanchao Xu, Kaidi Shao, Nikos Logothetis, Zhongwei Shen  
**会议**: ICML 2025  
**论文页面**: [PMLR](https://proceedings.mlr.press/v267/xu25y.html) · [OpenReview](https://openreview.net/forum?id=Svk7jjhlSu)  
**本地材料**: `reskoopnet.md`、`ResKoopNet-main/`

!!! info "证据边界"
    本页的算法解释同时核对了本地说明、README 和
    `solver_resdmd_torch.py`。原项目处理普通向量状态上的动力系统，不包含图传播、
    联邦聚合或外部力反馈；这些内容属于本项目的后续适配，不是论文已经验证的结论。

## 一句话总结

普通 EDMD 在一个预先给定的 dictionary 上拟合线性算子；ResKoopNet 交替执行
**闭式估计算子、谱分解、再用 eigenfunction residual 更新 dictionary**，使学习到的
observable 更直接地暴露可靠的 Koopman 谱结构。

<div class="research-flow">
<div class="flow-step"><span>01</span><strong>快照对</strong><small>$X,Y\in\mathbb R^{M\times d}$</small></div>
<div class="flow-arrow" aria-hidden="true">→</div>
<div class="flow-step"><span>02</span><strong>可学习字典</strong><small>$\Psi_\theta=[1,x,\psi_\theta(x)]$</small></div>
<div class="flow-arrow" aria-hidden="true">→</div>
<div class="flow-step"><span>03</span><strong>闭式 EDMD</strong><small>$K=(\Psi_X^\top\Psi_X+\sigma I)^\dagger\Psi_X^\top\Psi_Y$</small></div>
<div class="flow-arrow" aria-hidden="true">→</div>
<div class="flow-step"><span>04</span><strong>谱残差</strong><small>$(\Psi_Y-\Psi_XK)V$</small></div>
</div>

## 1. 输入、dictionary 与维度

给定一步 snapshot pairs：

$$
X=[x_1,\ldots,x_M]^\top\in\mathbb R^{M\times d},\qquad
Y=[y_1,\ldots,y_M]^\top\in\mathbb R^{M\times d},
$$

其中 $y_i=F(x_i)$。`KoopmanNNTorch.forward` 构造

$$
\boxed{
\Psi_\theta(x)=
\left[1,\ x_1,\ldots,x_d,\ \psi_{\theta,1}(x),\ldots,
\psi_{\theta,q}(x)\right]
}
$$

并得到

$$
\Psi_X,\Psi_Y\in\mathbb R^{M\times r},
\qquad r=1+d+q.
$$

固定常数项和原始坐标有两个直接作用：dictionary 不会完全塌缩为零；同时原状态
已经是 observable 的确定切片，后续可直接从该切片恢复状态预测，不必先假定一个
额外 decoder。

## 2. 闭式 Koopman operator

在当前 dictionary 下，代码求解右作用的离散算子

$$
\Psi_Y\approx \Psi_XK,
$$

$$
\boxed{
K=(\Psi_X^\top\Psi_X+\sigma I)^{\dagger}
\Psi_X^\top\Psi_Y
},\qquad K\in\mathbb R^{r\times r}.
$$

这里的 $K$ 不是独立训练的神经网络参数。每次 $\theta$ 攊变后都应重新计算
$\Psi_X,\Psi_Y$ 和 $K$。正则项 $\sigma I$ 控制病态 Gram matrix；使用伪逆不能替代
对 condition number、有效秩和非有限值的诊断。

## 3. Spectral residual

对 $K$ 做右特征分解：

$$
KV=V\Lambda.
$$

第 $j$ 个候选 eigenfunction 为

$$
\varphi_j(x)=\Psi_\theta(x)v_j.
$$

若它满足 Koopman 演化，应有

$$
\varphi_j(y)=\lambda_j\varphi_j(x).
$$

因此整体谱残差为

$$
R_\theta
=\Psi_YV-\Psi_XV\Lambda
=(\Psi_Y-\Psi_XK)V,
$$

$$
\boxed{
\mathcal L_{\mathrm{spec}}=
\frac{1}{M}\lVert R_\theta\rVert_F^2
}.
$$

与只压低 $\lVert\Psi_Y-\Psi_XK\rVert_F$ 相比，这个目标直接检查候选谱方向是否
服从一步演化。残差较小表示该 mode 在观测数据上更可信，但不自动表示它对节点分类
或跨客户端迁移有帮助。

## 4. 交替优化流程

一次外循环应当遵循：

1. 用当前 $\Psi_\theta$ 计算 $K$；
2. 在 CPU/NumPy 或受控的线性代数路径上得到 $(V,\Lambda)$；
3. 固定当前 $K,V,\Lambda$，用 $\mathcal L_{\mathrm{spec}}$ 更新 dictionary；
4. dictionary 更新后重新构造 $K$，继续下一轮。

也就是

$$
\boxed{
\theta\rightarrow\Psi_\theta\rightarrow K
\rightarrow(V,\Lambda)\rightarrow R_\theta\rightarrow\theta
}.
$$

这是一种交替识别，而不是让 autograd 一次穿过伪逆和复特征分解的端到端网络。
实现时必须明确哪些量处于 `no_grad`/`detach` 边界，不能靠隐式 tensor copy 决定梯度。

## 5. Koopman modes 与可靠性

由于原状态位于 dictionary 的第 $2$ 到第 $d+1$ 列，可定义选择矩阵

$$
B=
\begin{bmatrix}
0_{1\times d}\\
I_d\\
0_{q\times d}
\end{bmatrix}
\in\mathbb R^{r\times d},
\qquad x=\Psi(x)B.
$$

若 $K=V\Lambda V^{-1}$，原状态上的 modes 可由 $V^{-1}B$ 的相应布局得到。跨客户端
使用这些 modes 时仍需处理 eigenvector 缩放/符号、共轭对、重复特征值对应子空间和
mode permutation；按 eigenvalue 排序并不能解决坐标对齐。

## 6. 本地 PyTorch 源码审计

| 源码位置 | 实际行为 | 对重实现的要求 |
| --- | --- | --- |
| `KoopmanNNTorch.forward` | 拼接常数、原坐标和 MLP basis。 | 保留固定列顺序，并对输入状态做一致归一化。 |
| `compute_K` | 用 `pinv(reg I + Psi_X.T @ Psi_X)` 求 $K$。 | 使用当前 device/dtype，记录秩、条件数和 fit residual。 |
| `eig_decomp` | 转 NumPy 调用 `numpy.linalg.eig`。 | 明确这是外循环的冻结谱步骤，并正确处理复数。 |
| `KoopmanModelTorch.forward` | 意图计算 $\Psi_YV-\Psi_XV\Lambda$，但 `nn.Linear` 实际右乘所存权重的转置。 | 用显式矩阵乘法实现，并用已知 $V,\Lambda$ 的 shape/value test 固定方向。 |
| `compute_mode` | 通过原状态选择矩阵恢复 modes。 | 先写 shape test，再决定对外的 mode 轴布局。 |

!!! warning "发布代码不能直接作为项目模块导入"
    本地 `solver_resdmd_torch.py` 是研究原型：全局硬编码 `device='cuda'` 和默认
    `float64`；layer 构造中的 `layer_sizes[ii - 1]` 可疑；`forward` 通过
    `torch.tensor(psi_x, dtype=complex128)` 重新包装网络输出，会切断 dictionary 的
    梯度；`calc_psi_next` 引用了未定义的 `tf`；`build` 被定义两次；`layer_K` 被赋值
    但没有参与 residual forward。项目实现应复写数学核心并加入单元测试，而不是复制
    这个 solver 的控制流。

## 7. 映射到图传播轨迹

对客户端 $m$ 的图动力学轨迹

$$
\mathcal H_m=\left\lbrace H_m^0,\ldots,H_m^S\right\rbrace,
\qquad H_m^s\in\mathbb R^{N_m\times d},
$$

可把时间和节点轴合并为样本轴：

$$
X_m^{\mathrm{koop}}
=\operatorname{reshape}([H_m^0,\ldots,H_m^{S-1}])
\in\mathbb R^{SN_m\times d},
$$

$$
Y_m^{\mathrm{koop}}
=\operatorname{reshape}([H_m^1,\ldots,H_m^S])
\in\mathbb R^{SN_m\times d}.
$$

不同客户端可以有不同的 $N_m$，因为节点只占样本行；真正必须共享的是 hidden feature
坐标、状态归一化和 dictionary 的列语义。

例如 $S=16,N_m=600,d=64,q=31$ 时，$M=9600,r=96$：

| 对象 | 形状 | FP32 规模 |
| --- | --- | --- |
| $X_m^{\mathrm{koop}}$ | $9600\times64$ | 约 2.34 MiB |
| $\Psi_X$ | $9600\times96$ | 约 3.52 MiB |
| $K_m$ | $96\times96$ | 36 KiB |

这说明通信一个小 $K_m$ 很便宜，但本地 dictionary 激活和伪逆仍有实际计算成本。

## 8. 联邦迁移的硬边界

!!! danger "未对齐 dictionary 时不能平均 $K_m$"
    若每个客户端独立学习 $\Psi_{\theta_m}$，则 $K_m$ 分别表示在不同 basis 下的矩阵。
    即使形状相同，逐元素 FedAvg 也没有坐标不变性。第一版必须冻结同一个共享
    dictionary、固定 observable 顺序和归一化；否则应改做显式 basis transport、公共
    probe 响应对齐或 mode/subspace matching。

对 [Method V1.0](../methodv10.md) 最可迁移的部分是：

- 从现有 FedDGN rollout 构造 snapshot pairs；
- 用 $[1,H,\psi(H)]$ 保留可直接读取的原状态坐标；
- 闭式估计低维 $K_m$，并以 spectral residual 记录可靠性；
- 在共享坐标通过验证之后，才把 $K_m$ 作为联邦通信对象。

第一版不直接迁移独立 dictionary、复数 mode 聚类和 matrix-log generator。它们分别
需要坐标对齐、共轭/退化谱处理和实数稳定性门控。

## 引用

```bibtex
@inproceedings{xu2025reskoopnet,
  title     = {ResKoopNet: Learning Koopman Representations for Complex Dynamics with Spectral Residuals},
  author    = {Xu, Yuanchao and Shao, Kaidi and Logothetis, Nikos and Shen, Zhongwei},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  year      = {2025},
  publisher = {PMLR},
  series    = {ICML '25},
  url       = {https://openreview.net/forum?id=Svk7jjhlSu}
}
```
