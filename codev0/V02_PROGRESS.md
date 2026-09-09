# V0.2 进度保存：2026-09-09

状态：用户追问完整 100 轮联邦及算子正确性，并要求更新网页 V0.2。下方最新联邦状态优先于后面的历史快照。

## 最新：完整联邦验证与网页更新

- Cora 10 客户端、100 rounds、local epoch1、seed42、all-client equal FedAvg、持久 Adam。
- 原 V0.2 dissipative/pooled，lr.003 已完成100轮：ACC59.2913%，F1 18.0451%。
- V0.1 同学习率lr.003 已完成100轮：ACC68.8155%，F1 28.1005%。
- 历史 V0.1 lr.015 为69.8477%。上述全部是每客户端最佳验证轮配对测试均值。
- 改动版首次只完成81轮，worker exit=-9，保留全部中断日志，不计为100轮。
  此时同时运行3组、共15个训练worker；容器内存上限32GiB。已结束本次失败任务
  残留子进程，改用3worker从头重跑100轮；没有修改学习率、算法或数据。
- 正在运行的重跑目录：`logs/Cora_disjoint/clients_10/20260909_013223_method_v02_koopman_v02_full100_refined_retry_20260909`。
- 已完成目录末尾：`20260909_012502_method_v02_koopman_v02_full100_original_20260909`
  和 `20260909_012933_method_v01_linear_v01_full100_matched_lr003_20260909`。
- 新增 `scripts/audit_v02_federated.py`，重载客户端/服务器权重，核验100轮、分区hash和
  选模轮次，区分历史口径、round100本地更新模型与round100最终聚合模型。
- 已复算原 V0.2 全局末轮41.7494%；V0.1 .003 全局末轮40.8091%；历史 .015为53.9617%。
  不能将这些数字与69.85%那一列混用。
- 原版选定联邦checkpoint平均：recNMSE.58785，h16linNMSE.69542，predNMSE.66993，
  K=I推理ACC26.1722%，扰动响应NRMSE.94609。依赖K不等于学到Koopman闭合。
- 已更新网页方法、实验分析、首页和导航。明确完整图生成元/RK4、参考变化、
  与官方DeepKoopman目标梯度的差异、没有表达能力提升证据。
- 全部36项测试已通过（新增5项完整联邦记录核验测试）。完整汇总与最终网页构建待重跑结束。

## 历史：冻结诊断与两项可选改进

- 新增 `models/v02/diagnostics.py`、`scripts/diagnose_v02_frozen.py`、
  `tests/test_v02_diagnostics.py`。
- 冻结原 joint 的 epoch74 checkpoint，24/12/24 条独立种子生成的 5% 特征扰动
  train/validation/held-out 轨迹。只按扰动验证误差选模型，没有使用节点测试标签。
- 原目标潜态范数增长 2.46935 倍；真实图上的 16 步 RK4 算子范数仅 0.97098。
  固定该编码器及算子范数时，潜态 NMSE 下界为 0.36819，实际误差为 0.61231。
- 冻结 E/D 拟合算子：原 stable / stable 再拟合150次 / 普通线性 ridge / affine ridge
  的保留轨迹16步潜态NMSE分别为 0.61236 / 0.43104 / 0.02733 / 0.02709。
  常数项提升很小；没有把仿射项或状态条件网络接入主模型。
- 固定参考重训 E/D：原初态重建 NMSE0.40073，pooled重训后0.03284，
  per-time重训后0.01033；末态分别0.10142 / 0.00815 / 0.01132。
  32维PCA重建误差低，但这是观测状态重建，不是预测成绩。
- 冻结重训同时增加优化预算、去掉分类竞争；不能据此单独认定历史目标漂移是唯一原因。
- 新增 `--generator-mode bounded`：两个一般稠密矩阵、有范数上界，允许潜态增长，
  与默认耗散生成元从相同函数初始化；参数2r²，对照4r²。没有长期收缩保证。
- 新增 `--loss-normalization per_time`：rec/pred/lin 对各时刻分别归一化再平均。
- 两个参数均支持本地和联邦CLI；默认仍为 dissipative / pooled。
- 三组从头训练补充原对照组成2×2，同 client3、seed42、100ep、lr.003：

| Generator / normalization | Val ACC | Test ACC | F1 |
| --- | ---: | ---: | ---: |
| dissipative / pooled（旧结果） | 57.89% | 50.00% | 33.10% |
| bounded / pooled | 55.79% | 50.00% | 36.73% |
| dissipative / per_time | 53.68% | 46.81% | 32.21% |
| bounded / per_time | 57.89% | 53.19% | 38.82% |

组合改动相对原版 Test +3.19pp，但验证持平，仍低于 V0.1 57.45% 和 GCN55.32%。
不能按测试结果替换默认配置。分类与轨迹拟合收益不一致，仍未验证 backbone 优势。

最新结果目录：

- `codev0/run_logs/v02_diagnosis/20260909_frozen_joint_client3/report.md` / `report.json`
- `codev0/run_logs/v02_refinement/20260909_client3/summary.md` / `summary.json`
- `scripts/summarize_v02_pilot.py` 新增 `--reference-result`，可读取旧对照而不重复运行。

网页 `methodv02.md` / `experiments/methodv02.md` 和 `codev0/V02.md` 已补充本轮结果。
本轮结束验证：全部31项测试通过，MkDocs严格构建成功，网页本地预览返回HTTP200，
`git diff --check`通过。旧checkpoint兼容性、两种生成元相同初始函数、有界生成元的
线性/增长/范数上限及新损失的目标梯度隔离都已覆盖。
所有新训练进程均已结束，未运行新联邦实验。后续优先检查任务相关轨迹目标、
联合优化竞争，以及相同起点固定/变化参考的受控对照，不要重复已完成实验。

## 9 月 8 日快照（历史记录）

## 目标与已知背景

用户希望参考四篇 Koopman 论文改进 V0.1 backbone，让解码器参与任务并提高表达能力。
V0.1 的 Cora 10 客户端独立 local 平均 ACC 为 79.80%，100 轮 × local epoch 1 的
FedAvg 为 69.85%。这不是下述单分区 pilot 的比较口径。用户此前要求 Cora、CiteSeer
联邦实验均为 local epoch 1，并优先检验纯 local。

## 本次完成

- 新建 `codev0/models/v02/`：model、training、client、server、logger。
- 模型为共同从头训练的非线性参考轨迹 + 节点级 E/K/D 分类路径。
- 默认隐状态 64、潜态 32、16 步、步长 0.1；Encoder/Decoder 为非线性 MLP。
- 最终分类严格经过 Decoder；部署不需要参考转移或真实未来状态。
- 参考转移由 native CE 训练。辅助编码输入和目标 H 全部 detach，潜态一致性目标也
  detach。共享 Stem 仍接收两条 CE 梯度，因此参考系统在训练中变化，不能当作固定教师。
- 完整损失为 proxy CE + native CE + rec NMSE + prediction NMSE + 0.1 linearity NMSE。
  所有有效起点均执行自由预测，跨度 1/4/8/T；中途没有 teacher forcing。
- 图生成元 A0=S0-D0-D1、A1=S1+D1，Sj 反对称、Dj 半正定，统一正缩放限制范数，
  使用 RK4。连续耗散不等于无条件离散稳定；仅支持非负权对称图。
- 可选周期 E(D(Z))，默认关闭；辅助线性窗口始终不纠偏。
- `main.py`、`myparser.py` 已注册 `v02_koopman`。
- `configs/v02_koopman_gnn.json` 默认 100 rounds × local epoch 1，lr=0.003。
- 联邦保留现有完整模型 FedAvg/持久 Adam，参考分支也会上传；没有实现 MNIW。
- 新建本地 runner `scripts/run_v02_single_client.py`，支持 joint、identity、
  no_dynamics、no_aux、native、v01、gcn 七种对照。
- 新建 `scripts/summarize_v02_pilot.py`，核对协议并生成 summary.md/json。
- 实现文档：`codev0/V02.md`。
- 网页新增 `algorithm_design/docs/methodv02.md` 和
  `algorithm_design/docs/experiments/methodv02.md`；导航与首页已加入 V0.2。

## 验证完成

- 全部 26 项单元/集成测试通过：
  `/root/anaconda3/envs/torch/bin/python -m unittest discover -s codev0/tests -p 'test_*.py' -v`
- 包含 E/K/D 分类梯度、参考目标隔离、推理不读教师、所有窗口自由滚动、
  图置换等变、图生成元耗散、小图精确矩阵指数对照、CPU 客户端训练/日志接口。
- `mkdocs build --strict --config-file algorithm_design/mkdocs.yml` 成功。
- `git diff --check` 通过，CLI help 显示新入口。
- 没有运行 V0.2 多进程联邦训练；CPU 集成测试不等于完成联邦运行验证。
- 本次训练和构建进程均已正常结束；原有 MkDocs serve 属于用户已有预览服务，未停止。

## 已完成实验及准确口径

结果根目录：
`codev0/run_logs/v02_pilot/Cora_client3_seed42_lr003/`

Cora client 3，seed 42，100 local epochs，Adam lr=0.003，weight decay=1e-4，
clip=5；无数据增强，无聚合。严格最佳验证 ACC 选模，同分取更早 checkpoint。
测试标签在训练完成后评估。单客户端单种子诊断，不能与之前 10 客户端平均值直接比较。

| Variant | Best epoch | Val ACC | Test ACC | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| v01 | 59 | 63.16% | 57.45% | 55.23% |
| gcn | 100 | 57.89% | 55.32% | 36.43% |
| native | 69 | 67.37% | 53.19% | 39.39% |
| joint | 74 | 57.89% | 50.00% | 33.10% |
| no_dynamics | 51 | 52.63% | 50.00% | 31.48% |
| identity | 92 | 52.63% | 45.74% | 33.90% |
| no_aux | 33 | 36.84% | 43.62% | 23.04% |

no_dynamics 保留两条 CE 和 AE 重建，仅移除 prediction/linearity；no_aux 则只保留
proxy CE，参考分支没有训练。不能用 no_aux 对比直接归因于多步损失。
GCN 为此处实现的两层 PyG GCN（hidden64、ReLU、dropout0.5），不是论文结果复现。
GCN 在 CPU 上，其他在 GPU 上；更新数相同但参数/FLOPs/硬件未匹配，不作速度比较。

完整模型在 epoch74：

- rec NMSE=0.098891；16 步预测 NMSE=0.219499。
- 16 步增量 NRMSE=0.549051；1 步增量 NRMSE=3.613482。
- latent linearity NMSE：1 步 0.008451，16 步 0.612308。
- ZT/Z0=0.955033；潜态谱熵有效秩=2.440827（潜维32，不能解释为严格内在维度）。
- 一个保留 2% 乘性特征扰动：response NRMSE=0.799635，reference RMS=0.009506。
- 同一 checkpoint 推理干预：K=I ACC40.43%；潜态置零27.66%；每4步纠偏47.87%。
- 完整模型训练分配参数217582，部署路径208871；V0.1为187911；GCN为92231。

结论：代码解决了 decoder 不参与任务、没有独立转移监督的问题；尚未实现准确率提升。
完整模型和 no_dynamics 的 Test ACC 相同，多步监督带来分类增益也未被证实。
本次结果支持继续定位表示/目标/算子瓶颈，而不是直接宣称优于 GCN 或真正完成 Koopman 闭合。

## 恢复后的建议工作（尚未执行）

1. 先阅读 V02.md、网页实验分析和原始 result.json；保留当前试验结果。
2. 在固定参考 checkpoint 上区分 AE 表示瓶颈与固定线性算子瓶颈；检查目标漂移、
   初态重建偏差、低有效秩与多步误差，使用验证数据决定后续调整。
3. 若改进有依据，再做更多客户端/种子的 local 验证；当前不应把单分区 pilot 当完整性能表。
4. 本地骨干通过后才进入完整 100 轮联邦实验；保持 Cora/CiteSeer local epoch1。

## 恢复注意事项

- 可用 Python：`/root/anaconda3/envs/torch/bin/python`，torch1.13.1+cu117、PyG2.6.1。
  系统 `/usr/bin/python` 没有 torch。
- 三份官方参考代码在 `GitHub/DeepKoopman`、`GitHub/balanced-neural-odes`、
  `GitHub/MetaKoopman`；Course Correcting 没有已找到的官方仓库。
- 工作树原先已有大量用户暂存的删除/移动/修改，且本次部分新增修改也已被外部暂存；
  助手未执行 git add/commit/reset。不要清理或覆盖这些变更。
- `no_aux` 的早期已保存 result.json 没有后加的 reference_trained 标记；其参考动力学
  指标来自未训练分支，不可用于 Koopman 保真结论。新 runner 已明确写入该标记。
- 不需要重复已通过的测试/已完成的七组实验，除非有新修改或新的验证问题。
