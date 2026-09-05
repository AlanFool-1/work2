# Official A-DGN + FedAvg Baseline

> 当前默认数据集已切换为 `Synthetic`，默认场景为 `feature_shift`。虚拟图生成方式、异质性旋钮和可视化输出见 `SYNTHETIC_BENCHMARK.md`。原有真实数据集运行方式保持兼容，显式传入 `--dataset Cora` 等即可。

## 定义

- 图模型直接加载 `Anti-SymmetricDGN-main` 的官方 `GraphAntiSymmetricNN`；项目内不保留自研 A-DGN、GRAND 或 TANGO 分支。
- 联邦训练沿用 Fedrated 风格的 10 个常驻 worker、客户端持久 Adam 状态、每轮全局模型广播与全模型 FedAvg。
- 默认聚合为等权 FedAvg；评价指标取每个客户端验证集最佳轮对应的测试指标，再在客户端间求均值和标准差。
- A-DGN 使用共享向量场的 16 次固定步长 Euler 更新，`epsilon=0.1`、`gamma=0.1`、`tanh`。
- 官方 A-DGN 在模型内部直接展开固定步长 Euler，因此不依赖 `torchdiffeq`。

## 运行

```bash
bash /opt/data/private/xzc/work2/FunctionalDualEnergyGFL_true_tango/scripts/run_official_adgn_all11.sh
```

单数据集：

```bash
/root/anaconda3/envs/torch/bin/python \
  /opt/data/private/xzc/work2/FunctionalDualEnergyGFL_true_tango/main.py \
  --dataset Cora --gpu 0,1 --n-workers 10
```

## 已完成的正式结果

汇总文件：`BASELINE_RESULTS.json`。逐轮原始记录仍保存在全局
`logs/<dataset>_disjoint/clients_10/*official_adgn_k16_all11_20260903_022044*/`。

| Dataset | Metric | Mean ± client std (%) |
|---|---:|---:|
| Cora | ACC | 80.2680 ± 11.3101 |
| CiteSeer | ACC | 74.9667 ± 11.4351 |
| PubMed | ACC | 86.6107 ± 3.2900 |
| Computers | ACC | 84.2061 ± 9.8232 |
| Photo | ACC | 89.2568 ± 6.0502 |
| ogbn-arxiv | ACC | 66.3899 ± 6.6464 |
| Roman-empire | ACC | 70.4883 ± 1.7989 |
| Amazon-ratings | ACC | 41.8015 ± 5.0988 |
| Minesweeper | AUC | 87.2255 ± 3.1642 |
| Tolokers | AUC | 74.6877 ± 6.5675 |
| Questions | AUC | 68.7566 ± 4.8365 |
