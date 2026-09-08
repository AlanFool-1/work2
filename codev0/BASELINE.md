# Official A-DGN + FedAvg Baseline

## 定义

- 图模型直接加载 `Anti-SymmetricDGN-main` 的官方 `GraphAntiSymmetricNN`；项目内不保留自研 A-DGN、GRAND 或 TANGO 分支。
- 联邦训练沿用 Fedrated 风格的 10 个常驻 worker、客户端持久 Adam 状态、每轮全局模型广播与全模型 FedAvg。
- 默认聚合为等权 FedAvg；评价指标取每个客户端验证集最佳轮对应的测试指标，再在客户端间求均值和标准差。
- A-DGN 使用共享向量场的 16 次固定步长 Euler 更新，`epsilon=0.1`、`gamma=0.1`、`tanh`。
- 官方 A-DGN 在模型内部直接展开固定步长 Euler，因此不依赖 `torchdiffeq`。

## 运行

```bash
bash /opt/data/private/xzc/work2/codev0/scripts/run_official_adgn_all11.sh
```

单数据集：

```bash
/root/anaconda3/envs/torch/bin/python \
  /opt/data/private/xzc/work2/codev0/main.py \
  --dataset Cora --gpu 0,1 --n-workers 10
```

## 本次基线结果

本次运行目录为 `run_logs/official_adgn_k16_all11_20260908_040956/`；模型检查点和
完整训练日志保存在工作区根目录对应的 `checkpoints/` 与 `logs/` 下。

| Dataset | Metric | Mean ± client std (%) |
|---|---:|---:|
| Cora | ACC | 80.2680 ± 11.3101 |
| CiteSeer | ACC | 76.1752 ± 10.2351 |

结果按每个客户端最佳验证轮配对测试指标汇总，和官方 A-DGN/FedAvg 生命周期一致。

## V0 代理

代理不修改上述训练路径，只读取选定的 `0_state.pt` 和对应客户端图，运行：

```bash
/root/anaconda3/envs/torch/bin/python \
  /opt/data/private/xzc/work2/codev0/scripts/run_koopman_proxy_v0.py \
  --dataset Cora --client-id 0 --device cuda:0
```

`proxy/graph_koopman_v0.py` 实现图条件、变分编码、固定潜空间 `K`、节点解码和
状态/logit/响应评估。代理运行产物放在 `codev0/run_logs/proxy_v0/`；实验分析见算法
设计站的 `实验分析` 页面。
