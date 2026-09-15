# ResKoopNet algorithmic reference

The passive Koopman identification branch is an independent PyTorch
adaptation of the algorithmic workflow described by:

Yuanchao Xu, Kaidi Shao, Nikos Logothetis, and Zhongwei Shen,
"ResKoopNet: Learning Koopman Representations for Complex Dynamics with
Spectral Residuals," ICML 2025.

Reference implementation inspected during development:
`/opt/data/private/xzc/work2/ResKoopNet-main/solver_resdmd_torch.py`.

The integration, checkpoint lifecycle, deterministic node-time sampling,
normalized per-mode residuals, state-space mode comparison, and visualization
pipeline in this package are repository-specific implementations.
