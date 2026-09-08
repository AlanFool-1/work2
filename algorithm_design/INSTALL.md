# Algorithm Design 文档站安装与迁移

本项目使用 MkDocs Material。当前已安装到 `/root/anaconda3/envs/torch`，不要
安装到系统 Python 或另一个未记录的环境。

## 当前版本

- Python 3.9.23
- MkDocs 1.6.1
- MkDocs Material 9.7.7
- Python Markdown Extensions 10.21.3

直接依赖记录在 `requirements-docs.txt`。迁移到新服务器或更换镜像时执行：

```bash
TORCH_PYTHON=/root/anaconda3/envs/torch/bin/python
$TORCH_PYTHON -m pip install -r requirements-docs.txt \
  -i https://mirrors.aliyun.com/pypi/simple
$TORCH_PYTHON -m pip check
```

当前 `torch` 环境还包含项目原有的 GPU、数据处理和图学习依赖；如果 `pip check`
报告这些历史依赖冲突，不要为了文档站自动升级或删除它们。文档站自身的直接依赖
以 `requirements-docs.txt` 为准，可用 `pip show mkdocs-material` 检查安装版本。

如果新服务器的环境路径不同，只修改 `TORCH_PYTHON`，不要改页面源文件。

## 启动预览

```bash
/root/anaconda3/envs/torch/bin/mkdocs serve \
  --config-file /opt/data/private/xzc/work2/algorithm_design/mkdocs.yml \
  --dev-addr 127.0.0.1:8765
```

本地 SSH 转发：

```bash
ssh -N -L 7891:127.0.0.1:8765 <服务器登录目标>
```

浏览器打开 `http://127.0.0.1:7891/`。
