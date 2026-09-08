# 页面维护

算法设计页面的源文件都放在 `algorithm_design/docs/`。新增页面后，在
`algorithm_design/mkdocs.yml` 的 `nav` 中增加一项即可。页面中的公式使用标准
LaTeX 分隔符：行内公式写成 `$z_{t+1}=Kz_t$`，独立公式写成一对 `$$`。

图片放在 `algorithm_design/docs/assets/`，页面中用相对路径引用，例如：

```markdown
![机制图](assets/mechanism.png)
```

参考论文页面放在 `algorithm_design/docs/references/`，每篇论文独立维护；论文
框架图等阅读辅助图片放在 `algorithm_design/docs/assets/references/`。新增论文时，
同时在 `mkdocs.yml` 的 `nav` 中加入页面，并在 `references/index.md` 的清单中登记。

页面允许嵌入 HTML；可复用的交互逻辑放在
`algorithm_design/docs/javascripts/`，样式放在
`algorithm_design/docs/stylesheets/`。这样算法正文保持易读，复杂展示也不需要
把整页写成难以维护的 HTML。

## 本地预览

当前服务器端预览端口是 `8765`。在服务器上使用 `torch` 环境启动：

```bash
/root/anaconda3/envs/torch/bin/mkdocs serve \
  --config-file /opt/data/private/xzc/work2/algorithm_design/mkdocs.yml \
  --dev-addr 127.0.0.1:8765
```

本地电脑建立 SSH 转发：

```bash
ssh -N -L 7891:127.0.0.1:8765 <服务器登录目标>
```

浏览器访问 `http://127.0.0.1:7891/`。MkDocs 会监视 Markdown、图片、JavaScript
和 CSS 的修改并自动刷新页面。

## 迁移安装

当前文档环境使用 `/root/anaconda3/envs/torch`，直接依赖版本记录在
`algorithm_design/requirements-docs.txt`。更换镜像或迁移服务器时：

```bash
TORCH_PYTHON=/root/anaconda3/envs/torch/bin/python
$TORCH_PYTHON -m pip install -r \
  /opt/data/private/xzc/work2/algorithm_design/requirements-docs.txt \
  -i https://mirrors.aliyun.com/pypi/simple
```

安装后检查：

```bash
$TORCH_PYTHON -m mkdocs --version
$TORCH_PYTHON -m pip check
```

MathJax 默认从 CDN 加载。若目标机器或本地浏览器完全离线，需要把 MathJax
运行时下载到 `docs/javascripts/vendor/`，再将 `mkdocs.yml` 中的 CDN 路径替换
为本地文件；页面中的 LaTeX 写法不需要改变。
