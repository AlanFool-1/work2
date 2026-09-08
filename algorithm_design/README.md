# Algorithm Design Workspace

这里是算法设计文档站的固定维护目录。Markdown 页面位于 `docs/`，由 MkDocs
Material 生成网页；公式、图片、交互 HTML/JS 和自定义布局都可以在同一个站点中
维护。

常用入口：

- `docs/index.md`：站点首页
- `docs/methodv0.md`：Method V0 设计页面
- `docs/guide/authoring.md`：页面维护和迁移说明
- `INSTALL.md`：环境安装记录

服务器端预览使用 `127.0.0.1:8765`。本地电脑建立 SSH 转发后访问
`http://127.0.0.1:7891/`：

```bash
ssh -N -L 7891:127.0.0.1:8765 <服务器登录目标>
```

服务器端启动：

```bash
./algorithm_design/serve.sh 8765
```
