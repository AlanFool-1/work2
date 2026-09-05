# 双 Codex 协作工作区设计

## 1. 目标

当前开发环境中同时使用两个终端、两个 Linux 用户和两个不同能力层级的 Codex：

- `root` 用户：使用 GPT-6，主要负责高难度工作，包括算法设计、理论分析、实验诊断、idea 改进和下一步研究方案设计。
- `xzc` 用户：使用 GPT-5.6，主要负责代码实现、调试、运行实验、排查工程问题和完成相对确定的实现任务。

希望二者共享同一个项目上下文，使：

1. GPT-6 完成新的算法设计后，可以直接把设计方案写入共享工作区；
2. GPT-5.6 不需要用户重新解释上一端做了什么，而是自动从共享文件中获取最新任务；
3. GPT-5.6 实现或实验结束后，将结果、失败原因和待分析问题写回共享工作区；
4. GPT-6 下一次进入项目时，可以直接读取实现结果并继续分析；
5. 人只负责决定大的研究方向，而不再充当两个 Agent 之间的“消息转发器”。

核心原则是：

> 两个 Codex 不共享聊天上下文，而是共享一个持续更新的项目状态。

---

# 2. 总体架构

两个 Codex 使用不同的配置目录和模型，但进入同一个项目目录。

例如项目目录：

```bash
/opt/data/private/xzc/work2/FunctionalDualEnergyGFL_backbone
```

root 端：

```bash
export CODEX_HOME=/root/.codex-gpt6
cd /opt/data/private/xzc/work2/FunctionalDualEnergyGFL_backbone
codex
```

xzc 端：

```bash
export CODEX_HOME=/home/xzc/.codex
cd /opt/data/private/xzc/work2/FunctionalDualEnergyGFL_backbone
codex
```

因此：

```text
root / GPT-6
        │
        │
        ▼
 ┌───────────────────────────────┐
 │        Shared Repository      │
 │                               │
 │  source code                  │
 │  AGENTS.md                    │
 │  .collab/                     │
 │  git history                  │
 └───────────────────────────────┘
        ▲
        │
        │
xzc / GPT-5.6
```

两个 Codex 的：

- 账号配置独立；
- 模型独立；
- Codex 历史独立；
- Linux 用户独立；

但共享：

- 代码；
- Git 状态；
- 当前算法设计；
- 当前实验结果；
- 当前待实现任务；
- 关键设计决策。

---

# 3. 推荐目录结构

在项目根目录增加：

```text
FunctionalDualEnergyGFL_backbone/
│
├── AGENTS.md
│
├── .collab/
│   ├── PROJECT_STATE.md
│   ├── NEXT_TASK.md
│   ├── DECISIONS.md
│   ├── EXPERIMENTS.md
│   │
│   └── logs/
│       ├── research.md
│       └── implementation.md
│
├── src/
├── models/
├── server/
├── trainer/
├── ...
└── .git/
```

其中 `.collab/` 是两个 Codex 的主要通信区域。

不要把完整聊天记录存进去。

共享文件应该保存的是：

> 当前仍然有效、下一位 Agent 真正需要知道的信息。

---

# 4. 各共享文件职责

## 4.1 `PROJECT_STATE.md`

保存整个项目当前的高层状态。

它回答：

> “这个项目现在整体做到哪里了？”

例如：

```markdown
# Current Project State

## High-level Idea

ODE-GNN 将图联邦异质性拆成两个动力学入口：

- feature heterogeneity -> initial state H0
- structure heterogeneity -> propagation operator / vector field

联邦协作目标不是直接平均所有动力学，而是寻找跨客户端可共享的动力学知识。

## Current Backbone

- Backbone: A-DGN / TANGO B2
- Federation: FedAvg
- Optimizer: persistent Adam
- Task: node classification
- Evaluation: per-client best-validation paired test

## Feature Branch

Current version:
Feature Score-Dynamics

Status:
Experimental

Known issue:
Cross-client score alignment is unstable.

## Structure Branch

Status:
Not implemented.

## Synthetic Experiments

Current observations:
- dynamics alignment is relatively stable;
- descriptor residual decreases;
- cross-client generator alignment remains problematic.

## Known Problems

1. Feature correction is still unstable.
2. Structure branch has not been implemented.
3. Need matched no-injection baseline.
```

这个文件应该保持**高层、长期、稳定**。

不要每次小 bug 都写进去。

---

# 5. `NEXT_TASK.md`：核心通信接口

这是整个双 Agent 工作流中最重要的文件。

它回答：

> “下一位 Agent 现在具体应该做什么？”

推荐格式：

```markdown
# Current Handoff

Status: READY_FOR_IMPLEMENTATION

Owner: implementation-agent

## Goal

重新设计 feature heterogeneity，使其只作用于 ODE initial state。

## Problem

当前 DSM score 在跨客户端聚合过程中 alignment 不稳定。

## Motivation

Feature heterogeneity 从动力学角度应主要表现为初始状态分布差异。

因此新的设计应该：

- 保持原 ODE dynamics 不变；
- 只修正 H0；
- 避免重新引入 Gram/PCA/covariance 等旧路线。

## Proposed Design

1. ...
2. ...
3. ...

## Expected Code Changes

Likely relevant files:

- models/...
- trainer/...
- server/...

## Invariants

必须保持：

- FedAvg 流程不变；
- TANGO/A-DGN backbone 不变；
- persistent optimizer state 不变；
- evaluation protocol 不变。

## Diagnostics

新增：

- ...
- ...
- ...

## Acceptance Criteria

1. baseline 可以完全复现；
2. 新模块可通过 flag 完全关闭；
3. 新模块关闭时结果应与 backbone 一致；
4. 完成 synthetic smoke test；
5. 输出必要的 diagnostics。
```

这样 GPT-6 不再只给用户解释设计，而是在完成研究工作后主动写成一份：

> implementation-ready specification

GPT-5.6 直接按照这个文件编码。

---

# 6. 状态机

`NEXT_TASK.md` 的 `Status` 建议只使用有限几个固定状态。

## 6.1 `READY_FOR_IMPLEMENTATION`

表示：

> GPT-6 已经完成设计，等待 GPT-5.6 实现。

例如：

```text
Status: READY_FOR_IMPLEMENTATION
Owner: implementation-agent
```

---

## 6.2 `IMPLEMENTING`

表示：

> GPT-5.6 正在实现。

---

## 6.3 `IMPLEMENTED`

表示：

> 实现完成，实验结果可供 GPT-6 检查。

例如：

```text
Status: IMPLEMENTED
Owner: research-agent
```

---

## 6.4 `NEEDS_RESEARCH`

表示：

> 实现或实验发现设计问题，需要 GPT-6 重新分析。

此时必须记录：

```markdown
## Failure

Observed:
...

## Evidence

...

## Suspected Cause

...

## Research Question

需要判断：

1. ...
2. ...
```

这样 GPT-6 下次启动后无需重新询问 GPT-5.6。

---

## 6.5 `BLOCKED`

仅用于真正的工程阻塞，例如：

- 环境无法启动；
- 数据缺失；
- CUDA 问题；
- 依赖无法安装；
- 文件权限问题。

不要把“效果不好”写成 `BLOCKED`。

效果不好通常应该是：

```text
NEEDS_RESEARCH
```

---

# 7. `DECISIONS.md`

用于保存重要设计决策。

它回答：

> “哪些路线已经明确采用或明确放弃？”

例如：

```markdown
# Design Decisions

## D001 — ODE Two-Entry Heterogeneity Modeling

Decision:

Feature heterogeneity is modeled primarily through initial-state differences.

Structure heterogeneity is modeled primarily through propagation/vector-field differences.

Reason:

This provides a unified dynamics interpretation while preserving distinct physical roles.

Status:

Active.
```

再例如：

```markdown
## D002 — Do Not Use Class-level Gram Gate

Decision:

Do not continue the C x C Gram-based feature correction branch.

Reason:

Controlled missing-class experiments showed unstable gate behavior.

Status:

Abandoned.
```

这个文件非常重要，因为它可以防止另一个 Codex：

> 过几天又重新发明一个已经失败过的方案。

---

# 8. `EXPERIMENTS.md`

只保存已经运行过、对研究决策有价值的实验。

建议使用统一格式：

```markdown
# Experiments

## E014 — Synthetic Feature Shift

### Configuration

- clients: 10
- nodes/client: ~1000
- feature dim: 128
- classes: 8
- rounds: 100
- seed: ...

### Results

- paired best test ACC: ...
- final ACC: ...
- dynamics residual: ...
- descriptor residual: ...

### Interpretation

...

### Decision

...
```

重点不是保存所有 log。

原始 log 仍然放实验目录。

这里保存的是：

> “以后研究 Agent 需要知道的实验结论。”

---

# 9. `logs/research.md`

保存 GPT-6 每次研究工作的简短记录。

例如：

```markdown
## 2026-09-05

Analyzed current feature injection.

Main conclusion:

The current generator alignment problem is likely caused by attempting to align local score fields directly.

Proposed switching from explicit cross-client generator matching to ...

Updated:

- PROJECT_STATE.md
- NEXT_TASK.md
- DECISIONS.md
```

不需要很长。

它主要用于回答：

> 最近 GPT-6 做过什么？

---

# 10. `logs/implementation.md`

保存 GPT-5.6 的工程记录。

例如：

```markdown
## 2026-09-05

Implemented feature initial-state adapter.

Changed:

- models/feature_adapter.py
- trainer/client.py
- config/default.yaml

Tests:

- baseline smoke test: pass
- module-off equivalence: pass
- synthetic 20 rounds: pass

Issue:

Injection magnitude quickly reaches clipping threshold.

NEXT_TASK status changed to NEEDS_RESEARCH.
```

---

# 11. 项目级 `AGENTS.md`

推荐在项目根目录建立：

```text
AGENTS.md
```

内容如下。

```markdown
# Multi-Agent Collaboration Protocol

This repository is collaboratively maintained by two Codex agents.

# Mandatory Startup Procedure

Before doing substantial work:

1. Run `whoami`.

2. Read:

   - `.collab/PROJECT_STATE.md`
   - `.collab/NEXT_TASK.md`
   - `.collab/DECISIONS.md`

3. Inspect repository state:

   ```bash
   git status --short
   git log -5 --oneline
   ```

4. Inspect relevant source code before making implementation assumptions.

Never ask the human to restate information that is already available in `.collab`.

---

# Role Selection

Determine your role from:

```bash
whoami
```

If the current Linux user is `root`, follow the Research Agent protocol.

If the current Linux user is `xzc`, follow the Implementation Agent protocol.

---

# Research Agent — root

Primary responsibility:

- algorithm design;
- theoretical reasoning;
- difficult debugging and diagnosis;
- experiment interpretation;
- research direction;
- architecture review.

Before proposing an algorithm change, inspect the relevant implementation to ensure the proposal is implementable.

Normally avoid large implementation changes unless explicitly requested.

## Required Handoff

Before finishing substantial research/design work, update:

- `.collab/PROJECT_STATE.md` if the global design changed;
- `.collab/DECISIONS.md` if an important decision was made;
- `.collab/NEXT_TASK.md`;
- `.collab/logs/research.md`.

A `READY_FOR_IMPLEMENTATION` handoff must contain:

1. Problem
2. Motivation
3. Proposed algorithm
4. Expected affected modules
5. Invariants
6. Required diagnostics
7. Acceptance criteria

Do not leave important implementation decisions only in chat context.

---

# Implementation Agent — xzc

Primary responsibility:

- source-code implementation;
- debugging;
- running experiments;
- engineering validation;
- configuration;
- test and reproducibility checks.

At the start of work, inspect `.collab/NEXT_TASK.md`.

If:

```text
Status: READY_FOR_IMPLEMENTATION
```

treat the file as the current engineering specification unless the human provides a newer explicit instruction.

Before modifying code:

1. inspect relevant implementation;
2. identify the minimal code path required;
3. preserve all stated invariants.

After implementation, update:

- `.collab/PROJECT_STATE.md` if implementation status changed;
- `.collab/EXPERIMENTS.md` if meaningful experiments were run;
- `.collab/logs/implementation.md`;
- `.collab/NEXT_TASK.md`.

Set `NEXT_TASK.md` status to one of:

- IMPLEMENTED
- NEEDS_RESEARCH
- BLOCKED

If status is `NEEDS_RESEARCH`, explicitly record:

1. what failed;
2. experimental evidence;
3. suspected cause;
4. exact research question.

Do not require the human to manually summarize previous-agent work.

---

# Shared Rules

## Preserve Existing Backbone

Do not modify backbone/federated/evaluation behavior unless the current handoff explicitly requires it.

## Avoid Silent Architectural Changes

Any architectural change outside the current specification must be documented.

## Experiments

Record research-relevant experiment conclusions in:

`.collab/EXPERIMENTS.md`

Do not copy full raw logs there.

## Git

Before editing:

```bash
git status --short
```

Do not overwrite uncommitted work from another agent.

When unexpected modifications exist, inspect them before proceeding.

## Communication Principle

Chat context is temporary.

Repository state is authoritative.

Important knowledge must be written into `.collab`.
```

---

# 12. Linux 文件权限

这是双用户协作中必须处理的问题。

如果 root 创建文件：

```text
-rw-r--r-- root root NEXT_TASK.md
```

xzc 虽然可以读取，但可能无法修改。

推荐使用 ACL。

假设项目路径：

```bash
PROJECT=/opt/data/private/xzc/work2/FunctionalDualEnergyGFL_backbone
```

执行：

```bash
sudo setfacl -R -m u:xzc:rwX "$PROJECT"
sudo setfacl -R -d -m u:xzc:rwX "$PROJECT"
```

第一条：

```bash
setfacl -R -m
```

修改已有文件权限。

第二条：

```bash
setfacl -R -d -m
```

设置默认 ACL，使未来由 root 创建的新文件也自动允许 xzc 读写。

检查：

```bash
getfacl "$PROJECT/.collab"
```

---

# 13. 如果没有 `setfacl`

可以使用共享 Unix group。

例如：

```bash
sudo groupadd codexdev
sudo usermod -aG codexdev xzc
sudo usermod -aG codexdev root
```

然后：

```bash
sudo chgrp -R codexdev "$PROJECT"
sudo chmod -R g+rwX "$PROJECT"
sudo find "$PROJECT" -type d -exec chmod g+s {} \;
```

其中：

```bash
chmod g+s
```

使目录中之后创建的文件自动继承共享 group。

如果系统支持 ACL，优先使用 ACL。

---

# 14. Git 协作原则

因为两个 Codex 在同一个工作树上工作，必须避免同时修改同一个源码文件。

推荐职责划分：

```text
GPT-6 / root
    ↓
主要修改
.collab/*.md

GPT-5.6 / xzc
    ↓
主要修改
source code
+
.collab implementation state
```

GPT-6 如果只是研究和设计：

> 尽量不要同时修改实现代码。

GPT-5.6 每次开始前运行：

```bash
git status --short
```

如果发现已有未提交修改：

先理解是谁产生的，再继续工作。

不要自动覆盖。

---

# 15. 推荐标准工作流

完整流程如下。

## Stage 1 — GPT-6 研究

root：

```bash
cd PROJECT
codex
```

Agent 自动读取：

```text
AGENTS.md
.collab/PROJECT_STATE.md
.collab/NEXT_TASK.md
.collab/DECISIONS.md
```

然后：

```text
分析实验
    ↓
检查代码
    ↓
设计新算法
    ↓
验证数学与工程可行性
    ↓
更新 NEXT_TASK.md
```

最终：

```text
Status: READY_FOR_IMPLEMENTATION
```

---

# 16. Stage 2 — GPT-5.6 实现

xzc：

```bash
cd PROJECT
codex
```

无需用户重新输入 GPT-6 的方案。

只需要：

```text
继续
```

或者直接给当前工程需求。

Agent 根据 `AGENTS.md` 自动读取：

```text
.collab/NEXT_TASK.md
```

然后：

```text
检查代码
    ↓
实现
    ↓
debug
    ↓
smoke test
    ↓
实验
```

---

# 17. Stage 3 — 实现成功

如果实现正确：

```text
Status: IMPLEMENTED
Owner: research-agent
```

同时写：

```markdown
## Implementation Result

Implemented:
...

Tests:
...

Results:
...

Open Questions:
...
```

---

# 18. Stage 4 — 实验发现 idea 有问题

如果代码实现本身正常，但实验效果说明算法设计有问题：

不要继续盲目调参。

设置：

```text
Status: NEEDS_RESEARCH
Owner: research-agent
```

并写：

```markdown
## Observed Failure

...

## Evidence

...

## Engineering Verification

Confirmed:

- implementation follows specification;
- baseline unchanged;
- module-off equivalence passes.

## Suspected Cause

...

## Research Question

Why does ... ?

Possible directions requiring analysis:

1. ...
2. ...
```

然后 GPT-6 下一次打开项目就能直接继续。

---

# 19. 最终形成的闭环

```text
┌──────────────────────────────┐
│       GPT-6 / root           │
│                              │
│  theory                      │
│  idea                        │
│  algorithm                   │
│  diagnosis                   │
└──────────────┬───────────────┘
               │
               │ READY_FOR_IMPLEMENTATION
               ▼
        .collab/NEXT_TASK.md
               │
               ▼
┌──────────────────────────────┐
│       GPT-5.6 / xzc          │
│                              │
│  code                        │
│  debug                       │
│  tests                       │
│  experiments                 │
└──────────────┬───────────────┘
               │
               │ IMPLEMENTED
               │ or
               │ NEEDS_RESEARCH
               ▼
        .collab/NEXT_TASK.md
               │
               └──────────────► GPT-6
```

用户不再负责：

```text
复制 GPT-6 输出
    ↓
粘贴给 GPT-5.6
    ↓
复制实验结果
    ↓
粘贴给 GPT-6
```

而只负责：

```text
提出研究目标
    ↓
决定重要方向
    ↓
观察协作结果
```

---

# 20. 可选：进一步做全自动 Agent Relay

第一阶段不建议直接做。

先让：

```text
AGENTS.md
+
.collab
+
Git
+
ACL
```

稳定运行。

之后可以加入自动 watcher。

Linux 可以监听：

```text
.collab/NEXT_TASK.md
```

文件变化。

例如：

```bash
inotifywait -m .collab/NEXT_TASK.md
```

当检测到：

```text
Status: READY_FOR_IMPLEMENTATION
```

自动以 `xzc` 用户运行非交互 Codex：

```bash
codex exec \
  "Read AGENTS.md and execute the current handoff in .collab/NEXT_TASK.md"
```

GPT-5.6 完成后将：

```text
Status: IMPLEMENTED
```

或：

```text
Status: NEEDS_RESEARCH
```

写回。

另一个 watcher 可以检测 `NEEDS_RESEARCH`，自动触发 GPT-6。

最终可以形成：

```text
GPT-6
 │
 │ write task
 ▼
NEXT_TASK.md
 │
 │ filesystem event
 ▼
GPT-5.6
 │
 │ implementation / experiment
 ▼
NEXT_TASK.md
 │
 │ NEEDS_RESEARCH
 ▼
GPT-6
```

这时才真正成为一个自动双 Agent loop。

---

# 21. 为什么先使用文件协议，而不是直接让两个 Agent “聊天”

研究代码项目最重要的不是保存所有对话，而是维护：

```text
Current State
+
Current Decision
+
Current Task
+
Experimental Evidence
```

直接 Agent-to-Agent 对话容易出现：

- 上下文不断膨胀；
- 旧结论与新结论混杂；
- 已失败的路线重新被提出；
- 工程 Agent 不知道哪个方案是最终版；
- 聊天记录难以和代码版本对应。

而文件式协议具有：

- 可版本控制；
- 可 diff；
- 可搜索；
- 可人工检查；
- 可与代码同步；
- Agent 重启后仍然存在；
- 不依赖某个 Codex session。

因此：

> Chat 是临时推理空间，Repository 才应该是长期协作记忆。

---

# 22. 推荐第一阶段实际落地

当前暂时不要实现复杂自动化。

只完成下面四件事：

## Step 1

创建：

```text
AGENTS.md
```

## Step 2

创建：

```text
.collab/
├── PROJECT_STATE.md
├── NEXT_TASK.md
├── DECISIONS.md
├── EXPERIMENTS.md
└── logs/
    ├── research.md
    └── implementation.md
```

## Step 3

配置 root/xzc 共享权限：

```bash
sudo setfacl -R -m u:xzc:rwX "$PROJECT"
sudo setfacl -R -d -m u:xzc:rwX "$PROJECT"
```

## Step 4

之后统一规定：

GPT-6：

```text
研究结束 -> 必须更新 NEXT_TASK.md
```

GPT-5.6：

```text
开始实现 -> 必须先读 NEXT_TASK.md
实现结束 -> 必须更新 NEXT_TASK.md
```

仅这一套机制，就已经能够解决大部分双终端协作问题。

---

# 23. 最终推荐架构

当前阶段采用：

```text
Two Codex Agents
        +
One Shared Repository
        +
AGENTS.md
        +
.collab State Protocol
        +
Git
        +
Linux ACL
```

角色固定为：

```text
GPT-6 / root
=
Research Agent
=
Idea + Theory + Algorithm + Diagnosis
```

```text
GPT-5.6 / xzc
=
Implementation Agent
=
Code + Debug + Experiment + Engineering Validation
```

核心协作接口：

```text
.collab/NEXT_TASK.md
```

核心长期项目记忆：

```text
.collab/PROJECT_STATE.md
```

核心历史决策：

```text
.collab/DECISIONS.md
```

核心实验知识：

```text
.collab/EXPERIMENTS.md
```

整个机制的目标不是让两个模型拥有相同聊天上下文，而是：

> 让两个模型在任何时间进入项目，都能从 repository 本身恢复当前研究状态，并继续上一位 Agent 的工作。
