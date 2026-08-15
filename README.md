# Codewright

Codewright 1.0.0 是一个运行在终端中的 AI 编程助手。它提供流式对话、工具调用、权限控制、会话恢复、SubAgent 与长期 Agent Team 协作，并可通过 MCP 扩展外部工具。

## 能力

- 支持 OpenAI-compatible（包括 DeepSeek）和 Anthropic 协议。
- 内置文件读写、精确编辑、Shell、Glob、Grep 等开发工具。
- ReAct Agent Loop、流式 Markdown、上下文压缩、持久化 Session 与 Memory。
- 四档权限模式：`DEFAULT`、`ACCEPT EDITS`、`PLAN`、`BYPASS`。
- 项目路径保护、危险命令拦截、分层 allow/deny 规则与人工审批。
- Skills、Hooks、MCP、隔离 Worktree 和可恢复的后台 SubAgent。
- Agent Team：长期队员、共享任务、Mailbox、Plan 审批、续派与 Coordinator Mode。
- Textual TUI，支持任务状态、Token 用量、命令补全和会话恢复。

> Codewright 的权限控制属于应用层保护，不等同于操作系统或容器沙箱。请只运行可信命令并连接可信 MCP Server。

## 环境

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Linux、macOS 或 Windows 终端
- 可用的模型 API Key；Worktree/Team 功能建议在 Git 仓库根目录运行
- tmux 为可选依赖；没有 pane 环境时 Team 使用 in-process 后端

## 安装

```bash
uv sync --locked --all-groups
uv run codewright --version
```

也可以使用模块入口：

```bash
uv run python -m codewright --version
```

## 配置

复制配置模板并填写自己的凭据：

```bash
cp .codewright/config.yaml.example .codewright/config.yaml
```

最小配置示例：

```yaml
providers:
  - name: deepseek
    protocol: openai-compatible   # 或 anthropic
    api_key: "your-api-key"
    base_url: https://api.deepseek.com
    model: deepseek-chat
    stream: true

default_provider: deepseek
log_level: INFO

# 可选功能
enable_subagent_background: true
enable_coordinator_mode: false
enable_fork_teammate: false
```

常用 Provider 可选字段包括 `timeout_seconds`、`temperature`、`max_tokens`、`context_window` 和 `extra_params`。顶层还支持 `system_prompt`。

Coordinator 与 Fork Teammate 使用配置和环境变量双重开关：

```bash
export CODEWRIGHT_COORDINATOR_MODE=1
export CODEWRIGHT_FORK_TEAMMATE=1
```

只有对应 YAML 字段也为 `true` 时功能才会开启。

其他配置入口：

- MCP：`~/.codewright/config.yaml` 和项目根目录 `.codewright.yaml`
- 权限：`~/.codewright/settings.yaml`、`.codewright/settings.yaml`、`.codewright/settings.local.yaml`
- 项目指令：`codewright.md` 或 `.codewright/codewright.md`
- 示例：[MCP 配置](docs/ch06/mcp-servers.example.yaml)、[权限配置](.codewright/settings.yaml.example)

请勿提交 API Key、`.codewright/config.yaml` 或其他本机凭据。

## 启动与交互

```bash
uv run codewright
# 或
uv run python -m codewright
```

常用参数：

```text
--config PATH       指定配置文件
--provider NAME     选择 Provider
--log-level LEVEL   覆盖日志级别
--version           显示版本
```

常用交互：

- `Enter`：提交消息；`Esc`：取消当前回合；`Shift+Tab`：切换权限模式。
- `/help`、`/status`、`/exit`：帮助、状态与退出。
- `/plan`、`/do`、`/review`：计划、执行与审查。
- `/session`、`/resume`、`/compact`、`/clear`：会话管理。
- `/skill`、`/hooks`、`/memory`、`/permission`：扩展与运行状态。
- `/worktree`：管理隔离 Worktree。
- `/team list|info|use|delete|kill`：管理 Agent Team。

工具或命令需要额外权限时，界面会请求允许一次、永久允许或拒绝。

## 开发与测试

```bash
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src/codewright
uv run python -m compileall -q src tests
uv build
```

默认测试离线运行。DeepSeek 真实集成测试需要显式设置：

```bash
export CODEWRIGHT_RUN_DEEPSEEK_INTEGRATION=1
export DEEPSEEK_API_KEY=your-api-key
```

主要源码位于 `src/codewright/`。
