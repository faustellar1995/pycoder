# DeepSeek 助手（CLI + PyQt5）

基于 DeepSeek 官方 Chat Completions API 的 Python 示例与桌面助手：支持普通对话、流式输出、本地工作区工具（文件读写/搜索）、可选联网搜索（百度/搜狗结果解析，无需付费搜索 API）、本地 **SKILL.md** 技能与 ClawHub 技能市场，以及 Git 提交信息生成。

## 模块一览

| 文件 | 作用 |
|------|------|
| `deepseek_api.py` | API 核心：非流式/流式 SSE、`tools` 多轮函数调用；模型别名 `flash`/`pro` → `deepseek-v4-*`；可选 HTTP(S) 代理；流式请求禁用 gzip 以避免 SSE 被整块缓冲 |
| `deepseek_cli.py` | 命令行：单轮或交互循环；`--tools` 启用本地工具；`--proxy` / `--workspace` / `--model` 等 |
| `deepseek_harness.py` | 对话执行层：普通对话 / 工具循环 / 流式；`tools_system_hint`；AI 生成 Git commit message 与 `git commit` 辅助 |
| `deepseek_pyqt5_ui.py` | PyQt5 图形界面：多会话标签、系统提示词、工具与技能、技能市场、工作区与上下文附件、对话导出等 |
| `prompts_manager.py` | 按类型持久化提示词（`system` / `roleplay` / `tool`），对应 `prompts_*.json`，兼容旧版单文件 `prompts.json` |
| `web_search.py` | 联网搜索工具：抓取百度搜索页 / 搜狗搜索页提取标题与链接（无第三方 HTTP 库依赖） |
| `workspace_tools.py` | 工作区工具实现：`read_file`、`write_file`、`list_directory`、`search_replace`、`glob_file_search`、`grep_file`、可选 `web_search` 与 `run_command`（路径限制在工作区内） |
| `skills_registry.py` | 扫描并解析本地 `SKILL.md`（YAML frontmatter），按关键词/标签/正则对用户消息打分并挑选技能，注入 system 附加片段 |
| `skill_catalog.py` | ClawHub 风格注册表：搜索与下载技能（内存 TTL 缓存；可通过环境变量覆盖注册表 URL） |
| `test.py` | 独立小示例：PyQt5 实时时钟窗口（与主程序无关，可作界面片段参考） |

## 环境变量

### 必选

- **`DS_KEY`**：DeepSeek API Key（Bearer）。

### 可选（常用）

- **`DEEPSEEK_WORKSPACE`**：工具模式默认工作区根路径（缺省为当前工作目录）。
- **HTTP 代理**：CLI 使用 `--proxy`；UI 内可勾选「代理」；联网搜索与模型请求可共用代理逻辑。
- **`DEEPSEEK_DISABLE_COMMANDS=1`**：禁用 `run_command`（子进程执行）。
- **`DEEPSEEK_DISABLE_WEB_SEARCH=1`**：禁用 `web_search` 工具。
- **联网搜索**：`DEEPSEEK_WEB_SEARCH_ENGINE=baidu|sogou`、`DEEPSEEK_WEB_SEARCH_FALLBACK=1`、`DEEPSEEK_WEB_SEARCH_TIMEOUT`、`DEEPSEEK_WEB_SEARCH_RETRIES` 等（详见 `web_search.py` 顶部说明）。
- **技能扫描**：`DEEPSEEK_SKILLS_HOME`、`DEEPSEEK_SKILL_DIRS`（多路径用系统路径分隔符）；默认包含用户目录下 `~/.deepseek-assistant/skills` 与当前目录 `skills/`。
- **技能市场注册表**：`CLAWHUB_REGISTRY` 或 `CLAWDHUB_REGISTRY`（缺省使用内置默认 Convex 站点）。

## 安装依赖

```powershell
pip install -r requirements.txt
```

依赖包含 **PyQt5**、**PyYAML**（解析 `SKILL.md` frontmatter）。

## 配置 API Key

PowerShell（当前会话）：

```powershell
$env:DS_KEY = "your_deepseek_api_key"
```

PowerShell（当前用户持久化）：

```powershell
[System.Environment]::SetEnvironmentVariable("DS_KEY", "your_deepseek_api_key", "User")
```

持久化后需重新打开终端或 IDE。

## 命令行：`deepseek_cli.py`

```powershell
python deepseek_cli.py
```

- **交互模式**：不传参数时进入循环，输入 `exit` / `quit` 退出。
- **单次提问**：`python deepseek_cli.py 你的问题全文`

### 常用参数

| 参数 | 说明 |
|------|------|
| `--tools` | 启用本地工具（默认注册 `web_search`；可用 `--no-web-search` 关闭） |
| `--no-run-command` | 在 `--tools` 下不注册 `run_command` |
| `--no-stream` | 关闭流式输出 |
| `-m` / `--model` | 模型模式，与 UI 一致（默认 `flash`） |
| `-w` / `--workspace` | 工具模式工作区根目录（默认当前目录） |
| `--proxy` | HTTP/HTTPS 代理，例如 `http://127.0.0.1:7890` |

## 图形界面：`deepseek_pyqt5_ui.py`

```powershell
python deepseek_pyqt5_ui.py
```

### 对话与模型

- **模型**：Flash（`deepseek-v4-flash`）/ Pro（`deepseek-v4-pro`），与 `deepseek_api.MODEL_ALIASES` 一致。
- **多会话**：标签页管理多个会话；支持流式输出与停止按钮；可勾选 **Preview** 预览下一条将发给模型的消息。
- **代理**：可为模型请求与技能市场请求分别配置代理相关选项（界面中的「代理」复选框）。

### 系统提示词

- 内置类型：**system**、**roleplay**、**tool**（分别对应 `prompts_system.json`、`prompts_roleplay.json`、`prompts_tool.json`）。
- 预置模板示例：`default`、`code_expert`、`translator`、`writer`（system）；tool 类型含默认「工作区 + 联网搜索」说明。
- 对话框内可直接编辑内容；关闭对话框时会保存当前编辑。

### 本地工具（Function Calling）

勾选 **启用本地工具** 后，模型可按 DeepSeek `tools` 规范多轮调用：`workspace_tools` 中实现的读写、目录列表、`grep`、glob、可选 **联网搜索**、可选 **`run_command`**（工作区内子进程，`DEEPSEEK_DISABLE_COMMANDS=1` 可全局禁用）。启用工具时流式行为按实现可能与纯对话模式不同（以界面逻辑为准）。

### 技能（Skills）

- **本地**：递归扫描 `SKILL.md`，按用户输入自动匹配（可勾选「自动匹配」）；可将选中技能的说明注入 system。
- **技能市场**：连接 ClawHub 兼容注册表，搜索并安装技能到 `~/.deepseek-assistant/skills`（需网络；可选代理）。

### 工作区与上下文

- 可选择工作区目录；支持向本轮对话附加上下文文件/目录（用于强化本地代码问答）。

### Git

- 在工作区内可 **AI 生成 commit message**（基于 `git status` / `git diff --staged`），并可一键执行提交（具体按钮与流程见界面）。

### 对话记录

- 支持将对话导出到 `logs/` 等路径（带时间戳或按日序列命名，具体以界面「保存/加载」为准）。

## API 与行为说明

- 接口地址：`https://api.deepseek.com/chat/completions`。
- **System 提示**：每次请求可携带 `role="system"`；UI 会将工具说明、选中技能片段等与主系统提示组合。
- **多轮历史**：普通模式下用户与助手消息会持久参与后续请求；工具模式下由 `run_chat_with_tools` 维护消息与工具结果。
- **中断**：流式请求可被用户中断；中断时可能抛出 `StreamInterrupted`。

## 其他

- `test.py` 仅演示 PyQt5 时钟，运行：`python test.py`。
