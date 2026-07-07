# LocalHarness

本地多模型 AI 工作台（CLI + PyQt5）：统一对接 **DeepSeek**、**Kimi**、**Ollama**、**LM Studio** 等 OpenAI 兼容接口；支持流式输出、思考链展示、本地工作区工具（文件读写/搜索）、可选联网搜索、**SKILL.md** 技能与 ClawHub 技能市场，以及 Git 提交信息生成。

另含独立子应用 **[novelwriter/](novelwriter/)**（本地小说编辑器），与主工作台共用 API 层。

## 模块一览

| 文件 | 作用 |
|------|------|
| `localharness_api.py` | API 核心：多 Provider 非流式/流式 SSE、`tools` 多轮函数调用；模型别名；代理与本机绕过；Ollama / LM Studio 模型列表 |
| `localharness_cli.py` | 命令行：单轮或交互循环；`--tools` 启用本地工具；`--proxy` / `--workspace` / `--model` 等 |
| `localharness_harness.py` | 对话执行层：普通对话 / 工具循环 / 流式；`tools_system_hint`；AI 生成 Git commit message |
| `localharness_ui.py` | PyQt5 图形界面：多会话、多 Provider 模型选择、Temperature/Timeout、系统提示词、工具与技能、工作区与上下文附件 |
| `prompts_manager.py` | 按类型持久化提示词（`system` / `roleplay` / `tool`），对应 `prompts_*.json` |
| `web_search.py` | 联网搜索：百度/搜狗结果解析（无第三方 HTTP 库依赖） |
| `workspace_tools.py` | 工作区工具：`read_file`、`write_file`、`list_directory`、`search_replace`、`glob_file_search`、`grep_file`、可选 `web_search` 与 `run_command` |
| `skills_registry.py` | 扫描并解析本地 `SKILL.md`，按用户消息打分并挑选技能 |
| `skill_catalog.py` | ClawHub 风格注册表：搜索与下载技能 |
| `novelwriter/` | 独立小说编辑器（资产库 + Prompt 块合成）；入口见下文 |

## 环境变量

### API Keys（按使用的 Provider 配置）

| 变量 | Provider |
|------|----------|
| `DS_KEY` | DeepSeek |
| `KIMI_KEY` | Kimi（Moonshot） |
| （无需密钥） | Ollama（默认 `http://127.0.0.1:11434`） |
| `LMSTUDIO_API_BASE` | LM Studio（默认 `http://127.0.0.1:1234`） |
| `LMSTUDIO_API_KEY` | LM Studio（可选） |

也可用 `OLLAMA_HOST` 覆盖 Ollama 地址。

### 可选（常用）

- **`LOCALHARNESS_WORKSPACE`** 或 **`DEEPSEEK_WORKSPACE`**：工具模式默认工作区根路径。
- **HTTP 代理**：CLI `--proxy`；UI 内可勾选「代理」；访问本机 Ollama/LM Studio 时建议关闭代理。
- **`LOCALHARNESS_DISABLE_COMMANDS`** / **`DEEPSEEK_DISABLE_COMMANDS=1`**：禁用 `run_command`。
- **`LOCALHARNESS_DISABLE_WEB_SEARCH`** / **`DEEPSEEK_DISABLE_WEB_SEARCH=1`**：禁用 `web_search`。
- **联网搜索**：`DEEPSEEK_WEB_SEARCH_ENGINE=baidu|sogou` 等（详见 `web_search.py`）。
- **技能扫描**：`LOCALHARNESS_SKILLS_HOME`、`LOCALHARNESS_SKILL_DIRS`（兼容旧名 `DEEPSEEK_SKILLS_*`）；默认 `~/.localharness/skills`（仍扫描旧路径 `~/.deepseek-assistant/skills`）与 `./skills/`。
- **技能市场**：`CLAWHUB_REGISTRY` 或 `CLAWDHUB_REGISTRY`。

## 安装依赖

```powershell
pip install -r requirements.txt
```

依赖包含 **PyQt5**、**PyYAML**（解析 `SKILL.md` frontmatter）。

## 配置 API Key（示例：DeepSeek）

```powershell
$env:DS_KEY = "your_deepseek_api_key"
```

## 命令行：`localharness_cli.py`

```powershell
python localharness_cli.py
python localharness_cli.py 你的问题全文
```

| 参数 | 说明 |
|------|------|
| `--tools` | 启用本地工具 |
| `--no-run-command` | 不注册 `run_command` |
| `--no-stream` | 关闭流式输出 |
| `-m` / `--model` | 模型（如 `flash`、`pro` 或完整模型 id） |
| `-w` / `--workspace` | 工作区根目录 |
| `--proxy` | HTTP/HTTPS 代理 |

## 图形界面：`localharness_ui.py`

```powershell
python localharness_ui.py
```

- **多 Provider**：DeepSeek Flash/Pro、Kimi、本机 Ollama、LM Studio（须先在 LM Studio 中 Start server）。
- **多会话**、流式输出、**Temperature**、**Timeout**（`-1` = 无限；本机模型默认较长超时）。
- **系统提示词**：`prompts_system.json` / `prompts_roleplay.json` / `prompts_tool.json`。
- **本地工具**、**Skills**、**技能市场**、工作区与上下文附件、对话导出、Git commit 辅助。

## 小说编辑器：`novelwriter/`

```powershell
python -m novelwriter
```

资产与 Prompt 块默认位于 `novelwriter/novel_assets/`；可在界面中改自定义目录。

## 其他

- `test.py`：PyQt5 时钟小示例，与主程序无关。
