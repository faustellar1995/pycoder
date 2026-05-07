import sys
import urllib.error
import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QThread, Qt, pyqtSignal, QSettings
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QTextBrowser,
    QSplitter,
    QTabWidget,
    QShortcut,
    QVBoxLayout,
    QWidget,
    QListWidget,
    QListWidgetItem,
)

from deepseek_api import explain_http_error, get_api_key, StreamInterrupted
from deepseek_harness import (
    HarnessConfig,
    do_git_commit,
    generate_git_commit_message,
    run_harness,
    tools_system_hint,
)
from skill_catalog import SkillCatalog, shared_catalog
from skills_registry import (
    build_skills_system_addon,
    clear_skill_discovery_cache,
    default_skill_scan_dirs,
    discover_skills,
    parse_skill_md,
    rewrite_skill_md_name_for_install,
    select_skills_for_message,
)
from prompts_manager import (
    add_prompt,
    delete_prompt,
    get_all_prompts,
    save_prompts,
)
from workspace_tools import WorkspaceToolSession

class AskWorker(QThread):
    chunk = pyqtSignal(str)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        api_key: str,
        messages,
        model_mode: str,
        stream: bool,
        system_prompt: str,
        *,
        use_tools: bool = False,
        workspace: Optional[Path] = None,
        allow_run_command: bool = True,
        enable_web_search: bool = True,
        proxy_url: Optional[str] = None,
    ):
        super().__init__()
        self.api_key = api_key
        self.messages = messages
        self.model_mode = model_mode
        self.stream = stream
        self.system_prompt = system_prompt
        self.use_tools = use_tools
        self.workspace = workspace or Path.cwd()
        self.allow_run_command = allow_run_command
        self.enable_web_search = enable_web_search
        self.proxy_url = proxy_url
        self._should_stop = False

    def stop(self):
        """Signal the worker to stop."""
        self._should_stop = True

    def should_stop(self) -> bool:
        return self._should_stop

    def run(self):
        try:
            cfg = HarnessConfig(
                workspace=self.workspace,
                use_tools=self.use_tools,
                allow_run_command=self.allow_run_command,
                enable_web_search=self.enable_web_search,
                stream=self.stream,
                proxy_url=self.proxy_url,
            )
            answer = run_harness(
                api_key=self.api_key,
                messages=self.messages,
                model_mode=self.model_mode,
                config=cfg,
                should_stop=self.should_stop,
                on_stream_token=self.chunk.emit if self.stream else None,
            )
            self.done.emit(answer)
        except StreamInterrupted:
            self.failed.emit("[Stream interrupted by user]")
        except urllib.error.HTTPError as exc:
            self.failed.emit(explain_http_error(exc))
        except urllib.error.URLError as exc:
            self.failed.emit(f"Network error: {exc.reason}")
        except Exception as exc:
            self.failed.emit(f"Unexpected error: {exc}")


class CommitWorker(QThread):
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, *, api_key: str, model_mode: str, workspace: Path, proxy_url: Optional[str] = None):
        super().__init__()
        self.api_key = api_key
        self.model_mode = model_mode
        self.workspace = workspace
        self.proxy_url = proxy_url

    def run(self):
        try:
            subject, body, log = generate_git_commit_message(
                api_key=self.api_key,
                model_mode=self.model_mode,
                workspace=self.workspace,
                proxy_url=self.proxy_url,
            )
            payload = {"subject": subject, "body": body, "log": log}
            self.done.emit(json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            self.failed.emit(f"[commit] Unexpected error: {exc}")


@dataclass
class SessionState:
    title: str
    messages: List[Dict[str, Any]]
    pending_stream_text: str = ""
    awaiting_response: bool = False


class SystemPromptDialog(QDialog):
    """Dialog to manage system prompts."""

    PROMPT_TYPE = "system"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage System Prompts")
        self.resize(600, 400)

        layout = QVBoxLayout(self)

        # Prompt list
        layout.addWidget(QLabel("Saved Prompts:"))
        self.prompt_list = QListWidget()
        layout.addWidget(self.prompt_list)

        # Buttons
        button_row = QHBoxLayout()

        self.new_button = QPushButton("New")
        self.delete_button = QPushButton("Delete")
        self.use_button = QPushButton("Use (Close Dialog)")

        button_row.addWidget(self.new_button)
        button_row.addWidget(self.delete_button)
        button_row.addWidget(self.use_button)

        layout.addLayout(button_row)

        # Editable content area
        layout.addWidget(QLabel("Prompt Content (auto-saved on close):"))
        self.preview_text = QPlainTextEdit()
        self.preview_text.setFixedHeight(80)
        layout.addWidget(self.preview_text)

        self.new_button.clicked.connect(self.on_new)
        self.delete_button.clicked.connect(self.on_delete)
        self.use_button.clicked.connect(self.accept)
        self.prompt_list.itemSelectionChanged.connect(self.on_selection_changed)
        self.preview_text.textChanged.connect(self.on_content_changed)

        self.selected_prompt_name = None
        self.prompts = {}
        self._loading_content = False
        self.reload_list()

    def reload_list(self):
        self.prompts = get_all_prompts(prompt_type=self.PROMPT_TYPE)
        self.prompt_list.clear()
        for name in sorted(self.prompts.keys()):
            self.prompt_list.addItem(name)

    def on_selection_changed(self):
        items = self.prompt_list.selectedItems()
        if items:
            name = items[0].text()
            content = self.prompts.get(name, "")
            self._loading_content = True
            self.preview_text.setPlainText(content)
            self._loading_content = False
            self.selected_prompt_name = name
        else:
            self.selected_prompt_name = None

    def on_new(self):
        name, ok = self._input_name_dialog("Enter prompt name:")
        if not ok or not name:
            return

        name = name.strip()
        if not name:
            return

        if name in self.prompts:
            QMessageBox.warning(self, "Duplicate", f"Prompt '{name}' already exists.")
            return

        add_prompt(name, "", prompt_type=self.PROMPT_TYPE)
        self.reload_list()

        matched = self.prompt_list.findItems(name, Qt.MatchExactly)
        if matched:
            self.prompt_list.setCurrentItem(matched[0])

    def on_delete(self):
        if not self.selected_prompt_name:
            QMessageBox.warning(self, "No selection", "Please select a prompt to delete.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete prompt '{self.selected_prompt_name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            delete_prompt(self.selected_prompt_name, prompt_type=self.PROMPT_TYPE)
            self.reload_list()
            self.preview_text.clear()
            self.selected_prompt_name = None

    def on_content_changed(self):
        if self._loading_content or not self.selected_prompt_name:
            return

        self.prompts[self.selected_prompt_name] = self.preview_text.toPlainText()

    def _persist_changes(self):
        save_prompts(self.prompts, prompt_type=self.PROMPT_TYPE)

    def accept(self):
        self._persist_changes()
        super().accept()

    def closeEvent(self, event):
        self._persist_changes()
        super().closeEvent(event)

    def _input_name_dialog(self, prompt_text: str) -> tuple:
        from PyQt5.QtWidgets import QInputDialog
        return QInputDialog.getText(self, "Input", prompt_text)

    def get_selected_prompt(self) -> str:
        """Return the content of the selected prompt."""
        if self.selected_prompt_name:
            return self.prompts.get(self.selected_prompt_name, "")
        return ""


class SkillMarketDialog(QDialog):
    """ClawHub 技能市场：搜索使用与 ironclaw 相同的 TTL 内存缓存。"""

    def __init__(self, parent=None, *, proxy_url: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("技能市场 (ClawHub)")
        self.resize(720, 480)
        self._catalog = shared_catalog()
        self._catalog.set_proxy(proxy_url)
        self._entries = []

        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索关键词…")
        self.search_btn = QPushButton("搜索")
        self.clear_cache_btn = QPushButton("清除搜索缓存")
        row.addWidget(self.search_edit, 1)
        row.addWidget(self.search_btn)
        row.addWidget(self.clear_cache_btn)
        layout.addLayout(row)

        self.hint = QLabel("搜索命中进程内缓存时不会重复请求网络（默认 TTL 300 秒）。")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        self.result_list = QListWidget()
        layout.addWidget(self.result_list, 1)

        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setFixedHeight(96)
        layout.addWidget(self.detail)

        self.install_btn = QPushButton("安装选中技能到 ~/.deepseek-assistant/skills")
        layout.addWidget(self.install_btn)

        self.search_btn.clicked.connect(self.on_search)
        self.clear_cache_btn.clicked.connect(self.on_clear_cache)
        self.result_list.itemSelectionChanged.connect(self.on_select)
        self.install_btn.clicked.connect(self.on_install)

    def on_clear_cache(self):
        self._catalog.clear_cache()
        self.hint.setText("已清除市场搜索内存缓存（与 ironclaw SkillCatalog 行为一致）。")

    def on_search(self):
        q = self.search_edit.text().strip()
        if not q:
            QMessageBox.warning(self, "输入", "请输入搜索词。")
            return
        outcome = self._catalog.search(q)
        self._entries = outcome.results
        self.result_list.clear()
        if outcome.error:
            self.hint.setText(f"搜索失败: {outcome.error}")
        else:
            self.hint.setText(f"返回 {len(self._entries)} 条结果。")
        for e in self._entries:
            label = f"{e.slug} — {e.name or e.slug}"
            self.result_list.addItem(label)
        self.detail.clear()

    def on_select(self):
        row = self.result_list.currentRow()
        if row < 0 or row >= len(self._entries):
            return
        e = self._entries[row]
        self.detail.setPlainText(
            f"{e.name}\n{e.description}\n\n版本: {e.version}\nslug: {e.slug}\nscore: {e.score}"
        )

    def on_install(self):
        row = self.result_list.currentRow()
        if row < 0:
            QMessageBox.warning(self, "选择", "请先选择一项。")
            return
        slug = self._entries[row].slug
        raw, err = self._catalog.download_skill_md(slug)
        if err:
            QMessageBox.warning(self, "下载失败", err)
            return
        try:
            parse_skill_md(Path("__market__/SKILL.md"), raw)
        except Exception as exc:
            # 安装路径：对齐 ironclaw 的 install-recovery —— name 不合法时自动重写后继续
            msg_text = str(exc)
            if "无效 skill name" in msg_text or "Invalid skill name" in msg_text:
                try:
                    raw_fixed = rewrite_skill_md_name_for_install(raw, preferred=slug)
                    parse_skill_md(Path("__market__/SKILL.md"), raw_fixed)
                except Exception:
                    # fallthrough: show original error + preview
                    pass
                else:
                    # 修复成功：使用修复后的内容继续安装
                    raw = raw_fixed
                    exc = None

            if exc is not None:
                msg = f"{type(exc).__name__}: {exc}"
                env_hint = (
                    "排查建议：\n"
                    "1) 确认运行 UI 的 Python 与安装依赖的是同一个：python -c \"import sys; print(sys.executable)\"\n"
                    "2) 确认 PyYAML 可导入：python -c \"import yaml; print(yaml.__version__)\"\n"
                    "3) 重新安装依赖：pip install -r requirements.txt\n"
                )
                preview = raw.replace("\r\n", "\n").replace("\r", "\n")
                preview = preview[:800] + ("\n... [SKILL.md 已截断]" if len(preview) > 800 else "")
                QMessageBox.warning(
                    self,
                    "SKILL.md 解析失败",
                    msg + "\n\n" + env_hint + "\n---\nSKILL.md 预览（前 800 字符）:\n" + preview,
                )
                return
        safe = slug.replace("/", "_").replace("\\", "_")
        dest_dir = Path.home() / ".deepseek-assistant" / "skills" / safe
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "SKILL.md"
        dest.write_text(raw, encoding="utf-8", newline="\n")
        clear_skill_discovery_cache()
        QMessageBox.information(self, "完成", f"已安装到:\n{dest}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepSeek Multi-turn Chat with System Prompt")
        self.resize(900, 750)

        self.worker = None
        self.commit_worker = None
        self.messages = []
        self.current_system_prompt = "You are a helpful assistant."
        self.pending_stream_text = ""
        self.awaiting_response = False
        self.logs_dir = Path("./logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.settings = QSettings("DeepSeekAssistant", "PyQtClient")
        self._skill_catalog = []

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Main split: left config (Model/Harness/Skills) vs right results ──
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        left_tabs = QTabWidget()
        left_tabs.setDocumentMode(True)
        splitter.addWidget(left_tabs)

        # Right side: conversation + preview
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # 会话 Tabs（多会话切换）
        self.session_tabs = QTabWidget()
        self.session_tabs.setDocumentMode(True)
        self.session_tabs.setMovable(True)
        self.session_tabs.setTabsClosable(True)
        right_layout.addWidget(self.session_tabs)

        self.new_session_btn = QPushButton("+")
        self.new_session_btn.setFixedWidth(28)
        self.new_session_btn.setToolTip("新建会话")
        self.session_tabs.setCornerWidget(self.new_session_btn, Qt.TopRightCorner)

        right_layout.addWidget(QLabel("对话结果"))
        content_split = QSplitter(Qt.Horizontal)
        right_layout.addWidget(content_split, 1)

        self.chat_output = QTextBrowser()
        self.chat_output.setOpenExternalLinks(False)
        content_split.addWidget(self.chat_output)

        preview_wrap = QWidget()
        preview_layout = QVBoxLayout(preview_wrap)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(4)
        preview_layout.addWidget(QLabel("下一次请求上下文预览"))
        self.preview_context_box = QPlainTextEdit()
        self.preview_context_box.setReadOnly(True)
        self.preview_context_box.setPlaceholderText("开启 Preview 后可检查下一次请求的 messages。")
        preview_layout.addWidget(self.preview_context_box, 1)
        content_split.addWidget(preview_wrap)

        right_layout.addWidget(QLabel("输入"))
        self.input_box = QPlainTextEdit()
        self.input_box.setPlaceholderText("在这里输入你的问题…")
        self.input_box.setFixedHeight(96)
        right_layout.addWidget(self.input_box)

        buttons_row = QHBoxLayout()
        self.ask_button = QPushButton("Send")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.clear_history_button = QPushButton("Clear History")
        self.load_button = QPushButton("Load")
        self.save_button = QPushButton("Save")
        buttons_row.addWidget(self.ask_button)
        buttons_row.addWidget(self.stop_button)
        buttons_row.addWidget(self.clear_history_button)
        buttons_row.addWidget(self.load_button)
        buttons_row.addWidget(self.save_button)
        buttons_row.addStretch(1)
        right_layout.addLayout(buttons_row)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # ── Tab: Model+Harness ──────────────────────────────────────────────
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)
        model_layout.setContentsMargins(8, 8, 8, 8)
        model_layout.setSpacing(6)

        row_model = QHBoxLayout()
        row_model.addWidget(QLabel("Model"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("Flash (deepseek-v4-flash)", "flash")
        self.model_combo.addItem("Pro (deepseek-v4-pro)", "pro")
        row_model.addWidget(self.model_combo, 1)
        model_layout.addLayout(row_model)

        row_flags = QHBoxLayout()
        self.stream_checkbox = QCheckBox("Stream")
        self.stream_checkbox.setChecked(True)
        self.preview_checkbox = QCheckBox("Preview")
        self.preview_checkbox.setChecked(True)
        self.model_proxy_checkbox = QCheckBox("代理")
        self.model_proxy_addr = QLineEdit()
        self.model_proxy_addr.setPlaceholderText("http://127.0.0.1:7890")
        mp_on = self.settings.value("model_proxy_on", False)
        if isinstance(mp_on, str):
            mp_on = mp_on.lower() in ("1", "true", "yes")
        self.model_proxy_checkbox.setChecked(bool(mp_on))
        self.model_proxy_addr.setText(str(self.settings.value("model_proxy_addr", "http://127.0.0.1:7890")))
        self.model_proxy_checkbox.toggled.connect(lambda v: self.settings.setValue("model_proxy_on", bool(v)))
        self.model_proxy_addr.textChanged.connect(lambda t: self.settings.setValue("model_proxy_addr", t))
        row_flags.addWidget(self.stream_checkbox)
        row_flags.addWidget(self.preview_checkbox)
        row_flags.addWidget(self.model_proxy_checkbox)
        row_flags.addStretch(1)
        model_layout.addLayout(row_flags)
        model_layout.addWidget(self.model_proxy_addr)

        self.prompt_button = QPushButton("System Prompts…")
        self.prompt_button.clicked.connect(self.on_open_prompt_dialog)
        model_layout.addWidget(self.prompt_button)

        model_layout.addWidget(QLabel("当前 System Prompt（只读）"))
        self.current_prompt_display = QPlainTextEdit()
        self.current_prompt_display.setReadOnly(True)
        self.current_prompt_display.setFixedHeight(92)
        self.current_prompt_display.setPlainText(self.current_system_prompt)
        model_layout.addWidget(self.current_prompt_display)
        model_layout.addWidget(QLabel("工作区"))
        ws_row = QHBoxLayout()
        self.workspace_edit = QLineEdit()
        ws_default = self.settings.value("workspace", str(Path.cwd().resolve()))
        self.workspace_edit.setText(str(ws_default))
        self.workspace_browse = QPushButton("浏览…")
        ws_row.addWidget(self.workspace_edit, 1)
        ws_row.addWidget(self.workspace_browse)
        model_layout.addLayout(ws_row)

        # ── 当前目录/上下文 ────────────────────────────────────────────────
        model_layout.addWidget(QLabel("当前目录/上下文（手动加入）"))
        ctx_row = QHBoxLayout()
        self.ctx_add_file_btn = QPushButton("添加文件…")
        self.ctx_add_dir_btn = QPushButton("添加目录…")
        self.ctx_clear_btn = QPushButton("清空")
        ctx_row.addWidget(self.ctx_add_file_btn)
        ctx_row.addWidget(self.ctx_add_dir_btn)
        ctx_row.addWidget(self.ctx_clear_btn)
        ctx_row.addStretch(1)
        model_layout.addLayout(ctx_row)

        self.context_list = QListWidget()
        self.context_list.setToolTip("将选中的文件/目录内容注入到 system prompt 的上下文中。")
        self.context_list.setMaximumHeight(120)
        model_layout.addWidget(self.context_list)

        self.tools_checkbox = QCheckBox("启用本地工具（文件读写/搜索）")
        self.tools_checkbox.setToolTip(
            "启用后模型可多轮调用工具：读写/搜索文件，并可选择在工作区内执行子进程命令（自动关闭流式输出）。"
        )
        model_layout.addWidget(self.tools_checkbox)

        self.allow_commands_checkbox = QCheckBox("允许执行命令（run_command）")
        self.allow_commands_checkbox.setToolTip(
            "向模型注册 run_command（shell=False，cwd 限制在工作区内）。可在环境中设置 DEEPSEEK_DISABLE_COMMANDS=1 禁用。"
        )
        ac_val = self.settings.value("allow_commands", True)
        if isinstance(ac_val, str):
            ac_val = ac_val.lower() in ("1", "true", "yes")
        self.allow_commands_checkbox.setChecked(bool(ac_val))
        self.allow_commands_checkbox.toggled.connect(
            lambda v: self.settings.setValue("allow_commands", bool(v))
        )
        model_layout.addWidget(self.allow_commands_checkbox)

        self.web_search_checkbox = QCheckBox("联网搜索（web_search）")
        self.web_search_checkbox.setToolTip(
            "启用后向模型注册 web_search（抓取百度/搜狗搜索结果，无需密钥）。"
            "DEEPSEEK_WEB_SEARCH_ENGINE=baidu|sogou；与模型代理共用 HTTP。DEEPSEEK_DISABLE_WEB_SEARCH=1 可禁用。"
        )
        ws_val = self.settings.value("enable_web_search", True)
        if isinstance(ws_val, str):
            ws_val = ws_val.lower() in ("1", "true", "yes")
        self.web_search_checkbox.setChecked(bool(ws_val))
        self.web_search_checkbox.toggled.connect(
            lambda v: self.settings.setValue("enable_web_search", bool(v))
        )
        model_layout.addWidget(self.web_search_checkbox)

        model_layout.addWidget(QLabel("AI Commit"))
        commit_row = QHBoxLayout()
        self.commit_gen_btn = QPushButton("生成 message")
        self.commit_do_btn = QPushButton("提交")
        self.commit_gen_btn.setToolTip("在后台读取 staged diff，生成 commit message（不提交）。")
        self.commit_do_btn.setToolTip("使用下方编辑框里的 message 执行 git commit。")
        commit_row.addWidget(self.commit_gen_btn)
        commit_row.addWidget(self.commit_do_btn)
        commit_row.addStretch(1)
        model_layout.addLayout(commit_row)

        self.commit_message_box = QPlainTextEdit()
        self.commit_message_box.setPlaceholderText("生成后会出现在这里，你也可以手动编辑。\n第一行为 subject，空行后为 body。")
        self.commit_message_box.setFixedHeight(92)
        model_layout.addWidget(self.commit_message_box)

        self.commit_log = QPlainTextEdit()
        self.commit_log.setReadOnly(True)
        self.commit_log.setPlaceholderText("Commit 日志输出…")
        self.commit_log.setFixedHeight(110)
        model_layout.addWidget(self.commit_log)
        model_layout.addStretch(1)
        left_tabs.addTab(model_tab, "Model+Harness")

        self.workspace_browse.clicked.connect(self.on_browse_workspace)
        self.tools_checkbox.toggled.connect(self._on_tools_toggled)
        self._on_tools_toggled(self.tools_checkbox.isChecked())
        self.ctx_add_file_btn.clicked.connect(self.on_add_context_files)
        self.ctx_add_dir_btn.clicked.connect(self.on_add_context_dirs)
        self.ctx_clear_btn.clicked.connect(self.on_clear_context_items)

        # ── Tab: Skills ─────────────────────────────────────────────────────
        skills_tab = QWidget()
        skills_layout = QVBoxLayout(skills_tab)
        skills_layout.setContentsMargins(8, 8, 8, 8)
        skills_layout.setSpacing(6)

        top_row = QHBoxLayout()
        self.auto_skill_checkbox = QCheckBox("自动匹配")
        self.refresh_skills_btn = QPushButton("刷新")
        self.market_skills_btn = QPushButton("市场…")
        self.clear_catalog_cache_btn = QPushButton("清缓存")
        self.skill_proxy_checkbox = QCheckBox("代理")
        self.skill_proxy_addr = QLineEdit()
        self.skill_proxy_addr.setPlaceholderText("http://127.0.0.1:7890")
        sp_on = self.settings.value("skill_proxy_on", False)
        if isinstance(sp_on, str):
            sp_on = sp_on.lower() in ("1", "true", "yes")
        self.skill_proxy_checkbox.setChecked(bool(sp_on))
        self.skill_proxy_addr.setText(str(self.settings.value("skill_proxy_addr", "http://127.0.0.1:7890")))
        self.skill_proxy_checkbox.toggled.connect(lambda v: self.settings.setValue("skill_proxy_on", bool(v)))
        self.skill_proxy_addr.textChanged.connect(lambda t: self.settings.setValue("skill_proxy_addr", t))
        top_row.addWidget(self.auto_skill_checkbox)
        top_row.addWidget(self.refresh_skills_btn)
        top_row.addWidget(self.market_skills_btn)
        top_row.addWidget(self.clear_catalog_cache_btn)
        top_row.addWidget(self.skill_proxy_checkbox)
        top_row.addStretch(1)
        skills_layout.addLayout(top_row)
        skills_layout.addWidget(self.skill_proxy_addr)

        self.skills_list = QListWidget()
        skills_layout.addWidget(self.skills_list, 1)
        left_tabs.addTab(skills_tab, "Skills")

        self.refresh_skills_btn.clicked.connect(lambda: self.refresh_skills_list(force_disk=True))
        self.market_skills_btn.clicked.connect(self.on_open_market)
        self.clear_catalog_cache_btn.clicked.connect(self.on_clear_catalog_cache)

        self.ask_button.clicked.connect(self.on_ask)
        self.stop_button.clicked.connect(self.on_stop)
        self.clear_history_button.clicked.connect(self.on_clear_history)
        self.load_button.clicked.connect(self.on_load_conversation)
        self.save_button.clicked.connect(self.on_save_conversation)
        self.input_box.textChanged.connect(self.update_next_context_preview)
        self.preview_checkbox.toggled.connect(self.update_next_context_preview)
        self.commit_gen_btn.clicked.connect(self.on_generate_commit_message)
        self.commit_do_btn.clicked.connect(self.on_do_git_commit)

        self.render_chat()
        self.update_next_context_preview()
        self.refresh_skills_list()

        self._init_sessions()
        self.new_session_btn.clicked.connect(self.new_session)
        self.session_tabs.currentChanged.connect(self.on_session_changed)
        self.session_tabs.tabCloseRequested.connect(self.close_session)

        # Chrome-like 会话快捷键
        QShortcut("Ctrl+T", self, activated=self.new_session)
        QShortcut("Ctrl+W", self, activated=lambda: self.close_session(self.session_tabs.currentIndex()))
        QShortcut("Ctrl+Tab", self, activated=lambda: self._cycle_session(1))
        QShortcut("Ctrl+Shift+Tab", self, activated=lambda: self._cycle_session(-1))

    def workspace_path(self) -> Path:
        text = self.workspace_edit.text().strip()
        if not text:
            return Path.cwd().resolve()
        return Path(text).expanduser().resolve()

    def skill_scan_dirs(self):
        roots = []
        seen = set()
        w = self.workspace_path()
        candidates = [w / "skills", *default_skill_scan_dirs()]
        for d in candidates:
            try:
                r = d.resolve()
            except OSError:
                continue
            key = str(r)
            if key in seen:
                continue
            seen.add(key)
            if r.is_dir():
                roots.append(r)
        return roots

    def refresh_skills_list(self, force_disk: bool = False):
        if force_disk:
            clear_skill_discovery_cache()
        self._skill_catalog = discover_skills(
            scan_dirs=self.skill_scan_dirs(),
            use_cache=not force_disk,
        )
        checked = set()
        for i in range(self.skills_list.count()):
            it = self.skills_list.item(i)
            if it.checkState() == Qt.Checked:
                n = it.data(Qt.UserRole)
                if n:
                    checked.add(n)
        self.skills_list.clear()
        for s in self._skill_catalog:
            item = QListWidgetItem(s.name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if s.name in checked else Qt.Unchecked)
            item.setData(Qt.UserRole, s.name)
            item.setToolTip(f"{s.description}\n{s.source_path}")
            self.skills_list.addItem(item)

    def on_browse_workspace(self):
        start = str(self.workspace_path())
        picked = QFileDialog.getExistingDirectory(self, "选择工作区根目录", start)
        if picked:
            self.workspace_edit.setText(picked)
            self.settings.setValue("workspace", picked)
            self.refresh_skills_list(force_disk=True)
            self._prune_context_items()

    def _workspace_relative(self, path: Path) -> Optional[str]:
        ws = self.workspace_path()
        try:
            rel = path.resolve().relative_to(ws.resolve())
        except Exception:
            return None
        return rel.as_posix()

    def _add_context_path(self, path: Path) -> None:
        rel = self._workspace_relative(path)
        if rel is None:
            QMessageBox.warning(self, "上下文", "只能加入工作区内的文件/目录。")
            return
        # 去重
        for i in range(self.context_list.count()):
            it = self.context_list.item(i)
            if it.data(Qt.UserRole) == rel:
                return
        label = rel + ("/" if path.is_dir() else "")
        item = QListWidgetItem(label)
        item.setData(Qt.UserRole, rel)
        item.setToolTip(str(path))
        self.context_list.addItem(item)

    def _prune_context_items(self) -> None:
        ws = self.workspace_path()
        keep: List[QListWidgetItem] = []
        for i in range(self.context_list.count()):
            it = self.context_list.item(i)
            rel = it.data(Qt.UserRole)
            if not isinstance(rel, str):
                continue
            p = (ws / rel).resolve()
            try:
                p.relative_to(ws)
            except Exception:
                continue
            if p.exists():
                keep.append(it)
        self.context_list.clear()
        for it in keep:
            self.context_list.addItem(it)

    def on_add_context_files(self) -> None:
        ws = self.workspace_path()
        files, _ = QFileDialog.getOpenFileNames(self, "选择要加入上下文的文件", str(ws))
        for f in files:
            self._add_context_path(Path(f))

    def on_add_context_dirs(self) -> None:
        ws = self.workspace_path()
        d = QFileDialog.getExistingDirectory(self, "选择要加入上下文的目录", str(ws))
        if d:
            self._add_context_path(Path(d))

    def on_clear_context_items(self) -> None:
        self.context_list.clear()

    def _build_manual_context_block(self) -> str:
        ws = self.workspace_path()
        if self.context_list.count() == 0:
            return ""
        session = WorkspaceToolSession(ws)
        blocks: List[str] = ["## Workspace Context (User-selected)", ""]
        for i in range(self.context_list.count()):
            it = self.context_list.item(i)
            rel = it.data(Qt.UserRole)
            if not isinstance(rel, str):
                continue
            abs_path = (ws / rel).resolve()
            if abs_path.is_dir():
                listing = session.execute("list_directory", json.dumps({"path": rel}, ensure_ascii=False))
                blocks.append(f"### Directory: {rel}/")
                blocks.append("```")
                blocks.append(listing)
                blocks.append("```")
                blocks.append("")
            else:
                text = session.execute("read_file", json.dumps({"path": rel}, ensure_ascii=False))
                blocks.append(f"### File: {rel}")
                blocks.append("```")
                blocks.append(text)
                blocks.append("```")
                blocks.append("")
        return "\n".join(blocks).strip()

    def on_open_market(self):
        proxy = self.skill_proxy_addr.text().strip() if self.skill_proxy_checkbox.isChecked() else None
        SkillMarketDialog(self, proxy_url=proxy).exec_()
        self.refresh_skills_list(force_disk=True)

    def on_clear_catalog_cache(self):
        shared_catalog().clear_cache()
        QMessageBox.information(
            self,
            "缓存",
            "已清除 ClawHub 搜索内存缓存（默认 TTL 300 秒、最多 50 条，与 ironclaw SkillCatalog 一致）。",
        )

    def _on_tools_toggled(self, on: bool) -> None:
        self.allow_commands_checkbox.setEnabled(on)
        self.web_search_checkbox.setEnabled(on)

    def _init_sessions(self) -> None:
        self.sessions: List[SessionState] = []
        self._active_session_index: int = -1
        self.new_session()

    def new_session(self) -> None:
        idx = len(self.sessions) + 1
        state = SessionState(title=f"会话 {idx}", messages=[])
        self.sessions.append(state)
        self.session_tabs.addTab(QWidget(), state.title)
        self.session_tabs.setCurrentIndex(len(self.sessions) - 1)

    def close_session(self, index: int) -> None:
        if index < 0 or index >= len(self.sessions):
            return
        if len(self.sessions) == 1:
            QMessageBox.information(self, "会话", "至少保留一个会话。")
            return
        if index == self._active_session_index:
            self._save_active_session_state()
        self.sessions.pop(index)
        self.session_tabs.removeTab(index)
        for i, s in enumerate(self.sessions, start=1):
            s.title = f"会话 {i}"
            self.session_tabs.setTabText(i - 1, s.title)
        self._active_session_index = -1
        self.on_session_changed(self.session_tabs.currentIndex())

    def _save_active_session_state(self) -> None:
        i = self._active_session_index
        if i < 0 or i >= len(self.sessions):
            return
        self.sessions[i].messages = list(self.messages)
        self.sessions[i].pending_stream_text = self.pending_stream_text
        self.sessions[i].awaiting_response = self.awaiting_response

    def _load_session_state(self, index: int) -> None:
        s = self.sessions[index]
        self.messages = list(s.messages)
        self.pending_stream_text = s.pending_stream_text
        self.awaiting_response = s.awaiting_response

    def on_session_changed(self, index: int) -> None:
        if index < 0 or index >= len(self.sessions):
            return
        if self._active_session_index == index:
            return
        self._save_active_session_state()
        self._active_session_index = index
        self._load_session_state(index)
        self.render_chat()
        self.update_next_context_preview()

    def _cycle_session(self, delta: int) -> None:
        n = self.session_tabs.count()
        if n <= 1:
            return
        cur = self.session_tabs.currentIndex()
        nxt = (cur + delta) % n
        self.session_tabs.setCurrentIndex(nxt)

    def on_ai_commit(self) -> None:
        # deprecated (kept for backward references)
        self.on_generate_commit_message()

    def on_commit_done(self, text: str) -> None:
        # text is a JSON payload from CommitWorker
        try:
            payload = json.loads(text)
        except Exception:
            self.commit_log.setPlainText(text)
            return
        log = str(payload.get("log") or "")
        subject = str(payload.get("subject") or "")
        body = payload.get("body")
        if body is None:
            msg = subject
        else:
            msg = subject + "\n\n" + str(body)
        if msg.strip():
            self.commit_message_box.setPlainText(msg.strip() + "\n")
        self.commit_log.setPlainText(log or "[commit] done")

    def on_commit_failed(self, text: str) -> None:
        self.commit_log.setPlainText(text)

    def _split_commit_box(self) -> tuple[str, Optional[str]]:
        raw = (self.commit_message_box.toPlainText() or "").strip("\n").strip()
        if not raw:
            return "", None
        lines = raw.splitlines()
        subject = lines[0].strip()
        rest = "\n".join(lines[1:]).strip()
        body = rest if rest else None
        return subject, body

    def on_generate_commit_message(self) -> None:
        if self.commit_worker and self.commit_worker.isRunning():
            return
        try:
            api_key = get_api_key()
        except ValueError as exc:
            QMessageBox.critical(self, "DS_KEY missing", str(exc))
            return
        model_mode = self.model_combo.currentData()
        ws = self.workspace_path()
        proxy = self.model_proxy_addr.text().strip() if self.model_proxy_checkbox.isChecked() else None
        self.commit_log.setPlainText("[commit] 生成中…")
        self.commit_gen_btn.setEnabled(False)
        self.commit_do_btn.setEnabled(False)
        self.commit_worker = CommitWorker(api_key=api_key, model_mode=model_mode, workspace=ws, proxy_url=proxy)
        self.commit_worker.done.connect(self.on_commit_done)
        self.commit_worker.failed.connect(self.on_commit_failed)
        self.commit_worker.finished.connect(self._on_commit_worker_finished)
        self.commit_worker.start()

    def _on_commit_worker_finished(self) -> None:
        self.commit_gen_btn.setEnabled(True)
        self.commit_do_btn.setEnabled(True)

    def on_do_git_commit(self) -> None:
        subject, body = self._split_commit_box()
        if not subject:
            QMessageBox.warning(self, "Commit", "commit message 为空（第一行 subject 必填）。")
            return
        ws = self.workspace_path()
        self.commit_log.setPlainText("[commit] 提交中…")
        out = do_git_commit(workspace=ws, subject=subject, body=body)
        self.commit_log.setPlainText(out)

    def compose_full_system_prompt(self, user_text: str) -> str:
        parts = [self.current_system_prompt]
        manual_ctx = self._build_manual_context_block()
        if manual_ctx:
            parts.append(manual_ctx)
        if self.tools_checkbox.isChecked():
            parts.append(
                tools_system_hint(
                    include_run_command=self.allow_commands_checkbox.isChecked(),
                    include_web_search=self.web_search_checkbox.isChecked(),
                )
            )
        name_to_skill = {s.name: s for s in self._skill_catalog}
        picked = []
        for i in range(self.skills_list.count()):
            it = self.skills_list.item(i)
            if it.checkState() == Qt.Checked:
                n = it.data(Qt.UserRole)
                if n and n in name_to_skill:
                    picked.append(name_to_skill[n])
        if self.auto_skill_checkbox.isChecked():
            auto = select_skills_for_message(user_text, self._skill_catalog)
            have = {s.name for s in picked}
            for s in auto:
                if s.name not in have:
                    picked.append(s)
                    have.add(s.name)
        addon = build_skills_system_addon(picked)
        if addon:
            parts.append(addon)
        return "\n\n".join(parts)

    def on_open_prompt_dialog(self):
        dialog = SystemPromptDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            prompt = dialog.get_selected_prompt()
            if prompt:
                self.current_system_prompt = prompt
                self.current_prompt_display.setPlainText(prompt)
                if self.messages and self.messages[0].get("role") == "system":
                    self.messages[0]["content"] = prompt
                self.update_next_context_preview()

    def on_clear_history(self):
        self._save_conversation_timestamped()
        self.messages = []
        self.pending_stream_text = ""
        self.awaiting_response = False
        self.render_chat()
        self.update_next_context_preview()

    def on_stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()

    def on_ask(self):
        if self.worker and self.worker.isRunning():
            return

        preview_enabled = self.preview_checkbox.isChecked()
        user_input = self.input_box.toPlainText().strip()

        if preview_enabled and not user_input:
            QMessageBox.warning(self, "Input required", "Please enter a message.")
            return

        try:
            api_key = get_api_key()
        except ValueError as exc:
            QMessageBox.critical(self, "DS_KEY missing", str(exc))
            return

        model_mode = self.model_combo.currentData()
        use_tools = self.tools_checkbox.isChecked()
        stream = self.stream_checkbox.isChecked()

        self.refresh_skills_list()
        full_system = self.compose_full_system_prompt(user_input)

        if preview_enabled:
            if not self.messages:
                self.messages = [{"role": "system", "content": full_system}]
            else:
                if self.messages[0].get("role") == "system":
                    self.messages[0]["content"] = full_system
                else:
                    self.messages.insert(0, {"role": "system", "content": full_system})
            self.messages.append({"role": "user", "content": user_input})
            api_messages = list(self.messages)
        else:
            try:
                api_messages = self._parse_preview_messages()
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid preview JSON", str(exc))
                return
            self.messages = list(api_messages)
            if self.messages and self.messages[0].get("role") == "system":
                self.messages[0]["content"] = full_system
            else:
                self.messages.insert(0, {"role": "system", "content": full_system})
            api_messages = list(self.messages)

        if not api_messages or api_messages[-1].get("role") != "user":
            QMessageBox.warning(self, "Invalid conversation", "Last message to send must be a user message.")
            return

        self.awaiting_response = True
        self.pending_stream_text = ""
        self.render_chat()

        self.input_box.clear()
        self.ask_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.update_next_context_preview()

        self.worker = AskWorker(
            api_key,
            api_messages,
            model_mode,
            stream,
            self.current_system_prompt,
            use_tools=use_tools,
            workspace=self.workspace_path(),
            allow_run_command=use_tools and self.allow_commands_checkbox.isChecked(),
            enable_web_search=use_tools and self.web_search_checkbox.isChecked(),
            proxy_url=self.model_proxy_addr.text().strip() if self.model_proxy_checkbox.isChecked() else None,
        )
        self.worker.chunk.connect(self.on_chunk)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_chunk(self, text: str):
        self.pending_stream_text += text
        self.render_chat()

    def on_done(self, full_answer: str):
        self.messages.append({"role": "assistant", "content": full_answer})
        self.awaiting_response = False
        self.pending_stream_text = ""
        self.render_chat()
        self.update_next_context_preview()

    def on_failed(self, message: str):
        self.messages.append({"role": "assistant", "content": f"[Error] {message}"})
        self.awaiting_response = False
        self.pending_stream_text = ""
        self.render_chat()
        self.update_next_context_preview()

    def on_worker_finished(self):
        self.ask_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_next_context_preview(self):
        if not self.preview_checkbox.isChecked():
            self.preview_context_box.setReadOnly(False)
            return

        self.preview_context_box.setReadOnly(True)

        next_user_input = self.input_box.toPlainText().strip()
        if self.messages:
            next_messages = list(self.messages)
        else:
            next_messages = [{"role": "system", "content": self.current_system_prompt}]

        if next_user_input:
            next_messages.append({"role": "user", "content": next_user_input})

        self.preview_context_box.setPlainText(
            json.dumps(next_messages, ensure_ascii=False, indent=2)
        )

    def render_chat(self):
        sections = []
        for msg in self.messages:
            role = msg.get("role", "")
            if role == "system":
                continue
            sections.append(self._bubble_html(role, msg.get("content", "")))

        if self.awaiting_response:
            typing_text = self.pending_stream_text if self.pending_stream_text else "..."
            sections.append(self._bubble_html("assistant", typing_text, pending=True))

        html_text = "".join([
            "<html><body style='font-family:Segoe UI,Microsoft YaHei,sans-serif; padding:12px; background:#f5f6f8;'>",
            "".join(sections) if sections else "<p style='color:#7a7a7a;'>No messages yet.</p>",
            "</body></html>",
        ])
        self.chat_output.setHtml(html_text)
        self.chat_output.moveCursor(QTextCursor.End)

    def _bubble_html(self, role: str, content: str, pending: bool = False) -> str:
        safe = html.escape(content).replace("\n", "<br>")

        if role == "user":
            label = "You"
            row_align = "right"
            bubble_bg = "#d8f0ff"
            text_color = "#1f2937"
        else:
            label = "DeepSeek"
            row_align = "left"
            bubble_bg = "#ffffff"
            text_color = "#111827"

        pending_badge = " <span style='color:#6b7280;'>(typing)</span>" if pending else ""
        return (
            f"<div style='text-align:{row_align}; margin:8px 0;'>"
            f"<div style='display:inline-block; max-width:78%; text-align:left;'>"
            f"<div style='font-size:12px; color:#6b7280; margin-bottom:4px;'>{label}{pending_badge}</div>"
            f"<div style='background:{bubble_bg}; color:{text_color}; border:1px solid #e5e7eb; border-radius:12px; padding:10px 12px; line-height:1.5;'>{safe}</div>"
            "</div>"
            "</div>"
        )

    def _conversation_payload(self) -> dict:
        return {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "system_prompt": self.current_system_prompt,
            "messages": self.messages,
        }

    def _write_payload(self, file_path: Path, payload: dict):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _next_timestamp_path(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = self.logs_dir / f"conversation_{stamp}.json"
        counter = 1
        while candidate.exists():
            candidate = self.logs_dir / f"conversation_{stamp}_{counter:02d}.json"
            counter += 1
        return candidate

    def _next_daily_seq_path(self) -> Path:
        day = datetime.now().strftime("%Y%m%d")
        existing = sorted(self.logs_dir.glob(f"conversation_{day}_*.json"))
        max_seq = 0
        for path in existing:
            stem = path.stem
            parts = stem.split("_")
            if len(parts) >= 3 and parts[-1].isdigit():
                max_seq = max(max_seq, int(parts[-1]))
        return self.logs_dir / f"conversation_{day}_{max_seq + 1:03d}.json"

    def _save_conversation_timestamped(self):
        if not self.messages:
            return
        path = self._next_timestamp_path()
        self._write_payload(path, self._conversation_payload())

    def _parse_preview_messages(self):
        raw = self.preview_context_box.toPlainText().strip()
        if not raw:
            raise ValueError("Preview is empty. Provide a JSON array of messages.")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON parse failed: {exc}")

        if not isinstance(data, list):
            raise ValueError("Preview JSON must be a list of messages.")

        cleaned = []
        for index, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Message #{index} must be an object.")
            role = item.get("role")
            content = item.get("content")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Message #{index} has invalid role: {role}")
            if not isinstance(content, str):
                raise ValueError(f"Message #{index} content must be a string.")
            cleaned.append({"role": role, "content": content})
        return cleaned

    def on_save_conversation(self):
        if self.preview_checkbox.isChecked():
            if not self.messages:
                QMessageBox.information(self, "Nothing to save", "No conversation to save.")
                return
            payload = self._conversation_payload()
        else:
            try:
                preview_messages = self._parse_preview_messages()
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid preview JSON", str(exc))
                return
            payload = {
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "system_prompt": self.current_system_prompt,
                "messages": preview_messages,
            }

        path = self._next_daily_seq_path()
        self._write_payload(path, payload)
        QMessageBox.information(self, "Saved", f"Saved to: {path}")

    def on_load_conversation(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Conversation",
            str(self.logs_dir.resolve()),
            "JSON Files (*.json)",
        )
        if not file_path:
            return

        try:
            with Path(file_path).open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            QMessageBox.warning(self, "Load failed", f"Cannot read file: {exc}")
            return

        if isinstance(data, dict) and "messages" in data:
            maybe_messages = data.get("messages")
            loaded_prompt = data.get("system_prompt")
        elif isinstance(data, list):
            maybe_messages = data
            loaded_prompt = None
        else:
            QMessageBox.warning(self, "Invalid file", "JSON must be a payload object or message list.")
            return

        try:
            if not isinstance(maybe_messages, list):
                raise ValueError("messages must be a list")
            cleaned = []
            for index, item in enumerate(maybe_messages, start=1):
                if not isinstance(item, dict):
                    raise ValueError(f"Message #{index} must be an object")
                role = item.get("role")
                content = item.get("content")
                if role not in {"system", "user", "assistant"}:
                    raise ValueError(f"Message #{index} has invalid role: {role}")
                if not isinstance(content, str):
                    raise ValueError(f"Message #{index} content must be a string")
                cleaned.append({"role": role, "content": content})
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid conversation", str(exc))
            return

        self.messages = cleaned
        if isinstance(loaded_prompt, str) and loaded_prompt.strip():
            self.current_system_prompt = loaded_prompt
        else:
            for msg in self.messages:
                if msg.get("role") == "system":
                    self.current_system_prompt = msg.get("content", self.current_system_prompt)
                    break

        self.current_prompt_display.setPlainText(self.current_system_prompt)
        self.pending_stream_text = ""
        self.awaiting_response = False
        self.render_chat()
        self.update_next_context_preview()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
