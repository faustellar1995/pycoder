import sys
import urllib.error
import html
import json
import base64
import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt5.QtCore import QBuffer, QIODevice, QMimeData, QThread, Qt, pyqtSignal, QSettings
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QKeySequence, QPixmap, QTextCursor
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
    QMenu,
    QPushButton,
    QPlainTextEdit,
    QTextBrowser,
    QSplitter,
    QTabWidget,
    QShortcut,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QInputDialog,
)

from localharness_api import (
    API_URL,
    PROVIDER_DEEPSEEK,
    PROVIDER_KIMI,
    PROVIDER_OLLAMA,
    PROVIDER_LMSTUDIO,
    STREAM_TIMEOUT_DEFAULT,
    STREAM_TIMEOUT_UNLIMITED,
    TOOL_MAX_ROUNDS_DEFAULT,
    TOOL_MAX_ROUNDS_MAX,
    TOOL_MAX_ROUNDS_MIN,
    VISION_MAX_IMAGE_BYTES,
    VISION_MAX_IMAGES,
    build_user_message_content,
    chat_completion,
    check_available,
    content_to_display_text,
    effective_stream_timeout,
    effective_temperature_for_resolved_model,
    explain_http_error,
    image_bytes_to_data_url,
    image_data_urls_from_content,
    provider_supports_vision,
    resolve_chat_endpoint,
    resolve_model,
    StreamInterrupted,
    validate_messages_for_provider,
    vision_unsupported_hint,
    redact_messages_for_preview,
)
from markdown_renderer import markdown_to_html
from localharness_harness import (
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
from workspace_tools import WorkspaceToolSession, openai_tool_specs
from todo_model import (
    TodoItem,
    extract_todos_from_reply,
    todos_execute_hint,
    todos_from_json_list,
    todos_system_hint,
    todos_to_json_list,
)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
CHAT_THUMB_MAX_W = 240
CHAT_THUMB_MAX_H = 180


def _data_url_cache_key(data_url: str) -> str:
    b64 = data_url.split(",", 1)[-1] if "," in data_url else data_url
    return hashlib.sha256(b64.encode("ascii", errors="ignore")).hexdigest()


def data_url_to_chat_thumbnail(data_url: str) -> Optional[Tuple[str, int, int]]:
    """
    为 QTextBrowser 生成小缩略图 data URL 及显示尺寸。
    Qt 富文本对 img 常按原图 intrinsic 尺寸排版，直接嵌入大图 base64 会产生大量留白。
    """
    if not data_url or not data_url.startswith("data:"):
        return None
    b64_part = data_url.split(",", 1)[-1] if "," in data_url else ""
    try:
        raw = base64.b64decode(b64_part, validate=False)
    except Exception:
        return None
    if not raw:
        return None
    pix = QPixmap()
    if not pix.loadFromData(raw):
        return None
    scaled = pix.scaled(
        CHAT_THUMB_MAX_W,
        CHAT_THUMB_MAX_H,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )
    w, h = scaled.width(), scaled.height()
    if w < 1 or h < 1:
        return None
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    if not scaled.save(buf, "JPEG", quality=82):
        return None
    thumb_url = image_bytes_to_data_url(bytes(buf.data()), "image/jpeg")
    return thumb_url, w, h


def _mime_for_path(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        return "image/jpeg"
    if ext == "gif":
        return "image/gif"
    if ext == "webp":
        return "image/webp"
    if ext == "bmp":
        return "image/bmp"
    return "image/png"


def _qimage_to_png_bytes(qimg: QImage) -> bytes:
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    scaled = qimg
    if scaled.width() > 2048 or scaled.height() > 2048:
        scaled = scaled.scaled(2048, 2048, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    if not scaled.save(buf, "PNG"):
        raise ValueError("无法将图片编码为 PNG")
    return bytes(buf.data())


class ChatInputEdit(QPlainTextEdit):
    """支持从剪贴板或拖拽粘贴图片的输入框。"""

    image_attached = pyqtSignal(str)

    def insertFromMimeData(self, source: QMimeData) -> None:
        if source.hasImage():
            img = source.imageData()
            if img is not None:
                qimg = img if isinstance(img, QImage) else img.toImage()
                if not qimg.isNull():
                    try:
                        data = _qimage_to_png_bytes(qimg)
                        self.image_attached.emit(image_bytes_to_data_url(data, "image/png"))
                    except ValueError:
                        pass
                    return
        if source.hasUrls():
            for url in source.urls():
                local = url.toLocalFile()
                if not local:
                    continue
                p = Path(local)
                if p.suffix.lower() in _IMAGE_SUFFIXES and p.is_file():
                    try:
                        raw = p.read_bytes()
                        if len(raw) > VISION_MAX_IMAGE_BYTES:
                            continue
                        self.image_attached.emit(
                            image_bytes_to_data_url(raw, _mime_for_path(p))
                        )
                    except OSError:
                        continue
                    return
        super().insertFromMimeData(source)


class AskWorker(QThread):
    chunk = pyqtSignal(str)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)
    interrupted = pyqtSignal()

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
        api_url: str = API_URL,
        provider: str = PROVIDER_DEEPSEEK,
        temperature: float = 0.7,
        timeout: int = STREAM_TIMEOUT_DEFAULT,
        tool_max_rounds: Optional[int] = None,
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
        self.api_url = api_url
        self.provider = provider
        self.temperature = temperature
        self.timeout = timeout
        self.tool_max_rounds = tool_max_rounds
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
                api_url=self.api_url,
                provider=self.provider,
                tool_max_rounds=self.tool_max_rounds,
            )
            result = run_harness(
                api_key=self.api_key,
                messages=self.messages,
                model_mode=self.model_mode,
                config=cfg,
                temperature=self.temperature,
                timeout=self.timeout,
                should_stop=self.should_stop,
                on_stream_token=self.chunk.emit if self.stream else None,
            )
            if result.messages is not None:
                self.done.emit(
                    json.dumps(
                        {"answer": result.answer, "messages": result.messages},
                        ensure_ascii=False,
                    )
                )
            else:
                self.done.emit(result.answer)
        except StreamInterrupted:
            self.interrupted.emit()
        except urllib.error.HTTPError as exc:
            self.failed.emit(explain_http_error(exc))
        except urllib.error.URLError as exc:
            self.failed.emit(f"Network error: {exc.reason}")
        except Exception as exc:
            self.failed.emit(f"Unexpected error: {exc}")


class CommitWorker(QThread):
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        api_key: str,
        model_mode: str,
        workspace: Path,
        proxy_url: Optional[str] = None,
        api_url: str = API_URL,
        provider: str = PROVIDER_DEEPSEEK,
    ):
        super().__init__()
        self.api_key = api_key
        self.model_mode = model_mode
        self.workspace = workspace
        self.proxy_url = proxy_url
        self.api_url = api_url
        self.provider = provider

    def run(self):
        try:
            subject, body, log = generate_git_commit_message(
                api_key=self.api_key,
                model_mode=self.model_mode,
                workspace=self.workspace,
                proxy_url=self.proxy_url,
                api_url=self.api_url,
                provider=self.provider,
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
    todos: List[Dict[str, Any]] = field(default_factory=list)
    todo_mode: bool = False


class TodoEditDialog(QDialog):
    """编辑单条 Todo（content + tags）。"""

    def __init__(self, parent=None, item: Optional[TodoItem] = None):
        super().__init__(parent)
        self.setWindowTitle("编辑 Todo")
        self.resize(480, 200)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("内容"))
        self.content_edit = QPlainTextEdit()
        self.content_edit.setFixedHeight(72)
        layout.addWidget(self.content_edit)
        layout.addWidget(QLabel("标签（逗号分隔）"))
        self.tags_edit = QLineEdit()
        self.tags_edit.setPlaceholderText("例如：工作, 紧急")
        layout.addWidget(self.tags_edit)
        row = QHBoxLayout()
        ok_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        row.addStretch(1)
        row.addWidget(ok_btn)
        row.addWidget(cancel_btn)
        layout.addLayout(row)
        self._item_id = item.id if item else uuid.uuid4().hex[:12]
        if item:
            self.content_edit.setPlainText(item.content)
            self.tags_edit.setText(", ".join(item.tags))

    def build_item(self) -> Optional[TodoItem]:
        content = self.content_edit.toPlainText().strip()
        if not content:
            return None
        tags = [t.strip() for t in self.tags_edit.text().split(",") if t.strip()]
        return TodoItem(content=content, tags=tags, id=self._item_id)


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

        self.install_btn = QPushButton("安装选中技能到 ~/.localharness/skills")
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
        dest_dir = Path.home() / ".localharness" / "skills" / safe
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "SKILL.md"
        dest.write_text(raw, encoding="utf-8", newline="\n")
        clear_skill_discovery_cache()
        QMessageBox.information(self, "完成", f"已安装到:\n{dest}")


SMART_SAVE_FILENAME_SYSTEM = (
    "你只根据下面给出的对话片段，输出一行「文件主名」（不要扩展名、路径、引号或任何解释）。"
    "用简短中文或英文描述对话主题，长度不超过 40 字符；仅输出这一行，无其他文字。"
)

# 仅用于请求模型命名：截断对话文本以控制 token（不可在 UI 编辑）
SMART_SAVE_TRANSCRIPT_MAX_CHARS = 20000


def _filename_stem_from_llm_reply(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    line = text.splitlines()[0].strip()
    line = line.strip("\"'[]「」")
    lower = line.lower()
    if lower.endswith(".json"):
        line = line[:-5]
    return _sanitize_filename_component(line)[:80]


def _flatten_messages_for_transcript(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for m in messages:
        role = str(m.get("role", "?"))
        content = m.get("content")
        if isinstance(content, str):
            text = content
        else:
            text = content_to_display_text(content)
            if content is not None and not isinstance(content, str):
                text = text + "\n" + json.dumps(content, ensure_ascii=False)
        parts.append(f"### {role}\n{text}")
    return "\n\n".join(parts)


def _sanitize_filename_component(name: str) -> str:
    name = name.strip()
    for c in '<>:"/\\|?*\n\r\t':
        name = name.replace(c, "_")
    return name[:120]


class SmartFilenameWorker(QThread):
    """仅请求模型输出一行适合作为文件名的主题串（不生成摘要）。"""

    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        api_key: str,
        model_mode: str,
        transcript: str,
        proxy_url: Optional[str] = None,
        api_url: str = API_URL,
        provider: str = PROVIDER_DEEPSEEK,
    ):
        super().__init__()
        self.api_key = api_key
        self.model_mode = model_mode
        self.transcript = transcript
        self.proxy_url = proxy_url
        self.api_url = api_url
        self.provider = provider

    def run(self) -> None:
        try:
            messages = [
                {"role": "system", "content": SMART_SAVE_FILENAME_SYSTEM},
                {
                    "role": "user",
                    "content": "对话片段（过长时仅含开头部分，仅供命名参考）：\n\n" + self.transcript,
                },
            ]
            reply = chat_completion(
                self.api_key,
                messages,
                model_mode=self.model_mode,
                temperature=0.2,
                timeout=120,
                proxy_url=self.proxy_url,
                api_url=self.api_url,
                provider=self.provider,
            )
            self.done.emit(reply)
        except Exception as exc:
            self.failed.emit(str(exc))


class ModelsRefreshWorker(QThread):
    """后台调用 check_available，拉取 DeepSeek / Kimi / 本地 Ollama 模型列表。"""

    done = pyqtSignal(list, list)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        proxy_url: Optional[str] = None,
        timeout: int = 45,
        ollama_ui_base: Optional[str] = None,
        lmstudio_ui_base: Optional[str] = None,
    ):
        super().__init__()
        self.proxy_url = proxy_url
        self.timeout = timeout
        self.ollama_ui_base = ollama_ui_base
        self.lmstudio_ui_base = lmstudio_ui_base

    def run(self) -> None:
        try:
            entries, notes = check_available(
                proxy_url=self.proxy_url,
                timeout=self.timeout,
                ollama_ui_base=self.ollama_ui_base,
                lmstudio_ui_base=self.lmstudio_ui_base,
            )
            self.done.emit(entries, notes)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LocalHarness — Multi-model Chat")
        self.setAcceptDrops(True)
        self.resize(900, 750)

        self.worker = None
        self.commit_worker = None
        self.smart_save_worker = None
        self._models_refresh_worker = None
        self._smart_save_pending: Optional[Dict[str, Any]] = None
        self.messages = []
        self.todos: List[TodoItem] = []
        self._todo_list_refreshing = False
        self._submitting_todo_execute = False
        self.current_system_prompt = "You are a helpful assistant."
        self.pending_stream_text = ""
        self.awaiting_response = False
        self._render_scheduled = False
        self._pending_image_urls: List[str] = []
        self._pending_image_thumbs: Dict[str, QPixmap] = {}
        self._chat_thumb_cache: Dict[str, Tuple[str, int, int]] = {}
        self.logs_dir = Path("./logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.settings = QSettings("LocalHarness", "PyQtClient")
        self._skill_catalog = []
        self._preview_debounce = QTimer(self)
        self._preview_debounce.setSingleShot(True)
        self._preview_debounce.timeout.connect(self.update_next_context_preview)

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

        # ── Markdown 渲染开关 ────────────────────────────────────────────
        result_header = QHBoxLayout()
        result_header.addWidget(QLabel("对话结果"))
        self.render_md_checkbox = QCheckBox("Markdown 渲染")
        md_val = self.settings.value("render_markdown", True)
        if isinstance(md_val, str):
            md_val = md_val.lower() in ("1", "true", "yes")
        self.render_md_checkbox.setChecked(bool(md_val))
        self.render_md_checkbox.setToolTip("关闭后显示原始 Markdown 文本（可选，便于调试格式）。")
        self.render_md_checkbox.toggled.connect(
            lambda v: (
                self.settings.setValue("render_markdown", bool(v)),
                self.render_chat(),
            )
        )
        result_header.addWidget(self.render_md_checkbox)
        result_header.addStretch(1)
        right_layout.addLayout(result_header)

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
        self.preview_context_box.setPlaceholderText(
            "开启 Preview 后显示下一次请求的摘要；可用下方开关隐藏不需要的段落（tools 默认关闭）。"
        )
        preview_layout.addWidget(self.preview_context_box, 1)

        preview_filters = QHBoxLayout()
        preview_filters.setSpacing(12)
        preview_filters.addWidget(QLabel("显示："))
        pm = self.settings.value("preview_show_meta", True)
        if isinstance(pm, str):
            pm = pm.lower() in ("1", "true", "yes")
        self.preview_show_meta_checkbox = QCheckBox("请求参数")
        self.preview_show_meta_checkbox.setChecked(bool(pm))
        self.preview_show_meta_checkbox.setToolTip(
            "model、model_mode、temperature、timeout、stream"
        )
        self.preview_show_meta_checkbox.toggled.connect(
            lambda v: self.settings.setValue("preview_show_meta", bool(v))
        )
        pmsg = self.settings.value("preview_show_messages", True)
        if isinstance(pmsg, str):
            pmsg = pmsg.lower() in ("1", "true", "yes")
        self.preview_show_messages_checkbox = QCheckBox("messages")
        self.preview_show_messages_checkbox.setChecked(bool(pmsg))
        self.preview_show_messages_checkbox.setToolTip("对话消息数组（通常必看）")
        self.preview_show_messages_checkbox.toggled.connect(
            lambda v: self.settings.setValue("preview_show_messages", bool(v))
        )
        pt = self.settings.value("preview_show_tools", False)
        if isinstance(pt, str):
            pt = pt.lower() in ("1", "true", "yes")
        self.preview_show_tools_checkbox = QCheckBox("tools")
        self.preview_show_tools_checkbox.setChecked(bool(pt))
        self.preview_show_tools_checkbox.setToolTip("函数定义列表，体积大；调 prompt 时可关掉")
        self.preview_show_tools_checkbox.toggled.connect(
            lambda v: self.settings.setValue("preview_show_tools", bool(v))
        )
        pi = self.settings.value("preview_show_images", False)
        if isinstance(pi, str):
            pi = pi.lower() in ("1", "true", "yes")
        self.preview_show_images_checkbox = QCheckBox("images")
        self.preview_show_images_checkbox.setChecked(bool(pi))
        self.preview_show_images_checkbox.setToolTip(
            "显示 messages 中的完整 image_url base64（体积大、会拖慢输入；默认隐藏为占位符）"
        )
        self.preview_show_images_checkbox.toggled.connect(
            lambda v: self.settings.setValue("preview_show_images", bool(v))
        )
        preview_filters.addWidget(self.preview_show_meta_checkbox)
        preview_filters.addWidget(self.preview_show_messages_checkbox)
        preview_filters.addWidget(self.preview_show_tools_checkbox)
        preview_filters.addWidget(self.preview_show_images_checkbox)
        preview_filters.addStretch(1)
        preview_layout.addLayout(preview_filters)

        content_split.addWidget(preview_wrap)

        right_layout.addWidget(QLabel("输入"))
        self.attachments_widget = QWidget()
        self.attachments_layout = QHBoxLayout(self.attachments_widget)
        self.attachments_layout.setContentsMargins(0, 0, 0, 0)
        self.attachments_layout.setSpacing(6)
        self.attachments_widget.setVisible(False)
        right_layout.addWidget(self.attachments_widget)

        self.input_box = ChatInputEdit()
        self.input_box.setPlaceholderText(
            "在这里输入你的问题…（Ctrl+V 可粘贴图片；亦可将图片拖入此框）"
        )
        self.input_box.setFixedHeight(96)
        self.input_box.setAcceptDrops(True)
        self.input_box.image_attached.connect(self._on_input_image_attached)
        right_layout.addWidget(self.input_box)

        buttons_row = QHBoxLayout()
        self.attach_image_btn = QPushButton("图片…")
        self.attach_image_btn.setToolTip(
            f"选择图片附件（最多 {VISION_MAX_IMAGES} 张，单张 ≤ {VISION_MAX_IMAGE_BYTES // (1024 * 1024)}MB）。"
            "Kimi / Ollama 视觉模型可用；DeepSeek 官方 API 暂不支持。"
        )
        self.clear_images_btn = QPushButton("清除图片")
        self.clear_images_btn.setEnabled(False)
        self.ask_button = QPushButton("Send")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.clear_history_button = QPushButton("Clear History")
        self.load_button = QPushButton("Load")
        self.save_button = QPushButton("Save")
        self.smart_save_button = QPushButton("Smart Save")
        self.smart_save_button.setToolTip("由模型根据对话命名并保存到 logs（无前缀摘要）")
        buttons_row.addWidget(self.attach_image_btn)
        buttons_row.addWidget(self.clear_images_btn)
        buttons_row.addWidget(self.ask_button)
        buttons_row.addWidget(self.stop_button)
        buttons_row.addWidget(self.clear_history_button)
        buttons_row.addWidget(self.load_button)
        buttons_row.addWidget(self.save_button)
        buttons_row.addWidget(self.smart_save_button)
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
        self.model_combo.setToolTip(
            "来自 DeepSeek / Kimi GET /v1/models、本地 Ollama（/api/tags）与 LM Studio（/v1/models）；"
            "标注 [DS]/[Kimi]/[Ollama]/[LM Studio]。"
            "Kimi 需 KIMI_KEY；DeepSeek 需 DS_KEY；Ollama / LM Studio 无需密钥（默认本机端口，可用下方地址或环境变量覆盖）。"
            "点击「刷新模型」合并列表。"
            "图片输入：Kimi 视觉模型与 Ollama / LM Studio 多模态模型支持；DeepSeek 官方 API 暂不支持。"
        )
        self.refresh_models_btn = QPushButton("刷新模型")
        self.refresh_models_btn.setToolTip(
            "合并 DeepSeek、Kimi、Ollama 与 LM Studio 模型；访问本机服务时建议关闭代理。"
        )
        row_model.addWidget(self.model_combo, 1)
        row_model.addWidget(self.refresh_models_btn)
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

        row_params = QHBoxLayout()
        row_params.addWidget(QLabel("Temperature"))
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setDecimals(2)
        temp_saved = self.settings.value("request_temperature", 0.7)
        try:
            temp_f = float(temp_saved)
        except (TypeError, ValueError):
            temp_f = 0.7
        self.temperature_spin.setValue(max(0.0, min(2.0, temp_f)))
        self.temperature_spin.setToolTip(
            "采样温度。部分 Kimi K2.5/K2.6 模型 API 仅允许 temperature=1，预览区会显示实际生效值。"
        )
        row_params.addWidget(self.temperature_spin)
        row_params.addWidget(QLabel("Timeout (s)"))
        self.request_timeout_spin = QSpinBox()
        self.request_timeout_spin.setRange(STREAM_TIMEOUT_UNLIMITED, 7200)
        self.request_timeout_spin.setSingleStep(30)
        self.request_timeout_spin.setSpecialValueText("∞")
        timeout_saved = self.settings.value("request_timeout", STREAM_TIMEOUT_DEFAULT)
        try:
            timeout_i = int(timeout_saved)
        except (TypeError, ValueError):
            timeout_i = STREAM_TIMEOUT_DEFAULT
        if timeout_i < 0:
            timeout_i = STREAM_TIMEOUT_UNLIMITED
        else:
            timeout_i = max(30, min(7200, timeout_i))
        self.request_timeout_spin.setValue(timeout_i)
        self.request_timeout_spin.setToolTip(
            "单次请求总超时（秒）。-1（∞）表示无限等待。"
            "正值时本地 Ollama 思考模型会自动不低于 API 下限（当前 3600s）；"
            "流式读轮询仍约 1s 以便 Stop 及时生效。"
        )
        row_params.addWidget(self.request_timeout_spin)
        row_params.addStretch(1)
        model_layout.addLayout(row_params)

        row_ollama = QHBoxLayout()
        row_ollama.addWidget(QLabel("Ollama 地址"))
        self.ollama_base_edit = QLineEdit()
        self.ollama_base_edit.setPlaceholderText(
            "留空 → OLLAMA_HOST / OLLAMA_API_BASE / 默认 http://127.0.0.1:11434"
        )
        ob_saved = self.settings.value("ollama_api_base", "")
        self.ollama_base_edit.setText(str(ob_saved) if ob_saved is not None else "")
        self.ollama_base_edit.setToolTip(
            "本地 Ollama HTTP 根地址（无路径）。用于扫描模型与选中 [Ollama] 时的对话请求。"
        )
        self.ollama_base_edit.textChanged.connect(lambda t: self.settings.setValue("ollama_api_base", t))
        row_ollama.addWidget(self.ollama_base_edit, 1)
        model_layout.addLayout(row_ollama)

        row_lmstudio = QHBoxLayout()
        row_lmstudio.addWidget(QLabel("LM Studio 地址"))
        self.lmstudio_base_edit = QLineEdit()
        self.lmstudio_base_edit.setPlaceholderText(
            "留空 → LMSTUDIO_API_BASE / 默认 http://127.0.0.1:1234"
        )
        ls_saved = self.settings.value("lmstudio_api_base", "")
        self.lmstudio_base_edit.setText(str(ls_saved) if ls_saved is not None else "")
        self.lmstudio_base_edit.setToolTip(
            "LM Studio 本地服务（默认 http://127.0.0.1:1234）。"
            "须先在 LM Studio → Developer 打开 Start server，再点「刷新模型」。"
            "访问本机时建议关闭代理。可选环境变量 LMSTUDIO_API_KEY。"
        )
        self.lmstudio_base_edit.textChanged.connect(
            lambda t: self.settings.setValue("lmstudio_api_base", t)
        )
        row_lmstudio.addWidget(self.lmstudio_base_edit, 1)
        model_layout.addLayout(row_lmstudio)

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
            "工具轮数上限见下方；也可用环境变量 DEEPSEEK_TOOL_MAX_ROUNDS 覆盖默认值。"
        )
        model_layout.addWidget(self.tools_checkbox)

        row_tool_rounds = QHBoxLayout()
        row_tool_rounds.addWidget(QLabel("工具轮数上限"))
        self.tool_max_rounds_spin = QSpinBox()
        self.tool_max_rounds_spin.setRange(TOOL_MAX_ROUNDS_MIN, TOOL_MAX_ROUNDS_MAX)
        tmr_saved = self.settings.value("tool_max_rounds", TOOL_MAX_ROUNDS_DEFAULT)
        try:
            tmr_i = int(tmr_saved)
        except (TypeError, ValueError):
            tmr_i = TOOL_MAX_ROUNDS_DEFAULT
        self.tool_max_rounds_spin.setValue(
            max(TOOL_MAX_ROUNDS_MIN, min(TOOL_MAX_ROUNDS_MAX, tmr_i))
        )
        self.tool_max_rounds_spin.setToolTip(
            "单次提问中模型↔工具 API 往返次数上限（默认 64）。"
            "达到上限时保留迄今 tool 上下文并提示继续追问，不再抛错清空。"
        )
        self.tool_max_rounds_spin.valueChanged.connect(
            lambda v: self.settings.setValue("tool_max_rounds", int(v))
        )
        row_tool_rounds.addWidget(self.tool_max_rounds_spin)
        row_tool_rounds.addStretch(1)
        model_layout.addLayout(row_tool_rounds)

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

        # ── Tab: Todo ───────────────────────────────────────────────────────
        todo_tab = QWidget()
        todo_layout = QVBoxLayout(todo_tab)
        todo_layout.setContentsMargins(8, 8, 8, 8)
        todo_layout.setSpacing(6)

        todo_top = QHBoxLayout()
        self.todo_mode_checkbox = QCheckBox("Todo 模式")
        self.todo_mode_checkbox.setToolTip(
            "启用后：system prompt 注入 Todo 输出约定；助手回复末尾的 ```todos``` 块"
            "会被解析并追加到列表（默认每轮 2 条；用户可要求 N 条）。正文中不显示该代码块。"
        )
        tm_default = self.settings.value("todo_mode_default", False)
        if isinstance(tm_default, str):
            tm_default = tm_default.lower() in ("1", "true", "yes")
        self.todo_mode_checkbox.setChecked(bool(tm_default))
        self.todo_mode_checkbox.toggled.connect(self._on_todo_mode_toggled)
        self.todo_add_btn = QPushButton("新增")
        self.todo_edit_btn = QPushButton("编辑")
        self.todo_delete_btn = QPushButton("删除")
        self.todo_clear_btn = QPushButton("清空")
        todo_top.addWidget(self.todo_mode_checkbox)
        todo_top.addStretch(1)
        todo_top.addWidget(self.todo_add_btn)
        todo_top.addWidget(self.todo_edit_btn)
        todo_top.addWidget(self.todo_delete_btn)
        todo_top.addWidget(self.todo_clear_btn)
        todo_layout.addLayout(todo_top)

        todo_layout.addWidget(
            QLabel("拖动调整优先级；双击提交询问；F2/右键编辑；Delete 删除。")
        )
        self.todo_list = QListWidget()
        self.todo_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.todo_list.setDefaultDropAction(Qt.MoveAction)
        self.todo_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.todo_list.setContextMenuPolicy(Qt.CustomContextMenu)
        todo_layout.addWidget(self.todo_list, 1)
        left_tabs.addTab(todo_tab, "Todo")

        self.todo_add_btn.clicked.connect(self.on_todo_add)
        self.todo_edit_btn.clicked.connect(self.on_todo_edit)
        self.todo_delete_btn.clicked.connect(self.on_todo_delete)
        self.todo_clear_btn.clicked.connect(self.on_todo_clear)
        self.todo_list.itemDoubleClicked.connect(self.on_todo_ask)
        self.todo_list.customContextMenuRequested.connect(self._on_todo_context_menu)
        self.todo_list.model().rowsMoved.connect(self._on_todos_reordered)
        QShortcut(QKeySequence("F2"), self.todo_list, activated=self.on_todo_edit)
        QShortcut(QKeySequence.Delete, self.todo_list, activated=self.on_todo_delete)
        QShortcut(QKeySequence.Backspace, self.todo_list, activated=self.on_todo_delete)

        # 须在创建 skills_list 之后再连接（初始化早期会调用 _on_tools_toggled，不能在其中刷新预览）
        self.tools_checkbox.toggled.connect(self.update_next_context_preview)

        self.refresh_skills_btn.clicked.connect(lambda: self.refresh_skills_list(force_disk=True))
        self.market_skills_btn.clicked.connect(self.on_open_market)
        self.clear_catalog_cache_btn.clicked.connect(self.on_clear_catalog_cache)

        self.ask_button.clicked.connect(self.on_ask)
        self.attach_image_btn.clicked.connect(self.on_attach_images)
        self.clear_images_btn.clicked.connect(self._clear_pending_images)
        self.stop_button.clicked.connect(self.on_stop)
        self.clear_history_button.clicked.connect(self.on_clear_history)
        self.load_button.clicked.connect(self.on_load_conversation)
        self.save_button.clicked.connect(self.on_save_conversation)
        self.smart_save_button.clicked.connect(self.on_smart_save)
        self.input_box.textChanged.connect(self._schedule_preview_update)
        self.preview_checkbox.toggled.connect(self.update_next_context_preview)
        self.model_combo.currentIndexChanged.connect(self.update_next_context_preview)
        self.model_combo.currentIndexChanged.connect(self._persist_model_choice)
        self.model_combo.currentIndexChanged.connect(self.render_chat)
        self.refresh_models_btn.clicked.connect(self.on_refresh_models)
        self.stream_checkbox.toggled.connect(self.update_next_context_preview)
        self.temperature_spin.valueChanged.connect(self._on_request_params_changed)
        self.request_timeout_spin.valueChanged.connect(self._on_request_params_changed)
        self.allow_commands_checkbox.toggled.connect(self.update_next_context_preview)
        self.web_search_checkbox.toggled.connect(self.update_next_context_preview)
        self.auto_skill_checkbox.toggled.connect(self.update_next_context_preview)
        self.workspace_edit.textChanged.connect(self._schedule_preview_update)
        self.skills_list.itemChanged.connect(self.update_next_context_preview)
        self.context_list.model().rowsInserted.connect(lambda *a: self.update_next_context_preview())
        self.context_list.model().rowsRemoved.connect(lambda *a: self.update_next_context_preview())
        self.preview_show_meta_checkbox.toggled.connect(self.update_next_context_preview)
        self.preview_show_messages_checkbox.toggled.connect(self.update_next_context_preview)
        self.preview_show_tools_checkbox.toggled.connect(self.update_next_context_preview)
        self.preview_show_images_checkbox.toggled.connect(self.update_next_context_preview)
        self.commit_gen_btn.clicked.connect(self.on_generate_commit_message)
        self.commit_do_btn.clicked.connect(self.on_do_git_commit)

        self._load_cached_models_into_combo()
        self.render_chat()
        self.update_next_context_preview()
        self.refresh_skills_list()

        # 必须先连接 currentChanged，再 _init_sessions；否则首个 Tab 创建时不会触发同步，
        # _active_session_index 会一直为 -1，切换会话时无法保存消息，表现为聊天记录与预览 JSON 错乱。
        self.session_tabs.currentChanged.connect(self.on_session_changed)
        self.session_tabs.tabCloseRequested.connect(self.close_session)
        self._init_sessions()
        self.new_session_btn.clicked.connect(self.new_session)

        # Chrome-like 会话快捷键
        QShortcut("Ctrl+T", self, activated=self.new_session)
        QShortcut("Ctrl+W", self, activated=lambda: self.close_session(self.session_tabs.currentIndex()))
        QShortcut("Ctrl+Tab", self, activated=lambda: self._cycle_session(1))
        QShortcut("Ctrl+Shift+Tab", self, activated=lambda: self._cycle_session(-1))

    def _schedule_render_chat(self, delay_ms: int = 60) -> None:
        """
        流式输出时避免每个 token 都触发 setHtml 全量重绘（UI 会明显卡顿/假死）。
        这里把渲染节流到固定帧率附近（默认 ~16FPS）。
        """
        if self._render_scheduled:
            return
        self._render_scheduled = True

        def _do() -> None:
            self._render_scheduled = False
            self.render_chat()

        QTimer.singleShot(max(0, int(delay_ms)), _do)

    def _schedule_preview_update(self, delay_ms: int = 300) -> None:
        """输入框等高频变更时节流预览刷新，避免每键 dump 含 base64 的巨大 JSON。"""
        if not self.preview_checkbox.isChecked():
            return
        self._preview_debounce.start(max(50, int(delay_ms)))

    def workspace_path(self) -> Path:
        text = self.workspace_edit.text().strip()
        if not text:
            return Path.cwd().resolve()
        return Path(text).expanduser().resolve()

    def _on_input_image_attached(self, data_url: str) -> None:
        self._add_pending_image_data_url(data_url)

    def _add_pending_image_data_url(self, data_url: str) -> None:
        if not data_url or not data_url.startswith("data:"):
            return
        if len(self._pending_image_urls) >= VISION_MAX_IMAGES:
            QMessageBox.warning(
                self,
                "图片",
                f"最多附加 {VISION_MAX_IMAGES} 张图片。",
            )
            return
        if data_url in self._pending_image_urls:
            return
        self._ensure_thumbnail_for_url(data_url)
        self._pending_image_urls.append(data_url)
        self._refresh_attachment_bar()
        self.update_next_context_preview()
        prov = self.current_provider() if self._model_choice_ready() else PROVIDER_DEEPSEEK
        if not provider_supports_vision(prov):
            self.statusBar().showMessage(vision_unsupported_hint(prov), 8000)

    def _ensure_thumbnail_for_url(self, data_url: str) -> None:
        if data_url in self._pending_image_thumbs:
            return
        pix = QPixmap()
        b64_part = data_url.split(",", 1)[-1] if "," in data_url else ""
        try:
            raw = base64.b64decode(b64_part, validate=False)
            loaded = pix.loadFromData(raw) if raw else False
        except Exception:
            loaded = False
        if loaded:
            self._pending_image_thumbs[data_url] = pix.scaled(
                56, 56, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

    def _clear_pending_images(self) -> None:
        self._pending_image_urls.clear()
        self._pending_image_thumbs.clear()
        self._refresh_attachment_bar()
        self.update_next_context_preview()

    def _refresh_attachment_bar(self) -> None:
        while self.attachments_layout.count():
            item = self.attachments_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if not self._pending_image_urls:
            self.attachments_widget.setVisible(False)
            self.clear_images_btn.setEnabled(False)
            return
        self.attachments_widget.setVisible(True)
        self.clear_images_btn.setEnabled(True)
        for idx, url in enumerate(self._pending_image_urls):
            wrap = QWidget()
            row = QVBoxLayout(wrap)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(2)
            thumb = QLabel()
            thumb.setFixedSize(56, 56)
            thumb.setStyleSheet("border:1px solid #ccc; border-radius:4px;")
            cached = self._pending_image_thumbs.get(url)
            if cached is not None and not cached.isNull():
                thumb.setPixmap(cached)
            else:
                thumb.setText("img")
                thumb.setAlignment(Qt.AlignCenter)
            rm = QPushButton("×")
            rm.setFixedWidth(24)
            rm.clicked.connect(lambda _checked=False, i=idx: self._remove_pending_image(i))
            row.addWidget(thumb)
            row.addWidget(rm, 0, Qt.AlignHCenter)
            self.attachments_layout.addWidget(wrap)
        self.attachments_layout.addStretch(1)

    def _remove_pending_image(self, index: int) -> None:
        if 0 <= index < len(self._pending_image_urls):
            url = self._pending_image_urls.pop(index)
            self._pending_image_thumbs.pop(url, None)
            self._refresh_attachment_bar()
            self.update_next_context_preview()

    def on_attach_images(self) -> None:
        if len(self._pending_image_urls) >= VISION_MAX_IMAGES:
            QMessageBox.warning(self, "图片", f"最多附加 {VISION_MAX_IMAGES} 张图片。")
            return
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图片",
            "",
            "Images (*.png *.jpg *.jpeg *.gif *.webp *.bmp)",
        )
        for f in files:
            if len(self._pending_image_urls) >= VISION_MAX_IMAGES:
                break
            p = Path(f)
            if p.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            try:
                raw = p.read_bytes()
            except OSError as exc:
                QMessageBox.warning(self, "图片", f"无法读取 {p.name}: {exc}")
                continue
            if len(raw) > VISION_MAX_IMAGE_BYTES:
                QMessageBox.warning(
                    self,
                    "图片",
                    f"{p.name} 超过 {VISION_MAX_IMAGE_BYTES // (1024 * 1024)}MB 上限。",
                )
                continue
            self._add_pending_image_data_url(
                image_bytes_to_data_url(raw, _mime_for_path(p))
            )

    def current_provider(self) -> str:
        d = self.model_combo.currentData()
        if isinstance(d, (tuple, list)) and len(d) >= 1:
            p = str(d[0])
            if p in ("__placeholder__", ""):
                return PROVIDER_DEEPSEEK
            return p
        return PROVIDER_DEEPSEEK

    def current_model_mode(self) -> str:
        d = self.model_combo.currentData()
        if isinstance(d, (tuple, list)) and len(d) >= 2:
            return str(d[1])
        return ""

    def _model_choice_ready(self) -> bool:
        d = self.model_combo.currentData()
        if not isinstance(d, (tuple, list)) or len(d) < 2:
            return False
        if str(d[0]) in ("__placeholder__", ""):
            return False
        return bool(str(d[1]).strip())

    def _persist_model_choice(self, *_args: Any) -> None:
        d = self.model_combo.currentData()
        if isinstance(d, (tuple, list)) and len(d) >= 2 and str(d[0]) not in ("__placeholder__", ""):
            self.settings.setValue("last_model_provider", str(d[0]))
            self.settings.setValue("last_model_id", str(d[1]))

    def _find_model_row(self, choice: Tuple[str, str]) -> int:
        for i in range(self.model_combo.count()):
            d = self.model_combo.itemData(i)
            if not isinstance(d, (tuple, list)) or len(d) < 2:
                continue
            if str(d[0]) == str(choice[0]) and str(d[1]) == str(choice[1]):
                return i
        return -1

    def _apply_model_entries(self, entries: List[Tuple[str, str]], *, persist: bool = True) -> None:
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for prov, mid in entries:
            if prov == PROVIDER_DEEPSEEK:
                tag = "DS"
            elif prov == PROVIDER_KIMI:
                tag = "Kimi"
            elif prov == PROVIDER_OLLAMA:
                tag = "Ollama"
            elif prov == PROVIDER_LMSTUDIO:
                tag = "LM Studio"
            else:
                tag = str(prov)[:8]
            self.model_combo.addItem(f"[{tag}] {mid}", (prov, mid))
        self.model_combo.blockSignals(False)
        if persist:
            self.settings.setValue(
                "available_models_cache_v1",
                json.dumps(entries, ensure_ascii=False),
            )
        if self.model_combo.count() > 0:
            self.model_combo.setCurrentIndex(0)

    def _restore_last_model_choice(self) -> None:
        lp = self.settings.value("last_model_provider", "")
        lm = self.settings.value("last_model_id", "")
        if not str(lp).strip() or not str(lm).strip():
            return
        idx = self._find_model_row((str(lp), str(lm)))
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

    def _load_cached_models_into_combo(self) -> None:
        raw = self.settings.value("available_models_cache_v1", "")
        if not raw or not str(raw).strip():
            self.model_combo.addItem("— 点击「刷新模型」—", ("__placeholder__", ""))
            return
        try:
            data = json.loads(str(raw))
        except Exception:
            self.model_combo.addItem("— 点击「刷新模型」—", ("__placeholder__", ""))
            return
        if not isinstance(data, list) or not data:
            self.model_combo.addItem("— 点击「刷新模型」—", ("__placeholder__", ""))
            return
        entries: List[Tuple[str, str]] = []
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                entries.append((str(item[0]), str(item[1])))
        if not entries:
            self.model_combo.addItem("— 点击「刷新模型」—", ("__placeholder__", ""))
            return
        self._apply_model_entries(entries, persist=False)
        self._restore_last_model_choice()

    def on_refresh_models(self) -> None:
        if self._models_refresh_worker and self._models_refresh_worker.isRunning():
            return
        proxy = self.model_proxy_addr.text().strip() if self.model_proxy_checkbox.isChecked() else None
        self.refresh_models_btn.setEnabled(False)
        self.statusBar().showMessage("正在拉取可用模型…")
        ollama_ui = self.ollama_base_edit.text().strip() or None
        lmstudio_ui = self.lmstudio_base_edit.text().strip() or None
        self._models_refresh_worker = ModelsRefreshWorker(
            proxy_url=proxy,
            timeout=60,
            ollama_ui_base=ollama_ui,
            lmstudio_ui_base=lmstudio_ui,
        )
        self._models_refresh_worker.done.connect(self._on_models_refresh_done)
        self._models_refresh_worker.failed.connect(self._on_models_refresh_failed)
        self._models_refresh_worker.finished.connect(self._on_models_refresh_finished)
        self._models_refresh_worker.start()

    def _cached_model_entries(self) -> List[Tuple[str, str]]:
        raw = self.settings.value("available_models_cache_v1", "")
        if not raw or not str(raw).strip():
            return []
        try:
            data = json.loads(str(raw))
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        out: List[Tuple[str, str]] = []
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append((str(item[0]), str(item[1])))
        return out

    def _merge_model_entries_with_cache(
        self,
        fresh: List[Tuple[str, str]],
        notes: List[str],
    ) -> List[Tuple[str, str]]:
        """
        若本次 DeepSeek/Kimi 的 list-models 请求失败，沿用设置里上次成功的条目，
        避免下拉框被 Ollama 等新数据源「全覆盖」后云端模型消失。
        """
        note_blob = "\n".join(str(n) for n in notes)
        skip_ds = "未设置 DS_KEY" in note_blob
        skip_kimi = "未设置 KIMI_KEY" in note_blob
        fail_ds = "DeepSeek list-models:" in note_blob
        fail_kimi = "Kimi list-models:" in note_blob

        def has_prov(entries: List[Tuple[str, str]], p: str) -> bool:
            return any(x[0] == p for x in entries)

        merged = list(fresh)
        seen = set(merged)
        cached = self._cached_model_entries()

        if fail_ds and not skip_ds and not has_prov(fresh, PROVIDER_DEEPSEEK):
            for t in cached:
                if t[0] == PROVIDER_DEEPSEEK and t not in seen:
                    merged.append(t)
                    seen.add(t)
        if fail_kimi and not skip_kimi and not has_prov(fresh, PROVIDER_KIMI):
            for t in cached:
                if t[0] == PROVIDER_KIMI and t not in seen:
                    merged.append(t)
                    seen.add(t)

        merged.sort(key=lambda x: (x[0], x[1]))
        return merged

    def _on_models_refresh_done(self, entries: list, notes: list) -> None:
        fresh_pairs: List[Tuple[str, str]] = []
        for item in entries or []:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                fresh_pairs.append((str(item[0]), str(item[1])))
        merged = self._merge_model_entries_with_cache(fresh_pairs, list(notes))
        if not merged:
            msg = "未获取到任何模型。"
            if notes:
                msg += "\n\n" + "\n".join(str(n) for n in notes)
            QMessageBox.warning(self, "刷新模型", msg)
            self.statusBar().showMessage("刷新模型：列表为空", 6000)
            return
        self._apply_model_entries(merged, persist=True)
        self._restore_last_model_choice()
        self.update_next_context_preview()
        self.render_chat()
        note_txt = ""
        if notes:
            note_txt = " · " + "; ".join(str(n) for n in notes[:4])
            if len(notes) > 4:
                note_txt += "…"
        self.statusBar().showMessage(f"已加载 {len(merged)} 个模型{note_txt}", 10000)

    def _on_models_refresh_failed(self, message: str) -> None:
        QMessageBox.warning(self, "刷新模型", message)
        self.statusBar().showMessage("刷新模型失败", 6000)

    def _on_models_refresh_finished(self) -> None:
        self.refresh_models_btn.setEnabled(True)

    def _assistant_display_name(self) -> str:
        if not self._model_choice_ready():
            return "Assistant"
        p = self.current_provider()
        if p == PROVIDER_KIMI:
            return "Kimi"
        if p == PROVIDER_OLLAMA:
            return "Ollama"
        if p == PROVIDER_LMSTUDIO:
            return "LM Studio"
        return "DeepSeek"

    def _ollama_ui_base_for_api(self) -> Optional[str]:
        t = self.ollama_base_edit.text().strip()
        return t if t else None

    def _lmstudio_ui_base_for_api(self) -> Optional[str]:
        t = self.lmstudio_base_edit.text().strip()
        return t if t else None

    def _ui_temperature(self) -> float:
        return float(self.temperature_spin.value())

    def _ui_request_timeout(self) -> int:
        return int(self.request_timeout_spin.value())

    def _effective_request_timeout(self) -> Optional[int]:
        prov = self.current_provider() if self._model_choice_ready() else PROVIDER_DEEPSEEK
        return effective_stream_timeout(self._ui_request_timeout(), provider=prov)

    @staticmethod
    def _timeout_for_preview(value: Optional[int]) -> Any:
        return "unlimited" if value is None else value

    def _on_request_params_changed(self, *_args: Any) -> None:
        self.settings.setValue("request_temperature", self._ui_temperature())
        self.settings.setValue("request_timeout", self._ui_request_timeout())
        self.update_next_context_preview()

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
        self.tool_max_rounds_spin.setEnabled(on)

    def _apply_todo_finalize_to_messages(
        self, msgs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        out = [dict(m) for m in msgs]
        if not self._todo_mode_enabled():
            return out
        for i in range(len(out) - 1, -1, -1):
            m = out[i]
            if m.get("role") != "assistant":
                continue
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                out[i] = {**m, "content": self._finalize_assistant_answer(c)}
            break
        return out

    def _parse_harness_done_payload(self, payload: str) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        text = payload or ""
        if not text.strip().startswith("{"):
            return text, None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text, None
        if not isinstance(data, dict) or "messages" not in data:
            return text, None
        msgs = data.get("messages")
        if not isinstance(msgs, list):
            return str(data.get("answer") or text), None
        return str(data.get("answer") or ""), msgs

    def _todo_mode_enabled(self) -> bool:
        return self.todo_mode_checkbox.isChecked()

    def _on_todo_mode_toggled(self, checked: bool) -> None:
        self.settings.setValue("todo_mode_default", bool(checked))
        if 0 <= self._active_session_index < len(self.sessions):
            self.sessions[self._active_session_index].todo_mode = bool(checked)
        self.update_next_context_preview()

    def _sync_todos_to_session(self) -> None:
        if self._active_session_index < 0 or self._active_session_index >= len(self.sessions):
            return
        self.sessions[self._active_session_index].todos = todos_to_json_list(self.todos)

    def _refresh_todo_list_ui(self) -> None:
        self._todo_list_refreshing = True
        self.todo_list.clear()
        for i, t in enumerate(self.todos):
            item = QListWidgetItem(f"{i + 1}. {t.display_line()}")
            item.setData(Qt.UserRole, t.to_dict())
            self.todo_list.addItem(item)
        self._todo_list_refreshing = False

    def _todos_from_list_widget(self) -> List[TodoItem]:
        out: List[TodoItem] = []
        for i in range(self.todo_list.count()):
            it = self.todo_list.item(i)
            if it is None:
                continue
            d = it.data(Qt.UserRole)
            if isinstance(d, dict):
                out.append(TodoItem.from_dict(d))
        return out

    def _on_todos_reordered(self, *_args: Any) -> None:
        if self._todo_list_refreshing:
            return
        self.todos = self._todos_from_list_widget()
        self._refresh_todo_list_ui()
        self._sync_todos_to_session()

    def _append_todos(self, items: List[TodoItem]) -> None:
        if not items:
            return
        self.todos.extend(items)
        self._sync_todos_to_session()
        self._refresh_todo_list_ui()

    def _finalize_assistant_answer(self, raw: str) -> str:
        if not self._todo_mode_enabled():
            return raw
        if self._submitting_todo_execute:
            display, _ = extract_todos_from_reply(raw)
            return display
        display, new_items = extract_todos_from_reply(raw)
        if new_items:
            self._append_todos(new_items)
        return display

    def _clear_todo_execute_submit_flag(self) -> None:
        self._submitting_todo_execute = False

    def on_todo_add(self) -> None:
        dlg = TodoEditDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        item = dlg.build_item()
        if item is None:
            QMessageBox.warning(self, "Todo", "内容不能为空。")
            return
        self.todos.append(item)
        self._sync_todos_to_session()
        self._refresh_todo_list_ui()

    def _todo_row_for_item(self, item: Optional[QListWidgetItem] = None) -> int:
        if item is not None:
            return self.todo_list.row(item)
        return self.todo_list.currentRow()

    def _query_text_for_todo(self, todo: TodoItem) -> str:
        lines = [
            "请优先完成或解答以下待办（直接给实质帮助，不要转而规划新待办）：",
            "",
            todo.content.strip(),
        ]
        if todo.tags:
            lines.extend(["", "标签：" + ", ".join(todo.tags)])
        return "\n".join(lines)

    def on_todo_ask(self, item: Optional[QListWidgetItem] = None) -> None:
        """将选中 Todo 填入输入框并立即 Send。"""
        row = self._todo_row_for_item(item)
        if row < 0 or row >= len(self.todos):
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "忙碌", "当前正在等待回复，请稍后再试。")
            return
        self.todo_list.setCurrentRow(row)
        self._submitting_todo_execute = True
        self.input_box.setPlainText(self._query_text_for_todo(self.todos[row]))
        self.on_ask()

    def _on_todo_context_menu(self, pos) -> None:
        item = self.todo_list.itemAt(pos)
        if item is not None:
            self.todo_list.setCurrentItem(item)
        row = self.todo_list.currentRow()
        if row < 0 or row >= len(self.todos):
            return
        menu = QMenu(self)
        ask_action = menu.addAction("提交询问")
        edit_action = menu.addAction("编辑")
        edit_action.setShortcut(QKeySequence("F2"))
        menu.addSeparator()
        delete_action = menu.addAction("删除")
        chosen = menu.exec_(self.todo_list.viewport().mapToGlobal(pos))
        if chosen is ask_action:
            self.on_todo_ask(item)
        elif chosen is edit_action:
            self.on_todo_edit()
        elif chosen is delete_action:
            self.on_todo_delete()

    def on_todo_edit(self, *_args: Any) -> None:
        row = self.todo_list.currentRow()
        if row < 0 or row >= len(self.todos):
            QMessageBox.information(self, "Todo", "请先选择一条待办。")
            return
        current = self.todos[row]
        dlg = TodoEditDialog(self, item=current)
        if dlg.exec_() != QDialog.Accepted:
            return
        item = dlg.build_item()
        if item is None:
            QMessageBox.warning(self, "Todo", "内容不能为空。")
            return
        self.todos[row] = item
        self._sync_todos_to_session()
        self._refresh_todo_list_ui()
        self.todo_list.setCurrentRow(row)

    def on_todo_delete(self) -> None:
        row = self.todo_list.currentRow()
        if row < 0 or row >= len(self.todos):
            QMessageBox.information(self, "Todo", "请先选择一条待办。")
            return
        self.todos.pop(row)
        self._sync_todos_to_session()
        self._refresh_todo_list_ui()

    def on_todo_clear(self) -> None:
        if not self.todos:
            return
        reply = QMessageBox.question(
            self,
            "清空 Todo",
            "确定清空当前会话的全部待办？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.todos.clear()
        self._sync_todos_to_session()
        self._refresh_todo_list_ui()

    def _init_sessions(self) -> None:
        self.sessions: List[SessionState] = []
        self._active_session_index: int = -1
        self.new_session()
        # 个别环境下首个 Tab 可能不会触发 currentChanged，兜底对齐活动会话索引
        if self._active_session_index < 0:
            self._active_session_index = self.session_tabs.currentIndex()

    def new_session(self) -> None:
        idx = len(self.sessions) + 1
        tm = self.settings.value("todo_mode_default", False)
        if isinstance(tm, str):
            tm = tm.lower() in ("1", "true", "yes")
        state = SessionState(title=f"会话 {idx}", messages=[], todo_mode=bool(tm))
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
        self.sessions[i].todos = todos_to_json_list(self.todos)
        self.sessions[i].todo_mode = self._todo_mode_enabled()

    def _load_session_state(self, index: int) -> None:
        s = self.sessions[index]
        self.messages = list(s.messages)
        self.pending_stream_text = s.pending_stream_text
        self.awaiting_response = s.awaiting_response
        self.todos = todos_from_json_list(s.todos)
        self.todo_mode_checkbox.blockSignals(True)
        self.todo_mode_checkbox.setChecked(bool(s.todo_mode))
        self.todo_mode_checkbox.blockSignals(False)
        self._refresh_todo_list_ui()

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

    # ── Drag & Drop: 拖拽文件/目录到窗口加入上下文 ─────────────────────
    def dragEnterEvent(self, event):
        """Accept drag if it carries file URLs."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        """Required to show the drop indicator."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Extract local file/directory paths and add to context."""
        added = 0
        for url in event.mimeData().urls():
            local_path = url.toLocalFile()
            if not local_path:
                continue
            p = Path(local_path)
            if not p.exists():
                continue
            self._add_context_path(p)
            added += 1
        event.acceptProposedAction()
        if added > 0:
            self.statusBar().showMessage(f"已添加 {added} 项到上下文", 3000)

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
        if not self._model_choice_ready():
            QMessageBox.warning(self, "模型", "请先点击「刷新模型」并选择具体模型。")
            return
        try:
            api_key, api_url = resolve_chat_endpoint(
                self.current_provider(),
                ollama_ui_base=self._ollama_ui_base_for_api(),
                lmstudio_ui_base=self._lmstudio_ui_base_for_api(),
            )
        except ValueError as exc:
            QMessageBox.critical(self, "API Key", str(exc))
            return
        model_mode = self.current_model_mode()
        ws = self.workspace_path()
        proxy = self.model_proxy_addr.text().strip() if self.model_proxy_checkbox.isChecked() else None
        self.commit_log.setPlainText("[commit] 生成中…")
        self.commit_gen_btn.setEnabled(False)
        self.commit_do_btn.setEnabled(False)
        self.commit_worker = CommitWorker(
            api_key=api_key,
            model_mode=model_mode,
            workspace=ws,
            proxy_url=proxy,
            api_url=api_url,
            provider=self.current_provider(),
        )
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
        if self._todo_mode_enabled():
            if self._submitting_todo_execute:
                parts.append(todos_execute_hint())
            else:
                parts.append(todos_system_hint())
        return "\n\n".join(parts)

    def _build_next_request_messages(
        self,
        user_input: str,
        image_urls: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """与点击 Send 时组装的 messages 一致（不修改 self.messages）。"""
        full_system = self.compose_full_system_prompt(user_input)
        if not self.messages:
            base: List[Dict[str, Any]] = [{"role": "system", "content": full_system}]
        else:
            base = [dict(m) for m in self.messages]
            if base and base[0].get("role") == "system":
                base[0] = {**base[0], "content": full_system}
            else:
                base.insert(0, {"role": "system", "content": full_system})
        urls = list(image_urls if image_urls is not None else self._pending_image_urls)
        u = user_input.strip()
        if u or urls:
            base.append(
                {
                    "role": "user",
                    "content": build_user_message_content(u, urls),
                }
            )
        return base

    def _next_request_preview_payload(self) -> Dict[str, Any]:
        """下一次 POST 与 localharness_api / harness 对齐的摘要（便于调 prompt）。"""
        next_user = self.input_box.toPlainText().strip()
        msgs = self._build_next_request_messages(next_user)
        prov = self.current_provider()
        mode_str = self.current_model_mode() if self._model_choice_ready() else ""
        resolved = resolve_model(mode_str, prov) if mode_str else ""
        req_temp = self._ui_temperature()
        eff_temp = (
            effective_temperature_for_resolved_model(resolved, req_temp, provider=prov)
            if resolved
            else req_temp
        )
        req_timeout = self._ui_request_timeout()
        eff_timeout = self._effective_request_timeout()
        payload: Dict[str, Any] = {
            "provider": prov,
            "model": resolved,
            "model_mode": mode_str,
            "temperature": eff_temp,
            "temperature_requested": req_temp,
            "timeout": self._timeout_for_preview(eff_timeout),
            "timeout_requested": self._timeout_for_preview(
                None if req_timeout < 0 else req_timeout
            ),
            "stream": self.stream_checkbox.isChecked(),
            "messages": msgs,
        }
        if self.tools_checkbox.isChecked():
            payload["tools"] = openai_tool_specs(
                enable_run_command=self.allow_commands_checkbox.isChecked(),
                enable_web_search=self.web_search_checkbox.isChecked(),
            )
            payload["tool_choice"] = "auto"
        return payload

    def _filtered_preview_payload(self) -> Any:
        """按预览区下方开关裁剪 JSON，减少刷屏。"""
        full = self._next_request_preview_payload()
        if not self.preview_show_images_checkbox.isChecked():
            full = dict(full)
            full["messages"] = redact_messages_for_preview(
                full.get("messages") or [],
                hide_images=True,
            )
        show_meta = self.preview_show_meta_checkbox.isChecked()
        show_msgs = self.preview_show_messages_checkbox.isChecked()
        show_tools = self.preview_show_tools_checkbox.isChecked()
        if show_meta and show_msgs and show_tools:
            return full
        out: Dict[str, Any] = {}
        if show_meta:
            for k in (
                "provider",
                "model",
                "model_mode",
                "temperature",
                "temperature_requested",
                "timeout",
                "timeout_requested",
                "stream",
            ):
                if k in full:
                    out[k] = full[k]
        if show_msgs:
            out["messages"] = full["messages"]
        if show_tools:
            if "tools" in full:
                out["tools"] = full["tools"]
            if "tool_choice" in full:
                out["tool_choice"] = full["tool_choice"]
        if not out:
            return {"_hint": "请至少勾选一项「显示」"}
        return out

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
        pending_images = list(self._pending_image_urls)

        if preview_enabled and not user_input and not pending_images:
            QMessageBox.warning(self, "Input required", "请输入文字或附加图片。")
            return

        if not self._model_choice_ready():
            QMessageBox.warning(self, "模型", "请先点击「刷新模型」并选择具体模型。")
            return

        try:
            api_key, api_url = resolve_chat_endpoint(
                self.current_provider(),
                ollama_ui_base=self._ollama_ui_base_for_api(),
                lmstudio_ui_base=self._lmstudio_ui_base_for_api(),
            )
        except ValueError as exc:
            QMessageBox.critical(self, "API Key", str(exc))
            return

        model_mode = self.current_model_mode()
        use_tools = self.tools_checkbox.isChecked()
        stream = self.stream_checkbox.isChecked()
        prov = self.current_provider()

        if pending_images and not provider_supports_vision(prov):
            QMessageBox.warning(self, "图片不支持", vision_unsupported_hint(prov))
            return

        self.refresh_skills_list()

        if preview_enabled:
            api_messages = self._build_next_request_messages(user_input, pending_images)
            self.messages = [dict(m) for m in api_messages]
        else:
            full_system = self.compose_full_system_prompt(user_input)
            try:
                parsed = self._parse_preview_messages()
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid preview JSON", str(exc))
                return
            self.messages = list(parsed)
            if self.messages and self.messages[0].get("role") == "system":
                self.messages[0]["content"] = full_system
            else:
                self.messages.insert(0, {"role": "system", "content": full_system})
            api_messages = list(self.messages)

        vision_err = validate_messages_for_provider(api_messages, prov)
        if vision_err:
            QMessageBox.warning(self, "图片不支持", vision_err)
            return

        if not api_messages or api_messages[-1].get("role") != "user":
            QMessageBox.warning(self, "Invalid conversation", "Last message to send must be a user message.")
            return

        last_content = api_messages[-1].get("content")
        if isinstance(last_content, str) and not last_content.strip() and not pending_images:
            QMessageBox.warning(self, "Input required", "请输入文字或附加图片。")
            return

        self.awaiting_response = True
        self.pending_stream_text = ""
        self.render_chat()

        self.input_box.clear()
        self._clear_pending_images()
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
            api_url=api_url,
            provider=self.current_provider(),
            temperature=self._ui_temperature(),
            timeout=self._ui_request_timeout(),
            tool_max_rounds=int(self.tool_max_rounds_spin.value()),
        )
        self.worker.chunk.connect(self.on_chunk)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.interrupted.connect(self.on_interrupted)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_chunk(self, text: str):
        self.pending_stream_text += text
        self._schedule_render_chat()

    def on_interrupted(self):
        """Stop：保留已流式产出或本轮已返回的正文，写入 messages，供后续轮次使用。"""
        self.awaiting_response = False
        partial = (self.pending_stream_text or "").strip()
        if partial:
            content = self._finalize_assistant_answer(partial)
            self.messages.append({"role": "assistant", "content": content + "\n\n[已中断]"})
        else:
            self.messages.append({"role": "assistant", "content": "[已中断]"})
        self.pending_stream_text = ""
        self._clear_todo_execute_submit_flag()
        self.render_chat()
        self.update_next_context_preview()

    def on_done(self, full_answer: str):
        answer, api_messages = self._parse_harness_done_payload(full_answer)
        self.awaiting_response = False
        self.pending_stream_text = ""
        self._clear_todo_execute_submit_flag()
        if api_messages is not None:
            self.messages = self._apply_todo_finalize_to_messages(api_messages)
        else:
            content = self._finalize_assistant_answer(answer)
            self.messages.append({"role": "assistant", "content": content})
        self.render_chat()
        self.update_next_context_preview()

    def on_failed(self, message: str):
        self.messages.append({"role": "assistant", "content": f"[Error] {message}"})
        self.awaiting_response = False
        self.pending_stream_text = ""
        self._clear_todo_execute_submit_flag()
        self.render_chat()
        self.update_next_context_preview()

    def on_worker_finished(self):
        self.ask_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._clear_todo_execute_submit_flag()

    def update_next_context_preview(self):
        if not self.preview_checkbox.isChecked():
            self.preview_context_box.setReadOnly(False)
            return

        self.preview_context_box.setReadOnly(True)

        payload = self._filtered_preview_payload()
        self.preview_context_box.setPlainText(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def render_chat(self):
        sections = []
        for msg in self.messages:
            role = msg.get("role", "")
            if role in ("system", "tool"):
                continue
            if role == "assistant" and msg.get("tool_calls"):
                c = msg.get("content")
                if not (isinstance(c, str) and c.strip()):
                    continue
            sections.append(self._bubble_html(role, msg.get("content")))

        if self.awaiting_response:
            typing_text = self.pending_stream_text if self.pending_stream_text else "..."
            if self._todo_mode_enabled() and typing_text:
                typing_text, _ = extract_todos_from_reply(typing_text)
            sections.append(self._bubble_html("assistant", typing_text, pending=True))

        html_text = "".join([
            "<html><head><style>",
            "body{font-family:Segoe UI,Microsoft YaHei,sans-serif;padding:12px;background:#f5f6f8;font-size:14px;line-height:1.6;}",
            # Markdown content styling
            ".md-content h1{font-size:1.4em;margin:0.5em 0 0.3em;border-bottom:1px solid #e0e0e0;padding-bottom:4px;}",
            ".md-content h2{font-size:1.2em;margin:0.4em 0 0.25em;color:#1f2937;}",
            ".md-content h3{font-size:1.1em;margin:0.3em 0 0.2em;}",
            ".md-content h4{font-size:1.0em;margin:0.25em 0 0.15em;}",
            ".md-content p{margin:0.4em 0;}",
            # Code blocks
            ".md-content pre{background:#1e1e2e;color:#cdd6f4;padding:10px 14px;border-radius:8px;overflow-x:auto;font-family:Consolas,'Courier New',monospace;font-size:13px;margin:0.5em 0;border:1px solid #313244;white-space:pre-wrap;word-wrap:break-word;}",
            ".md-content pre code{background:transparent;padding:0;border-radius:0;color:inherit;font-size:inherit;}",
            ".md-content code{background:#e8e8ee;padding:1px 5px;border-radius:4px;font-family:Consolas,'Courier New',monospace;font-size:13px;color:#c7254e;border:1px solid #ddd;}",
            # Blockquotes
            ".md-content blockquote{border-left:4px solid #a0aec0;padding:4px 0 4px 14px;margin:0.5em 0;color:#4a5568;background:#f0f4f8;border-radius:0 6px 6px 0;}",
            # Lists
            ".md-content ul,.md-content ol{margin:0.3em 0;padding-left:1.6em;}",
            ".md-content li{margin:0.15em 0;}",
            # Horizontal rule
            ".md-content hr{border:none;border-top:2px solid #e0e0e0;margin:0.8em 0;}",
            # Links
            ".md-content a{color:#2563eb;text-decoration:none;}",
            ".md-content a:hover{text-decoration:underline;}",
            # Images (chat bubble)
            ".chat-img{display:block;margin:4px 0 0 0;padding:0;border:0;}",
            ".md-content img{max-width:100%;height:auto;border-radius:6px;margin:0.5em 0;}",
            # Tables
            ".md-content table{border-collapse:collapse;margin:0.5em 0;width:100%;}",
            ".md-content th,.md-content td{border:1px solid #d0d0d0;padding:6px 10px;text-align:left;}",
            ".md-content th{background:#eef2f7;font-weight:600;}",
            ".md-content tr:nth-child(even){background:#f8fafc;}",
            "</style></head><body>",
            "".join(sections) if sections else "<p style='color:#7a7a7a;'>No messages yet.</p>",
            "</body></html>",
        ])
        self.chat_output.setHtml(html_text)
        self.chat_output.moveCursor(QTextCursor.End)

    def _chat_thumbnail_for_display(self, data_url: str) -> Optional[Tuple[str, int, int]]:
        key = _data_url_cache_key(data_url)
        cached = self._chat_thumb_cache.get(key)
        if cached is not None:
            return cached
        made = data_url_to_chat_thumbnail(data_url)
        if made is not None:
            self._chat_thumb_cache[key] = made
        return made

    def _bubble_html(self, role: str, content: Any, pending: bool = False) -> str:
        if role == "user":
            label = "You"
            row_align = "right"
            bubble_bg = "#d8f0ff"
            text_color = "#1f2937"
        else:
            label = self._assistant_display_name()
            row_align = "left"
            bubble_bg = "#ffffff"
            text_color = "#111827"

        img_urls = image_data_urls_from_content(content) if role == "user" else []
        display_text = content_to_display_text(content) if not isinstance(content, str) else content
        # 已有缩略图时不重复显示「[N 张图片]」占位文字
        if img_urls and display_text:
            display_text = re.sub(r"\n?\[\d+ 张图片\]\s*$", "", display_text).strip()

        # Pending (streaming) text uses plain escaping to avoid partial-Markdown glitches
        if pending:
            body = html.escape(display_text).replace("\n", "<br>") if display_text else ""
        elif self.render_md_checkbox.isChecked() and isinstance(content, str):
            body = markdown_to_html(display_text) if display_text else ""
        else:
            body = html.escape(display_text).replace("\n", "<br>") if display_text else ""

        if img_urls:
            img_parts: List[str] = []
            for u in img_urls:
                thumb = self._chat_thumbnail_for_display(u)
                if thumb is None:
                    img_parts.append(
                        "<div style='color:#6b7280;font-size:12px;margin-top:4px;'>[图片]</div>"
                    )
                    continue
                turl, tw, th = thumb
                img_parts.append(
                    f"<img class='chat-img' src='{html.escape(turl, quote=True)}' "
                    f"width='{tw}' height='{th}' "
                    f"style='width:{tw}px;height:{th}px;' alt='image'/>"
                )
            if img_parts:
                body = (body + "".join(img_parts)) if body else "".join(img_parts)

        if not body:
            body = "<span style='color:#9ca3af;'>…</span>"

        pending_badge = " <span style='color:#6b7280;'>(typing)</span>" if pending else ""
        return (
            f"<div style='text-align:{row_align}; margin:8px 0;'>"
            f"<div style='display:inline-block; max-width:78%; text-align:left;'>"
            f"<div style='font-size:12px; color:#6b7280; margin-bottom:4px;'>{label}{pending_badge}</div>"
            f"<div style='background:{bubble_bg}; color:{text_color}; border:1px solid #e5e7eb; border-radius:12px; padding:10px 12px; line-height:1.5;'>{body}</div>"
            "</div>"
            "</div>"
        )

    def _conversation_payload(self) -> dict:
        return {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "system_prompt": self.current_system_prompt,
            "messages": self.messages,
            "todos": todos_to_json_list(self.todos),
            "todo_mode": self._todo_mode_enabled(),
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

    def _normalize_message_content(self, content: Any, index: int) -> Any:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for pi, part in enumerate(content, start=1):
                if not isinstance(part, dict):
                    raise ValueError(f"Message #{index} content part #{pi} must be an object.")
                ptype = part.get("type")
                if ptype == "text":
                    if not isinstance(part.get("text"), str):
                        raise ValueError(f"Message #{index} text part #{pi} needs string 'text'.")
                elif ptype == "image_url":
                    iu = part.get("image_url")
                    url = iu.get("url") if isinstance(iu, dict) else iu
                    if not isinstance(url, str) or not url.strip():
                        raise ValueError(f"Message #{index} image_url part #{pi} needs url.")
                else:
                    raise ValueError(
                        f"Message #{index} content part #{pi} has unsupported type: {ptype}"
                    )
            return content
        raise ValueError(f"Message #{index} content must be a string or array.")

    def _parse_preview_messages(self):
        raw = self.preview_context_box.toPlainText().strip()
        if not raw:
            raise ValueError("Preview is empty. Provide a JSON array of messages.")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON parse failed: {exc}")

        if isinstance(data, dict):
            inner = data.get("messages")
            if inner is None:
                raise ValueError('JSON 对象需包含 "messages" 数组（可与 Preview 开启时格式一致）。')
            data = inner

        if not isinstance(data, list):
            raise ValueError("messages 必须是 JSON 数组。")

        cleaned = []
        for index, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Message #{index} must be an object.")
            role = item.get("role")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Message #{index} has invalid role: {role}")
            content = self._normalize_message_content(item.get("content"), index)
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
            loaded_todos = data.get("todos")
            loaded_todo_mode = data.get("todo_mode")
        elif isinstance(data, list):
            maybe_messages = data
            loaded_prompt = None
            loaded_todos = None
            loaded_todo_mode = None
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
                if role not in {"system", "user", "assistant"}:
                    raise ValueError(f"Message #{index} has invalid role: {role}")
                content = self._normalize_message_content(item.get("content"), index)
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
                    sp = msg.get("content", self.current_system_prompt)
                    self.current_system_prompt = (
                        sp if isinstance(sp, str) else content_to_display_text(sp)
                    )
                    break

        self.current_prompt_display.setPlainText(self.current_system_prompt)
        if loaded_todos is not None:
            self.todos = todos_from_json_list(loaded_todos)
        else:
            self.todos = []
        if loaded_todo_mode is not None:
            self.todo_mode_checkbox.blockSignals(True)
            self.todo_mode_checkbox.setChecked(bool(loaded_todo_mode))
            self.todo_mode_checkbox.blockSignals(False)
        if 0 <= self._active_session_index < len(self.sessions):
            self.sessions[self._active_session_index].todos = todos_to_json_list(self.todos)
            self.sessions[self._active_session_index].todo_mode = self._todo_mode_enabled()
        self._refresh_todo_list_ui()
        self.pending_stream_text = ""
        self.awaiting_response = False
        self.render_chat()
        self.update_next_context_preview()

    def _messages_source_like_save(self) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """与 Save 相同的消息来源。"""
        if self.preview_checkbox.isChecked():
            if not self.messages:
                return None, "empty"
            return list(self.messages), None
        try:
            return self._parse_preview_messages(), None
        except ValueError as exc:
            return None, str(exc)

    def _unique_path_in_logs(self, stem: str, stamp: str) -> Path:
        base = f"{stem}_{stamp}.json"
        path = self.logs_dir / base
        counter = 1
        while path.exists():
            path = self.logs_dir / f"{stem}_{stamp}_{counter:02d}.json"
            counter += 1
        return path

    def on_smart_save(self) -> None:
        if self.smart_save_worker and self.smart_save_worker.isRunning():
            return

        msgs, err = self._messages_source_like_save()
        if err == "empty":
            QMessageBox.information(self, "Nothing to save", "No conversation to save.")
            return
        if err:
            QMessageBox.warning(self, "Invalid preview JSON", err)
            return
        if not msgs:
            QMessageBox.information(self, "Nothing to save", "No conversation to save.")
            return

        transcript = _flatten_messages_for_transcript(msgs)
        transcript_for_api = transcript[:SMART_SAVE_TRANSCRIPT_MAX_CHARS]

        if not self._model_choice_ready():
            QMessageBox.warning(self, "模型", "请先点击「刷新模型」并选择具体模型。")
            return

        try:
            api_key, api_url = resolve_chat_endpoint(
                self.current_provider(),
                ollama_ui_base=self._ollama_ui_base_for_api(),
                lmstudio_ui_base=self._lmstudio_ui_base_for_api(),
            )
        except ValueError as exc:
            QMessageBox.critical(self, "API Key", str(exc))
            return

        self._smart_save_pending = {"messages": msgs}
        proxy = self.model_proxy_addr.text().strip() if self.model_proxy_checkbox.isChecked() else None
        self.smart_save_button.setEnabled(False)
        self.smart_save_worker = SmartFilenameWorker(
            api_key=api_key,
            model_mode=self.current_model_mode(),
            transcript=transcript_for_api,
            proxy_url=proxy,
            api_url=api_url,
            provider=self.current_provider(),
        )
        self.smart_save_worker.done.connect(self._on_smart_save_filename_ok)
        self.smart_save_worker.failed.connect(self._on_smart_save_fail)
        self.smart_save_worker.finished.connect(self._on_smart_save_worker_finished)
        self.smart_save_worker.start()
        self.statusBar().showMessage("Smart Save：正在生成文件名…")

    def _on_smart_save_filename_ok(self, raw_reply: str) -> None:
        pending = self._smart_save_pending
        if not pending:
            return

        stem = _filename_stem_from_llm_reply(raw_reply) or "conversation"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._unique_path_in_logs(stem, stamp)

        payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "title": stem,
            "system_prompt": self.current_system_prompt,
            "messages": pending["messages"],
        }

        try:
            self._write_payload(path, payload)
            QMessageBox.information(self, "Saved", f"已保存:\n{path.resolve()}")
            self.statusBar().showMessage(f"Smart Save → {path.name}", 5000)
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))

        self._smart_save_pending = None

    def _on_smart_save_fail(self, message: str) -> None:
        self._smart_save_pending = None
        QMessageBox.warning(self, "Smart Save 失败", message)
        self.statusBar().showMessage("Smart Save 失败", 5000)

    def _on_smart_save_worker_finished(self) -> None:
        self.smart_save_button.setEnabled(True)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
