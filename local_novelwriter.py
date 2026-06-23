"""本地小说编辑器 — 独立 PyQt5 客户端。"""

from __future__ import annotations

import html
import json
import sys
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PyQt5.QtCore import Qt, QSettings, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QListWidget,
    QListWidgetItem,
)

from deepseek_api import (
    API_URL,
    PROVIDER_DEEPSEEK,
    PROVIDER_KIMI,
    PROVIDER_OLLAMA,
    STREAM_TIMEOUT_DEFAULT,
    STREAM_TIMEOUT_UNLIMITED,
    StreamInterrupted,
    check_available,
    effective_stream_timeout,
    effective_temperature_for_resolved_model,
    explain_http_error,
    resolve_chat_endpoint,
    resolve_model,
)
from deepseek_harness import HarnessConfig, run_harness
from markdown_renderer import markdown_to_html
from novelwriter_assets import (
    ASSET_CHARACTER,
    ASSET_EVENT,
    ASSET_STORY_MODE,
    ASSET_TYPES,
    ASSET_WORLD,
    CATEGORY_LABELS,
    AssetStore,
    NovelAsset,
    default_assets_root,
)
from novelwriter_compose import build_chat_messages, compose_system_prompt, preview_payload
from novelwriter_dialogs import AssetEditDialog, PromptBlockEditDialog
from novelwriter_prompts import PromptBlock, PromptStore


class NovelAskWorker(QThread):
    chunk = pyqtSignal(str)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)
    interrupted = pyqtSignal()

    def __init__(
        self,
        *,
        api_key: str,
        messages: List[Dict[str, Any]],
        model_mode: str,
        provider: str,
        api_url: str,
        stream: bool,
        temperature: float,
        timeout: int,
        proxy_url: Optional[str],
    ):
        super().__init__()
        self.api_key = api_key
        self.messages = messages
        self.model_mode = model_mode
        self.provider = provider
        self.api_url = api_url
        self.stream = stream
        self.temperature = temperature
        self.timeout = timeout
        self.proxy_url = proxy_url
        self._should_stop = False

    def stop(self) -> None:
        self._should_stop = True

    def should_stop(self) -> bool:
        return self._should_stop

    def run(self) -> None:
        try:
            cfg = HarnessConfig(
                workspace=Path.cwd(),
                use_tools=False,
                stream=self.stream,
                proxy_url=self.proxy_url,
                api_url=self.api_url,
                provider=self.provider,
            )
            answer = run_harness(
                api_key=self.api_key,
                messages=self.messages,
                model_mode=self.model_mode,
                config=cfg,
                temperature=self.temperature,
                timeout=self.timeout,
                should_stop=self.should_stop,
                on_stream_token=self.chunk.emit if self.stream else None,
            )
            self.done.emit(answer)
        except StreamInterrupted:
            self.interrupted.emit()
        except urllib.error.HTTPError as exc:
            self.failed.emit(explain_http_error(exc))
        except urllib.error.URLError as exc:
            self.failed.emit(f"Network error: {exc.reason}")
        except Exception as exc:
            self.failed.emit(f"Unexpected error: {exc}")


class ModelsRefreshWorker(QThread):
    done = pyqtSignal(list, list)
    failed = pyqtSignal(str)

    def __init__(self, *, proxy_url: Optional[str], ollama_ui_base: Optional[str]):
        super().__init__()
        self.proxy_url = proxy_url
        self.ollama_ui_base = ollama_ui_base

    def run(self) -> None:
        try:
            entries, notes = check_available(
                proxy_url=self.proxy_url,
                timeout=60,
                ollama_ui_base=self.ollama_ui_base,
            )
            self.done.emit(entries, notes)
        except Exception as exc:
            self.failed.emit(str(exc))


class NovelWriterWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Local Novel Writer")
        self.resize(1180, 820)

        self.settings = QSettings("LocalNovelWriter", "PyQtClient")
        assets_path = self.settings.value("assets_root", "")
        self.assets_root = Path(str(assets_path)) if assets_path else default_assets_root()
        self.asset_store = AssetStore(self.assets_root)
        self.prompt_store = PromptStore(self.assets_root)

        self.messages: List[Dict[str, str]] = []
        self.pending_stream_text = ""
        self._pending_user_text = ""
        self.awaiting_response = False
        self._render_scheduled = False
        self.logs_dir = self.assets_root / "sessions"
        self.worker: Optional[NovelAskWorker] = None
        self._models_worker: Optional[ModelsRefreshWorker] = None
        self._asset_lists: Dict[str, QListWidget] = {}

        self._build_ui()
        self._reload_stores()
        self.render_chat()
        self.update_preview()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        top = QHBoxLayout()
        top.addWidget(QLabel("资产目录"))
        self.assets_dir_edit = QLineEdit(str(self.assets_root))
        browse_btn = QPushButton("浏览…")
        browse_btn.clicked.connect(self.on_browse_assets)
        reload_btn = QPushButton("重新加载")
        reload_btn.clicked.connect(self._reload_stores)
        import_btn = QPushButton("从 prompts_system 导入角色")
        import_btn.setToolTip("读取项目内 prompts_system.json 的 card* 角色卡")
        import_btn.clicked.connect(self.on_import_cards)
        top.addWidget(self.assets_dir_edit, 1)
        top.addWidget(browse_btn)
        top.addWidget(reload_btn)
        top.addWidget(import_btn)
        layout.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        # ── 左：资产 ──
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("资产库（勾选注入上下文）"))
        self.asset_tabs = QTabWidget()
        for asset_type in ASSET_TYPES:
            lw = QListWidget()
            lw.setSelectionMode(lw.ExtendedSelection)
            lw.itemChanged.connect(self.update_preview)
            self._asset_lists[asset_type] = lw
            self.asset_tabs.addTab(lw, CATEGORY_LABELS.get(asset_type, asset_type))
        left_layout.addWidget(self.asset_tabs, 1)

        asset_btns = QHBoxLayout()
        add_asset_btn = QPushButton("添加")
        edit_asset_btn = QPushButton("编辑")
        del_asset_btn = QPushButton("删除")
        add_asset_btn.clicked.connect(self.on_add_asset)
        edit_asset_btn.clicked.connect(self.on_edit_asset)
        del_asset_btn.clicked.connect(self.on_delete_asset)
        asset_btns.addWidget(add_asset_btn)
        asset_btns.addWidget(edit_asset_btn)
        asset_btns.addWidget(del_asset_btn)
        left_layout.addLayout(asset_btns)
        splitter.addWidget(left)

        # ── 中：对话 ──
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("创作对话"))
        self.render_md_checkbox = QCheckBox("Markdown")
        self.render_md_checkbox.setChecked(True)
        self.render_md_checkbox.toggled.connect(self.render_chat)
        hdr.addWidget(self.render_md_checkbox)
        hdr.addStretch(1)
        center_layout.addLayout(hdr)

        self.chat_output = QTextBrowser()
        self.chat_output.setOpenExternalLinks(False)
        center_layout.addWidget(self.chat_output, 2)

        center_layout.addWidget(QLabel("请求预览"))
        self.preview_box = QPlainTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setMaximumHeight(160)
        center_layout.addWidget(self.preview_box)

        center_layout.addWidget(QLabel("输入"))
        self.input_box = QPlainTextEdit()
        self.input_box.setFixedHeight(88)
        self.input_box.textChanged.connect(self.update_preview)
        center_layout.addWidget(self.input_box)
        splitter.addWidget(center)

        # ── 右：Prompt 块 ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("Prompt 块（可随时增删，勾选生效）"))
        self.prompt_list = QListWidget()
        self.prompt_list.itemChanged.connect(self.on_prompt_item_changed)
        right_layout.addWidget(self.prompt_list, 1)

        prow = QHBoxLayout()
        add_p = QPushButton("添加")
        edit_p = QPushButton("编辑")
        del_p = QPushButton("删除")
        up_p = QPushButton("↑")
        down_p = QPushButton("↓")
        add_p.clicked.connect(self.on_add_prompt)
        edit_p.clicked.connect(self.on_edit_prompt)
        del_p.clicked.connect(self.on_delete_prompt)
        up_p.clicked.connect(lambda: self.on_move_prompt(-1))
        down_p.clicked.connect(lambda: self.on_move_prompt(1))
        for b in (add_p, edit_p, del_p, up_p, down_p):
            prow.addWidget(b)
        right_layout.addLayout(prow)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setStretchFactor(2, 3)

        # ── 底：模型与发送 ──
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        refresh_btn = QPushButton("刷新模型")
        refresh_btn.clicked.connect(self.on_refresh_models)
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(refresh_btn)

        self.stream_checkbox = QCheckBox("Stream")
        self.stream_checkbox.setChecked(True)
        self.proxy_checkbox = QCheckBox("代理")
        self.proxy_edit = QLineEdit("http://127.0.0.1:7890")
        model_row.addWidget(self.stream_checkbox)
        model_row.addWidget(self.proxy_checkbox)
        model_row.addWidget(self.proxy_edit)
        layout.addLayout(model_row)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Temperature"))
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(0.85)
        param_row.addWidget(self.temperature_spin)
        param_row.addWidget(QLabel("Timeout"))
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(STREAM_TIMEOUT_UNLIMITED, 7200)
        self.timeout_spin.setSpecialValueText("∞")
        self.timeout_spin.setValue(STREAM_TIMEOUT_UNLIMITED)
        param_row.addWidget(self.timeout_spin)
        param_row.addWidget(QLabel("Ollama"))
        self.ollama_edit = QLineEdit()
        self.ollama_edit.setPlaceholderText("http://127.0.0.1:11434")
        param_row.addWidget(self.ollama_edit, 1)
        layout.addLayout(param_row)

        btn_row = QHBoxLayout()
        self.send_btn = QPushButton("Send")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        clear_btn = QPushButton("清空对话")
        save_btn = QPushButton("保存对话")
        load_btn = QPushButton("加载对话")
        self.send_btn.clicked.connect(self.on_send)
        self.stop_btn.clicked.connect(self.on_stop)
        clear_btn.clicked.connect(self.on_clear_chat)
        save_btn.clicked.connect(self.on_save_chat)
        load_btn.clicked.connect(self.on_load_chat)
        btn_row.addWidget(self.send_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(load_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.model_combo.currentIndexChanged.connect(self.update_preview)
        self.temperature_spin.valueChanged.connect(self.update_preview)
        self.timeout_spin.valueChanged.connect(self.update_preview)

        self._load_cached_models()

    def _reload_stores(self) -> None:
        path = Path(self.assets_dir_edit.text().strip() or str(default_assets_root()))
        self.assets_root = path
        self.logs_dir = path / "sessions"
        self.settings.setValue("assets_root", str(path))
        self.asset_store = AssetStore(path)
        self.prompt_store = PromptStore(path)
        self.asset_store.load_all()
        prompts_src = path.parent / "prompts_system.json"
        if not prompts_src.is_file():
            prompts_src = Path("prompts_system.json")
        if not self.asset_store.list_assets(ASSET_CHARACTER):
            n = self.asset_store.import_from_prompts_system(prompts_src)
            if n:
                self.statusBar().showMessage(f"已导入 {n} 个角色卡", 5000)
        self.prompt_store.load()
        self._refresh_asset_lists()
        self._refresh_prompt_list()
        self.update_preview()

    def _refresh_asset_lists(self) -> None:
        for asset_type, lw in self._asset_lists.items():
            lw.blockSignals(True)
            lw.clear()
            for asset in self.asset_store.list_assets(asset_type):
                item = QListWidgetItem(asset.name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                item.setData(Qt.UserRole, (asset_type, asset.id))
                tip = asset.summary or asset.content[:120]
                item.setToolTip(tip)
                lw.addItem(item)
            lw.blockSignals(False)

    def _refresh_prompt_list(self) -> None:
        self.prompt_list.blockSignals(True)
        self.prompt_list.clear()
        for block in self.prompt_store.list_blocks():
            item = QListWidgetItem(block.name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if block.enabled else Qt.Unchecked)
            item.setData(Qt.UserRole, block.id)
            item.setToolTip(block.content[:200])
            self.prompt_list.addItem(item)
        self.prompt_list.blockSignals(False)

    def _current_asset_tab_type(self) -> str:
        idx = self.asset_tabs.currentIndex()
        if 0 <= idx < len(ASSET_TYPES):
            return ASSET_TYPES[idx]
        return ASSET_CHARACTER

    def _current_asset_list(self) -> QListWidget:
        return self._asset_lists[self._current_asset_tab_type()]

    def _selected_assets(self) -> Tuple[List[NovelAsset], List[NovelAsset], List[NovelAsset], List[NovelAsset]]:
        chars: List[NovelAsset] = []
        worlds: List[NovelAsset] = []
        events: List[NovelAsset] = []
        modes: List[NovelAsset] = []
        mapping = {
            ASSET_CHARACTER: chars,
            ASSET_WORLD: worlds,
            ASSET_EVENT: events,
            ASSET_STORY_MODE: modes,
        }
        for asset_type, lw in self._asset_lists.items():
            bucket = mapping[asset_type]
            for i in range(lw.count()):
                it = lw.item(i)
                if it.checkState() != Qt.Checked:
                    continue
                data = it.data(Qt.UserRole)
                if not isinstance(data, tuple) or len(data) != 2:
                    continue
                a = self.asset_store.get(data[0], data[1])
                if a:
                    bucket.append(a)
        return chars, worlds, events, modes

    def _enabled_prompt_blocks(self) -> List[PromptBlock]:
        enabled_ids: Set[str] = set()
        for i in range(self.prompt_list.count()):
            it = self.prompt_list.item(i)
            if it.checkState() == Qt.Checked:
                bid = it.data(Qt.UserRole)
                if bid:
                    enabled_ids.add(str(bid))
        blocks = []
        for b in self.prompt_store.list_blocks():
            if b.id in enabled_ids:
                blocks.append(b)
        return blocks

    def compose_system(self) -> str:
        c, w, e, m = self._selected_assets()
        return compose_system_prompt(
            prompt_blocks=self._enabled_prompt_blocks(),
            characters=c,
            worlds=w,
            events=e,
            story_modes=m,
        )

    def update_preview(self) -> None:
        user = self.input_box.toPlainText().strip()
        system = self.compose_system()
        msgs = build_chat_messages(system_prompt=system, history=self.messages, user_input=user)
        prov = self.current_provider()
        mode = self.current_model_mode()
        resolved = resolve_model(mode, prov) if mode else ""
        temp = self.temperature_spin.value()
        eff_temp = (
            effective_temperature_for_resolved_model(resolved, temp, provider=prov)
            if resolved
            else temp
        )
        payload = preview_payload(
            system_prompt=system,
            messages=msgs,
            meta={
                "provider": prov,
                "model": resolved,
                "temperature": eff_temp,
                "temperature_requested": temp,
                "timeout": self._timeout_for_preview(
                    effective_stream_timeout(self.timeout_spin.value(), provider=prov)
                ),
                "timeout_requested": self._timeout_for_preview(
                    None if self.timeout_spin.value() < 0 else self.timeout_spin.value()
                ),
                "selected_assets": {
                    t: [it.text() for it in self._checked_names(t)] for t in ASSET_TYPES
                },
            },
        )
        self.preview_box.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))

    @staticmethod
    def _timeout_for_preview(value: Optional[int]) -> Any:
        return "unlimited" if value is None else value

    def _checked_names(self, asset_type: str) -> List[QListWidgetItem]:
        lw = self._asset_lists[asset_type]
        out = []
        for i in range(lw.count()):
            it = lw.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it)
        return out

    def current_provider(self) -> str:
        d = self.model_combo.currentData()
        if isinstance(d, (tuple, list)) and len(d) >= 1:
            return str(d[0])
        return PROVIDER_DEEPSEEK

    def current_model_mode(self) -> str:
        d = self.model_combo.currentData()
        if isinstance(d, (tuple, list)) and len(d) >= 2:
            return str(d[1])
        return ""

    def _model_ready(self) -> bool:
        d = self.model_combo.currentData()
        return isinstance(d, (tuple, list)) and len(d) >= 2 and bool(str(d[1]).strip())

    def _proxy_url(self) -> Optional[str]:
        if self.proxy_checkbox.isChecked():
            t = self.proxy_edit.text().strip()
            return t or None
        return None

    def _load_cached_models(self) -> None:
        raw = self.settings.value("novel_models_cache", "")
        if not raw:
            self.model_combo.addItem("— 点击刷新模型 —", ("", ""))
            return
        try:
            data = json.loads(str(raw))
        except Exception:
            return
        self._fill_models(data)

    def _fill_models(self, entries: list) -> None:
        self.model_combo.clear()
        for item in entries or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            prov, mid = str(item[0]), str(item[1])
            tag = {"deepseek": "DS", "kimi": "Kimi", "ollama": "Ollama"}.get(prov, prov[:6])
            self.model_combo.addItem(f"[{tag}] {mid}", (prov, mid))

    def on_refresh_models(self) -> None:
        if self._models_worker and self._models_worker.isRunning():
            return
        ollama = self.ollama_edit.text().strip() or None
        self._models_worker = ModelsRefreshWorker(proxy_url=self._proxy_url(), ollama_ui_base=ollama)
        self._models_worker.done.connect(self._on_models_done)
        self._models_worker.failed.connect(lambda m: QMessageBox.warning(self, "模型", m))
        self._models_worker.start()

    def _on_models_done(self, entries: list, notes: list) -> None:
        pairs = [[str(a), str(b)] for a, b in entries if isinstance(a, str)]
        self.settings.setValue("novel_models_cache", json.dumps(pairs))
        self._fill_models(pairs)
        if notes:
            self.statusBar().showMessage("; ".join(str(n) for n in notes[:3]), 8000)

    def on_browse_assets(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择资产目录", str(self.assets_root))
        if d:
            self.assets_dir_edit.setText(d)
            self._reload_stores()

    def on_import_cards(self) -> None:
        prompts_src = self.assets_root.parent / "prompts_system.json"
        if not prompts_src.is_file():
            prompts_src = Path("prompts_system.json")
        n = self.asset_store.import_from_prompts_system(prompts_src)
        self._refresh_asset_lists()
        QMessageBox.information(self, "导入", f"新导入 {n} 个角色卡。")

    def on_add_asset(self) -> None:
        dlg = AssetEditDialog(self, asset_type=self._current_asset_tab_type())
        if dlg.exec_() != QDialog.Accepted:
            return
        asset = dlg.result_asset()
        if not asset:
            return
        self.asset_store.upsert(asset)
        self._refresh_asset_lists()
        self.update_preview()

    def on_edit_asset(self) -> None:
        it = self._current_asset_list().currentItem()
        if not it:
            return
        data = it.data(Qt.UserRole)
        if not isinstance(data, tuple):
            return
        asset = self.asset_store.get(data[0], data[1])
        if not asset:
            return
        dlg = AssetEditDialog(self, asset=asset)
        if dlg.exec_() != QDialog.Accepted:
            return
        updated = dlg.result_asset()
        if not updated:
            return
        updated.id = asset.id
        self.asset_store.upsert(updated)
        self._refresh_asset_lists()
        self.update_preview()

    def on_delete_asset(self) -> None:
        it = self._current_asset_list().currentItem()
        if not it:
            return
        data = it.data(Qt.UserRole)
        if not isinstance(data, tuple):
            return
        if QMessageBox.question(self, "删除", f"删除「{it.text()}」？") != QMessageBox.Yes:
            return
        self.asset_store.delete(data[0], data[1])
        self._refresh_asset_lists()
        self.update_preview()

    def on_prompt_item_changed(self, item: QListWidgetItem) -> None:
        bid = item.data(Qt.UserRole)
        if bid:
            self.prompt_store.toggle(str(bid), item.checkState() == Qt.Checked)
        self.update_preview()

    def on_add_prompt(self) -> None:
        dlg = PromptBlockEditDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        block = dlg.result_block()
        if block:
            self.prompt_store.upsert(block)
            self._refresh_prompt_list()
            self.update_preview()

    def on_edit_prompt(self) -> None:
        it = self.prompt_list.currentItem()
        if not it:
            return
        block = self.prompt_store.get(str(it.data(Qt.UserRole)))
        if not block:
            return
        dlg = PromptBlockEditDialog(self, block=block)
        if dlg.exec_() != QDialog.Accepted:
            return
        updated = dlg.result_block()
        if not updated:
            return
        updated.id = block.id
        updated.enabled = block.enabled
        self.prompt_store.upsert(updated)
        self._refresh_prompt_list()
        self.update_preview()

    def on_delete_prompt(self) -> None:
        it = self.prompt_list.currentItem()
        if not it:
            return
        bid = str(it.data(Qt.UserRole))
        if QMessageBox.question(self, "删除", f"删除 Prompt「{it.text()}」？") != QMessageBox.Yes:
            return
        self.prompt_store.delete(bid)
        self._refresh_prompt_list()
        self.update_preview()

    def on_move_prompt(self, delta: int) -> None:
        row = self.prompt_list.currentRow()
        if row < 0:
            return
        new_row = row + delta
        if new_row < 0 or new_row >= self.prompt_list.count():
            return
        it = self.prompt_list.takeItem(row)
        self.prompt_list.insertItem(new_row, it)
        self.prompt_list.setCurrentRow(new_row)
        ids = [str(self.prompt_list.item(i).data(Qt.UserRole)) for i in range(self.prompt_list.count())]
        self.prompt_store.reorder(ids)
        self._refresh_prompt_list()
        self.prompt_list.setCurrentRow(new_row)

    def on_send(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        user = self.input_box.toPlainText().strip()
        if not user:
            QMessageBox.warning(self, "输入", "请输入内容。")
            return
        if not self._model_ready():
            QMessageBox.warning(self, "模型", "请先刷新并选择模型。")
            return
        try:
            api_key, api_url = resolve_chat_endpoint(
                self.current_provider(),
                ollama_ui_base=self.ollama_edit.text().strip() or None,
            )
        except ValueError as exc:
            QMessageBox.critical(self, "API Key", str(exc))
            return

        self._pending_user_text = user
        system = self.compose_system()
        api_messages = build_chat_messages(
            system_prompt=system,
            history=self.messages,
            user_input=user,
        )
        self.awaiting_response = True
        self.pending_stream_text = ""
        self.input_box.clear()
        self.update_preview()
        self.render_chat()
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.worker = NovelAskWorker(
            api_key=api_key,
            messages=api_messages,
            model_mode=self.current_model_mode(),
            provider=self.current_provider(),
            api_url=api_url,
            stream=self.stream_checkbox.isChecked(),
            temperature=self.temperature_spin.value(),
            timeout=self.timeout_spin.value(),
            proxy_url=self._proxy_url(),
        )
        self.worker.chunk.connect(self.on_chunk)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.interrupted.connect(self.on_interrupted)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def on_stop(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.stop()

    def on_chunk(self, text: str) -> None:
        self.pending_stream_text += text
        self._schedule_render()

    def on_done(self, answer: str) -> None:
        if self._pending_user_text:
            self.messages.append({"role": "user", "content": self._pending_user_text})
        self.messages.append({"role": "assistant", "content": answer})
        self._pending_user_text = ""
        self.awaiting_response = False
        self.pending_stream_text = ""
        self.render_chat()
        self.update_preview()

    def on_failed(self, msg: str) -> None:
        if self._pending_user_text:
            self.messages.append({"role": "user", "content": self._pending_user_text})
            self._pending_user_text = ""
        self.messages.append({"role": "assistant", "content": f"[Error] {msg}"})
        self.awaiting_response = False
        self.pending_stream_text = ""
        self.render_chat()
        self.update_preview()

    def on_interrupted(self) -> None:
        partial = (self.pending_stream_text or "").strip()
        if self._pending_user_text:
            self.messages.append({"role": "user", "content": self._pending_user_text})
        self.messages.append(
            {
                "role": "assistant",
                "content": (partial + "\n\n[已中断]") if partial else "[已中断]",
            }
        )
        self._pending_user_text = ""
        self.awaiting_response = False
        self.pending_stream_text = ""
        self.render_chat()
        self.update_preview()

    def _on_worker_finished(self) -> None:
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _schedule_render(self) -> None:
        if self._render_scheduled:
            return
        self._render_scheduled = True
        QTimer.singleShot(60, self._do_render)

    def _do_render(self) -> None:
        self._render_scheduled = False
        self.render_chat()

    def render_chat(self) -> None:
        sections = []
        for msg in self.messages:
            sections.append(self._bubble(msg.get("role", ""), msg.get("content", "")))
        if self.awaiting_response:
            text = self.pending_stream_text or "..."
            sections.append(self._bubble("assistant", text, pending=True))
        body = "".join(sections) if sections else "<p style='color:#888;'>尚无对话。</p>"
        html_text = (
            "<html><head><style>"
            "body{font-family:Segoe UI,Microsoft YaHei,sans-serif;padding:12px;"
            "background:#f8f6f2;font-size:14px;line-height:1.6;}"
            ".md-content pre{background:#1e1e2e;color:#cdd6f4;padding:8px;border-radius:6px;}"
            "</style></head><body>"
            f"{body}</body></html>"
        )
        self.chat_output.setHtml(html_text)
        self.chat_output.moveCursor(QTextCursor.End)

    def _bubble(self, role: str, content: str, pending: bool = False) -> str:
        if role == "user":
            label, align, bg = "You", "right", "#dbeafe"
        else:
            label, align, bg = "Assistant", "left", "#ffffff"
        if pending:
            body = html.escape(content).replace("\n", "<br>")
        elif self.render_md_checkbox.isChecked():
            body = markdown_to_html(content)
        else:
            body = html.escape(content).replace("\n", "<br>")
        badge = " <span style='color:#888;'>(typing)</span>" if pending else ""
        return (
            f"<div style='text-align:{align};margin:8px 0;'>"
            f"<div style='display:inline-block;max-width:82%;text-align:left;'>"
            f"<div style='font-size:12px;color:#666;margin-bottom:4px;'>{label}{badge}</div>"
            f"<div style='background:{bg};border:1px solid #e5e7eb;border-radius:10px;"
            f"padding:10px 12px;' class='md-content'>{body}</div></div></div>"
        )

    def on_clear_chat(self) -> None:
        self.messages = []
        self.pending_stream_text = ""
        self._pending_user_text = ""
        self.awaiting_response = False
        self.render_chat()
        self.update_preview()

    def on_save_chat(self) -> None:
        if not self.messages:
            QMessageBox.information(self, "保存", "没有可保存的对话。")
            return
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.logs_dir / f"session_{stamp}.json"
        payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "system_prompt": self.compose_system(),
            "messages": self.messages,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        QMessageBox.information(self, "保存", f"已保存到\n{path}")

    def on_load_chat(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self, "加载对话", str(self.logs_dir), "JSON (*.json)"
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.warning(self, "加载失败", str(exc))
            return
        msgs = data.get("messages") if isinstance(data, dict) else data
        if not isinstance(msgs, list):
            QMessageBox.warning(self, "加载失败", "无效的消息列表。")
            return
        cleaned = []
        for m in msgs:
            if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
                cleaned.append({"role": m["role"], "content": str(m.get("content") or "")})
        self.messages = cleaned
        self.render_chat()
        self.update_preview()


def main() -> int:
    app = QApplication(sys.argv)
    win = NovelWriterWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
