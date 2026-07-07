"""小说编辑器对话框。"""

from __future__ import annotations

from typing import Optional, Tuple

from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
)

from .assets import ASSET_TYPES, CATEGORY_LABELS, NovelAsset
from .prompts import PROMPT_ROLES, ROLE_LABELS, PromptBlock


class AssetEditDialog(QDialog):
    def __init__(self, parent=None, *, asset: Optional[NovelAsset] = None, asset_type: str = "character"):
        super().__init__(parent)
        self.setWindowTitle("编辑资产" if asset else "新建资产")
        self.resize(560, 480)
        self._asset_type = asset.asset_type if asset else asset_type

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.type_combo = QComboBox()
        for t in ASSET_TYPES:
            self.type_combo.addItem(CATEGORY_LABELS.get(t, t), t)
        idx = self.type_combo.findData(self._asset_type)
        if idx >= 0:
            self.type_combo.setCurrentIndex(idx)
        if asset:
            self.type_combo.setEnabled(False)

        self.name_edit = QLineEdit(asset.name if asset else "")
        self.summary_edit = QLineEdit(asset.summary if asset else "")
        self.tags_edit = QLineEdit(", ".join(asset.tags) if asset else "")
        self.content_edit = QPlainTextEdit(asset.content if asset else "")
        self.content_edit.setPlaceholderText("正文：角色设定、世界观、事件梗概、故事模式说明等")

        form.addRow("类型", self.type_combo)
        form.addRow("名称", self.name_edit)
        form.addRow("摘要", self.summary_edit)
        form.addRow("标签", self.tags_edit)
        layout.addLayout(form)
        layout.addWidget(QLabel("内容"))
        layout.addWidget(self.content_edit, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._existing_id = asset.id if asset else ""

    def result_asset(self) -> Optional[NovelAsset]:
        name = self.name_edit.text().strip()
        if not name:
            return None
        tags = [t.strip() for t in self.tags_edit.text().split(",") if t.strip()]
        return NovelAsset(
            id=self._existing_id,
            name=name,
            asset_type=str(self.type_combo.currentData()),
            summary=self.summary_edit.text().strip(),
            content=self.content_edit.toPlainText(),
            tags=tags,
        )


class PromptBlockEditDialog(QDialog):
    def __init__(self, parent=None, *, block: Optional[PromptBlock] = None):
        super().__init__(parent)
        self.setWindowTitle("编辑 Prompt" if block else "新建 Prompt")
        self.resize(520, 400)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.name_edit = QLineEdit(block.name if block else "")
        self.role_combo = QComboBox()
        for r in PROMPT_ROLES:
            self.role_combo.addItem(ROLE_LABELS.get(r, r), r)
        if block:
            ri = self.role_combo.findData(block.role)
            if ri >= 0:
                self.role_combo.setCurrentIndex(ri)
        self.order_edit = QLineEdit(str(block.order if block else 50))
        self.content_edit = QPlainTextEdit(block.content if block else "")

        form.addRow("名称", self.name_edit)
        form.addRow("类别", self.role_combo)
        form.addRow("排序", self.order_edit)
        layout.addLayout(form)
        layout.addWidget(QLabel("内容"))
        layout.addWidget(self.content_edit, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._existing_id = block.id if block else ""

    def result_block(self) -> Optional[PromptBlock]:
        name = self.name_edit.text().strip()
        if not name:
            return None
        try:
            order = int(self.order_edit.text().strip() or "0")
        except ValueError:
            order = 0
        return PromptBlock(
            id=self._existing_id,
            name=name,
            role=str(self.role_combo.currentData()),
            content=self.content_edit.toPlainText(),
            enabled=True,
            order=order,
        )
