"""小说编辑器：可自由增删的 Prompt 块。"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROMPT_ROLE_SYSTEM = "system"
PROMPT_ROLE_STYLE = "style"
PROMPT_ROLE_CONSTRAINT = "constraint"
PROMPT_ROLE_CUSTOM = "custom"

PROMPT_ROLES = (
    PROMPT_ROLE_SYSTEM,
    PROMPT_ROLE_STYLE,
    PROMPT_ROLE_CONSTRAINT,
    PROMPT_ROLE_CUSTOM,
)

ROLE_LABELS: Dict[str, str] = {
    PROMPT_ROLE_SYSTEM: "系统/设定",
    PROMPT_ROLE_STYLE: "文风",
    PROMPT_ROLE_CONSTRAINT: "约束",
    PROMPT_ROLE_CUSTOM: "自定义",
}


@dataclass
class PromptBlock:
    id: str
    name: str
    role: str = PROMPT_ROLE_CUSTOM
    content: str = ""
    enabled: bool = True
    order: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "content": self.content,
            "enabled": self.enabled,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptBlock":
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:12]),
            name=str(data.get("name") or "未命名"),
            role=str(data.get("role") or PROMPT_ROLE_CUSTOM),
            content=str(data.get("content") or ""),
            enabled=bool(data.get("enabled", True)),
            order=int(data.get("order") or 0),
        )


DEFAULT_BLOCKS: List[PromptBlock] = [
    PromptBlock(
        id="base_novel",
        name="小说助手基础",
        role=PROMPT_ROLE_SYSTEM,
        order=0,
        content=(
            "你是专业小说创作与润色助手。协助用户推进剧情、描写场景、塑造人物对白。"
            "保持叙事连贯，尊重已设定的世界观与角色性格。用中文回答，除非用户要求其他语言。"
        ),
    ),
    PromptBlock(
        id="style_literary",
        name="文学描写",
        role=PROMPT_ROLE_STYLE,
        order=10,
        content="注重感官细节与心理刻画；对白简洁有个性；避免说明书式旁白。",
    ),
    PromptBlock(
        id="constraint_ooc",
        name="角色一致性",
        role=PROMPT_ROLE_CONSTRAINT,
        order=20,
        content="不要让人物 OOC；不要替用户角色做决定，除非用户明确要求。",
    ),
]


class PromptStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.path = self.root / "prompt_blocks.json"
        self._blocks: Dict[str, PromptBlock] = {}

    def ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        self.ensure_layout()
        if not self.path.is_file():
            self._blocks = {b.id: b for b in DEFAULT_BLOCKS}
            self.save()
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._blocks = {b.id: b for b in DEFAULT_BLOCKS}
            return
        items = raw.get("blocks") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            self._blocks = {b.id: b for b in DEFAULT_BLOCKS}
            return
        self._blocks = {}
        for item in items:
            if isinstance(item, dict):
                b = PromptBlock.from_dict(item)
                self._blocks[b.id] = b

    def save(self) -> None:
        self.ensure_layout()
        blocks = sorted(self._blocks.values(), key=lambda b: (b.order, b.name))
        payload = {"version": 1, "blocks": [b.to_dict() for b in blocks]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_blocks(self) -> List[PromptBlock]:
        return sorted(self._blocks.values(), key=lambda b: (b.order, b.name))

    def enabled_blocks(self) -> List[PromptBlock]:
        return [b for b in self.list_blocks() if b.enabled]

    def get(self, block_id: str) -> Optional[PromptBlock]:
        return self._blocks.get(block_id)

    def upsert(self, block: PromptBlock) -> PromptBlock:
        if not block.id:
            block.id = uuid.uuid4().hex[:12]
        self._blocks[block.id] = block
        self.save()
        return block

    def delete(self, block_id: str) -> bool:
        if block_id not in self._blocks:
            return False
        del self._blocks[block_id]
        self.save()
        return True

    def reorder(self, ordered_ids: List[str]) -> None:
        for i, bid in enumerate(ordered_ids):
            b = self._blocks.get(bid)
            if b is not None:
                b.order = i * 10
        self.save()

    def toggle(self, block_id: str, enabled: bool) -> None:
        b = self._blocks.get(block_id)
        if b is None:
            return
        b.enabled = enabled
        self.save()
