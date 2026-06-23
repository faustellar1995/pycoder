"""本地小说资产库：角色卡 / 世界观 / 事件 / 故事模式。"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ASSET_CHARACTER = "character"
ASSET_WORLD = "world"
ASSET_EVENT = "event"
ASSET_STORY_MODE = "story_mode"

ASSET_TYPES = (ASSET_CHARACTER, ASSET_WORLD, ASSET_EVENT, ASSET_STORY_MODE)

CATEGORY_FILES: Dict[str, str] = {
    ASSET_CHARACTER: "characters.json",
    ASSET_WORLD: "worlds.json",
    ASSET_EVENT: "events.json",
    ASSET_STORY_MODE: "story_modes.json",
}

CATEGORY_LABELS: Dict[str, str] = {
    ASSET_CHARACTER: "角色",
    ASSET_WORLD: "世界",
    ASSET_EVENT: "事件",
    ASSET_STORY_MODE: "故事模式",
}


def _slug_id(name: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", (name or "").strip())
    s = s.strip("_") or "item"
    return s[:48]


@dataclass
class NovelAsset:
    id: str
    name: str
    asset_type: str
    summary: str = ""
    content: str = ""
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.asset_type,
            "summary": self.summary,
            "content": self.content,
            "tags": list(self.tags),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, asset_type: str) -> "NovelAsset":
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:12]),
            name=str(data.get("name") or "未命名"),
            asset_type=str(data.get("type") or asset_type),
            summary=str(data.get("summary") or ""),
            content=str(data.get("content") or ""),
            tags=[str(t) for t in (data.get("tags") or [])],
            extra=dict(data.get("extra") or {}),
        )


def character_card_to_asset(card: Dict[str, Any], *, asset_id: Optional[str] = None) -> NovelAsset:
    """SillyTavern 风格角色卡 → NovelAsset。"""
    name = str(card.get("name") or "未命名角色")
    lines = [
        f"## 角色：{name}",
        "",
        card.get("description") and f"**外貌/背景**：{card['description']}",
        card.get("personality") and f"**性格**：{card['personality']}",
        card.get("scenario") and f"**场景**：{card['scenario']}",
        card.get("system_prompt") and f"**扮演指令**：{card['system_prompt']}",
        card.get("example_dialogue") and f"**示例对话**：\n{card['example_dialogue']}",
        card.get("post_history_instructions")
        and f"**后续指令**：{card['post_history_instructions']}",
        card.get("first_mes") and f"**开场白**：{card['first_mes']}",
    ]
    content = "\n\n".join(x for x in lines if x)
    return NovelAsset(
        id=asset_id or _slug_id(name),
        name=name,
        asset_type=ASSET_CHARACTER,
        summary=str(card.get("description") or "")[:200],
        content=content,
        tags=[str(t) for t in (card.get("tags") or [])],
        extra={"card": card},
    )


class AssetStore:
    """读写 ``<root>/characters.json`` 等分类文件。"""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._cache: Dict[str, Dict[str, NovelAsset]] = {t: {} for t in ASSET_TYPES}

    def ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, asset_type: str) -> Path:
        fn = CATEGORY_FILES.get(asset_type, f"{asset_type}.json")
        return self.root / fn

    def load_all(self) -> None:
        self.ensure_layout()
        for asset_type in ASSET_TYPES:
            self._cache[asset_type] = self._load_category(asset_type)

    def _load_category(self, asset_type: str) -> Dict[str, NovelAsset]:
        path = self.path_for(asset_type)
        if not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        items = raw.get("items") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            return {}
        out: Dict[str, NovelAsset] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            asset = NovelAsset.from_dict(item, asset_type=asset_type)
            out[asset.id] = asset
        return out

    def save_category(self, asset_type: str) -> None:
        self.ensure_layout()
        items = [a.to_dict() for a in sorted(self._cache[asset_type].values(), key=lambda x: x.name)]
        payload = {"version": 1, "items": items}
        self.path_for(asset_type).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list_assets(self, asset_type: str) -> List[NovelAsset]:
        return sorted(self._cache.get(asset_type, {}).values(), key=lambda a: a.name)

    def get(self, asset_type: str, asset_id: str) -> Optional[NovelAsset]:
        return self._cache.get(asset_type, {}).get(asset_id)

    def upsert(self, asset: NovelAsset) -> NovelAsset:
        if not asset.id:
            asset.id = _slug_id(asset.name) + "_" + uuid.uuid4().hex[:6]
        self._cache.setdefault(asset.asset_type, {})[asset.id] = asset
        self.save_category(asset.asset_type)
        return asset

    def delete(self, asset_type: str, asset_id: str) -> bool:
        bucket = self._cache.get(asset_type, {})
        if asset_id not in bucket:
            return False
        del bucket[asset_id]
        self.save_category(asset_type)
        return True

    def import_from_prompts_system(self, prompts_path: Path) -> int:
        """从 prompts_system.json 的 card* 键导入角色卡。"""
        if not prompts_path.is_file():
            return 0
        try:
            prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return 0
        added = 0
        for key, val in prompts.items():
            if not str(key).lower().startswith("card"):
                continue
            if not isinstance(val, str) or not val.strip():
                continue
            try:
                card = json.loads(val)
            except json.JSONDecodeError:
                continue
            if not isinstance(card, dict):
                continue
            aid = _slug_id(str(card.get("name") or key))
            if aid in self._cache[ASSET_CHARACTER]:
                continue
            self.upsert(character_card_to_asset(card, asset_id=aid))
            added += 1
        return added


def default_assets_root() -> Path:
    return Path(__file__).resolve().parent / "novel_assets"
