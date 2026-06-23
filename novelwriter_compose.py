"""将 Prompt 块与选中资产合成为请求上下文。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from novelwriter_assets import (
    ASSET_CHARACTER,
    ASSET_EVENT,
    ASSET_STORY_MODE,
    ASSET_WORLD,
    CATEGORY_LABELS,
    NovelAsset,
)
from novelwriter_prompts import PromptBlock, ROLE_LABELS


def _section(title: str, body: str) -> str:
    body = (body or "").strip()
    if not body:
        return ""
    return f"## {title}\n\n{body}"


def format_asset_block(asset: NovelAsset) -> str:
    label = CATEGORY_LABELS.get(asset.asset_type, asset.asset_type)
    header = f"{label}：{asset.name}"
    if asset.summary.strip():
        return _section(header, f"{asset.summary.strip()}\n\n{asset.content.strip()}".strip())
    return _section(header, asset.content)


def compose_system_prompt(
    *,
    prompt_blocks: Sequence[PromptBlock],
    characters: Sequence[NovelAsset],
    worlds: Sequence[NovelAsset],
    events: Sequence[NovelAsset],
    story_modes: Sequence[NovelAsset],
    extra_system: str = "",
) -> str:
    parts: List[str] = []

    for block in sorted(prompt_blocks, key=lambda b: (b.order, b.name)):
        if not block.enabled or not block.content.strip():
            continue
        role_label = ROLE_LABELS.get(block.role, block.role)
        parts.append(_section(f"Prompt · {block.name} ({role_label})", block.content))

    for sm in story_modes:
        parts.append(format_asset_block(sm))
    for w in worlds:
        parts.append(format_asset_block(w))
    for c in characters:
        parts.append(format_asset_block(c))
    for e in events:
        parts.append(format_asset_block(e))

    extra = (extra_system or "").strip()
    if extra:
        parts.append(extra)

    return "\n\n".join(p for p in parts if p).strip()


def build_chat_messages(
    *,
    system_prompt: str,
    history: List[Dict[str, str]],
    user_input: str,
) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    sys_text = (system_prompt or "").strip() or "You are a helpful writing assistant."
    msgs.append({"role": "system", "content": sys_text})
    for m in history:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            msgs.append({"role": role, "content": content})
    u = (user_input or "").strip()
    if u:
        msgs.append({"role": "user", "content": u})
    return msgs


def preview_payload(
    *,
    system_prompt: str,
    messages: List[Dict[str, str]],
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "system_prompt": system_prompt,
        "messages": messages,
    }
    if meta:
        out.update(meta)
    return out
