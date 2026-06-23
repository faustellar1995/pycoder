"""会话 Todo：数据结构、系统 prompt 约定、从助手回复中解析。"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

TODO_FENCE = "todos"

_TODO_BLOCK_RE = re.compile(
    r"```\s*todos?\s*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class TodoItem:
    content: str
    tags: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TodoItem":
        content = str(data.get("content") or "").strip()
        raw_tags = data.get("tags")
        tags: List[str] = []
        if isinstance(raw_tags, list):
            tags = [str(t).strip() for t in raw_tags if str(t).strip()]
        tid = str(data.get("id") or "").strip() or uuid.uuid4().hex[:12]
        return cls(content=content, tags=tags, id=tid)

    def display_line(self) -> str:
        tag_part = " ".join(f"#{t}" for t in self.tags) if self.tags else ""
        if tag_part:
            return f"{self.content}  [{tag_part}]"
        return self.content


def todos_system_hint() -> str:
    return (
        "## Todo 输出约定（当前已启用 Todo 模式，必须遵守）\n"
        "在完成面向用户的正文之后，**另起一段**输出恰好 **2 条**待办，放入独立 fenced 代码块。\n"
        "代码块语言标记为 `todos`，内容为 JSON 数组；此块不计入正文，正文中不要重复列出这 2 条。\n\n"
        "格式示例：\n"
        "```todos\n"
        '[\n'
        '  {"content": "一条具体、可执行的待办", "tags": ["标签1", "标签2"]},\n'
        '  {"content": "另一条待办", "tags": ["标签"]}\n'
        "]\n"
        "```\n\n"
        "规则：\n"
        "- 每条对象仅含 `content`（字符串）与 `tags`（字符串数组，可为 `[]`）\n"
        "- 必须输出 2 条，且与本轮对话相关、可独立执行\n"
        "- tags 用简短词（中文或英文），便于归类\n"
    )


def parse_todo_items(payload: Any) -> List[TodoItem]:
    if not isinstance(payload, list):
        return []
    out: List[TodoItem] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        todo = TodoItem.from_dict(item)
        if todo.content:
            out.append(todo)
    return out


def extract_todos_from_reply(text: str, *, max_items: int = 2) -> Tuple[str, List[TodoItem]]:
    """
    从助手全文提取 ```todos``` 块，返回 (去掉该块后的展示正文, 解析出的 Todo 列表)。
    """
    raw = text or ""
    match = _TODO_BLOCK_RE.search(raw)
    if not match:
        return raw.strip(), []
    block = match.group(1).strip()
    display = (raw[: match.start()] + raw[match.end() :]).strip()
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return raw.strip(), []
    items = parse_todo_items(data)
    if max_items > 0:
        items = items[:max_items]
    return display, items


def todos_from_json_list(data: Any) -> List[TodoItem]:
    if not isinstance(data, list):
        return []
    return [TodoItem.from_dict(x) for x in data if isinstance(x, dict)]


def todos_to_json_list(items: List[TodoItem]) -> List[Dict[str, Any]]:
    return [t.to_dict() for t in items]
