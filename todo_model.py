"""会话 Todo：数据结构、系统 prompt 约定、从助手回复中解析。"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

TODO_FENCE = "todos"
TODO_DEFAULT_COUNT = 2
TODO_MAX_COUNT = 20

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
        "## Todo 输出约定（已启用，必须遵守）\n"
        "\n"
        "### 回复结构\n"
        "1. **正文**（可含思考/推理过程，本地自用无需刻意隐藏）。\n"
        "2. 在全文**最末尾**追加**一个** fenced 代码块，语言标记为 `todos`，内含 JSON 数组。\n"
        "\n"
        "### 待办条数\n"
        f"- **默认**：每轮 `todos` 块内恰好 **{TODO_DEFAULT_COUNT}** 条。\n"
        "- **用户本轮明确要求 N 条时**（如「输出5个待办」）：输出恰好 **N** 条，"
        f"不必拘泥于默认 {TODO_DEFAULT_COUNT} 条；N 建议 1–{TODO_MAX_COUNT}。\n"
        "- 若用户未指定条数，一律按默认条数。\n"
        "\n"
        "### `todos` 代码块格式\n"
        "仅含 JSON 数组，每项对象只有两个字段：\n"
        '- `content`（字符串）：具体、可独立执行的一步。\n'
        '- `tags`（字符串数组，可为 `[]`）：简短归类词。\n'
        "\n"
        "示例（默认 2 条）：\n"
        "```todos\n"
        '[\n'
        '  {"content": "阅读某章并做笔记", "tags": ["泛函", "学习"]},\n'
        '  {"content": "完成课后习题 1–5", "tags": ["练习"]}\n'
        "]\n"
        "```\n"
        "\n"
        "待办 JSON **只**放在上述 `todos` 块中；全回复中该块**只能出现一次**，且必须在最后。\n"
        "（用户从待办列表双击提交某条以**执行**时，不适用本节，见「执行已有待办」约定。）\n"
    )


def todos_execute_hint() -> str:
    """用户双击待办列表提交询问时：优先解题，本轮不产出新待办。"""
    return (
        "## 执行已有待办（本轮模式，优先于上文 Todo 输出约定）\n"
        "用户从待办列表选中了**一条已有待办**并提交，请你：\n"
        "1. **首要任务**：直接完成、解答或推进该事项（讲解、推导、步骤、代码、示例、资料等**实质内容**）。\n"
        "2. 不要把回复重心放在「再列一批新待办」或任务拆解清单上；先解决眼前这条。\n"
        "3. 本轮回复**末尾不要**输出 `todos` fenced 代码块（除非用户明确要求追加待办）。\n"
        "4. 若该事项已做完，在正文中说明结论即可。\n"
        "正文可含思考/推理过程。\n"
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


def extract_todos_from_reply(
    text: str,
    *,
    max_items: int = 0,
) -> Tuple[str, List[TodoItem]]:
    """
    从助手全文提取 ```todos``` 块，返回 (去掉该块后的展示正文, 解析出的 Todo 列表)。
    max_items<=0 表示不截断（至多 TODO_MAX_COUNT 条安全上限）。
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
    cap = max_items if max_items > 0 else TODO_MAX_COUNT
    items = items[:cap]
    return display, items


def todos_from_json_list(data: Any) -> List[TodoItem]:
    if not isinstance(data, list):
        return []
    return [TodoItem.from_dict(x) for x in data if isinstance(x, dict)]


def todos_to_json_list(items: List[TodoItem]) -> List[Dict[str, Any]]:
    return [t.to_dict() for t in items]
