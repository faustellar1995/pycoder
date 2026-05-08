"""Lightweight Markdown → HTML renderer (zero extra dependencies).

Supports:
  - Headings h1–h6（行尾可选 #）
  - Bold **text** / __text__、Italic *text* / _text_
  - Strikethrough ~~text~~
  - Inline code `code` / Fenced code blocks ```lang
  - Unordered / Ordered lists（可选任务项 `- [ ]` / `- [x]`）
  - Blockquotes（`>` / `> ` 及续行）
  - Horizontal rules ---
  - Tables（管道表格 + 分隔行）；非法表格降级为段落而非错乱重复
  - Links [text](url) & Images ![alt](url)（href/src 转义 + 拦截 javascript: 等）
  - HTML escaping throughout
"""

from __future__ import annotations

import html
import re
from typing import List

# Qt QTextBrowser 无外部 CSS 时便于辨认表格与代码块
_TABLE_ATTRS = ' border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;border:1px solid #ccc;"'
_CODE_BLOCK_STYLE = "background:#f6f8fa;padding:12px;border-radius:6px;overflow-x:auto;"


def _safe_url(url: str) -> str:
    """属性用 URL：转义并拦截危险 scheme。"""
    u = url.strip()
    if not u:
        return "#"
    low = u.lower()
    if low.startswith(("javascript:", "vbscript:", "data:")):
        return "#"
    if re.match(r"^[a-z][a-z0-9+.-]*:", u, re.I):
        scheme = u.split(":", 1)[0].lower()
        if scheme not in ("http", "https", "mailto", "ftp"):
            return "#"
    return html.escape(u, quote=True)


def _strip_heading_trailing_hashes(content: str) -> str:
    """去掉标题末尾可选的 # 与空格（CommonMark 风格）。"""
    return re.sub(r"(?:\s+#)+\s*$", "", content).rstrip()


def markdown_to_html(text: str) -> str:
    """
    Convert a full Markdown document to safe HTML fragment.
    Output is wrapped in a <div class='md-content'> for scoped styling.
    """
    lines = text.split("\n")
    out: List[str] = []
    i, n = 0, len(lines)

    in_code = False
    code_lang = ""
    code_lines: List[str] = []

    while i < n:
        line = lines[i]

        # ── Fenced code blocks ──────────────────────────────────────────
        if line.startswith("```"):
            if not in_code:
                in_code = True
                code_lang = line[3:].strip()
                code_lines = []
            else:
                in_code = False
                lang_attr = (
                    f' class="language-{html.escape(code_lang)}"'
                    if code_lang
                    else ""
                )
                escaped = html.escape("\n".join(code_lines))
                out.append(
                    f'<pre style="{_CODE_BLOCK_STYLE}"><code{lang_attr}>'
                    f"{escaped}</code></pre>\n"
                )
                code_lines = []
                code_lang = ""
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # ── Blank line ──────────────────────────────────────────────────
        stripped = line.strip()
        if not stripped:
            out.append("\n")
            i += 1
            continue

        # ── Horizontal rule ─────────────────────────────────────────────
        if re.match(r"^(-{3,}|\*{3,}|_{3,})\s*$", stripped):
            out.append("<hr>\n")
            i += 1
            continue

        # ── Headings ────────────────────────────────────────────────────
        hm = re.match(r"^(#{1,6})\s+(.+)$", line)
        if hm:
            level = len(hm.group(1))
            raw_title = _strip_heading_trailing_hashes(hm.group(2).strip())
            content = _inline(raw_title)
            out.append(f"<h{level}>{content}</h{level}>\n")
            i += 1
            continue

        # ── Blockquotes（支持 `>` 与 `> `）─────────────────────────────
        _ls = line.lstrip()
        if _ls.startswith(">"):
            bq_lines: List[str] = []
            while i < n:
                raw = lines[i]
                lead = raw.lstrip()
                if not lead.startswith(">"):
                    break
                rest = lead[1:]
                if rest.startswith(" "):
                    rest = rest[1:]
                bq_lines.append(rest)
                i += 1
            bq_html = "<br>".join(_inline(l) for l in bq_lines)
            out.append(
                '<blockquote style="margin:8px 0;padding-left:12px;'
                'border-left:3px solid #cbd5e1;">'
                f"{bq_html}</blockquote>\n"
            )
            continue

        # ── Table (| ... |) ────────────────────────────────────────────
        if line.lstrip().startswith("|") and "|" in line[1:]:
            rows: List[List[str]] = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(
                    [c.strip() for c in lines[i].strip().strip("|").split("|")]
                )
                i += 1
            if len(rows) >= 2 and re.match(
                r"^[\s:|:-]+$", "".join(rows[1])
            ):
                # Second row is the separator → real table
                header = rows[0]
                out.append(f"<table{_TABLE_ATTRS}>\n<thead>\n<tr>")
                for h in header:
                    out.append(f"<th>{_inline(h)}</th>")
                out.append("</tr>\n</thead>\n<tbody>\n")
                for row in rows[2:]:
                    out.append("<tr>")
                    for cell in row:
                        out.append(f"<td>{_inline(cell)}</td>")
                    out.append("</tr>\n")
                out.append("</tbody>\n</table>\n")
            else:
                # 无合法分隔行：已读完各行，合并为一段落（避免 i 回溯错误）
                flat = [" | ".join(r) for r in rows]
                para_text = " ".join(_inline(l) for l in flat)
                out.append(f"<p>{para_text}</p>\n")
            continue

        # ── Unordered list ──────────────────────────────────────────────
        ul_match = re.match(r"^(\s*)[-*+]\s+(.+)$", line)
        if ul_match:
            out.append("<ul>\n")
            while i < n:
                raw = lines[i]
                tm = re.match(r"^(\s*)[-*+]\s+\[([ xX])\]\s+(.+)$", raw)
                if tm:
                    tick = "&#9745; " if tm.group(2).strip().lower() == "x" else "&#9744; "
                    out.append(f"  <li>{tick}{_inline(tm.group(3))}</li>\n")
                    i += 1
                    continue
                m = re.match(r"^(\s*)[-*+]\s+(.+)$", raw)
                if not m:
                    break
                out.append(f"  <li>{_inline(m.group(2))}</li>\n")
                i += 1
            out.append("</ul>\n")
            continue

        # ── Ordered list ────────────────────────────────────────────────
        ol_match = re.match(r"^\s*(\d+)\.\s+(.+)$", line)
        if ol_match:
            out.append("<ol>\n")
            while i < n:
                m = re.match(r"^\s*(\d+)\.\s+(.+)$", lines[i])
                if not m:
                    break
                out.append(f"  <li>{_inline(m.group(2))}</li>\n")
                i += 1
            out.append("</ol>\n")
            continue

        # ── Regular paragraph (join until blank line) ──────────────────
        para: List[str] = [line]
        i += 1
        while i < n and lines[i].strip():
            para.append(lines[i])
            i += 1
        para_text = " ".join(_inline(l) for l in para if l.strip())
        out.append(f"<p>{para_text}</p>\n")

    # Close dangling code block
    if in_code and code_lines:
        escaped = html.escape("\n".join(code_lines))
        out.append(
            f'<pre style="{_CODE_BLOCK_STYLE}"><code>{escaped}</code></pre>\n'
        )

    return '<div class="md-content">\n' + "".join(out) + "</div>\n"


def _inline_img(m: re.Match) -> str:
    alt = m.group(1)
    url = _safe_url(m.group(2))
    return (
        f'<img src="{url}" alt="{alt}" '
        'style="max-width:100%;height:auto;">'
    )


def _inline_link(m: re.Match) -> str:
    label = m.group(1)
    url = _safe_url(m.group(2))
    return (
        f'<a href="{url}" target="_blank" rel="noopener noreferrer">'
        f"{label}</a>"
    )


def _inline(text: str) -> str:
    """Process inline formatting on a single line (HTML-escaped output)."""
    text = html.escape(text)

    # 1) Inline code（须在其它模式之前，避免误匹配）
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)

    # 2) Images、3) Links（URL 安全处理）
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _inline_img, text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _inline_link, text)

    # 4) Strikethrough
    text = re.sub(r"~~(.+?)~~", r"<del>\1</del>", text)

    # 5) Bold ** ** / __ __
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__(.+?)__", r"<strong>\1</strong>", text)

    # 6) Italic * *（不与 ** 冲突）
    text = re.sub(r"(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", text)
    # 7) Italic _ _（不与 __ 冲突）
    text = re.sub(r"(?<!_)_(?!_)([^_]+?)(?<!_)_(?!_)", r"<em>\1</em>", text)

    return text
