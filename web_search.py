"""联网搜索工具（无第三方依赖，urllib + 百度/搜狗网页结果解析）。

通过抓取 ``www.baidu.com`` / ``www.sogou.com`` 的搜索结果页提取标题与链接（无需付费 API）。
页面结构可能变更；若解析为空可切换引擎或更换关键词。

环境变量 ``DEEPSEEK_WEB_SEARCH_ENGINE``：``baidu``（默认）或 ``sogou``。
可选 ``DEEPSEEK_WEB_SEARCH_FALLBACK=1``：主引擎无条目时自动尝试另一引擎（耗时会增加一次请求）。
"""

from __future__ import annotations

import errno
import html as html_module
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple

MAX_SINGLE_TIMEOUT_SEC = 10
DEFAULT_TIMEOUT = 10
DEFAULT_RETRIES = 1
MAX_OUTPUT_CHARS = 24_000

ResultRow = Tuple[str, str, str]  # title, url, snippet


def _disabled_by_env() -> bool:
    v = os.getenv("DEEPSEEK_DISABLE_WEB_SEARCH", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _timeout_seconds() -> int:
    raw = os.getenv("DEEPSEEK_WEB_SEARCH_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_TIMEOUT
    try:
        t = int(raw)
        return max(3, min(t, MAX_SINGLE_TIMEOUT_SEC))
    except ValueError:
        return DEFAULT_TIMEOUT


def _retry_count() -> int:
    raw = os.getenv("DEEPSEEK_WEB_SEARCH_RETRIES", "").strip()
    if not raw:
        return DEFAULT_RETRIES
    try:
        n = int(raw)
        return max(1, min(n, 3))
    except ValueError:
        return DEFAULT_RETRIES


def _engine_name() -> str:
    v = (os.getenv("DEEPSEEK_WEB_SEARCH_ENGINE") or "baidu").strip().lower()
    if v in ("baidu", "sogou"):
        return v
    return "baidu"


def _fallback_enabled() -> bool:
    return os.getenv("DEEPSEEK_WEB_SEARCH_FALLBACK", "").strip().lower() in ("1", "true", "yes", "on")


def _is_transient_net_error(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (429, 500, 502, 503, 504)
    if isinstance(exc, urllib.error.URLError):
        r = exc.reason
        if isinstance(r, (socket.timeout, TimeoutError)):
            return True
        if isinstance(r, OSError):
            if r.errno in (
                errno.ETIMEDOUT,
                errno.ECONNRESET,
                errno.ECONNREFUSED,
                errno.EPIPE,
                errno.EHOSTUNREACH,
                errno.ENETUNREACH,
            ):
                return True
            if getattr(r, "winerror", None) == 10060:
                return True
        return False
    return isinstance(exc, (TimeoutError, socket.timeout, ConnectionResetError, BrokenPipeError))


def _build_opener(proxy_url: Optional[str]) -> urllib.request.OpenerDirector:
    if not proxy_url:
        return urllib.request.build_opener()
    proxy = {"http": proxy_url, "https": proxy_url}
    return urllib.request.build_opener(urllib.request.ProxyHandler(proxy))


def _read_url(
    req: urllib.request.Request,
    *,
    proxy_url: Optional[str],
    base_timeout: int,
    retries: int,
) -> bytes:
    opener = _build_opener(proxy_url)
    last: Optional[BaseException] = None
    cap = min(base_timeout, MAX_SINGLE_TIMEOUT_SEC)
    for attempt in range(retries):
        per_try = cap
        try:
            with opener.open(req, timeout=per_try) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if attempt + 1 < retries and _is_transient_net_error(exc):
                time.sleep(0.35 * (2**attempt))
                continue
            raise
        except Exception as exc:
            last = exc
            if attempt + 1 < retries and _is_transient_net_error(exc):
                time.sleep(0.35 * (2**attempt))
                continue
            raise
    assert last is not None
    raise last


def _decode_html(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gb18030", errors="replace")


def _strip_tags(s: str) -> str:
    t = re.sub(r"<[^>]+>", " ", s)
    t = html_module.unescape(t)
    return re.sub(r"\s+", " ", t).strip()


def _browser_headers(referer: str) -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": referer,
        "Cache-Control": "no-cache",
    }


def _fetch_search_html(
    url: str,
    referer: str,
    *,
    proxy_url: Optional[str],
    base_timeout: int,
    retries: int,
) -> str:
    req = urllib.request.Request(url, headers=_browser_headers(referer), method="GET")
    raw = _read_url(req, proxy_url=proxy_url, base_timeout=base_timeout, retries=retries)
    return _decode_html(raw)


def _parse_baidu(html: str, max_results: int) -> List[ResultRow]:
    """从百度桌面结果页抽取标题与链接（启发式，页面改版可能导致为空）。"""
    if "安全验证" in html or "请输入验证码" in html:
        return []

    results: List[ResultRow] = []
    seen: set[str] = set()

    # 常见：h3.t > a
    pat = re.compile(
        r'<h3[^>]*class="[^"]*\bt\b[^"]*"[^>]*>\s*'
        r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        re.I | re.S,
    )
    for m in pat.finditer(html):
        if len(results) >= max_results:
            break
        url = _strip_tags(m.group(1))
        title = _strip_tags(m.group(2))
        if not url or not title:
            continue
        if url.startswith("//"):
            url = "https:" + url
        if not url.startswith("http"):
            continue
        if url in seen:
            continue
        seen.add(url)
        results.append((title, url, ""))

    if results:
        return results

    # 兜底：部分样式下 class 顺序不同
    pat2 = re.compile(
        r'<h3[^>]*>\s*<a[^>]*href="(https?://[^"]+)"[^>]*class="[^"]*\bt\b[^"]*"[^>]*>(.*?)</a>',
        re.I | re.S,
    )
    for m in pat2.finditer(html):
        if len(results) >= max_results:
            break
        url = _strip_tags(m.group(1))
        title = _strip_tags(m.group(2))
        if url and title and url not in seen:
            seen.add(url)
            results.append((title, url, ""))

    return results[:max_results]


def _parse_sogou(html: str, max_results: int) -> List[ResultRow]:
    """从搜狗网页搜索抽取标题与链接。"""
    if "请输入验证码" in html or "请输入右侧字符" in html:
        return []

    results: List[ResultRow] = []
    seen: set[str] = set()

    patterns = [
        # vr 结果块标题
        re.compile(
            r'<h3[^>]*class="[^"]*vr-title[^"]*"[^>]*>\s*'
            r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            re.I | re.S,
        ),
        # 通用 h3 > a（过滤站内）
        re.compile(
            r'<h3[^>]*>\s*<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>',
            re.I | re.S,
        ),
    ]

    for pat in patterns:
        for m in pat.finditer(html):
            if len(results) >= max_results:
                return results
            url = _strip_tags(m.group(1))
            title = _strip_tags(m.group(2))
            if not url or not title:
                continue
            if "sogou.com/web/tn" in url and "query=" not in url[:80]:
                continue
            if url in seen:
                continue
            seen.add(url)
            results.append((title, url, ""))
        if results:
            break

    return results[:max_results]


def _run_engine(
    engine: str,
    query: str,
    mr: int,
    *,
    proxy_url: Optional[str],
    base_timeout: int,
    retries: int,
) -> Tuple[str, List[ResultRow]]:
    """返回 (引擎展示名, 结果行列表)。失败时抛出异常由上层处理。"""
    q = urllib.parse.quote(query)
    if engine == "sogou":
        url = f"https://www.sogou.com/web?query={q}"
        html = _fetch_search_html(url, "https://www.sogou.com/", proxy_url=proxy_url, base_timeout=base_timeout, retries=retries)
        rows = _parse_sogou(html, mr)
        return ("搜狗", rows)

    url = f"https://www.baidu.com/s?wd={q}&ie=utf-8"
    html = _fetch_search_html(url, "https://www.baidu.com/", proxy_url=proxy_url, base_timeout=base_timeout, retries=retries)
    rows = _parse_baidu(html, mr)
    return ("百度", rows)


def run_web_search(
    query: str,
    *,
    max_results: int = 8,
    proxy_url: Optional[str] = None,
    timeout: Optional[int] = None,
) -> str:
    """
    抓取百度或搜狗搜索结果页并解析标题与链接（免费、无需密钥）。

    - ``DEEPSEEK_WEB_SEARCH_ENGINE``：``baidu``（默认）或 ``sogou``
    - ``DEEPSEEK_WEB_SEARCH_FALLBACK=1``：主引擎无条目时再请求另一引擎
    - ``DEEPSEEK_WEB_SEARCH_TIMEOUT`` / ``DEEPSEEK_WEB_SEARCH_RETRIES``：同前
    """
    if _disabled_by_env():
        return "[tool error] 已禁用联网搜索（环境变量 DEEPSEEK_DISABLE_WEB_SEARCH）"

    q = (query or "").strip()
    if not q:
        return "[tool error] query 不能为空"

    raw_t = timeout if timeout is not None else _timeout_seconds()
    base_timeout = min(int(raw_t), MAX_SINGLE_TIMEOUT_SEC)
    retries = _retry_count()
    mr = max(1, min(int(max_results or 8), 15))

    primary = _engine_name()
    secondary = "sogou" if primary == "baidu" else "baidu"

    lines: List[str] = [f"[web_search] query: {q}", ""]

    engines_tried: List[str] = []
    all_rows: List[ResultRow] = []

    for eng in (primary, secondary):
        if eng != primary and not (_fallback_enabled() and not all_rows):
            break
        if eng != primary and all_rows:
            break
        try:
            label, rows = _run_engine(eng, q, mr, proxy_url=proxy_url, base_timeout=base_timeout, retries=retries)
            engines_tried.append(label)
            if rows:
                all_rows = rows
                lines[1] = f"(engine={label})"
                break
            lines.append(f"({label} 解析未得到有效条目，页面结构可能已变更或触发风控。)")
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:
                body = ""
            lines.append(f"[{eng}] 请求失败: HTTP {exc.code} {body}")
        except Exception as exc:
            lines.append(f"[{eng}] 请求失败: {exc}")

        if eng == primary and not _fallback_enabled():
            break

    if all_rows:
        lines.append(f"## 网页搜索结果（{engines_tried[-1]}）")
        for i, (title, url, snippet) in enumerate(all_rows[:mr], start=1):
            lines.append(f"{i}. {title}")
            lines.append(f"   {url}")
            if snippet:
                lines.append(f"   {snippet[:400]}")
    else:
        lines.append("## 网页搜索结果")
        lines.append(
            "(未解析到链接。可尝试：更换关键词；设置 DEEPSEEK_WEB_SEARCH_ENGINE=sogou；"
            "开启 HTTP 代理；或设置 DEEPSEEK_WEB_SEARCH_FALLBACK=1 自动换引擎。)"
        )

    out = "\n".join(lines).strip()
    if len(out) > MAX_OUTPUT_CHARS:
        return out[:MAX_OUTPUT_CHARS] + "\n\n... [web_search 输出已截断]"
    return out
