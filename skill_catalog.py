"""ClawHub 技能市场 API（与 ironclaw `ironclaw_skills::catalog` 对齐，含搜索缓存）。"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import gzip
import io
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_REGISTRY_URL = "https://wry-manatee-359.convex.site"
REQUEST_TIMEOUT = 15

# 与 ironclaw catalog.rs 一致：5 分钟 TTL，最多保留 50 条查询缓存
CACHE_TTL_SEC = 300.0
MAX_CACHE_ENTRIES = 50


def registry_base_url() -> str:
    return (
        os.getenv("CLAWHUB_REGISTRY", "").strip()
        or os.getenv("CLAWDHUB_REGISTRY", "").strip()
        or DEFAULT_REGISTRY_URL
    ).rstrip("/")


@dataclass
class CatalogEntry:
    slug: str
    name: str
    description: str
    version: str = ""
    score: float = 0.0


@dataclass
class CatalogSearchOutcome:
    """与 ironclaw `CatalogSearchOutcome` 对应：结果 + 可选错误信息。"""

    results: List[CatalogEntry]
    error: Optional[str] = None


def _build_opener(proxy_url: Optional[str]):
    if not proxy_url:
        return urllib.request.build_opener()
    proxy = {"http": proxy_url, "https": proxy_url}
    return urllib.request.build_opener(urllib.request.ProxyHandler(proxy))


def _http_get_json(url: str, *, proxy_url: Optional[str]) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "localharness/1.0"})
    opener = _build_opener(proxy_url)
    with opener.open(req, timeout=REQUEST_TIMEOUT) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _parse_search_payload(data: Any) -> Tuple[List[CatalogEntry], Optional[str]]:
    raw_list: List[Dict[str, Any]]
    if isinstance(data, dict) and "results" in data:
        raw_list = data["results"]  # type: ignore[assignment]
    elif isinstance(data, list):
        raw_list = data  # type: ignore[assignment]
    else:
        return [], "注册表返回格式无法识别"

    out: List[CatalogEntry] = []
    for item in raw_list[:25]:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug") or "").strip()
        if not slug:
            continue
        name = str(item.get("displayName") or item.get("display_name") or item.get("name") or "")
        desc = str(item.get("summary") or item.get("description") or "")
        ver = str(item.get("version") or "")
        score = float(item.get("score") or 0.0)
        out.append(CatalogEntry(slug=slug, name=name, description=desc, version=ver, score=score))
    return out, None


def _fetch_search_raw(base: str, query_lower: str, *, proxy_url: Optional[str]) -> CatalogSearchOutcome:
    q = urllib.parse.urlencode({"q": query_lower})
    url = f"{base}/api/v1/search?{q}"
    try:
        data = _http_get_json(url, proxy_url=proxy_url)
    except urllib.error.HTTPError as exc:
        return CatalogSearchOutcome(results=[], error=f"HTTP {exc.code}")
    except urllib.error.URLError as exc:
        return CatalogSearchOutcome(results=[], error=str(exc.reason))
    except json.JSONDecodeError as exc:
        return CatalogSearchOutcome(results=[], error=f"JSON 解析失败: {exc}")
    except Exception as exc:
        return CatalogSearchOutcome(results=[], error=str(exc))

    results, parse_err = _parse_search_payload(data)
    if parse_err:
        return CatalogSearchOutcome(results=[], error=parse_err)
    return CatalogSearchOutcome(results=results, error=None)


class SkillCatalog:
    """带内存 TTL 缓存的目录客户端（行为对齐 ironclaw `SkillCatalog::search`）。"""

    def __init__(self, registry_url: Optional[str] = None) -> None:
        self.registry_url = (registry_url or registry_base_url()).rstrip("/")
        self.proxy_url: Optional[str] = None
        self._lock = threading.Lock()
        # (query_lower, monotonic_fetched_at, outcome)
        self._cache: List[Tuple[str, float, CatalogSearchOutcome]] = []

    def set_proxy(self, proxy_url: Optional[str]) -> None:
        self.proxy_url = (proxy_url or "").strip() or None

    def search(self, query: str) -> CatalogSearchOutcome:
        query_lower = query.lower().strip()
        now = time.monotonic()
        with self._lock:
            for q, fetched_at, outcome in self._cache:
                if q == query_lower and now - fetched_at < CACHE_TTL_SEC:
                    return CatalogSearchOutcome(
                        results=list(outcome.results),
                        error=outcome.error,
                    )

        outcome = _fetch_search_raw(self.registry_url, query_lower, proxy_url=self.proxy_url)

        with self._lock:
            self._cache = [(q, t, o) for q, t, o in self._cache if q != query_lower]
            if len(self._cache) >= MAX_CACHE_ENTRIES:
                self._cache.pop(0)
            self._cache.append((query_lower, now, outcome))
        return CatalogSearchOutcome(results=list(outcome.results), error=outcome.error)

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def skill_download_url(self, slug: str) -> str:
        encoded = urllib.parse.quote(slug, safe="")
        return f"{self.registry_url}/api/v1/download?slug={encoded}"

    def download_skill_md(self, slug: str) -> Tuple[str, Optional[str]]:
        url = self.skill_download_url(slug)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "localharness/1.0"})
            opener = _build_opener(self.proxy_url)
            with opener.open(req, timeout=REQUEST_TIMEOUT) as resp:
                data = resp.read()

            # 处理常见压缩/打包：zip / gzip
            if data[:4] == b"PK\x03\x04":
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    names = zf.namelist()
                    pick = None
                    for n in names:
                        if n.endswith("SKILL.md"):
                            pick = n
                            break
                    if pick is None:
                        for n in names:
                            if n.lower().endswith(".md"):
                                pick = n
                                break
                    if pick is None and names:
                        pick = names[0]
                    if pick is None:
                        return "", "zip 包为空，未找到 SKILL.md"
                    raw = zf.read(pick)
                    return raw.decode("utf-8", errors="replace"), None

            if len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B:
                try:
                    raw = gzip.decompress(data)
                    return raw.decode("utf-8", errors="replace"), None
                except Exception:
                    pass

            return data.decode("utf-8", errors="replace"), None
        except Exception as exc:
            return "", str(exc)


_default_catalog: Optional[SkillCatalog] = None
_default_catalog_lock = threading.Lock()


def shared_catalog() -> SkillCatalog:
    """进程内共享实例，使 UI / 后台线程复用同一份搜索缓存。"""
    global _default_catalog
    with _default_catalog_lock:
        if _default_catalog is None:
            _default_catalog = SkillCatalog()
        return _default_catalog


def search_catalog(query: str) -> Tuple[List[CatalogEntry], Optional[str]]:
    """便捷函数：走共享目录缓存。"""
    o = shared_catalog().search(query)
    return o.results, o.error


def skill_download_url(slug: str) -> str:
    return shared_catalog().skill_download_url(slug)


def download_skill_md(slug: str) -> Tuple[str, Optional[str]]:
    return shared_catalog().download_skill_md(slug)
