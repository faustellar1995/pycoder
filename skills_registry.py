"""本地 SKILL.md 发现、解析与自动选择（对齐 OpenClaw/IronClaw 风格）。"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Tuple

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

# 本地扫描缓存（减轻频繁遍历目录；可按需 clear）
_DISCOVERY_LOCK = threading.Lock()
_DISCOVERY_CACHE: Optional[Tuple[float, Tuple[str, ...], int, Tuple["LoadedSkill", ...]]] = None
DEFAULT_DISCOVERY_CACHE_TTL = 45.0

# 与 ironclaw 校验类似的宽松检测
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")

def normalize_skill_identifier(value: str) -> Optional[str]:
    """将任意字符串规范化为合法 skill name（尽量对齐 ironclaw 的规则）。"""
    raw = (value or "").strip()
    if not raw:
        return None
    # 常见：把空白转为 '-'
    raw = re.sub(r"\s+", "-", raw)
    # 过滤为允许字符集合
    cleaned = "".join(ch for ch in raw if re.match(r"[a-zA-Z0-9._-]", ch))
    cleaned = cleaned.strip("-.")
    if not cleaned:
        return None
    # 限长并确保首字符为字母/数字
    cleaned = cleaned[:64]
    if not re.match(r"^[a-zA-Z0-9]", cleaned):
        cleaned = re.sub(r"^[^a-zA-Z0-9]+", "", cleaned)
    cleaned = cleaned[:64]
    return cleaned if cleaned and _NAME_RE.match(cleaned) else None


def rewrite_skill_md_name_for_install(content: str, *, preferred: Optional[str] = None) -> str:
    """
    安装时修复不合法的 name（例如包含空格）。
    优先使用 preferred（如 catalog slug），否则从 frontmatter.slug/name 推导。
    """
    manifest, body = _split_frontmatter(content)
    slug = str(manifest.get("slug") or "").strip()
    name = str(manifest.get("name") or "").strip()
    candidate = normalize_skill_identifier(preferred or "") or normalize_skill_identifier(slug) or normalize_skill_identifier(name)
    if not candidate:
        candidate = "skill"
    manifest["name"] = candidate
    if yaml is None:
        raise RuntimeError("缺少 PyYAML，无法重写 SKILL.md")
    front = yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{front}\n---\n\n{body.rstrip()}\n"


@dataclass
class LoadedSkill:
    """单个已加载技能。"""

    name: str
    source_path: Path
    description: str
    prompt_body: str
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    raw_manifest: Dict[str, Any] = field(default_factory=dict)
    # ironclaw selector：正则模式在加载时预编译，避免每轮用户消息重复 compile
    _pattern_re: List[Pattern[str]] = field(default_factory=list, repr=False, compare=False)

    def score_against_message(self, text: str) -> int:
        """确定性打分（简化版 ironclaw prefilter）：关键词/标签/预编译正则。"""
        t = text.lower()
        score = 0
        for kw in self.keywords:
            kl = kw.lower()
            if kl and kl in t:
                score += 5
        for tag in self.tags:
            tl = tag.lower()
            if tl and tl in t:
                score += 3
        if self.name.lower() in t:
            score += 10
        desc = self.description.lower()
        if desc:
            for word in re.findall(r"[\w\u4e00-\u9fff]+", desc):
                if len(word) > 1 and word in t:
                    score += 2
        pat_hits = 0
        for cre in self._pattern_re:
            if cre.search(text):
                pat_hits += 1
                score += 20
                if pat_hits >= 2:
                    break
        return score


def default_skill_scan_dirs() -> List[Path]:
    base = Path(
        os.getenv("LOCALHARNESS_SKILLS_HOME") or os.getenv("DEEPSEEK_SKILLS_HOME", "")
    ).expanduser()
    dirs: List[Path] = []
    extra = os.getenv("LOCALHARNESS_SKILL_DIRS") or os.getenv("DEEPSEEK_SKILL_DIRS", "")
    if extra:
        for part in extra.split(os.pathsep):
            p = Path(part.strip()).expanduser()
            if p:
                dirs.append(p.resolve())
    home_skills = Path.home() / ".localharness" / "skills"
    legacy_home = Path.home() / ".deepseek-assistant" / "skills"
    dirs.append(home_skills)
    if legacy_home != home_skills:
        dirs.append(legacy_home)
    cwd_skills = Path.cwd() / "skills"
    dirs.append(cwd_skills.resolve())
    if base:
        dirs.insert(0, base.resolve())
    # 去重保持顺序
    seen = set()
    out: List[Path] = []
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _split_frontmatter(content: str) -> tuple[Dict[str, Any], str]:
    if yaml is None:
        raise RuntimeError(
            "缺少 YAML 解析库：请先安装 PyYAML（pip install PyYAML 或 pip install -r requirements.txt）"
        )
    text = content.replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("\ufeff"):
        text = text[1:]
    if not text.startswith("---"):
        raise ValueError("缺少 YAML frontmatter（应以 --- 开头）")
    rest = text[3:].lstrip("\n")
    end = rest.find("\n---")
    if end < 0:
        raise ValueError("缺少 closing --- 的 YAML frontmatter")
    yaml_src = rest[:end]
    body = rest[end + 4 :].lstrip("\n")
    data = yaml.safe_load(yaml_src) or {}
    if not isinstance(data, dict):
        raise ValueError("frontmatter 必须是 YAML 对象")
    return data, body


def parse_skill_md(path: Path, content: str) -> LoadedSkill:
    manifest, body = _split_frontmatter(content)
    name = str(manifest.get("name") or "").strip()
    if not name or not _NAME_RE.match(name):
        raise ValueError(f"无效 skill name: {name!r}")
    desc = str(manifest.get("description") or "").strip()
    activation = manifest.get("activation") or {}
    if not isinstance(activation, dict):
        activation = {}
    keywords = activation.get("keywords") or manifest.get("keywords") or []
    tags = activation.get("tags") or manifest.get("tags") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    if isinstance(tags, str):
        tags = [tags]
    keywords = [str(k) for k in keywords if str(k).strip()]
    tags = [str(t) for t in tags if str(t).strip()]
    patterns_raw = activation.get("patterns") or []
    if isinstance(patterns_raw, str):
        patterns_raw = [patterns_raw]
    compiled: List[Pattern[str]] = []
    for pr in patterns_raw:
        p = str(pr).strip()
        if not p:
            continue
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error:
            continue
    return LoadedSkill(
        name=name,
        source_path=path.resolve(),
        description=desc,
        prompt_body=body.strip(),
        keywords=keywords,
        tags=tags,
        raw_manifest=manifest,
        _pattern_re=compiled,
    )


def clear_skill_discovery_cache() -> None:
    """强制下次重新扫描磁盘（例如安装新技能、刷新列表）。"""
    global _DISCOVERY_CACHE
    with _DISCOVERY_LOCK:
        _DISCOVERY_CACHE = None


def _discover_skills_impl(roots: List[Path], max_depth: int) -> List[LoadedSkill]:
    found: Dict[str, LoadedSkill] = {}

    def walk(root: Path, depth: int) -> None:
        if depth > max_depth or not root.is_dir():
            return
        skill_md = root / "SKILL.md"
        if skill_md.is_file():
            try:
                raw = skill_md.read_text(encoding="utf-8")
                skill = parse_skill_md(skill_md, raw)
                if skill.name not in found:
                    found[skill.name] = skill
            except Exception:
                pass
        try:
            for child in root.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    walk(child, depth + 1)
        except OSError:
            pass

    for r in roots:
        if r.is_dir():
            walk(r.resolve(), 0)

    return sorted(found.values(), key=lambda s: s.name.lower())


def discover_skills(
    scan_dirs: Optional[List[Path]] = None,
    max_depth: int = 4,
    *,
    use_cache: bool = True,
    cache_ttl: float = DEFAULT_DISCOVERY_CACHE_TTL,
) -> List[LoadedSkill]:
    """在若干根目录下递归查找 **/SKILL.md（带短期内存缓存）。"""
    global _DISCOVERY_CACHE
    roots = scan_dirs if scan_dirs is not None else default_skill_scan_dirs()
    key_roots = tuple(str(p.resolve()) for p in roots if p.is_dir())
    now = time.monotonic()
    if use_cache:
        with _DISCOVERY_LOCK:
            cached = _DISCOVERY_CACHE
            if cached is not None:
                t0, k, d, skills = cached
                if k == key_roots and d == max_depth and now - t0 < cache_ttl:
                    return list(skills)

    skills_list = _discover_skills_impl(roots, max_depth)
    with _DISCOVERY_LOCK:
        _DISCOVERY_CACHE = (now, key_roots, max_depth, tuple(skills_list))
    return skills_list


def select_skills_for_message(
    message: str,
    catalog: List[LoadedSkill],
    max_skills: int = 5,
    min_score: int = 3,
) -> List[LoadedSkill]:
    """根据用户输入自动挑选技能（无 LLM）。"""
    scored = [(s.score_against_message(message), s) for s in catalog]
    scored.sort(key=lambda x: (-x[0], x[1].name.lower()))
    picked: List[LoadedSkill] = []
    for score, s in scored:
        if score < min_score:
            continue
        picked.append(s)
        if len(picked) >= max_skills:
            break
    return picked


def build_skills_system_addon(skills: List[LoadedSkill]) -> str:
    """注入 system 提示的片段（类似 ironclaw Active Skills）。"""
    if not skills:
        return ""
    lines = ["## Active Skills", ""]
    for s in skills:
        lines.append(f"### Skill: {s.name}")
        if s.description:
            lines.append(s.description)
        lines.append("")
        lines.append(s.prompt_body)
        lines.append("")
    return "\n".join(lines).strip()
