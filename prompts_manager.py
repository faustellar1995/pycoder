import json
import os
from typing import Dict

DEFAULT_PROMPT_TYPE = "system"

PROMPT_FILES: Dict[str, str] = {
    "system": "prompts_system.json",
    "roleplay": "prompts_roleplay.json",
    "tool": "prompts_tool.json",
}

LEGACY_PROMPTS_FILE = "prompts.json"

DEFAULT_PROMPTS_BY_TYPE: Dict[str, Dict[str, str]] = {
    "system": {
        "default": "You are a helpful assistant.",
        "code_expert": "You are an expert programmer. Help the user with code, debugging, and technical explanations.",
        "translator": "You are a professional translator. Help the user translate between languages, maintaining tone and context.",
        "writer": "You are a creative writer. Help the user with writing, editing, and storytelling.",
    },
    "roleplay": {},
    "tool": {
        "default": (
            "你拥有工作区文件工具与 web_search（解析百度/搜狗搜索结果页，无需 API 密钥）。"
            "需要公开资料、版本说明或新闻时用 web_search；本地代码仍以 read_file / grep_file 为准并交叉验证。"
        ),
    },
}


def _resolve_prompt_file(prompt_type: str) -> str:
    """Resolve prompt type to on-disk filename in current working folder."""
    prompt_type = (prompt_type or DEFAULT_PROMPT_TYPE).strip().lower()
    return PROMPT_FILES.get(prompt_type, f"prompts_{prompt_type}.json")


def _default_prompts(prompt_type: str) -> Dict[str, str]:
    defaults = DEFAULT_PROMPTS_BY_TYPE.get(prompt_type, {})
    return defaults.copy()


def load_prompts(prompt_type: str = DEFAULT_PROMPT_TYPE) -> Dict[str, str]:
    """Load prompts by type from file or return defaults if file doesn't exist."""
    prompt_type = (prompt_type or DEFAULT_PROMPT_TYPE).strip().lower()
    prompts_file = _resolve_prompt_file(prompt_type)

    if os.path.exists(prompts_file):
        try:
            with open(prompts_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return _default_prompts(prompt_type)

    # Backward compatibility: migrate legacy single-file prompts to system type.
    if prompt_type == "system" and os.path.exists(LEGACY_PROMPTS_FILE):
        try:
            with open(LEGACY_PROMPTS_FILE, "r", encoding="utf-8") as f:
                prompts = json.load(f)
            save_prompts(prompts, prompt_type="system")
            return prompts
        except (json.JSONDecodeError, IOError):
            return _default_prompts(prompt_type)

    return _default_prompts(prompt_type)


def save_prompts(prompts: Dict[str, str], prompt_type: str = DEFAULT_PROMPT_TYPE) -> None:
    """Save prompts by type to file."""
    prompts_file = _resolve_prompt_file(prompt_type)
    with open(prompts_file, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)


def get_all_prompts(prompt_type: str = DEFAULT_PROMPT_TYPE) -> Dict[str, str]:
    """Get all prompts by type."""
    return load_prompts(prompt_type=prompt_type)


def get_prompt(name: str, prompt_type: str = DEFAULT_PROMPT_TYPE) -> str:
    """Get a specific prompt by name and type."""
    prompts = load_prompts(prompt_type=prompt_type)
    return prompts.get(name, "You are a helpful assistant.")


def add_prompt(name: str, content: str, prompt_type: str = DEFAULT_PROMPT_TYPE) -> None:
    """Add or update a prompt by type."""
    prompts = load_prompts(prompt_type=prompt_type)
    prompts[name] = content
    save_prompts(prompts, prompt_type=prompt_type)


def delete_prompt(name: str, prompt_type: str = DEFAULT_PROMPT_TYPE) -> bool:
    """Delete a prompt by type. Returns True if deleted, False if not found."""
    prompts = load_prompts(prompt_type=prompt_type)
    if name in prompts:
        del prompts[name]
        save_prompts(prompts, prompt_type=prompt_type)
        return True
    return False


def rename_prompt(
    old_name: str,
    new_name: str,
    prompt_type: str = DEFAULT_PROMPT_TYPE,
) -> bool:
    """Rename a prompt by type. Returns True on success."""
    prompts = load_prompts(prompt_type=prompt_type)
    if old_name not in prompts or new_name in prompts:
        return False
    prompts[new_name] = prompts.pop(old_name)
    save_prompts(prompts, prompt_type=prompt_type)
    return True
