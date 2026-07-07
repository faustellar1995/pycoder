"""LocalHarness：将对话执行逻辑从 UI 分离。

- 非流式：普通 chat completion
- 流式：SSE 增量 token
- 工具模式：DeepSeek tools/function calling + 本地工具执行（含可选 run_command）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import subprocess

from localharness_api import (
    API_URL,
    PROVIDER_DEEPSEEK,
    StreamInterrupted,
    chat_completion,
    chat_completion_stream,
    run_chat_with_tools,
    validate_messages_for_provider,
)
from workspace_tools import WorkspaceToolSession, openai_tool_specs


def tools_system_hint(*, include_run_command: bool, include_web_search: bool = True) -> str:
    parts = [
        "你可以调用工具完成本地查看与修改：read_file 与 list_directory 对「绝对路径」默认可读/可列本机任意目录（无副作用，类似 type/dir）；"
        "相对路径仍相对工作区根。write_file、search_replace 及 run_command 的 cwd 仅允许工作区根或环境变量 "
        "DEEPSEEK_WORKSPACE_EXTRA_ROOTS（分号分隔的额外根，用于写入 D:/bin 等）。"
        "glob_file_search 与 grep_file 仅扫描工作区根。",
    ]
    if include_web_search:
        parts.append(
            "需要实时资讯、文档版本或公开网页摘要时，可调用 web_search（解析百度/搜狗搜索结果页，无需付费密钥），"
            "query 用简短关键词或问句；可用 DEEPSEEK_WEB_SEARCH_ENGINE=sogou 切换引擎。"
            "返回标题与链接，仍需 read_file 等核对本地代码；DEEPSEEK_DISABLE_WEB_SEARCH=1 可禁用。"
        )
    if include_run_command:
        parts.append(
            "还可调用 run_command：在工作区内以子进程执行命令（shell=False，无管道）。"
            "argv 为字符串数组，例如 [\"python\", \"-m\", \"pytest\", \"-q\"] 或 Windows 下 [\"cmd\", \"/c\", \"dir\"]；"
            "cwd 可选，为相对工作区的子目录，缺省为工作区根；timeout_sec 默认 120、最大 600。"
            "宿主可用环境变量 DEEPSEEK_DISABLE_COMMANDS=1 彻底禁用命令执行。"
        )
    return "\n".join(parts)


@dataclass
class HarnessConfig:
    workspace: Path
    use_tools: bool = False
    allow_run_command: bool = True
    enable_web_search: bool = True
    stream: bool = True
    proxy_url: Optional[str] = None
    api_url: str = API_URL
    provider: str = PROVIDER_DEEPSEEK
    tool_max_rounds: Optional[int] = None


@dataclass
class HarnessResult:
    """run_harness 返回值。工具模式下 messages 含完整 assistant/tool 往返，供 UI 写回上下文。"""

    answer: str
    messages: Optional[List[Dict[str, Any]]] = None


def run_harness(
    *,
    api_key: str,
    messages: List[Dict[str, Any]],
    model_mode: str,
    config: HarnessConfig,
    temperature: float = 0.7,
    timeout: int = 180,
    should_stop: Optional[Callable[[], bool]] = None,
    on_stream_token: Optional[Callable[[str], None]] = None,
) -> HarnessResult:
    """执行一次“发问→得到回答”，按 config 决定是否走工具循环/流式。"""
    vision_err = validate_messages_for_provider(messages, config.provider)
    if vision_err:
        raise ValueError(vision_err)

    if config.use_tools:
        session = WorkspaceToolSession(config.workspace, http_proxy=config.proxy_url)
        tools = openai_tool_specs(
            enable_run_command=config.allow_run_command,
            enable_web_search=config.enable_web_search,
        )

        def _exec(name: str, args: str) -> str:
            return session.execute(name, args)

        msgs_out, answer = run_chat_with_tools(
            api_key,
            messages,
            model_mode,
            tools,
            _exec,
            temperature=temperature,
            timeout=timeout,
            max_rounds=config.tool_max_rounds,
            should_stop=should_stop,
            proxy_url=config.proxy_url,
            api_url=config.api_url,
            provider=config.provider,
            on_stream_token=on_stream_token,
            stream_tokens=config.stream,
        )
        return HarnessResult(answer=answer, messages=msgs_out)

    if config.stream:
        full: List[str] = []
        for token in chat_completion_stream(
            api_key=api_key,
            messages=messages,  # type: ignore[arg-type]
            model_mode=model_mode,
            temperature=temperature,
            timeout=timeout,
            should_stop=should_stop,
            proxy_url=config.proxy_url,
            api_url=config.api_url,
            provider=config.provider,
        ):
            full.append(token)
            if on_stream_token:
                on_stream_token(token)
        return HarnessResult(answer="".join(full))

    return HarnessResult(
        answer=chat_completion(
            api_key=api_key,
            messages=messages,  # type: ignore[arg-type]
            model_mode=model_mode,
            temperature=temperature,
            timeout=timeout,
            proxy_url=config.proxy_url,
            api_url=config.api_url,
            provider=config.provider,
        )
    )


def _run_git(args: List[str], *, cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        timeout=float(timeout),
    )


def _split_commit_message(text: str) -> Tuple[str, Optional[str]]:
    t = (text or "").strip("\n").strip()
    if not t:
        return "", None
    lines = t.splitlines()
    subject = lines[0].strip()
    rest = "\n".join(lines[1:]).strip()
    if rest.startswith("\n"):
        rest = rest.lstrip("\n")
    body = rest if rest else None
    return subject, body


def ai_git_commit(
    *,
    api_key: str,
    model_mode: str,
    workspace: Path,
    temperature: float = 0.2,
    timeout: int = 180,
    max_diff_chars: int = 180_000,
) -> str:
    """兼容旧入口：先 generate 再 do commit。"""
    subject, body, log = generate_git_commit_message(
        api_key=api_key,
        model_mode=model_mode,
        workspace=workspace,
        temperature=temperature,
        timeout=timeout,
        max_diff_chars=max_diff_chars,
    )
    if not subject:
        return log
    result = do_git_commit(workspace=workspace, subject=subject, body=body, timeout=timeout)
    return f"{log}\n\n---\n\n{result}".strip()


def generate_git_commit_message(
    *,
    api_key: str,
    model_mode: str,
    workspace: Path,
    proxy_url: Optional[str] = None,
    api_url: str = API_URL,
    provider: str = PROVIDER_DEEPSEEK,
    temperature: float = 0.2,
    timeout: int = 180,
    max_diff_chars: int = 180_000,
) -> Tuple[str, Optional[str], str]:
    """
    生成 commit message（不提交）。
    返回 (subject, body, log_text)。
    """
    ws = workspace.resolve()
    probe = _run_git(["rev-parse", "--is-inside-work-tree"], cwd=ws, timeout=30)
    if probe.returncode != 0:
        return "", None, f"[commit] 当前目录不是 git 仓库：\n{probe.stderr or probe.stdout}"

    add = _run_git(["add", "-A"], cwd=ws, timeout=timeout)
    if add.returncode != 0:
        return "", None, f"[commit] git add 失败：\n{add.stderr or add.stdout}"

    status = _run_git(["status", "--porcelain"], cwd=ws, timeout=timeout)
    if status.returncode != 0:
        return "", None, f"[commit] git status 失败：\n{status.stderr or status.stdout}"
    if not (status.stdout or "").strip():
        return "", None, "[commit] 没有可提交的变更（working tree clean）。"

    diff = _run_git(["diff", "--staged"], cwd=ws, timeout=timeout)
    if diff.returncode != 0:
        return "", None, f"[commit] git diff --staged 失败：\n{diff.stderr or diff.stdout}"
    diff_text = diff.stdout or ""
    if len(diff_text) > max_diff_chars:
        diff_text = diff_text[:max_diff_chars] + "\n... [diff 已截断]"

    sys = (
        "你是一个资深工程师。请基于给定的 git status 与 git diff --staged，生成一个高质量的 git commit message。\n"
        "要求：\n"
        "- 只输出 commit message 本身，不要加解释、引号、markdown。\n"
        "- 默认使用中文撰写（除非 diff 中出现必须保留的英文专有名词）。\n"
        "- 第一行是简短 subject（<=72 字符），使用动词开头。\n"
        "- 如需补充信息，用空行后再写 body（每行尽量 <=72 字符）。\n"
        "- 不要包含敏感信息。\n"
    )
    user = f"## git status --porcelain\n{status.stdout}\n\n## git diff --staged\n{diff_text}\n"
    msg = chat_completion(
        api_key=api_key,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        model_mode=model_mode,
        temperature=temperature,
        timeout=timeout,
        proxy_url=proxy_url,
        api_url=api_url,
        provider=provider,
    )
    subject, body = _split_commit_message(msg)
    if not subject:
        return "", None, "[commit] 模型没有生成有效的 commit message。"
    log = f"[commit] 已生成 message：\n{subject}\n\n{body or ''}".rstrip()
    return subject, body, log


def do_git_commit(
    *,
    workspace: Path,
    subject: str,
    body: Optional[str] = None,
    timeout: int = 180,
) -> str:
    """执行 git commit（假定已 git add -A）。"""
    ws = workspace.resolve()
    probe = _run_git(["rev-parse", "--is-inside-work-tree"], cwd=ws, timeout=30)
    if probe.returncode != 0:
        return f"[commit] 当前目录不是 git 仓库：\n{probe.stderr or probe.stdout}"

    add = _run_git(["add", "-A"], cwd=ws, timeout=timeout)
    if add.returncode != 0:
        return f"[commit] git add 失败：\n{add.stderr or add.stdout}"

    status = _run_git(["status", "--porcelain"], cwd=ws, timeout=timeout)
    if status.returncode != 0:
        return f"[commit] git status 失败：\n{status.stderr or status.stdout}"
    if not (status.stdout or "").strip():
        return "[commit] 没有可提交的变更（working tree clean）。"

    subj = (subject or "").strip()
    if not subj:
        return "[commit] subject 为空，拒绝提交。"

    cmd = ["commit", "-m", subj]
    b = (body or "").strip()
    if b:
        cmd += ["-m", b]
    commit = _run_git(cmd, cwd=ws, timeout=timeout)
    if commit.returncode != 0:
        return f"[commit] git commit 失败：\n{commit.stderr or commit.stdout}"
    return f"[commit] 已提交：\n{commit.stdout or commit.stderr}".strip()

