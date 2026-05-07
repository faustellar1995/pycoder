"""本地工作区工具：供 DeepSeek function calling 执行（路径限制在工作区内）。"""

from __future__ import annotations

import fnmatch
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def default_workspace_root() -> Path:
    return Path(os.getenv("DEEPSEEK_WORKSPACE", os.getcwd())).resolve()


def openai_tool_specs(
    *,
    enable_run_command: bool = True,
    enable_web_search: bool = True,
) -> List[Dict[str, Any]]:
    """OpenAI/DeepSeek `tools` 列表格式。"""
    specs: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "读取工作区内单个 UTF-8 文本文件的全部内容。路径为相对工作区根目录。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "相对工作区根目录的文件路径，使用正斜杠。",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "覆盖写入工作区内文本文件（自动创建父目录）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "相对工作区根目录的文件路径。",
                        },
                        "content": {"type": "string", "description": "完整文件内容。"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "列出工作区内某目录下的文件与子目录名称（非递归）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "相对工作区根目录的目录路径，空字符串表示根目录。",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_replace",
                "description": "在文本文件中将唯一出现的 old_string 替换为 new_string（用于精确局部编辑）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对工作区根目录的文件路径。"},
                        "old_string": {"type": "string", "description": "要被替换的原始片段，必须在文件中唯一存在。"},
                        "new_string": {"type": "string", "description": "替换后的新内容。"},
                    },
                    "required": ["path", "old_string", "new_string"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "glob_file_search",
                "description": "在工作区根目录下按 glob 模式搜索文件名（如 *.py），返回最多 50 条相对路径。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "glob_pattern": {
                            "type": "string",
                            "description": "glob 模式，例如 **/*.py 或 *.md",
                        }
                    },
                    "required": ["glob_pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep_file",
                "description": (
                    "在工作区内按 glob 模式搜索文件内容（类似 grep -r）。"
                    "支持正则表达式或简单字符串匹配，返回匹配行及行号。"
                    "适合代码搜索、日志分析、查找特定调用等场景。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "搜索模式（默认视为正则表达式，若 use_regex=false 则为纯文本子串匹配）。",
                        },
                        "glob": {
                            "type": "string",
                            "description": "文件 glob 模式，例如 '**/*.py'、'*.md'、'logs/*.log'、'src/**/*.ts'。",
                        },
                        "use_regex": {
                            "type": "boolean",
                            "description": "是否将 pattern 视为正则表达式；false 则 pattern 按纯文本子串搜索。默认 true。",
                        },
                        "max_matches": {
                            "type": "integer",
                            "description": "最多返回多少条匹配行，默认 100。防止输出过大。",
                        },
                        "max_file_size": {
                            "type": "integer",
                            "description": "跳过超过此大小的文件（字节），默认 1_048_576（1 MiB）。",
                        },
                    },
                    "required": ["pattern", "glob"],
                },
            },
        },
    ]

    if enable_web_search:
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "联网搜索（百度/搜狗网页结果，无需 API 密钥）：获取与查询相关的标题与链接。"
                        "环境变量 DEEPSEEK_WEB_SEARCH_ENGINE=baidu|sogou 选择引擎，"
                        "DEEPSEEK_WEB_SEARCH_FALLBACK=1 可在无结果时自动换引擎。结果来自公开搜索结果页解析，仅供辅助。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索关键词或完整问句（建议中文或英文关键词）。",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "最多返回多少条网页结果，默认 8，最大 15。",
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        )

    if enable_run_command:
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": (
                        "在工作区内启动子进程执行命令（不使用系统 shell 解析整条字符串，避免注入）。"
                        "参数 argv 为程序名及参数的数组，例如 Windows: [\"cmd\", \"/c\", \"dir\"]，"
                        "或 [\"python\", \"-m\", \"pytest\", \"-q\"]。"
                        "工作目录默认为工作区根；可通过 cwd 传入相对工作区的子目录。"
                        "适合运行测试、格式化、构建脚本等；长任务请合理设置 timeout_sec。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "argv": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "可执行文件及参数列表，不得为空。",
                            },
                            "cwd": {
                                "type": "string",
                                "description": "相对工作区的目录作为子进程当前目录；默认 \".\" 表示工作区根。",
                            },
                            "timeout_sec": {
                                "type": "integer",
                                "description": "超时秒数，默认 120，最大 600。",
                            },
                        },
                        "required": ["argv"],
                    },
                },
            }
        )

    return specs


class WorkspaceSandboxError(Exception):
    pass


class WorkspaceToolSession:
    """将相对路径解析到单一根目录下并执行工具。"""

    def __init__(self, root: Optional[Path] = None, *, http_proxy: Optional[str] = None):
        self.root = (root or default_workspace_root()).resolve()
        self._http_proxy = (http_proxy or "").strip() or None

    def _resolve(self, rel: str) -> Path:
        if not isinstance(rel, str):
            raise WorkspaceSandboxError("path 必须是字符串")
        cleaned = rel.replace("\\", "/").strip().lstrip("/")
        candidate = (self.root / cleaned).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceSandboxError(f"路径越界（必须在工作区内）: {rel}") from exc
        return candidate

    def execute(self, name: str, arguments: str) -> str:
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as exc:
            return f"[tool error] 参数不是合法 JSON: {exc}"

        try:
            if name == "read_file":
                return self._read_file(args.get("path", ""))
            if name == "write_file":
                return self._write_file(args.get("path", ""), args.get("content", ""))
            if name == "list_directory":
                return self._list_directory(args.get("path", ""))
            if name == "search_replace":
                return self._search_replace(
                    args.get("path", ""),
                    args.get("old_string", ""),
                    args.get("new_string", ""),
                )
            if name == "glob_file_search":
                return self._glob_file_search(args.get("glob_pattern", ""))
            if name == "grep_file":
                return self._grep_file(
                    args.get("pattern", ""),
                    args.get("glob", ""),
                    use_regex=args.get("use_regex", True),
                    max_matches=args.get("max_matches", 100),
                    max_file_size=args.get("max_file_size", 1_048_576),
                )
            if name == "run_command":
                return self._run_command(args)
            if name == "web_search":
                from web_search import run_web_search

                q = str(args.get("query") or "").strip()
                mr = args.get("max_results", 8)
                try:
                    mr_int = int(mr)
                except (TypeError, ValueError):
                    mr_int = 8
                return run_web_search(q, max_results=mr_int, proxy_url=self._http_proxy)
            return f"[tool error] 未知工具: {name}"
        except WorkspaceSandboxError as exc:
            return f"[tool error] {exc}"
        except OSError as exc:
            return f"[tool error] 文件系统错误: {exc}"

    def _read_file(self, rel: str) -> str:
        path = self._resolve(rel)
        if not path.is_file():
            return f"[tool error] 不是文件或不存在: {rel}"
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"[tool error] 非 UTF-8 文本，拒绝读取: {rel}"
        # 避免超大文件撑爆上下文
        max_chars = 200_000
        if len(text) > max_chars:
            return text[:max_chars] + f"\n\n... [已截断，原文件约 {len(text)} 字符]"
        return text

    def _write_file(self, rel: str, content: str) -> str:
        path = self._resolve(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        return f"已写入 {rel}（{len(content)} 字符）"

    def _list_directory(self, rel: str) -> str:
        rel = rel.strip().replace("\\", "/")
        path = self.root if not rel else self._resolve(rel)
        if not path.is_dir():
            return f"[tool error] 不是目录或不存在: {rel or '.'}"
        names = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = []
        for p in names:
            suffix = "/" if p.is_dir() else ""
            lines.append(f"{p.name}{suffix}")
        return "\n".join(lines) if lines else "(空目录)"

    def _search_replace(self, rel: str, old: str, new: str) -> str:
        path = self._resolve(rel)
        if not path.is_file():
            return f"[tool error] 不是文件或不存在: {rel}"
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            return "[tool error] old_string 在文件中未找到"
        if count > 1:
            return f"[tool error] old_string 出现 {count} 次，必须唯一"
        path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")
        return f"已在 {rel} 中完成唯一替换"

    def _glob_file_search(self, pattern: str) -> str:
        if not pattern or not isinstance(pattern, str):
            return "[tool error] glob_pattern 无效"
        matches: List[str] = []
        for p in self.root.rglob("*"):
            try:
                rel = p.relative_to(self.root)
            except ValueError:
                continue
            if p.is_file() and fnmatch.fnmatch(rel.as_posix().lower(), pattern.lower()):
                matches.append(rel.as_posix())
            if len(matches) >= 50:
                break
        matches.sort()
        return "\n".join(matches) if matches else "(无匹配)"

    def _grep_file(
        self,
        pattern: str,
        glob_pattern: str,
        use_regex: bool = True,
        max_matches: int = 100,
        max_file_size: int = 1_048_576,
    ) -> str:
        """纯 Python 实现的内容搜索（跨平台，不依赖外部 grep 命令）。"""
        import re as _re_module

        if not pattern or not glob_pattern:
            return "[tool error] pattern 和 glob 均不能为空"

        # 编译正则
        if use_regex:
            try:
                re_obj = _re_module.compile(pattern, _re_module.MULTILINE)
            except _re_module.error as exc:
                return f"[tool error] 正则表达式无效: {exc}"
        else:
            re_obj = _re_module.compile(_re_module.escape(pattern), _re_module.MULTILINE)

        matched_lines: list[str] = []
        skipped_binary = 0
        skipped_size = 0
        scanned = 0

        for p in self.root.rglob("*"):
            if not p.is_file():
                continue
            try:
                rel = p.relative_to(self.root).as_posix()
            except ValueError:
                continue
            if not fnmatch.fnmatch(rel.lower(), glob_pattern.lower()):
                continue

            # 跳过超大文件
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size > max_file_size:
                skipped_size += 1
                continue

            if len(matched_lines) >= max_matches:
                break
            scanned += 1

            # 尝试以 UTF-8 读入，兜底用 latin-1
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                try:
                    text = p.read_text(encoding="latin-1")
                except OSError:
                    skipped_binary += 1
                    continue

            for lineno, line in enumerate(text.splitlines(), 1):
                if len(matched_lines) >= max_matches:
                    break
                if re_obj.search(line):
                    matched_lines.append(f"{rel}:{lineno}:{line.rstrip()}")

        if not matched_lines:
            summary = f"(无匹配，已扫描 {scanned} 个文件)"
            if skipped_binary:
                summary += f"，跳过 {skipped_binary} 个二进制/不可读文件"
            if skipped_size:
                summary += f"，跳过 {skipped_size} 个超大文件"
            return summary

        result = "\n".join(matched_lines)
        summary = f"匹配 {len(matched_lines)} 行 (上限 {max_matches})，扫描 {scanned} 个文件"
        if skipped_binary:
            summary += f"，跳过 {skipped_binary} 个二进制文件"
        if skipped_size:
            summary += f"，跳过 {skipped_size} 个超大文件"

        max_result_chars = 100_000
        if len(result) > max_result_chars:
            result = result[:max_result_chars] + "\n... [已截断]"
        return f"{summary}\n\n{result}"

    def _run_command(self, args: Dict[str, Any]) -> str:
        flag = os.getenv("DEEPSEEK_DISABLE_COMMANDS", "").strip().lower()
        if flag in ("1", "true", "yes", "on"):
            return "[tool error] 已禁用命令执行（环境变量 DEEPSEEK_DISABLE_COMMANDS）"

        raw_argv = args.get("argv")
        if not isinstance(raw_argv, list) or len(raw_argv) == 0:
            return "[tool error] argv 必须为非空数组"
        cmd: List[str] = []
        for i, x in enumerate(raw_argv):
            if not isinstance(x, str):
                return f"[tool error] argv[{i}] 必须是字符串"
            if "\x00" in x:
                return "[tool error] argv 含非法字符"
            cmd.append(x)
        if sum(len(s) for s in cmd) > 32_000:
            return "[tool error] 参数总长度过大"

        raw_timeout = args.get("timeout_sec", 120)
        try:
            timeout_sec = int(raw_timeout)
        except (TypeError, ValueError):
            timeout_sec = 120
        timeout_sec = max(1, min(timeout_sec, 600))

        cwd_rel = args.get("cwd", ".")
        if cwd_rel is None:
            cwd_rel = "."
        if not isinstance(cwd_rel, str):
            return "[tool error] cwd 必须是字符串"
        cwd_path = self._resolve(cwd_rel.strip().replace("\\", "/"))
        if not cwd_path.is_dir():
            return f"[tool error] cwd 不是目录: {cwd_rel}"

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd_path),
                capture_output=True,
                text=True,
                timeout=float(timeout_sec),
                shell=False,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return f"[tool error] 超时 ({timeout_sec}s): {cmd!r}"
        except OSError as exc:
            return f"[tool error] 无法启动进程: {exc}"

        out = proc.stdout or ""
        err = proc.stderr or ""
        max_chars = 120_000
        if len(out) > max_chars:
            out = out[:max_chars] + "\n... [stdout 已截断]"
        if len(err) > max_chars:
            err = err[:max_chars] + "\n... [stderr 已截断]"

        lines = [
            f"exit_code: {proc.returncode}",
            f"command: {cmd!r}",
            f"cwd: {cwd_path}",
            "--- stdout ---",
            out if out else "(empty)",
            "--- stderr ---",
            err if err else "(empty)",
        ]
        return "\n".join(lines)


ToolExecutor = Callable[[str, str], str]
