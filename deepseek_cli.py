import argparse
import sys
import urllib.error
from pathlib import Path
from typing import Optional

from deepseek_api import (
    build_simple_messages,
    chat_completion,
    explain_http_error,
    get_api_key,
)
from deepseek_harness import HarnessConfig, run_harness


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepSeek 命令行：普通对话或 --tools 本地工具 + 联网搜索。")
    p.add_argument(
        "--tools",
        action="store_true",
        help="启用本地工具（含默认开启的 web_search，可用 --no-web-search 关闭）",
    )
    p.add_argument(
        "--no-web-search",
        action="store_true",
        help="在 --tools 模式下不注册 web_search",
    )
    p.add_argument(
        "--no-run-command",
        action="store_true",
        help="在 --tools 模式下不注册 run_command",
    )
    p.add_argument(
        "--no-stream",
        action="store_true",
        help="关闭流式输出（含 --tools 时每轮对话也不再增量打印）",
    )
    p.add_argument(
        "--model",
        "-m",
        default="flash",
        help="模型模式，与 UI 一致（默认 flash）",
    )
    p.add_argument(
        "--workspace",
        "-w",
        type=Path,
        default=Path.cwd(),
        help="工具模式下的工作区根目录（默认当前目录）",
    )
    p.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="HTTP/HTTPS 代理，例如 http://127.0.0.1:7890；缺省为直连",
    )
    p.add_argument(
        "question",
        nargs="*",
        help="问题文本；缺省则进入交互循环",
    )
    return p.parse_args()


def _one_shot(
    api_key: str,
    question: str,
    *,
    use_tools: bool,
    model_mode: str,
    workspace: Path,
    proxy_url: Optional[str],
    stream: bool,
    allow_run_command: bool,
    enable_web_search: bool,
) -> str:
    messages = build_simple_messages(question)
    if use_tools:

        def _emit(t: str) -> None:
            sys.stdout.write(t)
            sys.stdout.flush()

        cfg = HarnessConfig(
            workspace=workspace.resolve(),
            use_tools=True,
            allow_run_command=allow_run_command,
            enable_web_search=enable_web_search,
            stream=stream,
            proxy_url=proxy_url,
        )
        return run_harness(
            api_key=api_key,
            messages=messages,
            model_mode=model_mode,
            config=cfg,
            on_stream_token=_emit if stream else None,
        )
    return chat_completion(
        api_key=api_key,
        messages=messages,
        model_mode=model_mode,
        proxy_url=proxy_url,
    ) if not stream else _stream_then_join(
        api_key, messages, model_mode, proxy_url
    )


def _stream_then_join(api_key: str, messages, model_mode: str, proxy_url: Optional[str]) -> str:
    from deepseek_api import chat_completion_stream

    chunks: list[str] = []
    for token in chat_completion_stream(
        api_key=api_key,
        messages=messages,
        model_mode=model_mode,
        proxy_url=proxy_url,
    ):
        chunks.append(token)
        sys.stdout.write(token)
        sys.stdout.flush()
    sys.stdout.write("\n")
    return "".join(chunks)


def main() -> int:
    args = _parse_args()
    try:
        api_key = get_api_key()
    except ValueError as exc:
        print(f"Error: {exc}")
        print("Please set DS_KEY first, then run again.")
        return 1

    use_tools = bool(args.tools)
    allow_run_command = use_tools and not args.no_run_command
    enable_web_search = use_tools and not args.no_web_search
    stream = not args.no_stream
    proxy_url = (args.proxy or "").strip() or None
    workspace: Path = args.workspace
    model_mode = str(args.model or "flash").strip() or "flash"

    if args.question:
        q = " ".join(args.question).strip()
        if not q:
            print("Empty question.")
            return 1
        try:
            ans = _one_shot(
                api_key,
                q,
                use_tools=use_tools,
                model_mode=model_mode,
                workspace=workspace,
                proxy_url=proxy_url,
                stream=stream,
                allow_run_command=allow_run_command,
                enable_web_search=enable_web_search,
            )
        except urllib.error.HTTPError as exc:
            print(explain_http_error(exc), file=sys.stderr)
            return 1
        except urllib.error.URLError as exc:
            print(f"Network error: {exc.reason}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Unexpected error: {exc}", file=sys.stderr)
            return 1
        if use_tools and stream:
            sys.stdout.write("\n")
        elif use_tools or args.no_stream:
            print(ans)
        return 0

    print("DeepSeek CLI ready. Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            return 0

        try:
            ans = _one_shot(
                api_key,
                question,
                use_tools=use_tools,
                model_mode=model_mode,
                workspace=workspace,
                proxy_url=proxy_url,
                stream=stream,
                allow_run_command=allow_run_command,
                enable_web_search=enable_web_search,
            )
            if use_tools and stream:
                sys.stdout.write("\n")
            elif use_tools or args.no_stream:
                print(f"\nDeepSeek: {ans}")
            else:
                print()
        except urllib.error.HTTPError as exc:
            print(f"\n{explain_http_error(exc)}")
            continue
        except urllib.error.URLError as exc:
            print(f"\nNetwork error: {exc.reason}")
            continue
        except Exception as exc:
            print(f"\nUnexpected error: {exc}")
            continue

    return 0


if __name__ == "__main__":
    sys.exit(main())
