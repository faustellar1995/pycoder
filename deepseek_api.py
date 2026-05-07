import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

API_URL = "https://api.deepseek.com/chat/completions"
MODEL_ALIASES: Dict[str, str] = {
    "flash": "deepseek-v4-flash",
    "pro": "deepseek-v4-pro",
}


class StreamInterrupted(Exception):
    """Raised when stream is interrupted by user."""


def get_api_key() -> str:
    api_key = os.getenv("DS_KEY", "").strip()
    if not api_key:
        raise ValueError("DS_KEY environment variable is not set.")
    return api_key


def resolve_model(model_mode: str) -> str:
    key = model_mode.strip().lower()
    return MODEL_ALIASES.get(key, model_mode)


def _build_payload(
    messages: List[Dict[str, str]],
    model_mode: str,
    temperature: float,
    stream: bool,
) -> Dict[str, object]:
    return {
        "model": resolve_model(model_mode),
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }


def _post_json(payload: Dict[str, object], api_key: str, timeout: int):
    return _post_json_with_proxy(payload, api_key, timeout, proxy_url=None)


def _build_opener(proxy_url: Optional[str]):
    if not proxy_url:
        return urllib.request.build_opener()
    proxy = {"http": proxy_url, "https": proxy_url}
    return urllib.request.build_opener(urllib.request.ProxyHandler(proxy))


def _request_headers(api_key: str) -> Dict[str, str]:
    """统一请求头：禁止 gzip，避免 urllib 对流式 SSE 先解压再整块读出（表现为「流式失效」）。"""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept-Encoding": "identity",
    }


def _post_json_with_proxy(
    payload: Dict[str, object],
    api_key: str,
    timeout: int,
    *,
    proxy_url: Optional[str],
):
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=_request_headers(api_key),
        method="POST",
    )
    opener = _build_opener(proxy_url)
    return opener.open(request, timeout=timeout)


def chat_completion(
    api_key: str,
    messages: List[Dict[str, str]],
    model_mode: str = "flash",
    temperature: float = 0.7,
    timeout: int = 60,
    proxy_url: Optional[str] = None,
) -> str:
    payload = _build_payload(messages, model_mode, temperature, stream=False)
    with _post_json_with_proxy(payload, api_key, timeout, proxy_url=proxy_url) as response:
        data = json.loads(response.read().decode("utf-8"))

    choices = data.get("choices", [])
    if not choices:
        raise ValueError(f"Unexpected API response: {data}")

    message = choices[0].get("message", {})
    content = message.get("content")
    if not content:
        raise ValueError(f"Empty response content: {data}")
    return content.strip()


def _normalize_delta_piece(val: Any) -> str:
    """将 delta 里的文本片段统一成 str（兼容 OpenAI 风格 content 数组等）。"""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        parts: List[str] = []
        for item in val:
            if isinstance(item, dict):
                t = item.get("text")
                if t is not None:
                    parts.append(str(t))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(val)


def chat_completion_stream(
    api_key: str,
    messages: List[Dict[str, str]],
    model_mode: str = "flash",
    temperature: float = 0.7,
    timeout: int = 120,
    should_stop: Optional[Callable[[], bool]] = None,
    proxy_url: Optional[str] = None,
) -> Iterator[str]:
    """流式输出；消费 delta 内 reasoning / content 等字段。

    思考模式往往先出 reasoning_content（或 reasoning），再出 content；只读其一可能长时间无输出。
    请求使用 Accept-Encoding: identity，避免 gzip + urllib 把整段 SSE 缓冲后才解压（界面表现为非流式）。
    """
    payload = _build_payload(messages, model_mode, temperature, stream=True)
    with _post_json_with_proxy(payload, api_key, timeout, proxy_url=proxy_url) as response:
        while True:
            raw_line = response.readline()
            if not raw_line:
                break
            if should_stop and should_stop():
                raise StreamInterrupted("Stream interrupted by user")

            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue

            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                break

            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = event.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta") or {}
            # 思考链（字段名以服务端为准）
            r_piece = delta.get("reasoning_content")
            if r_piece is None:
                r_piece = delta.get("reasoning")
            r = _normalize_delta_piece(r_piece)
            if r:
                yield r
            c = _normalize_delta_piece(delta.get("content"))
            if c:
                yield c


def build_simple_messages(
    question: str,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    system_content = system_prompt or "You are a helpful assistant."
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]


def add_user_message(messages: List[Dict[str, str]], text: str) -> List[Dict[str, str]]:
    return messages + [{"role": "user", "content": text}]


def add_assistant_message(messages: List[Dict[str, str]], text: str) -> List[Dict[str, str]]:
    return messages + [{"role": "assistant", "content": text}]


def explain_http_error(exc: urllib.error.HTTPError) -> str:
    body = exc.read().decode("utf-8", errors="replace")
    return f"DeepSeek HTTP error {exc.code}: {body}"


def _post_json_return_dict(payload: Dict[str, Any], api_key: str, timeout: int) -> Dict[str, Any]:
    return _post_json_return_dict_with_proxy(payload, api_key, timeout, proxy_url=None)


def _post_json_return_dict_with_proxy(
    payload: Dict[str, Any],
    api_key: str,
    timeout: int,
    *,
    proxy_url: Optional[str],
) -> Dict[str, Any]:
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=_request_headers(api_key),
        method="POST",
    )
    opener = _build_opener(proxy_url)
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _build_tool_payload(
    messages: List[Dict[str, Any]],
    model_mode: str,
    temperature: float,
    stream: bool,
    tools: List[Dict[str, Any]],
    tool_choice: str = "auto",
) -> Dict[str, Any]:
    return {
        "model": resolve_model(model_mode),
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
        "tools": tools,
        "tool_choice": tool_choice,
    }


def chat_completion_message(
    api_key: str,
    messages: List[Dict[str, Any]],
    model_mode: str = "flash",
    temperature: float = 0.7,
    timeout: int = 120,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    proxy_url: Optional[str] = None,
) -> Dict[str, Any]:
    """非流式请求，返回 API JSON 根对象（用于工具调用）。"""
    assert tools is not None
    payload = _build_tool_payload(messages, model_mode, temperature, False, tools, tool_choice)
    return _post_json_return_dict_with_proxy(payload, api_key, timeout, proxy_url=proxy_url)


def chat_completion_message_stream(
    api_key: str,
    messages: List[Dict[str, Any]],
    model_mode: str,
    temperature: float,
    timeout: int,
    tools: List[Dict[str, Any]],
    *,
    tool_choice: str = "auto",
    proxy_url: Optional[str] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    on_stream_token: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """单次 stream=true + tools 请求，解析 SSE 直至 [DONE]，组装与非流式相同的 assistant message。"""
    payload = _build_tool_payload(messages, model_mode, temperature, True, tools, tool_choice)
    reasoning_parts: List[str] = []
    content_parts: List[str] = []
    tool_calls_acc: Dict[int, Dict[str, Any]] = {}

    with _post_json_with_proxy(payload, api_key, timeout, proxy_url=proxy_url) as response:
        while True:
            raw_line = response.readline()
            if not raw_line:
                break
            if should_stop and should_stop():
                raise StreamInterrupted("Stream interrupted by user")

            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue

            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                break

            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = event.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta") or {}

            r_piece = delta.get("reasoning_content")
            if r_piece is None:
                r_piece = delta.get("reasoning")
            rs = _normalize_delta_piece(r_piece)
            if rs:
                reasoning_parts.append(rs)
                if on_stream_token:
                    on_stream_token(rs)

            cs = _normalize_delta_piece(delta.get("content"))
            if cs:
                content_parts.append(cs)
                if on_stream_token:
                    on_stream_token(cs)

            for tc in delta.get("tool_calls") or []:
                idx = int(tc.get("index", 0))
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc.get("id"):
                    tool_calls_acc[idx]["id"] = tc["id"]
                if tc.get("type"):
                    tool_calls_acc[idx]["type"] = tc["type"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    tool_calls_acc[idx]["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    tool_calls_acc[idx]["function"]["arguments"] += fn["arguments"]

    reasoning_joined = "".join(reasoning_parts)
    content_joined = "".join(content_parts)

    tool_calls_list: List[Dict[str, Any]] = []
    for idx in sorted(tool_calls_acc.keys()):
        tc = tool_calls_acc[idx]
        tid = str(tc.get("id") or "").strip()
        name = str((tc.get("function") or {}).get("name") or "").strip()
        args = str((tc.get("function") or {}).get("arguments") or "")
        if not tid or not name:
            continue
        tool_calls_list.append(
            {
                "id": tid,
                "type": tc.get("type") or "function",
                "function": {"name": name, "arguments": args},
            }
        )

    msg: Dict[str, Any] = {"role": "assistant"}
    if reasoning_joined:
        msg["reasoning_content"] = reasoning_joined

    if tool_calls_list:
        msg["tool_calls"] = tool_calls_list
        msg["content"] = content_joined if content_joined else None
    else:
        msg["content"] = content_joined

    return msg


def run_chat_with_tools(
    api_key: str,
    messages: List[Dict[str, Any]],
    model_mode: str,
    tools: List[Dict[str, Any]],
    tool_executor: Callable[[str, str], str],
    *,
    temperature: float = 0.7,
    timeout: int = 180,
    max_rounds: int = 24,
    should_stop: Optional[Callable[[], bool]] = None,
    on_tool_round: Optional[Callable[[str, str, str], None]] = None,
    proxy_url: Optional[str] = None,
    on_stream_token: Optional[Callable[[str], None]] = None,
    stream_tokens: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    执行多轮 tool_calls，直到模型返回纯文本。
    返回 (完整 messages 副本, 展示用最终助手文本)。

    若 ``stream_tokens`` 与 ``on_stream_token`` 同时启用，则每轮请求使用 ``stream: true``，
    在解析出最终 assistant message 的同时把 reasoning/content 片段交给回调（工具轮与终答均可增量显示）。
    """
    msgs: List[Dict[str, Any]] = [dict(m) for m in messages]
    trace_lines: List[str] = []
    use_stream = bool(stream_tokens and on_stream_token)

    for _ in range(max_rounds):
        if should_stop and should_stop():
            raise StreamInterrupted("Stream interrupted by user")

        try:
            if use_stream:
                msg = chat_completion_message_stream(
                    api_key,
                    msgs,
                    model_mode=model_mode,
                    temperature=temperature,
                    timeout=timeout,
                    tools=tools,
                    proxy_url=proxy_url,
                    should_stop=should_stop,
                    on_stream_token=on_stream_token,
                )
            else:
                data = chat_completion_message(
                    api_key,
                    msgs,
                    model_mode=model_mode,
                    temperature=temperature,
                    timeout=timeout,
                    tools=tools,
                    tool_choice="auto",
                    proxy_url=proxy_url,
                )
                choices = data.get("choices", [])
                if not choices:
                    raise ValueError(f"Unexpected API response: {data}")
                msg = choices[0].get("message") or {}
        except urllib.error.HTTPError as exc:
            raise RuntimeError(explain_http_error(exc)) from exc
        tool_calls = msg.get("tool_calls")
        # DeepSeek thinking mode: must roundtrip `reasoning_content` back to API.
        reasoning_content = msg.get("reasoning_content")

        if tool_calls:
            assistant_entry: Dict[str, Any] = {
                "role": "assistant",
                "content": msg.get("content"),
                "tool_calls": tool_calls,
            }
            if reasoning_content is not None:
                assistant_entry["reasoning_content"] = reasoning_content
            msgs.append(assistant_entry)

            for tc in tool_calls:
                if should_stop and should_stop():
                    raise StreamInterrupted("Stream interrupted by user")
                fn = (tc.get("function") or {})
                name = fn.get("name") or ""
                args = fn.get("arguments") or ""
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                tid = tc.get("id") or ""
                result = tool_executor(name, args)
                if on_tool_round:
                    on_tool_round(name, args, result)
                trace_lines.append(f"[tool:{name}] {result[:800]}{'…' if len(result) > 800 else ''}")
                msgs.append({"role": "tool", "tool_call_id": tid, "content": result})
            continue

        content = msg.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        assistant_entry2: Dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning_content is not None:
            assistant_entry2["reasoning_content"] = reasoning_content
        msgs.append(assistant_entry2)
        final = content.strip()
        if trace_lines:
            final = "\n".join(trace_lines) + "\n\n---\n\n" + final
        return msgs, final

    raise RuntimeError(f"工具调用超过最大轮数 limit={max_rounds}")
