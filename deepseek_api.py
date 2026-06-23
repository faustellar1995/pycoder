import base64
import errno
import json
import os
import select
import socket
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

# 流式 SSE：短轮询便于 Stop；总等待时长由 opener.open(timeout=…) 控制，思考模型需足够大
STREAM_READ_POLL_SEC = 1.0
STREAM_TIMEOUT_DEFAULT = 180
STREAM_TIMEOUT_OLLAMA = 3600
STREAM_TIMEOUT_UNLIMITED = -1

API_URL = "https://api.deepseek.com/chat/completions"
MODEL_ALIASES: Dict[str, str] = {
    "flash": "deepseek-v4-flash",
    "pro": "deepseek-v4-pro",
}

# Kimi（Moonshot 开放平台 · 中国站）：OpenAI 兼容 Chat Completions / List Models
# 控制台与密钥：https://platform.moonshot.cn/
# 默认 API 主机为 api.moonshot.cn（在国际站 platform.moonshot.ai 创建的密钥与中国站不互通）
# 可选：设置环境变量 KIMI_API_BASE（根 URL，无尾斜杠，如 https://api.moonshot.cn）覆盖默认主机
KIMI_API_BASE_DEFAULT = "https://api.moonshot.cn"


def kimi_api_base() -> str:
    b = os.getenv("KIMI_API_BASE", "").strip().rstrip("/")
    return b if b else KIMI_API_BASE_DEFAULT


def kimi_chat_completions_url() -> str:
    return f"{kimi_api_base()}/v1/chat/completions"


def kimi_models_list_url() -> str:
    return f"{kimi_api_base()}/v1/models"


# 兼容旧代码或静态引用（等于默认中国站，不反映 KIMI_API_BASE 覆盖）
KIMI_API_URL = f"{KIMI_API_BASE_DEFAULT}/v1/chat/completions"
KIMI_MODELS_URL = f"{KIMI_API_BASE_DEFAULT}/v1/models"

# OpenAI 兼容：GET /v1/models（两家均有）
DEEPSEEK_MODELS_URLS = (
    "https://api.deepseek.com/v1/models",
    "https://api.deepseek.com/models",
)
KIMI_MODEL_ALIASES: Dict[str, str] = {
    "flash": "kimi-k2-turbo-preview",
    "pro": "kimi-k2.6",
    "k2": "kimi-k2.6",
    "k2.6": "kimi-k2.6",
    "k2.5": "kimi-k2.5",
}

PROVIDER_DEEPSEEK = "deepseek"
PROVIDER_KIMI = "kimi"
PROVIDER_OLLAMA = "ollama"

# 多模态：单次 user 消息最多图片数；单张原始字节上限（base64 后更大）
VISION_MAX_IMAGES = 4
VISION_MAX_IMAGE_BYTES = 4 * 1024 * 1024

# OpenAI 兼容 content part
ContentPart = Dict[str, Any]
UserContent = Union[str, List[ContentPart]]
ChatMessage = Dict[str, Any]

# 本地 Ollama OpenAI 兼容端点：POST /v1/chat/completions；列举模型：GET /api/tags 或 GET /v1/models
# 可通过环境变量 OLLAMA_API_BASE / OLLAMA_HOST（任一为根 URL，无尾斜杠）或 UI 覆盖默认
OLLAMA_API_BASE_DEFAULT = "http://127.0.0.1:11434"


def ollama_api_base(ui_override: Optional[str] = None) -> str:
    """解析 Ollama HTTP 根地址（不含路径）。ui_override 非空时优先（来自界面输入）。"""
    u = (ui_override or "").strip().rstrip("/")
    if u:
        return u
    for env in ("OLLAMA_API_BASE", "OLLAMA_HOST"):
        v = os.getenv(env, "").strip().rstrip("/")
        if v:
            return v
    return OLLAMA_API_BASE_DEFAULT


def ollama_chat_completions_url(ui_override: Optional[str] = None) -> str:
    return f"{ollama_api_base(ui_override)}/v1/chat/completions"


class StreamInterrupted(Exception):
    """Raised when stream is interrupted by user."""


def get_api_key() -> str:
    api_key = os.getenv("DS_KEY", "").strip()
    if not api_key:
        raise ValueError("DS_KEY environment variable is not set.")
    return api_key


def get_kimi_key() -> str:
    api_key = os.getenv("KIMI_KEY", "").strip()
    if not api_key:
        raise ValueError("KIMI_KEY environment variable is not set.")
    return api_key


def resolve_chat_endpoint(
    provider: str,
    *,
    ollama_ui_base: Optional[str] = None,
) -> Tuple[str, str]:
    """按提供方返回 (api_key, chat_completions_url)。Ollama 本地无需密钥，返回空字符串。"""
    p = (provider or PROVIDER_DEEPSEEK).strip().lower()
    if p == PROVIDER_KIMI:
        return get_kimi_key(), kimi_chat_completions_url()
    if p == PROVIDER_OLLAMA:
        return "", ollama_chat_completions_url(ollama_ui_base)
    return get_api_key(), API_URL


def resolve_model(model_mode: str, provider: str = PROVIDER_DEEPSEEK) -> str:
    key = model_mode.strip().lower()
    p = (provider or PROVIDER_DEEPSEEK).strip().lower()
    if p == PROVIDER_KIMI:
        return KIMI_MODEL_ALIASES.get(key, model_mode)
    if p == PROVIDER_OLLAMA:
        return model_mode.strip()
    return MODEL_ALIASES.get(key, model_mode)


def effective_temperature_for_resolved_model(
    resolved_model_id: str,
    temperature: float,
    *,
    provider: str = PROVIDER_DEEPSEEK,
) -> float:
    """
    Kimi K2.5 / K2.6 等部分模型仅允许 temperature=1，否则 API 返回
    ``invalid temperature: only 1 is allowed for this model``（HTTP 400）。
    """
    p = (provider or PROVIDER_DEEPSEEK).strip().lower()
    if p != PROVIDER_KIMI:
        return temperature
    s = (resolved_model_id or "").lower()
    if "k2.5" in s or "k2.6" in s:
        return 1.0
    return temperature


def _request_headers_get(api_key: str) -> Dict[str, str]:
    h = {
        "Accept": "application/json",
        "Accept-Encoding": "identity",
    }
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def _get_json_simple(
    url: str,
    timeout: int,
    proxy_url: Optional[str],
) -> Dict[str, Any]:
    """GET JSON，无 Bearer（用于本地 Ollama /api/tags 等）。"""
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        },
        method="GET",
    )
    opener = _build_opener(proxy_url)
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json_with_proxy(
    url: str,
    api_key: str,
    timeout: int,
    proxy_url: Optional[str],
) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers=_request_headers_get(api_key),
        method="GET",
    )
    opener = _build_opener(proxy_url)
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_openai_models_list(payload: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        if isinstance(mid, str) and mid.strip():
            out.append(mid.strip())
    return sorted(set(out))


def _try_list_models_urls(
    urls: Tuple[str, ...],
    api_key: str,
    timeout: int,
    proxy_url: Optional[str],
) -> List[str]:
    """依次尝试多个 URL（兼容部分网关路径差异），返回模型 id 列表。"""
    last_exc: Optional[BaseException] = None
    for url in urls:
        try:
            payload = _get_json_with_proxy(url, api_key, timeout, proxy_url)
            ids = _parse_openai_models_list(payload)
            if ids:
                return ids
        except BaseException as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    return []


def list_ollama_models(
    *,
    base_url: str,
    timeout: int = 15,
    proxy_url: Optional[str] = None,
) -> List[str]:
    """
    扫描本地 Ollama：优先 GET /api/tags，其次 GET /v1/models（OpenAI 兼容）。
    返回已安装模型名（含 tag，如 ``llama3:latest``）。
    """
    base = base_url.rstrip("/")
    last_exc: Optional[BaseException] = None

    tags_url = f"{base}/api/tags"
    try:
        payload = _get_json_simple(tags_url, timeout, proxy_url)
        names: List[str] = []
        for m in payload.get("models") or []:
            if isinstance(m, dict):
                n = m.get("name")
                if isinstance(n, str) and n.strip():
                    names.append(n.strip())
        if names:
            return sorted(set(names))
    except BaseException as exc:
        last_exc = exc

    try:
        payload = _get_json_simple(f"{base}/v1/models", timeout, proxy_url)
        ids = _parse_openai_models_list(payload)
        if ids:
            return ids
    except BaseException as exc:
        last_exc = exc

    if last_exc is not None:
        raise last_exc
    return []


def check_available(
    *,
    proxy_url: Optional[str] = None,
    timeout: int = 45,
    ds_key: Optional[str] = None,
    kimi_key: Optional[str] = None,
    ollama_ui_base: Optional[str] = None,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    通过 HTTP 拉取 DeepSeek、Kimi 可用模型及本地 Ollama 已安装模型。
    返回 ( [(provider, model_id), ...], [警告/跳过说明] )。
    """
    combined: List[Tuple[str, str]] = []
    notes: List[str] = []

    sk = (ds_key if ds_key is not None else os.getenv("DS_KEY", "")).strip()
    if sk:
        try:
            ids = _try_list_models_urls(DEEPSEEK_MODELS_URLS, sk, timeout, proxy_url)
            for mid in ids:
                combined.append((PROVIDER_DEEPSEEK, mid))
        except urllib.error.HTTPError as exc:
            notes.append(f"DeepSeek list-models: {explain_http_error(exc)}")
        except Exception as exc:
            notes.append(f"DeepSeek list-models: {exc}")
    else:
        notes.append("DeepSeek：未设置 DS_KEY，已跳过")

    kk = (kimi_key if kimi_key is not None else os.getenv("KIMI_KEY", "")).strip()
    if kk:
        try:
            payload = _get_json_with_proxy(kimi_models_list_url(), kk, timeout, proxy_url)
            for mid in _parse_openai_models_list(payload):
                combined.append((PROVIDER_KIMI, mid))
        except urllib.error.HTTPError as exc:
            notes.append(f"Kimi list-models: {explain_http_error(exc)}")
        except Exception as exc:
            notes.append(f"Kimi list-models: {exc}")
    else:
        notes.append("Kimi：未设置 KIMI_KEY，已跳过")

    base_ollama = ollama_api_base(ollama_ui_base)
    try:
        for mid in list_ollama_models(
            base_url=base_ollama,
            timeout=min(timeout, 25),
            proxy_url=proxy_url,
        ):
            combined.append((PROVIDER_OLLAMA, mid))
    except urllib.error.URLError as exc:
        notes.append(f"Ollama（{base_ollama}）: {exc}")
    except Exception as exc:
        notes.append(f"Ollama（{base_ollama}）: {exc}")

    combined.sort(key=lambda x: (x[0], x[1]))
    return combined, notes


def _build_payload(
    messages: List[Dict[str, str]],
    model_mode: str,
    temperature: float,
    stream: bool,
    provider: str = PROVIDER_DEEPSEEK,
) -> Dict[str, object]:
    mid = resolve_model(model_mode, provider)
    temp = effective_temperature_for_resolved_model(
        mid, temperature, provider=provider
    )
    return {
        "model": mid,
        "messages": messages,
        "temperature": temp,
        "stream": stream,
    }


def _post_json(
    payload: Dict[str, object], api_key: str, timeout: int, *, api_url: str = API_URL
):
    return _post_json_with_proxy(payload, api_key, timeout, proxy_url=None, api_url=api_url)


def _build_opener(proxy_url: Optional[str]):
    if not proxy_url:
        return urllib.request.build_opener()
    proxy = {"http": proxy_url, "https": proxy_url}
    return urllib.request.build_opener(urllib.request.ProxyHandler(proxy))


def _request_headers(api_key: str) -> Dict[str, str]:
    """统一请求头：禁止 gzip，避免 urllib 对流式 SSE 先解压再整块读出（表现为「流式失效」）。"""
    h: Dict[str, str] = {
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
    }
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def _post_json_with_proxy(
    payload: Dict[str, object],
    api_key: str,
    timeout: Optional[int],
    *,
    proxy_url: Optional[str],
    api_url: str = API_URL,
):
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=_request_headers(api_key),
        method="POST",
    )
    opener = _build_opener(proxy_url)
    return opener.open(request, timeout=timeout)


def effective_stream_timeout(
    timeout: Optional[int],
    *,
    provider: str = PROVIDER_DEEPSEEK,
) -> Optional[int]:
    """
    解析请求总超时（秒），供 urllib opener.open 使用。
    ``STREAM_TIMEOUT_UNLIMITED``（-1）或任意负值 → ``None``（无限等待）。
    本地 Ollama 思考阶段可能数分钟无首包，正值下限显著长于云端 API。
    """
    if timeout is not None and timeout < 0:
        return None
    base = timeout if timeout is not None and timeout > 0 else STREAM_TIMEOUT_DEFAULT
    p = (provider or PROVIDER_DEEPSEEK).strip().lower()
    if p == PROVIDER_OLLAMA:
        return max(base, STREAM_TIMEOUT_OLLAMA)
    return base


def _get_response_socket(response: Any) -> Optional[socket.socket]:
    """从 urllib HTTPResponse 解析底层 socket（Windows / 各 Python 版本结构略有差异）。"""
    seen: set[int] = set()
    obj: Any = response
    for _ in range(10):
        if obj is None or id(obj) in seen:
            break
        seen.add(id(obj))
        if isinstance(obj, socket.socket):
            return obj
        nxt = getattr(obj, "fp", None) or getattr(obj, "_fp", None)
        if nxt is not None and nxt is not obj:
            obj = nxt
            continue
        raw = getattr(obj, "raw", None)
        if raw is not None and raw is not obj:
            obj = raw
            continue
        sock = getattr(obj, "_sock", None)
        if sock is not None and sock is not obj:
            obj = sock
            continue
        break
    return obj if isinstance(obj, socket.socket) else None


def _clear_socketio_timeout_flag(response: Any) -> None:
    """
    urllib 经 SocketIO 读超时后会置 ``_timeout_occurred``，此后读会直接
    ``OSError: cannot read from timed out object``。SSE 轮询需在每次可恢复
    的超时后清除该标记。
    """
    fp = getattr(response, "fp", None) or getattr(response, "_fp", None)
    if fp is None:
        return
    for obj in (fp, getattr(fp, "raw", None)):
        if obj is not None and hasattr(obj, "_timeout_occurred"):
            try:
                obj._timeout_occurred = False
            except Exception:
                pass


def _is_read_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return True
    if isinstance(exc, OSError):
        msg = str(exc).lower()
        if "timed out object" in msg:
            return True
        if exc.errno in (errno.ETIMEDOUT, errno.EWOULDBLOCK):
            return True
        # Windows WSAETIMEDOUT
        if getattr(exc, "winerror", None) == 10060:
            return True
    return False


def _iter_sse_stream_lines(
    response: Any,
    *,
    should_stop: Optional[Callable[[], bool]] = None,
    read_poll_sec: float = STREAM_READ_POLL_SEC,
) -> Iterator[bytes]:
    """
    逐行读取 SSE。

    用 ``select`` 做短轮询（便于 Stop），避免对同一 SocketIO 反复 ``readline``
    触发「cannot read from timed out object」（见 socket.SocketIO._timeout_occurred）。
    """
    sock = _get_response_socket(response)
    _clear_socketio_timeout_flag(response)
    if sock is not None:
        try:
            sock.settimeout(None)
        except OSError:
            pass

    while True:
        if should_stop and should_stop():
            raise StreamInterrupted("Stream interrupted by user")

        if sock is not None:
            try:
                readable, _, _ = select.select([sock], [], [], read_poll_sec)
            except (ValueError, OSError):
                break
            if not readable:
                continue

        try:
            raw_line = response.readline()
        except BaseException as exc:
            if _is_read_timeout_error(exc):
                _clear_socketio_timeout_flag(response)
                if should_stop and should_stop():
                    raise StreamInterrupted("Stream interrupted by user")
                continue
            raise
        if not raw_line:
            break
        if should_stop and should_stop():
            raise StreamInterrupted("Stream interrupted by user")
        yield raw_line


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


def _extract_reasoning_from_delta(delta: Dict[str, Any]) -> str:
    """思考链字段：DeepSeek reasoning_content、Ollama/OpenAI reasoning、原生 thinking。"""
    for key in ("reasoning_content", "reasoning", "thinking"):
        piece = delta.get(key)
        if piece is None:
            continue
        s = _normalize_delta_piece(piece)
        if s:
            return s
    return ""


def chat_completion(
    api_key: str,
    messages: List[Dict[str, str]],
    model_mode: str = "flash",
    temperature: float = 0.7,
    timeout: int = 60,
    proxy_url: Optional[str] = None,
    *,
    api_url: str = API_URL,
    provider: str = PROVIDER_DEEPSEEK,
) -> str:
    payload = _build_payload(messages, model_mode, temperature, stream=False, provider=provider)
    req_timeout = effective_stream_timeout(timeout, provider=provider)
    with _post_json_with_proxy(
        payload, api_key, req_timeout, proxy_url=proxy_url, api_url=api_url
    ) as response:
        data = json.loads(response.read().decode("utf-8"))

    choices = data.get("choices", [])
    if not choices:
        raise ValueError(f"Unexpected API response: {data}")

    message = choices[0].get("message", {})
    content = message.get("content")
    if not content:
        raise ValueError(f"Empty response content: {data}")
    return content.strip()


def chat_completion_stream(
    api_key: str,
    messages: List[Dict[str, str]],
    model_mode: str = "flash",
    temperature: float = 0.7,
    timeout: int = 120,
    should_stop: Optional[Callable[[], bool]] = None,
    proxy_url: Optional[str] = None,
    *,
    api_url: str = API_URL,
    provider: str = PROVIDER_DEEPSEEK,
) -> Iterator[str]:
    """流式输出；消费 delta 内 reasoning / content 等字段。

    思考模式往往先出 reasoning_content（或 reasoning / thinking），再出 content。
    读超时仅用于轮询 Stop；总等待由 effective_stream_timeout 控制（Ollama 默认更长）。
    请求使用 Accept-Encoding: identity，避免 gzip + urllib 把整段 SSE 缓冲后才解压（界面表现为非流式）。
    """
    payload = _build_payload(messages, model_mode, temperature, stream=True, provider=provider)
    req_timeout = effective_stream_timeout(timeout, provider=provider)
    with _post_json_with_proxy(
        payload, api_key, req_timeout, proxy_url=proxy_url, api_url=api_url
    ) as response:
        for raw_line in _iter_sse_stream_lines(response, should_stop=should_stop):
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
            r = _extract_reasoning_from_delta(delta)
            if r:
                yield r
            c = _normalize_delta_piece(delta.get("content"))
            if c:
                yield c


def provider_supports_vision(provider: str) -> bool:
    """
    当前集成是否可向 API 发送 image_url 多模态 user 消息。

    DeepSeek 官方 Chat Completions（deepseek-v4-*）截至文档仍仅接受纯文本 content；
    发送 image_url 会返回 400（unknown variant image_url, expected text）。
    Kimi（Moonshot）与本地 Ollama 视觉模型支持 OpenAI 风格 content 数组。
    """
    p = (provider or PROVIDER_DEEPSEEK).strip().lower()
    return p in (PROVIDER_KIMI, PROVIDER_OLLAMA)


def vision_unsupported_hint(provider: str) -> str:
    p = (provider or PROVIDER_DEEPSEEK).strip().lower()
    if p == PROVIDER_DEEPSEEK:
        return (
            "DeepSeek 官方 API（deepseek-v4-flash / deepseek-v4-pro）目前不支持图片输入；"
            "网页版 chat 有视觉能力但 API 尚未开放。"
            "请切换到 Kimi 视觉模型或本地 Ollama 多模态模型。"
        )
    return f"提供方 {p} 未标记为支持视觉输入。"


def image_bytes_to_data_url(data: bytes, mime: str = "image/png") -> str:
    enc = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{enc}"


def build_user_message_content(text: str, image_data_urls: Optional[List[str]] = None) -> UserContent:
    """构造 user message 的 content：纯文本 str，或含 image_url + text 的数组。"""
    urls = [u.strip() for u in (image_data_urls or []) if isinstance(u, str) and u.strip()]
    t = (text or "").strip()
    if not urls:
        return t
    parts: List[ContentPart] = []
    for url in urls[:VISION_MAX_IMAGES]:
        parts.append({"type": "image_url", "image_url": {"url": url}})
    if t:
        parts.append({"type": "text", "text": t})
    return parts


def redact_image_url_for_preview(url: str) -> str:
    """预览 JSON 用：用短占位符替代 data URL，避免 megabytes 级 dumps。"""
    u = (url or "").strip()
    if not u:
        return "[empty image]"
    if not u.startswith("data:"):
        return u if len(u) <= 96 else u[:80] + "…"
    header, _, b64 = u.partition(",")
    mime = header.split(":", 1)[-1].split(";", 1)[0] if ":" in header else "image"
    approx = len(b64) * 3 // 4 if b64 else 0
    if approx >= 1024 * 1024:
        size = f"{approx / (1024 * 1024):.1f}MB"
    elif approx >= 1024:
        size = f"{approx / 1024:.0f}KB"
    else:
        size = f"{approx}B"
    return f"[{mime} {size}; omitted in preview]"


def redact_content_for_preview(content: Any, *, hide_images: bool = True) -> Any:
    if not hide_images or isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content
    out: List[ContentPart] = []
    for item in content:
        if not isinstance(item, dict):
            out.append(item)
            continue
        if item.get("type") == "image_url":
            iu = item.get("image_url")
            url = iu.get("url") if isinstance(iu, dict) else iu
            if hide_images:
                out.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": redact_image_url_for_preview(str(url or "")),
                        },
                    }
                )
            else:
                out.append(dict(item))
        else:
            out.append(dict(item))
    return out


def redact_messages_for_preview(
    messages: List[Dict[str, Any]],
    *,
    hide_images: bool = True,
) -> List[Dict[str, Any]]:
    if not hide_images:
        return messages
    return [
        {
            **m,
            "content": redact_content_for_preview(m.get("content"), hide_images=True),
        }
        for m in messages
    ]


def count_images_in_content(content: Any) -> int:
    if not isinstance(content, list):
        return 0
    n = 0
    for item in content:
        if isinstance(item, dict) and item.get("type") == "image_url":
            n += 1
    return n


def content_to_display_text(content: Any) -> str:
    """将 message content 转为聊天区可显示的纯文本摘要。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                t = item.get("text")
                if t is not None:
                    texts.append(str(t))
        img_n = count_images_in_content(content)
        body = "\n".join(texts)
        if img_n:
            suffix = f"\n[{img_n} 张图片]" if body else f"[{img_n} 张图片]"
            return body + suffix
        return body
    return str(content)


def image_data_urls_from_content(content: Any) -> List[str]:
    """从 multimodal content 提取 data URL（用于 UI 缩略图）。"""
    if not isinstance(content, list):
        return []
    out: List[str] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "image_url":
            continue
        iu = item.get("image_url")
        if isinstance(iu, dict):
            url = iu.get("url")
            if isinstance(url, str) and url.strip():
                out.append(url.strip())
        elif isinstance(iu, str) and iu.strip():
            out.append(iu.strip())
    return out


def validate_messages_for_provider(
    messages: List[Dict[str, Any]],
    provider: str,
) -> Optional[str]:
    """若 messages 含图片但提供方不支持视觉，返回错误说明；否则 None。"""
    if provider_supports_vision(provider):
        return None
    for m in messages:
        if m.get("role") != "user":
            continue
        if count_images_in_content(m.get("content")) > 0:
            return vision_unsupported_hint(provider)
    return None


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
    return f"HTTP error {exc.code}: {body}"


def _post_json_return_dict(
    payload: Dict[str, Any], api_key: str, timeout: int, *, api_url: str = API_URL
) -> Dict[str, Any]:
    return _post_json_return_dict_with_proxy(
        payload, api_key, timeout, proxy_url=None, api_url=api_url
    )


def _post_json_return_dict_with_proxy(
    payload: Dict[str, Any],
    api_key: str,
    timeout: Optional[int],
    *,
    proxy_url: Optional[str],
    api_url: str = API_URL,
) -> Dict[str, Any]:
    request = urllib.request.Request(
        api_url,
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
    provider: str = PROVIDER_DEEPSEEK,
) -> Dict[str, Any]:
    mid = resolve_model(model_mode, provider)
    temp = effective_temperature_for_resolved_model(
        mid, temperature, provider=provider
    )
    return {
        "model": mid,
        "messages": messages,
        "temperature": temp,
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
    *,
    api_url: str = API_URL,
    provider: str = PROVIDER_DEEPSEEK,
) -> Dict[str, Any]:
    """非流式请求，返回 API JSON 根对象（用于工具调用）。"""
    assert tools is not None
    payload = _build_tool_payload(
        messages, model_mode, temperature, False, tools, tool_choice, provider=provider
    )
    req_timeout = effective_stream_timeout(timeout, provider=provider)
    return _post_json_return_dict_with_proxy(
        payload, api_key, req_timeout, proxy_url=proxy_url, api_url=api_url
    )


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
    api_url: str = API_URL,
    provider: str = PROVIDER_DEEPSEEK,
    should_stop: Optional[Callable[[], bool]] = None,
    on_stream_token: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """单次 stream=true + tools 请求，解析 SSE 直至 [DONE]，组装与非流式相同的 assistant message。"""
    payload = _build_tool_payload(
        messages, model_mode, temperature, True, tools, tool_choice, provider=provider
    )
    reasoning_parts: List[str] = []
    content_parts: List[str] = []
    tool_calls_acc: Dict[int, Dict[str, Any]] = {}

    req_timeout = effective_stream_timeout(timeout, provider=provider)
    with _post_json_with_proxy(
        payload, api_key, req_timeout, proxy_url=proxy_url, api_url=api_url
    ) as response:
        for raw_line in _iter_sse_stream_lines(response, should_stop=should_stop):
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

            rs = _extract_reasoning_from_delta(delta)
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
    api_url: str = API_URL,
    provider: str = PROVIDER_DEEPSEEK,
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
                    api_url=api_url,
                    provider=provider,
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
                    api_url=api_url,
                    provider=provider,
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
