from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence
from urllib import error, parse, request


@dataclass
class LLMConfig:
    enabled: bool = False
    provider: str = "openai_compatible"
    base_url: str = "https://api.xiaomimimo.com/v1"
    chat_path: str = "/chat/completions"
    model: str = "mimo-v2-pro"
    api_key_env: str = "MIMO_API_KEY"
    timeout_seconds: int = 60
    temperature: float = 0.2
    max_tokens: int = 800
    top_p: float = 1.0
    include_response_format: bool = False
    response_format_type: str = "json_object"
    extra_headers: Dict[str, str] = field(default_factory=dict)

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "").strip()

    @property
    def endpoint(self) -> str:
        if not self.base_url:
            return ""
        return parse.urljoin(self.base_url.rstrip("/") + "/", self.chat_path.lstrip("/"))

    def is_ready(self) -> bool:
        return self.enabled and bool(self.endpoint) and bool(self.api_key)


class LLMClientError(RuntimeError):
    pass


class OpenAICompatibleLLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def is_ready(self) -> bool:
        return self.config.is_ready()

    def generate(self, packet: Mapping[str, object]) -> Dict[str, object]:
        if not self.is_ready():
            raise LLMClientError(
                f"LLM client is not ready. Check llm.enabled, llm.base_url and env var {self.config.api_key_env}."
            )

        payload = self._build_payload(packet)
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json; charset=utf-8",
        }
        headers.update(self.config.extra_headers)

        req = request.Request(
            self.config.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                raw_text = response.read().decode("utf-8", errors="replace")
                raw = json.loads(raw_text)
        except error.HTTPError as exc:  # pragma: no cover - depends on provider
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(f"LLM HTTP error {exc.code}: {detail}") from exc
        except error.URLError as exc:  # pragma: no cover - depends on network
            raise LLMClientError(f"LLM connection error: {exc.reason}") from exc
        except json.JSONDecodeError as exc:  # pragma: no cover - provider bug
            raise LLMClientError("LLM returned invalid JSON") from exc

        content = _extract_message_content(raw)
        parsed_content = _try_parse_json(content)
        usage = raw.get("usage", {})
        choice = (raw.get("choices") or [{}])[0]

        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "endpoint": self.config.endpoint,
            "content": content,
            "parsed_content": parsed_content,
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
            "raw": raw,
        }

    def _build_payload(self, packet: Mapping[str, object]) -> Dict[str, object]:
        messages = packet.get("messages", [])
        payload: Dict[str, object] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        if self.config.include_response_format and packet.get("response_schema"):
            payload["response_format"] = {"type": self.config.response_format_type}
        return payload


def create_llm_client(config: LLMConfig | None) -> OpenAICompatibleLLMClient | None:
    if config is None:
        return None
    return OpenAICompatibleLLMClient(config)


def load_llm_config(raw: Mapping[str, object] | None) -> LLMConfig:
    raw = raw or {}
    extra_headers = raw.get("extra_headers", {})
    return LLMConfig(
        enabled=bool(raw.get("enabled", False)),
        provider=str(raw.get("provider", "openai_compatible")),
        base_url=str(raw.get("base_url", "")),
        chat_path=str(raw.get("chat_path", "/chat/completions")),
        model=str(raw.get("model", "mimo-v2-pro")),
        api_key_env=str(raw.get("api_key_env", "MIMO_API_KEY")),
        timeout_seconds=int(raw.get("timeout_seconds", 60)),
        temperature=float(raw.get("temperature", 0.2)),
        max_tokens=int(raw.get("max_tokens", 800)),
        top_p=float(raw.get("top_p", 1.0)),
        include_response_format=bool(raw.get("include_response_format", False)),
        response_format_type=str(raw.get("response_format_type", "json_object")),
        extra_headers=dict(extra_headers) if isinstance(extra_headers, Mapping) else {},
    )


def _extract_message_content(raw: Mapping[str, object]) -> str:
    choices = raw.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts = []
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _try_parse_json(content: str) -> object:
    text = content.strip()
    if not text:
        return None
    if not ((text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]"))):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
