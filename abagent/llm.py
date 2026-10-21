"""Shared LLM access: LiteLLM in online mode, a deterministic MockLLM offline.

Offline demo mode is used when ``LLM_MODEL=mock`` or no API key is set for the
configured provider, so the demo, the Streamlit app and the tests run with no key
and no network. Online runs track token usage and estimated cost.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

DEFAULT_MODEL = "anthropic/claude-sonnet-4-5"
OFFLINE_BANNER = "Running in offline demo mode (no LLM key) — outputs are rule-based"

_PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "azure": "AZURE_API_KEY",
}


def configured_model() -> str:
    return os.getenv("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def is_offline() -> bool:
    """True when we must fall back to the rule-based MockLLM."""
    model = configured_model()
    if model.lower() == "mock":
        return True
    provider = model.split("/", 1)[0] if "/" in model else "openai"
    key_var = _PROVIDER_KEYS.get(provider)
    if key_var is None:  # unknown provider: accept any of the common keys
        return not any(os.getenv(v) for v in _PROVIDER_KEYS.values())
    return not os.getenv(key_var)


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)

    def as_message(self) -> dict[str, Any]:
        """OpenAI-format assistant message to append to the conversation."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in self.tool_calls
            ]
        return msg


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "estimated_cost_usd": round(self.cost_usd, 6)}


class LLMClient(Protocol):
    model: str
    offline: bool
    usage: Usage

    def chat(self, messages: list[dict[str, Any]],
             tools: list[dict[str, Any]] | None = None) -> LLMResponse: ...


def tool_message(call_id: str, result: dict[str, Any]) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)}


class LiteLLMClient:
    """Real model via LiteLLM function calling."""

    offline = False

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        self.model = model or configured_model()
        self.temperature = temperature
        self.usage = Usage()

    def chat(self, messages: list[dict[str, Any]],
             tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        import litellm  # imported lazily: slow import, unused offline

        kwargs: dict[str, Any] = {"model": self.model, "messages": messages,
                                  "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = litellm.completion(**kwargs)
        self._track(litellm, response)
        message = response.choices[0].message
        calls = []
        for tc in message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCallRequest(tc.id, tc.function.name, args))
        return LLMResponse(content=message.content, tool_calls=calls)

    def _track(self, litellm: Any, response: Any) -> None:
        self.usage.calls += 1
        usage = getattr(response, "usage", None)
        if usage:
            self.usage.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.usage.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        try:
            self.usage.cost_usd += float(litellm.completion_cost(completion_response=response))
        except Exception:  # pricing unknown for this model: tokens are still tracked
            pass


MockPolicy = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], LLMResponse]


class MockLLM:
    """Deterministic stand-in for an LLM.

    The calling agent supplies a ``policy`` (a rule-based planner) that maps the
    conversation so far to the next response, so the exact same agent loop runs
    offline and online.
    """

    offline = True

    def __init__(self, policy: MockPolicy | None = None):
        self.model = "mock"
        self.policy = policy
        self.usage = Usage()

    def chat(self, messages: list[dict[str, Any]],
             tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        self.usage.calls += 1
        if self.policy is None:
            last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            return LLMResponse(content=f"[mock] {last[:200]}")
        return self.policy(messages, tools)


def get_llm(mock_policy: MockPolicy | None = None) -> LLMClient:
    """LiteLLM client when a key is configured, otherwise the MockLLM."""
    if is_offline():
        return MockLLM(mock_policy)
    return LiteLLMClient()
