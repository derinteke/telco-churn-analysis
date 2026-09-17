"""Minimal Ollama chat client with tool-calling support.

Talks to the ``/api/chat`` endpoint directly over HTTP, so no SDK is needed.
Anything that implements ``chat(messages, tools) -> dict`` can stand in for it
(the tests use a scripted fake).
"""
from __future__ import annotations

from typing import Protocol

import requests

from .. import config


_THINKING_MODELS = ("qwen3", "deepseek-r1", "gpt-oss")


class LLMError(RuntimeError):
    pass


class ChatModel(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Return the assistant message: {"role", "content", "tool_calls"?}."""


class OllamaChat:
    def __init__(
        self,
        model: str = config.OLLAMA_MODEL,
        host: str = config.OLLAMA_HOST,
        temperature: float = config.LLM_TEMPERATURE,
        timeout: int = config.LLM_TIMEOUT,
    ):
        self.model, self.host = model, host.rstrip("/")
        self.temperature, self.timeout = temperature, timeout

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": config.LLM_NUM_CTX,
                "num_predict": config.LLM_NUM_PREDICT,
                "repeat_penalty": config.LLM_REPEAT_PENALTY,
            },
        }
        if tools:
            payload["tools"] = tools
        if self.model.startswith(_THINKING_MODELS):
            # Reasoning models spend hundreds of tokens "thinking" before answering;
            # with num_predict capped that can leave an empty answer.
            payload["think"] = False
        try:
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        except requests.ConnectionError as e:
            raise LLMError(
                f"Cannot reach Ollama at {self.host}. Is `ollama serve` running?"
            ) from e
        except requests.Timeout as e:
            raise LLMError(
                f"Ollama did not answer within {self.timeout}s. Try a smaller model "
                f"(e.g. OLLAMA_MODEL=qwen2.5:3b) or raise LLM_TIMEOUT."
            ) from e
        if r.status_code == 404:
            raise LLMError(f"Model '{self.model}' not found. Run `ollama pull {self.model}`.")
        if not r.ok:
            raise LLMError(f"Ollama error {r.status_code}: {r.text[:300]}")
        return r.json()["message"]

    def health(self) -> dict:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            return {"reachable": True, "model": self.model, "model_available": self.model in models}
        except requests.RequestException:
            return {"reachable": False, "model": self.model, "model_available": False}
