"""Language Model integration for LUMI — Ollama backend with extensible adapter."""

from __future__ import annotations

import abc
import logging
import time
from typing import Generator, Optional

import httpx

from lumi_voice.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMBackend(abc.ABC):
    """Abstract base class for LLM backends."""

    @abc.abstractmethod
    def check_availability(self) -> tuple[bool, str]:
        """Check if the backend is available. Returns (available, message)."""
        ...

    @abc.abstractmethod
    def generate(
        self, messages: list[dict], stream: bool = False
    ) -> str:
        """Generate a response from conversation messages."""
        ...

    @abc.abstractmethod
    def generate_stream(
        self, messages: list[dict]
    ) -> Generator[str, None, None]:
        """Stream response tokens from conversation messages."""
        ...


class OllamaBackend(LLMBackend):
    """Ollama local LLM backend."""

    def __init__(self, host: str, model: str, timeout_s: int = 60) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._client = httpx.Client(timeout=timeout_s)

    def check_availability(self) -> tuple[bool, str]:
        """Check Ollama server and model availability."""
        try:
            resp = self._client.get(f"{self.host}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]

            # Check if exact model name or model:tag matches
            model_found = False
            for m in models:
                if m == self.model or m.startswith(f"{self.model}:") or self.model.startswith(f"{m.split(':')[0]}"):
                    model_found = True
                    break

            # Also check without tag suffix
            if not model_found:
                model_base = self.model.split(":")[0]
                for m in models:
                    if m.split(":")[0] == model_base:
                        model_found = True
                        break

            if model_found:
                return True, f"Ollama ready with model '{self.model}'"
            else:
                available = ", ".join(models[:10]) if models else "none"
                return False, (
                    f"Model '{self.model}' not found. "
                    f"Available models: {available}. "
                    f"Pull it with: ollama pull {self.model}"
                )
        except httpx.ConnectError:
            return False, (
                f"Cannot connect to Ollama at {self.host}. "
                "Make sure Ollama is running: ollama serve"
            )
        except Exception as e:
            return False, f"Ollama check failed: {e}"

    def generate(self, messages: list[dict], stream: bool = False) -> str:
        """Generate a complete response."""
        try:
            start = time.monotonic()
            resp = self._client.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()
            elapsed = time.monotonic() - start
            logger.info("LLM response generated in %.2fs (%d chars)", elapsed, len(content))
            return content
        except httpx.TimeoutException:
            logger.error("LLM request timed out after %ds", self.timeout_s)
            return "I'm sorry, I took too long to think about that. Could you try again?"
        except httpx.ConnectError:
            logger.error("Lost connection to Ollama")
            return "I'm having trouble connecting to my language model. Please check that Ollama is running."
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return "I encountered an error while generating a response. Please try again."

    def generate_stream(self, messages: list[dict]) -> Generator[str, None, None]:
        """Stream response tokens."""
        try:
            import json as json_module
            start = time.monotonic()
            with self._client.stream(
                "POST",
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                },
                timeout=self.timeout_s,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json_module.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                    except json_module.JSONDecodeError:
                        continue
            elapsed = time.monotonic() - start
            logger.info("LLM stream completed in %.2fs", elapsed)
        except httpx.TimeoutException:
            logger.error("LLM stream timed out")
            yield "I'm sorry, I took too long. Could you try again?"
        except httpx.ConnectError:
            logger.error("Lost connection to Ollama during stream")
            yield "I lost connection to my language model."
        except Exception as e:
            logger.error("LLM stream failed: %s", e)
            yield "I encountered an error. Please try again."

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


class LLMEngine:
    """High-level LLM engine that wraps backend selection."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._backend: Optional[LLMBackend] = None
        self._setup_backend()

    def _setup_backend(self) -> None:
        """Initialize the configured backend."""
        if self.config.backend == "ollama":
            self._backend = OllamaBackend(
                host=self.config.ollama.host,
                model=self.config.ollama.model,
                timeout_s=self.config.ollama.timeout_s,
            )
        else:
            logger.error("Unknown LLM backend: %s", self.config.backend)
            raise ValueError(f"Unsupported LLM backend: {self.config.backend}")

    @property
    def backend(self) -> Optional[LLMBackend]:
        return self._backend

    def check_availability(self) -> tuple[bool, str]:
        """Check if the LLM backend is available."""
        if self._backend is None:
            return False, "No LLM backend configured"
        return self._backend.check_availability()

    def generate(self, messages: list[dict]) -> str:
        """Generate a response."""
        if self._backend is None:
            return "Language model is not configured."
        if self.config.ollama.stream:
            # Collect streamed tokens into full response
            tokens = list(self._backend.generate_stream(messages))
            return "".join(tokens)
        return self._backend.generate(messages)

    def generate_stream(self, messages: list[dict]) -> Generator[str, None, None]:
        """Stream response tokens."""
        if self._backend is None:
            yield "Language model is not configured."
            return
        yield from self._backend.generate_stream(messages)

    def change_model(self, model: str) -> None:
        """Switch to a different model."""
        if isinstance(self._backend, OllamaBackend):
            self._backend.model = model
            self.config.ollama.model = model
            logger.info("LLM model changed to: %s", model)

    def change_host(self, host: str) -> None:
        """Switch to a different Ollama host."""
        if isinstance(self._backend, OllamaBackend):
            self._backend.host = host.rstrip("/")
            self.config.ollama.host = host
            logger.info("Ollama host changed to: %s", host)
