import asyncio
from dataclasses import asdict, dataclass
import time
from typing import AsyncIterator, Protocol

from errors import (
    InvalidProviderResponseError,
    InvalidRequestError,
    ProviderConfigurationError,
    ProviderTimeoutError,
    RateLimitError,
    TransientProviderError,
)
@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    provider: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    finish_reason: str = "stop"

    def to_dict(self) -> dict:
        return asdict(self)
    
class GeminiProvider:
    """Adaptador opcional para Google Gen AI SDK."""

    def __init__(self, api_key: str | None, model: str):
        if not api_key:
            raise ProviderConfigurationError("Falta GEMINI_API_KEY")
        from google import genai
        self.model = model
        self.client = genai.Client(api_key=api_key)

    @staticmethod
    def _map_exception(exc: Exception) -> Exception:
        text = str(exc).lower()
        if "429" in text or "rate limit" in text or "resource exhausted" in text:
            return RateLimitError(str(exc))
        if "timeout" in text or "timed out" in text:
            return ProviderTimeoutError(str(exc))
        if "401" in text or "403" in text or "api key" in text:
            return ProviderConfigurationError(str(exc))
        return TransientProviderError(str(exc))

    @staticmethod
    def _config(system: str | None, temperature: float):
        from google.genai import types
        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
        )

    @staticmethod
    def _usage(response) -> tuple[int, int]:
        usage = getattr(response, "usage_metadata", None)
        return (
            int(getattr(usage, "prompt_token_count", 0) or 0),
            int(getattr(usage, "candidates_token_count", 0) or 0),
        )

    def _normalize(self, response, started: float) -> GenerationResult:
        text = response.text or ""
        if not text.strip():
            raise InvalidProviderResponseError("Gemini devolvió texto vacío")
        input_tokens, output_tokens = self._usage(response)
        return GenerationResult(
            text=text,
            model=self.model,
            provider="gemini",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.2) -> GenerationResult:
        if not prompt.strip():
            raise InvalidRequestError("El prompt no puede estar vacío")
        started = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self._config(system, temperature),
            )
            return self._normalize(response, started)
        except Exception as exc:
            if isinstance(exc, InvalidProviderResponseError):
                raise
            raise self._map_exception(exc) from exc

    async def agenerate(self, prompt: str, *, system: str | None = None, temperature: float = 0.2) -> GenerationResult:
        if not prompt.strip():
            raise InvalidRequestError("El prompt no puede estar vacío")
        started = time.perf_counter()
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self._config(system, temperature),
            )
            return self._normalize(response, started)
        except Exception as exc:
            if isinstance(exc, InvalidProviderResponseError):
                raise
            raise self._map_exception(exc) from exc

    async def astream(self, prompt: str, *, system: str | None = None, temperature: float = 0.2) -> AsyncIterator[str]:
        if not prompt.strip():
            raise InvalidRequestError("El prompt no puede estar vacío")
        try:
            stream = await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=self._config(system, temperature),
            )
            async for chunk in stream:
                text = chunk.text or ""
                if text:
                    yield text
        except Exception as exc:
            raise self._map_exception(exc) from exc