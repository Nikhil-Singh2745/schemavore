"""Provider adapter for providers exposing an OpenAI-compatible chat completions API."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .base import (
    ProviderMalformedResponseError,
    ProviderQuotaExceededError,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
    Transport,
    UrllibTransport,
    read_api_key,
)

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


@dataclass(slots=True)
class OpenAICompatibleProvider:
    model: str
    api_key_env: str
    base_url: str = _DEFAULT_BASE_URL
    timeout_seconds: float = 60.0
    transport: Transport = field(default_factory=UrllibTransport)

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        api_key = read_api_key(self.api_key_env)

        payload = {
            "model": self.model,
            "messages": [
                {"role": message.role.value, "content": message.content}
                for message in request.messages
            ],
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        url = f"{self.base_url.rstrip('/')}/chat/completions"

        started = time.monotonic()
        response = self.transport.post_json(
            url,
            headers=headers,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        latency_seconds = time.monotonic() - started

        if response.status_code == 429:
            raise ProviderQuotaExceededError("provider reported a quota or rate limit failure")
        if response.status_code >= 400:
            raise ProviderMalformedResponseError(
                f"provider returned an error status: {response.status_code}"
            )

        try:
            body = json.loads(response.body)
            text = body["choices"][0]["message"]["content"]
            usage_payload = body["usage"]
            usage = ProviderUsage(
                input_tokens=usage_payload["prompt_tokens"],
                output_tokens=usage_payload["completion_tokens"],
            )
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderMalformedResponseError(
                "provider response did not match the expected schema"
            ) from exc

        return ProviderResponse(text=text, usage=usage, latency_seconds=latency_seconds)
