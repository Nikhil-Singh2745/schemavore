"""Gemini native API provider adapter."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .base import (
    MessageRole,
    ProviderMalformedResponseError,
    ProviderQuotaExceededError,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
    Transport,
    UrllibTransport,
    read_api_key,
)

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


@dataclass(slots=True)
class GeminiProvider:
    model: str
    api_key_env: str
    base_url: str = _DEFAULT_BASE_URL
    timeout_seconds: float = 60.0
    transport: Transport = field(default_factory=UrllibTransport)

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        api_key = read_api_key(self.api_key_env)

        system_parts = [
            message.content for message in request.messages if message.role is MessageRole.SYSTEM
        ]
        turn_messages = [
            message for message in request.messages if message.role is not MessageRole.SYSTEM
        ]

        payload: dict[str, object] = {
            "contents": [
                {
                    "role": "model" if message.role is MessageRole.ASSISTANT else "user",
                    "parts": [{"text": message.content}],
                }
                for message in turn_messages
            ],
            "generationConfig": {
                "maxOutputTokens": request.max_output_tokens,
                "temperature": request.temperature,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

        url = f"{self.base_url}/models/{self.model}:generateContent?key={api_key}"
        started = time.monotonic()
        response = self.transport.post_json(
            url,
            headers={"Content-Type": "application/json"},
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )
        latency_seconds = time.monotonic() - started

        if response.status_code == 429:
            raise ProviderQuotaExceededError("Gemini reported a quota or rate limit failure")
        if response.status_code >= 400:
            raise ProviderMalformedResponseError(
                f"Gemini returned an error status: {response.status_code}"
            )

        try:
            body = json.loads(response.body)
            text = "".join(part["text"] for part in body["candidates"][0]["content"]["parts"])
            usage_metadata = body["usageMetadata"]
            usage = ProviderUsage(
                input_tokens=usage_metadata["promptTokenCount"],
                output_tokens=usage_metadata["candidatesTokenCount"],
            )
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderMalformedResponseError(
                "Gemini response did not match the expected schema"
            ) from exc

        return ProviderResponse(text=text, usage=usage, latency_seconds=latency_seconds)
