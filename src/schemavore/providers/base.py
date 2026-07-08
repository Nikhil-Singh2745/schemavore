"""Provider protocol, normalized request and response contracts, and transport abstraction."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class ProviderError(Exception):
    """Base error for provider failures."""


class ProviderAuthenticationError(ProviderError):
    """Raised when the configured API key environment variable is missing or empty."""


class ProviderQuotaExceededError(ProviderError):
    """Raised when the provider reports a rate limit or quota failure."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request exceeds its timeout."""


class ProviderMalformedResponseError(ProviderError):
    """Raised when a provider response cannot be parsed into the normalized contract."""


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ProviderMessage:
    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("message content must not be empty")


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    messages: tuple[ProviderMessage, ...]
    max_output_tokens: int
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("request must contain at least one message")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts must not be negative")


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    text: str
    usage: ProviderUsage
    latency_seconds: float

    def __post_init__(self) -> None:
        if self.latency_seconds < 0:
            raise ValueError("latency_seconds must not be negative")


class Provider(Protocol):
    def generate(self, request: ProviderRequest) -> ProviderResponse: ...


@dataclass(frozen=True, slots=True)
class TransportResponse:
    status_code: int
    body: bytes


class Transport(Protocol):
    def post_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> TransportResponse: ...


class UrllibTransport:
    """Default transport using only the standard library."""

    def post_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> TransportResponse:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return TransportResponse(status_code=response.status, body=response.read())
        except urllib.error.HTTPError as exc:
            return TransportResponse(status_code=exc.code, body=exc.read())
        except TimeoutError as exc:
            raise ProviderTimeoutError("provider request timed out") from exc


def read_api_key(api_key_env: str) -> str:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ProviderAuthenticationError(f"environment variable {api_key_env} is not set")
    return api_key
