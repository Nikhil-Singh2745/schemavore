"""Provider abstraction: normalized contracts and adapters."""

from .base import (
    MessageRole,
    Provider,
    ProviderAuthenticationError,
    ProviderError,
    ProviderMalformedResponseError,
    ProviderMessage,
    ProviderQuotaExceededError,
    ProviderRequest,
    ProviderResponse,
    ProviderTimeoutError,
    ProviderUsage,
    Transport,
    TransportResponse,
    UrllibTransport,
)
from .gemini import GeminiProvider
from .openai_compatible import OpenAICompatibleProvider

__all__ = [
    "MessageRole",
    "Provider",
    "ProviderAuthenticationError",
    "ProviderError",
    "ProviderMalformedResponseError",
    "ProviderMessage",
    "ProviderQuotaExceededError",
    "ProviderRequest",
    "ProviderResponse",
    "ProviderTimeoutError",
    "ProviderUsage",
    "Transport",
    "TransportResponse",
    "UrllibTransport",
    "GeminiProvider",
    "OpenAICompatibleProvider",
]
