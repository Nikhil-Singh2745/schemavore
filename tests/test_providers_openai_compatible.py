from __future__ import annotations

import json
import os
import unittest
from typing import Any

from schemavore.providers import (
    MessageRole,
    OpenAICompatibleProvider,
    ProviderAuthenticationError,
    ProviderMalformedResponseError,
    ProviderMessage,
    ProviderQuotaExceededError,
    ProviderRequest,
    ProviderTimeoutError,
    TransportResponse,
)

_KEY_ENV = "SCHEMAVORE_TEST_OPENAI_COMPATIBLE_KEY"


class FakeTransport:
    def __init__(
        self,
        response: TransportResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def post_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> TransportResponse:
        self.calls.append(
            {"url": url, "headers": headers, "payload": payload, "timeout_seconds": timeout_seconds}
        )
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def _request() -> ProviderRequest:
    return ProviderRequest(
        messages=(
            ProviderMessage(role=MessageRole.SYSTEM, content="Be concise."),
            ProviderMessage(role=MessageRole.USER, content="Say hello."),
        ),
        max_output_tokens=64,
    )


class OpenAICompatibleProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ[_KEY_ENV] = "test-key-value"
        self.addCleanup(os.environ.pop, _KEY_ENV, None)

    def test_generate_returns_normalized_response_on_success(self) -> None:
        body = json.dumps(
            {
                "choices": [{"message": {"role": "assistant", "content": "Hello there."}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        ).encode("utf-8")
        transport = FakeTransport(response=TransportResponse(status_code=200, body=body))
        provider = OpenAICompatibleProvider(
            model="local-compatible-model", api_key_env=_KEY_ENV, transport=transport
        )

        response = provider.generate(_request())

        self.assertEqual(response.text, "Hello there.")
        self.assertEqual(response.usage.input_tokens, 10)
        self.assertEqual(response.usage.output_tokens, 5)
        self.assertGreaterEqual(response.latency_seconds, 0.0)
        self.assertEqual(transport.calls[0]["headers"]["Authorization"], "Bearer test-key-value")

    def test_generate_raises_on_quota_exceeded(self) -> None:
        body = json.dumps({"error": {"type": "insufficient_quota"}}).encode("utf-8")
        transport = FakeTransport(response=TransportResponse(status_code=429, body=body))
        provider = OpenAICompatibleProvider(
            model="local-compatible-model", api_key_env=_KEY_ENV, transport=transport
        )

        with self.assertRaises(ProviderQuotaExceededError):
            provider.generate(_request())

    def test_generate_raises_on_timeout(self) -> None:
        transport = FakeTransport(error=ProviderTimeoutError("timed out"))
        provider = OpenAICompatibleProvider(
            model="local-compatible-model", api_key_env=_KEY_ENV, transport=transport
        )

        with self.assertRaises(ProviderTimeoutError):
            provider.generate(_request())

    def test_generate_raises_on_malformed_response(self) -> None:
        transport = FakeTransport(response=TransportResponse(status_code=200, body=b"not json"))
        provider = OpenAICompatibleProvider(
            model="local-compatible-model", api_key_env=_KEY_ENV, transport=transport
        )

        with self.assertRaises(ProviderMalformedResponseError):
            provider.generate(_request())

    def test_generate_requires_configured_api_key(self) -> None:
        provider = OpenAICompatibleProvider(
            model="local-compatible-model",
            api_key_env="SCHEMAVORE_TEST_MISSING_OPENAI_KEY",
            transport=FakeTransport(),
        )

        with self.assertRaises(ProviderAuthenticationError):
            provider.generate(_request())


if __name__ == "__main__":
    unittest.main()
