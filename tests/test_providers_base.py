from __future__ import annotations

import os
import unittest

from schemavore.providers import (
    MessageRole,
    ProviderAuthenticationError,
    ProviderMessage,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
)
from schemavore.providers.base import read_api_key


class ProviderContractTests(unittest.TestCase):
    def test_message_rejects_empty_content(self) -> None:
        with self.assertRaisesRegex(ValueError, "content must not be empty"):
            ProviderMessage(role=MessageRole.USER, content="")

    def test_request_rejects_empty_messages(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one message"):
            ProviderRequest(messages=(), max_output_tokens=16)

    def test_request_rejects_invalid_max_output_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_output_tokens"):
            ProviderRequest(
                messages=(ProviderMessage(role=MessageRole.USER, content="hi"),),
                max_output_tokens=0,
            )

    def test_request_rejects_invalid_temperature(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature"):
            ProviderRequest(
                messages=(ProviderMessage(role=MessageRole.USER, content="hi"),),
                max_output_tokens=16,
                temperature=3.0,
            )

    def test_usage_rejects_negative_token_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            ProviderUsage(input_tokens=-1, output_tokens=0)

    def test_response_rejects_negative_latency(self) -> None:
        with self.assertRaisesRegex(ValueError, "latency_seconds"):
            ProviderResponse(
                text="hi",
                usage=ProviderUsage(input_tokens=1, output_tokens=1),
                latency_seconds=-0.1,
            )


class ReadApiKeyTests(unittest.TestCase):
    def test_missing_env_var_raises_authentication_error(self) -> None:
        env_name = "SCHEMAVORE_TEST_MISSING_KEY"
        os.environ.pop(env_name, None)

        with self.assertRaises(ProviderAuthenticationError):
            read_api_key(env_name)

    def test_present_env_var_is_returned(self) -> None:
        env_name = "SCHEMAVORE_TEST_PRESENT_KEY"
        os.environ[env_name] = "secret-value"
        self.addCleanup(os.environ.pop, env_name, None)

        self.assertEqual(read_api_key(env_name), "secret-value")


if __name__ == "__main__":
    unittest.main()
