from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from schemavore.config import (
    ConfigError,
    LimitsConfig,
    PrivacyConfig,
    ProviderConfig,
    ProviderName,
    SchemavoreConfig,
    VerificationCommandConfig,
    config_from_mapping,
    load_config,
    save_config,
)


class ConfigTests(unittest.TestCase):
    def test_missing_config_uses_safe_defaults_without_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(Path(directory) / ".schemavore" / "config.toml")

        self.assertEqual(config.provider.name, ProviderName.GEMINI)
        self.assertEqual(config.provider.api_key_env, "GEMINI_API_KEY")
        self.assertEqual(config.verification_commands, ())
        self.assertFalse(config.privacy.send_git_history)

    def test_config_round_trips_without_api_key(self) -> None:
        config = SchemavoreConfig(
            provider=ProviderConfig(
                name=ProviderName.OPENAI_COMPATIBLE,
                model="local-compatible-model",
                api_key_env="SCHEMAVORE_API_KEY",
                base_url="https://example.test/v1",
            ),
            verification_commands=(
                VerificationCommandConfig(
                    name="unit",
                    command=("python3", "-m", "unittest", "discover", "-s", "tests"),
                    timeout_seconds=120,
                ),
            ),
            limits=LimitsConfig(
                max_context_files=12,
                max_file_bytes=50_000,
                max_output_bytes=8_000,
                max_repair_attempts=1,
            ),
            privacy=PrivacyConfig(
                send_git_history=True,
                allow_local_embeddings=True,
                include_untracked_files=False,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".schemavore" / "config.toml"
            save_config(path, config)
            loaded = load_config(path)

        self.assertEqual(loaded, config)
        self.assertNotIn("secret", config.to_toml().lower())

    def test_rejects_inline_api_key_values(self) -> None:
        with self.assertRaisesRegex(ConfigError, "must not contain API key"):
            config_from_mapping(
                {
                    "provider": {
                        "name": "gemini",
                        "model": "gemini-2.5-flash",
                        "api_key": "not-allowed",
                    }
                }
            )

    def test_rejects_shell_string_verification_commands(self) -> None:
        with self.assertRaisesRegex(ConfigError, "list of strings"):
            config_from_mapping(
                {
                    "verification": {
                        "commands": [
                            {
                                "name": "tests",
                                "command": "python3 -m unittest",
                            }
                        ]
                    }
                }
            )

    def test_rejects_invalid_limits_and_duplicate_commands(self) -> None:
        with self.assertRaisesRegex(ConfigError, "max_repair_attempts"):
            config_from_mapping({"limits": {"max_repair_attempts": 3}})

        with self.assertRaisesRegex(ConfigError, "duplicate verification command"):
            config_from_mapping(
                {
                    "verification": {
                        "commands": [
                            {"name": "tests", "command": ["python3"], "timeout_seconds": 1},
                            {"name": "tests", "command": ["python3"], "timeout_seconds": 1},
                        ]
                    }
                }
            )

    def test_rejects_unknown_keys(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unknown provider keys"):
            config_from_mapping({"provider": {"unexpected": True}})


if __name__ == "__main__":
    unittest.main()
