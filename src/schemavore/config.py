"""Configuration schema, defaults, loading, and validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
import re
import tomllib


class ConfigError(ValueError):
    """Raised when a Schemavore configuration file is invalid."""


class ProviderName(StrEnum):
    GEMINI = "gemini"
    OPENAI_COMPATIBLE = "openai_compatible"


_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    name: ProviderName = ProviderName.GEMINI
    model: str = "gemini-2.5-flash"
    api_key_env: str = "GEMINI_API_KEY"
    base_url: str | None = None

    def validate(self) -> None:
        if not self.model.strip():
            raise ConfigError("provider.model must not be empty")
        if not _ENV_NAME_RE.fullmatch(self.api_key_env):
            raise ConfigError("provider.api_key_env must be an environment variable name")
        if self.base_url is not None and not self.base_url.startswith(("https://", "http://")):
            raise ConfigError("provider.base_url must start with http:// or https://")


@dataclass(frozen=True, slots=True)
class VerificationCommandConfig:
    name: str
    command: tuple[str, ...]
    timeout_seconds: int = 60

    def validate(self) -> None:
        if not self.name.strip():
            raise ConfigError("verification command name must not be empty")
        if not self.command or any(not part.strip() for part in self.command):
            raise ConfigError("verification command must contain non-empty parts")
        if not 1 <= self.timeout_seconds <= 3600:
            raise ConfigError("verification timeout_seconds must be between 1 and 3600")


@dataclass(frozen=True, slots=True)
class LimitsConfig:
    max_context_files: int = 40
    max_file_bytes: int = 200_000
    max_output_bytes: int = 64_000
    max_repair_attempts: int = 2

    def validate(self) -> None:
        _validate_positive("limits.max_context_files", self.max_context_files)
        _validate_positive("limits.max_file_bytes", self.max_file_bytes)
        _validate_positive("limits.max_output_bytes", self.max_output_bytes)
        if not 0 <= self.max_repair_attempts <= 2:
            raise ConfigError("limits.max_repair_attempts must be between 0 and 2")


@dataclass(frozen=True, slots=True)
class PrivacyConfig:
    send_git_history: bool = False
    allow_local_embeddings: bool = False
    include_untracked_files: bool = False


@dataclass(frozen=True, slots=True)
class SchemavoreConfig:
    provider: ProviderConfig = ProviderConfig()
    verification_commands: tuple[VerificationCommandConfig, ...] = ()
    limits: LimitsConfig = LimitsConfig()
    privacy: PrivacyConfig = PrivacyConfig()

    def validate(self) -> None:
        self.provider.validate()
        self.limits.validate()
        names: set[str] = set()
        for command in self.verification_commands:
            command.validate()
            if command.name in names:
                raise ConfigError(f"duplicate verification command name: {command.name}")
            names.add(command.name)

    def to_toml(self) -> str:
        self.validate()
        lines = [
            "[provider]",
            f'name = "{_escape(self.provider.name.value)}"',
            f'model = "{_escape(self.provider.model)}"',
            f'api_key_env = "{_escape(self.provider.api_key_env)}"',
        ]
        if self.provider.base_url is not None:
            lines.append(f'base_url = "{_escape(self.provider.base_url)}"')
        lines.extend(
            [
                "",
                "[limits]",
                f"max_context_files = {self.limits.max_context_files}",
                f"max_file_bytes = {self.limits.max_file_bytes}",
                f"max_output_bytes = {self.limits.max_output_bytes}",
                f"max_repair_attempts = {self.limits.max_repair_attempts}",
                "",
                "[privacy]",
                f"send_git_history = {_toml_bool(self.privacy.send_git_history)}",
                f"allow_local_embeddings = {_toml_bool(self.privacy.allow_local_embeddings)}",
                f"include_untracked_files = {_toml_bool(self.privacy.include_untracked_files)}",
            ]
        )
        for command in self.verification_commands:
            lines.extend(
                [
                    "",
                    "[[verification.commands]]",
                    f'name = "{_escape(command.name)}"',
                    f"command = [{', '.join(_quoted(part) for part in command.command)}]",
                    f"timeout_seconds = {command.timeout_seconds}",
                ]
            )
        return "\n".join(lines) + "\n"


def default_config() -> SchemavoreConfig:
    return SchemavoreConfig()


def load_config(path: Path) -> SchemavoreConfig:
    if not path.exists():
        config = default_config()
        config.validate()
        return config
    with path.open("rb") as file:
        raw_config = tomllib.load(file)
    return config_from_mapping(raw_config)


def save_config(path: Path, config: SchemavoreConfig) -> None:
    config.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.to_toml(), encoding="utf-8")


def config_from_mapping(raw_config: dict[str, Any]) -> SchemavoreConfig:
    _reject_unknown(raw_config, {"provider", "verification", "limits", "privacy"}, "config")
    if _contains_forbidden_secret_key(raw_config):
        raise ConfigError("configuration must not contain API key values")

    provider = _provider_from_mapping(_table(raw_config, "provider", required=False))
    limits = _limits_from_mapping(_table(raw_config, "limits", required=False))
    privacy = _privacy_from_mapping(_table(raw_config, "privacy", required=False))
    verification_commands = _verification_from_mapping(
        _table(raw_config, "verification", required=False)
    )
    config = SchemavoreConfig(
        provider=provider,
        verification_commands=verification_commands,
        limits=limits,
        privacy=privacy,
    )
    config.validate()
    return config


def _provider_from_mapping(raw_provider: dict[str, Any]) -> ProviderConfig:
    _reject_unknown(raw_provider, {"name", "model", "api_key_env", "base_url"}, "provider")
    defaults = ProviderConfig()
    raw_name = raw_provider.get("name", defaults.name.value)
    if not isinstance(raw_name, str):
        raise ConfigError("provider.name must be a string")
    try:
        name = ProviderName(raw_name)
    except ValueError as exc:
        raise ConfigError(f"unsupported provider.name: {raw_name}") from exc
    model = _string(raw_provider, "model", defaults.model)
    api_key_env = _string(raw_provider, "api_key_env", defaults.api_key_env)
    base_url = _optional_string(raw_provider, "base_url", defaults.base_url)
    return ProviderConfig(
        name=name,
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
    )


def _verification_from_mapping(raw_verification: dict[str, Any]) -> tuple[VerificationCommandConfig, ...]:
    _reject_unknown(raw_verification, {"commands"}, "verification")
    raw_commands = raw_verification.get("commands", [])
    if not isinstance(raw_commands, list):
        raise ConfigError("verification.commands must be a list")
    commands = []
    for index, raw_command in enumerate(raw_commands):
        if not isinstance(raw_command, dict):
            raise ConfigError(f"verification.commands[{index}] must be a table")
        _reject_unknown(raw_command, {"name", "command", "timeout_seconds"}, f"verification.commands[{index}]")
        name = _string(raw_command, "name")
        command_parts = raw_command.get("command")
        if not isinstance(command_parts, list) or not all(
            isinstance(part, str) for part in command_parts
        ):
            raise ConfigError(f"verification.commands[{index}].command must be a list of strings")
        timeout_seconds = _integer(raw_command, "timeout_seconds", 60)
        commands.append(
            VerificationCommandConfig(
                name=name,
                command=tuple(command_parts),
                timeout_seconds=timeout_seconds,
            )
        )
    return tuple(commands)


def _limits_from_mapping(raw_limits: dict[str, Any]) -> LimitsConfig:
    _reject_unknown(
        raw_limits,
        {"max_context_files", "max_file_bytes", "max_output_bytes", "max_repair_attempts"},
        "limits",
    )
    defaults = LimitsConfig()
    return LimitsConfig(
        max_context_files=_integer(raw_limits, "max_context_files", defaults.max_context_files),
        max_file_bytes=_integer(raw_limits, "max_file_bytes", defaults.max_file_bytes),
        max_output_bytes=_integer(raw_limits, "max_output_bytes", defaults.max_output_bytes),
        max_repair_attempts=_integer(
            raw_limits,
            "max_repair_attempts",
            defaults.max_repair_attempts,
        ),
    )


def _privacy_from_mapping(raw_privacy: dict[str, Any]) -> PrivacyConfig:
    _reject_unknown(
        raw_privacy,
        {"send_git_history", "allow_local_embeddings", "include_untracked_files"},
        "privacy",
    )
    defaults = PrivacyConfig()
    return PrivacyConfig(
        send_git_history=_boolean(raw_privacy, "send_git_history", defaults.send_git_history),
        allow_local_embeddings=_boolean(
            raw_privacy,
            "allow_local_embeddings",
            defaults.allow_local_embeddings,
        ),
        include_untracked_files=_boolean(
            raw_privacy,
            "include_untracked_files",
            defaults.include_untracked_files,
        ),
    )


def _table(raw: dict[str, Any], key: str, *, required: bool) -> dict[str, Any]:
    value = raw.get(key)
    if value is None:
        if required:
            raise ConfigError(f"{key} table is required")
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a table")
    return value


def _string(raw: dict[str, Any], key: str, default: str | None = None) -> str:
    value = raw.get(key, default)
    if value is None:
        raise ConfigError(f"{key} is required")
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string")
    return value


def _optional_string(raw: dict[str, Any], key: str, default: str | None = None) -> str | None:
    value = raw.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string")
    return value


def _integer(raw: dict[str, Any], key: str, default: int | None = None) -> int:
    value = raw.get(key, default)
    if value is None:
        raise ConfigError(f"{key} is required")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{key} must be an integer")
    return value


def _boolean(raw: dict[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean")
    return value


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ConfigError(f"{name} must be at least 1")


def _reject_unknown(raw: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"unknown {context} keys: {', '.join(unknown)}")


def _contains_forbidden_secret_key(value: Any) -> bool:
    if isinstance(value, dict):
        for raw_key, raw_value in value.items():
            if raw_key.lower() in {"api_key", "token", "secret"}:
                return True
            if _contains_forbidden_secret_key(raw_value):
                return True
    if isinstance(value, list):
        return any(_contains_forbidden_secret_key(item) for item in value)
    return False


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _quoted(value: str) -> str:
    return f'"{_escape(value)}"'


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"
