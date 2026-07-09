"""Ignore rules, secret and binary detection, safe path resolution, and context manifests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

from schemavore.config import LimitsConfig

IGNORE_FILE_NAME = ".schemavoreignore"

DEFAULT_IGNORE_PATTERNS: tuple[str, ...] = (
    ".git/",
    ".schemavore/",
    ".venv/",
    "venv/",
    "__pycache__/",
    "*.pyc",
    "node_modules/",
    "dist/",
    "build/",
    "*.egg-info/",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.pfx",
    "*.p12",
    "id_rsa",
    "id_rsa.pub",
    "id_ed25519",
    "id_ed25519.pub",
    "*credentials*.json",
    "*.sqlite",
    "*.sqlite3",
    "*.db",
)

SECRET_FILENAME_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*_rsa",
    "*_rsa.pub",
    "*_ed25519",
    "*_ed25519.pub",
    "*.pfx",
    "*.p12",
    "*credentials*",
    "*secret*",
    ".npmrc",
    ".pypirc",
    ".netrc",
)


class PathSafetyError(ValueError):
    """Raised when a candidate path is unsafe to resolve inside the repository root."""


class PathTraversalError(PathSafetyError):
    """Raised when a candidate path is absolute, empty, or contains traversal segments."""


class SymlinkEscapeError(PathSafetyError):
    """Raised when a resolved path escapes the repository root, typically through a symlink."""


def resolve_repository_path(repo_root: Path, relative_path: str) -> Path:
    """Resolve `relative_path` inside `repo_root`, rejecting traversal and symlink escapes."""
    posix_path = PurePosixPath(relative_path)
    if not posix_path.parts:
        raise PathTraversalError("path must not be empty")
    if posix_path.is_absolute():
        raise PathTraversalError(f"path must be relative: {relative_path}")
    if ".." in posix_path.parts:
        raise PathTraversalError(f"path must not contain traversal segments: {relative_path}")

    repo_root_resolved = repo_root.resolve()
    candidate = repo_root_resolved.joinpath(*posix_path.parts)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repo_root_resolved)
    except ValueError as exc:
        raise SymlinkEscapeError(f"path escapes the repository root: {relative_path}") from exc
    return resolved


def is_secret_filename(relative_path: str) -> bool:
    name = PurePosixPath(relative_path).name.lower()
    return any(_fnmatch_lower(name, pattern) for pattern in SECRET_FILENAME_PATTERNS)


def looks_binary(data: bytes) -> bool:
    if not data:
        return False
    if b"\x00" in data:
        return True
    text_bytes = bytes(range(32, 127)) + b"\n\r\t\f\b"
    nontext = data.translate(None, delete=text_bytes)
    return (len(nontext) / len(data)) > 0.30


@dataclass(frozen=True, slots=True)
class _CompiledPattern:
    regex: re.Pattern[str]
    negation: bool


class IgnoreMatcher:
    """A simplified gitignore-style matcher for `.schemavoreignore` files."""

    def __init__(self, patterns: Sequence[str]) -> None:
        self._compiled = [
            _compile_pattern(pattern) for pattern in patterns if _is_meaningful(pattern)
        ]

    @classmethod
    def load(cls, repo_root: Path, *, extra_patterns: Sequence[str] = ()) -> "IgnoreMatcher":
        lines: list[str] = list(DEFAULT_IGNORE_PATTERNS)
        ignore_file = repo_root / IGNORE_FILE_NAME
        if ignore_file.is_file():
            lines.extend(ignore_file.read_text(encoding="utf-8").splitlines())
        lines.extend(extra_patterns)
        return cls(lines)

    def is_ignored(self, relative_path: str) -> bool:
        posix_path = relative_path.replace("\\", "/").strip("/")
        ignored = False
        for compiled in self._compiled:
            if compiled.regex.match(posix_path):
                ignored = not compiled.negation
        return ignored


class ExclusionReason(StrEnum):
    MISSING = "missing"
    NOT_A_FILE = "not_a_file"
    TRAVERSAL = "traversal"
    SYMLINK_ESCAPE = "symlink_escape"
    IGNORED = "ignored"
    SECRET = "secret"
    BINARY = "binary"
    OVERSIZED = "oversized"


@dataclass(frozen=True, slots=True)
class ContextEntry:
    path: str
    included: bool
    reason: ExclusionReason | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.included and self.reason is not None:
            raise ValueError("included entries must not carry an exclusion reason")
        if not self.included and self.reason is None:
            raise ValueError("excluded entries must carry an exclusion reason")


@dataclass(frozen=True, slots=True)
class ContextManifest:
    entries: tuple[ContextEntry, ...]

    @property
    def included_paths(self) -> tuple[str, ...]:
        return tuple(entry.path for entry in self.entries if entry.included)

    @property
    def excluded_entries(self) -> tuple[ContextEntry, ...]:
        return tuple(entry for entry in self.entries if not entry.included)

    def to_report(self) -> str:
        lines = ["Context manifest:"]
        for entry in self.entries:
            if entry.included:
                lines.append(f"  include  {entry.path} ({entry.size_bytes} bytes)")
            else:
                assert entry.reason is not None
                lines.append(f"  exclude  {entry.path} [{entry.reason.value}]")
        return "\n".join(lines)


class RepositoryPrivacyGuard:
    """Decides which repository paths may be read and sent to a provider."""

    def __init__(self, repo_root: Path, limits: LimitsConfig, ignore_matcher: IgnoreMatcher) -> None:
        self._repo_root = repo_root.resolve()
        self._limits = limits
        self._ignore_matcher = ignore_matcher

    @classmethod
    def load(cls, repo_root: Path, limits: LimitsConfig) -> "RepositoryPrivacyGuard":
        return cls(repo_root, limits, IgnoreMatcher.load(repo_root))

    def evaluate(self, relative_path: str) -> ContextEntry:
        try:
            resolved = resolve_repository_path(self._repo_root, relative_path)
        except PathTraversalError:
            return ContextEntry(path=relative_path, included=False, reason=ExclusionReason.TRAVERSAL)
        except SymlinkEscapeError:
            return ContextEntry(
                path=relative_path, included=False, reason=ExclusionReason.SYMLINK_ESCAPE
            )

        if is_secret_filename(relative_path):
            return ContextEntry(path=relative_path, included=False, reason=ExclusionReason.SECRET)
        if self._ignore_matcher.is_ignored(relative_path):
            return ContextEntry(path=relative_path, included=False, reason=ExclusionReason.IGNORED)
        if not resolved.exists():
            return ContextEntry(path=relative_path, included=False, reason=ExclusionReason.MISSING)
        if resolved.is_symlink() or not resolved.is_file():
            return ContextEntry(path=relative_path, included=False, reason=ExclusionReason.NOT_A_FILE)

        size_bytes = resolved.stat().st_size
        if size_bytes > self._limits.max_file_bytes:
            return ContextEntry(path=relative_path, included=False, reason=ExclusionReason.OVERSIZED)

        with resolved.open("rb") as file:
            head = file.read(8192)
        if looks_binary(head):
            return ContextEntry(path=relative_path, included=False, reason=ExclusionReason.BINARY)

        return ContextEntry(path=relative_path, included=True, size_bytes=size_bytes)

    def build_manifest(self, relative_paths: Iterable[str]) -> ContextManifest:
        return ContextManifest(tuple(self.evaluate(path) for path in relative_paths))


def _is_meaningful(pattern: str) -> bool:
    stripped = pattern.strip()
    return bool(stripped) and not stripped.startswith("#")


def _compile_pattern(pattern: str) -> _CompiledPattern:
    stripped = pattern.strip()
    negation = stripped.startswith("!")
    if negation:
        stripped = stripped[1:]

    had_leading_slash = stripped.startswith("/")
    core = stripped[1:] if had_leading_slash else stripped
    dir_only = core.endswith("/")
    if dir_only:
        core = core[:-1]
    anchored = had_leading_slash or "/" in core

    segments = core.split("/")
    regex_segments = [_segment_to_regex(segment) for segment in segments]
    joined = "/".join(regex_segments)

    if anchored:
        regex_str = f"^{joined}(/.*)?$"
    else:
        regex_str = f"^(.*/)?{joined}(/.*)?$"

    return _CompiledPattern(regex=re.compile(regex_str), negation=negation)


def _segment_to_regex(segment: str) -> str:
    if segment == "**":
        return ".*"
    result = []
    index = 0
    while index < len(segment):
        char = segment[index]
        if char == "*":
            result.append("[^/]*")
        elif char == "?":
            result.append("[^/]")
        else:
            result.append(re.escape(char))
        index += 1
    return "".join(result)


def _fnmatch_lower(name: str, pattern: str) -> bool:
    regex = re.compile(f"^{_segment_to_regex(pattern.lower())}$")
    return regex.match(name) is not None
