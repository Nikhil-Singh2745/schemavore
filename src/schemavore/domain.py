"""Typed domain contracts for Schemavore."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath


class EvidenceKind(StrEnum):
    CONFIG = "config"
    FILE = "file"
    GIT_COMMIT = "git_commit"
    CORRECTION = "correction"
    TEST = "test"


class MemoryCategory(StrEnum):
    NAMING = "naming"
    TYPING = "typing"
    IMPORTS = "imports"
    EXCEPTIONS = "exceptions"
    TESTING = "testing"
    STRUCTURE = "structure"
    STYLE = "style"


class MemoryStatus(StrEnum):
    CANDIDATE = "candidate"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class EditKind(StrEnum):
    CREATE = "create"
    REPLACE = "replace"
    DELETE = "delete"


class SessionStatus(StrEnum):
    CREATED = "created"
    GENERATED = "generated"
    VERIFIED = "verified"
    FAILED = "failed"
    APPLIED = "applied"


class VerificationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    SKIPPED = "skipped"


def utc_now() -> datetime:
    return datetime.now(UTC)


def _validate_relative_path(path: str, field_name: str) -> None:
    if not path:
        raise ValueError(f"{field_name} must not be empty")
    posix_path = PurePosixPath(path)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise ValueError(f"{field_name} must be a relative path inside the repository")


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    id: str
    kind: EvidenceKind
    source: str
    summary: str
    path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    commit: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("evidence id must not be empty")
        if not self.source:
            raise ValueError("evidence source must not be empty")
        if not self.summary:
            raise ValueError("evidence summary must not be empty")
        if self.path is not None:
            _validate_relative_path(self.path, "evidence path")
        if self.line_start is not None and self.line_start < 1:
            raise ValueError("line_start must be at least 1")
        if self.line_end is not None and self.line_end < 1:
            raise ValueError("line_end must be at least 1")
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("line_end must be greater than or equal to line_start")


@dataclass(frozen=True, slots=True)
class Memory:
    id: str
    statement: str
    category: MemoryCategory
    scope: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    confidence: float
    status: MemoryStatus = MemoryStatus.CANDIDATE
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("memory id must not be empty")
        if not self.statement:
            raise ValueError("memory statement must not be empty")
        if not self.scope:
            raise ValueError("memory scope must not be empty")
        for path_scope in self.scope:
            if path_scope != "*":
                _validate_relative_path(path_scope, "memory scope")
        if not self.evidence_ids:
            raise ValueError("memory evidence_ids must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("memory confidence must be between 0.0 and 1.0")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be earlier than created_at")


@dataclass(frozen=True, slots=True)
class EditOperation:
    id: str
    kind: EditKind
    path: str
    content: str | None = None
    find: str | None = None
    replace: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("edit id must not be empty")
        _validate_relative_path(self.path, "edit path")
        if self.kind is EditKind.CREATE and self.content is None:
            raise ValueError("create edits require content")
        if self.kind is EditKind.REPLACE and (self.find is None or self.replace is None):
            raise ValueError("replace edits require find and replace values")
        if self.kind is EditKind.DELETE and any(
            value is not None for value in (self.content, self.find, self.replace)
        ):
            raise ValueError("delete edits must not include content, find, or replace")


@dataclass(frozen=True, slots=True)
class VerificationCommandResult:
    command: tuple[str, ...]
    exit_code: int | None
    duration_seconds: float
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    def __post_init__(self) -> None:
        if not self.command or any(not part for part in self.command):
            raise ValueError("verification command must contain non-empty parts")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must not be negative")
        if self.timed_out and self.exit_code is not None:
            raise ValueError("timed out commands must not have an exit code")


@dataclass(frozen=True, slots=True)
class VerificationResult:
    status: VerificationStatus
    commands: tuple[VerificationCommandResult, ...] = ()
    summary: str = ""

    def __post_init__(self) -> None:
        if self.status is VerificationStatus.SKIPPED and self.commands:
            raise ValueError("skipped verification must not include command results")


@dataclass(frozen=True, slots=True)
class Session:
    id: str
    task: str
    worktree_path: str
    status: SessionStatus
    edits: tuple[EditOperation, ...] = ()
    verification: VerificationResult | None = None
    context_paths: tuple[str, ...] = ()
    approved_memory_ids: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("session id must not be empty")
        if not self.task:
            raise ValueError("session task must not be empty")
        if not self.worktree_path:
            raise ValueError("session worktree_path must not be empty")
        for path in self.context_paths:
            _validate_relative_path(path, "context path")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be earlier than created_at")
