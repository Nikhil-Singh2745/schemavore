"""Human-readable persistence and lifecycle management for memories."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

from schemavore.domain import Memory, MemoryCategory, MemoryStatus, utc_now


class MemoryStoreError(ValueError):
    """Raised when stored memories are invalid or a transition is unsafe."""


class MemoryNotFoundError(MemoryStoreError):
    """Raised when a requested memory does not exist."""


class MemoryConflictError(MemoryStoreError):
    """Raised when approval would create conflicting active memories."""


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> tuple[Memory, ...]:
        if not self.path.exists():
            return ()
        try:
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise MemoryStoreError(f"could not read memory store: {exc}") from exc
        if raw is None:
            return ()
        if not isinstance(raw, dict) or set(raw) != {"version", "memories"}:
            raise MemoryStoreError("memory store must contain only version and memories")
        if raw["version"] != 1:
            raise MemoryStoreError(f"unsupported memory store version: {raw['version']}")
        if not isinstance(raw["memories"], list):
            raise MemoryStoreError("memories must be a list")
        memories = tuple(_memory_from_mapping(item, index) for index, item in enumerate(raw["memories"]))
        ids = [memory.id for memory in memories]
        if len(ids) != len(set(ids)):
            raise MemoryStoreError("memory ids must be unique")
        return memories

    def save(self, memories: tuple[Memory, ...]) -> None:
        ids = [memory.id for memory in memories]
        if len(ids) != len(set(ids)):
            raise MemoryStoreError("memory ids must be unique")
        document = {
            "version": 1,
            "memories": [_memory_to_mapping(memory) for memory in sorted(memories, key=lambda item: item.id)],
        }
        content = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as file:
                file.write(content)
                temporary_path = Path(file.name)
            temporary_path.replace(self.path)
        except OSError as exc:
            raise MemoryStoreError(f"could not write memory store: {exc}") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def add_candidates(self, candidates: tuple[Memory, ...]) -> tuple[Memory, ...]:
        memories = {memory.id: memory for memory in self.load()}
        for candidate in candidates:
            if candidate.status is not MemoryStatus.CANDIDATE:
                raise MemoryStoreError("new inferred memories must be candidates")
            existing = memories.get(candidate.id)
            if existing is not None and existing.status is not MemoryStatus.CANDIDATE:
                continue
            memories[candidate.id] = candidate
        result = tuple(memories.values())
        self.save(result)
        return tuple(sorted(result, key=lambda item: item.id))

    def get(self, memory_id: str) -> Memory:
        return _find(self.load(), memory_id)

    def approve(self, memory_id: str, *, now: datetime | None = None) -> Memory:
        memories = self.load()
        memory = _find(memories, memory_id)
        if memory.status is MemoryStatus.APPROVED:
            return memory
        if memory.status is not MemoryStatus.CANDIDATE:
            raise MemoryStoreError(f"cannot approve a {memory.status.value} memory")
        conflicts = find_conflicts(memory, memories)
        if conflicts:
            conflict_ids = ", ".join(item.id for item in conflicts)
            raise MemoryConflictError(f"memory conflicts with approved memories: {conflict_ids}")
        updated = replace(memory, status=MemoryStatus.APPROVED, updated_at=now or utc_now())
        self.save(_replace_memory(memories, updated))
        return updated

    def reject(self, memory_id: str, *, now: datetime | None = None) -> Memory:
        return self._transition(memory_id, MemoryStatus.CANDIDATE, MemoryStatus.REJECTED, now)

    def supersede(self, memory_id: str, *, now: datetime | None = None) -> Memory:
        return self._transition(memory_id, MemoryStatus.APPROVED, MemoryStatus.SUPERSEDED, now)

    def approved(self) -> tuple[Memory, ...]:
        return tuple(memory for memory in self.load() if memory.status is MemoryStatus.APPROVED)

    def _transition(
        self,
        memory_id: str,
        expected: MemoryStatus,
        target: MemoryStatus,
        now: datetime | None,
    ) -> Memory:
        memories = self.load()
        memory = _find(memories, memory_id)
        if memory.status is target:
            return memory
        if memory.status is not expected:
            raise MemoryStoreError(f"cannot mark a {memory.status.value} memory as {target.value}")
        updated = replace(memory, status=target, updated_at=now or utc_now())
        self.save(_replace_memory(memories, updated))
        return updated


def find_conflicts(memory: Memory, memories: tuple[Memory, ...]) -> tuple[Memory, ...]:
    return tuple(
        item
        for item in memories
        if item.id != memory.id
        and item.status is MemoryStatus.APPROVED
        and item.category is memory.category
        and item.statement != memory.statement
        and _scopes_overlap(item.scope, memory.scope)
    )


def _scopes_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return any(_scope_pair_overlaps(a, b) for a in left for b in right)


def _scope_pair_overlaps(left: str, right: str) -> bool:
    if "*" in (left, right):
        return True
    left_parts = PurePosixPath(left).parts
    right_parts = PurePosixPath(right).parts
    common_length = min(len(left_parts), len(right_parts))
    return left_parts[:common_length] == right_parts[:common_length]


def _find(memories: tuple[Memory, ...], memory_id: str) -> Memory:
    try:
        return next(memory for memory in memories if memory.id == memory_id)
    except StopIteration as exc:
        raise MemoryNotFoundError(f"memory not found: {memory_id}") from exc


def _replace_memory(memories: tuple[Memory, ...], updated: Memory) -> tuple[Memory, ...]:
    return tuple(updated if memory.id == updated.id else memory for memory in memories)


def _memory_to_mapping(memory: Memory) -> dict[str, Any]:
    return {
        "id": memory.id,
        "statement": memory.statement,
        "category": memory.category.value,
        "scope": list(memory.scope),
        "evidence_ids": list(memory.evidence_ids),
        "confidence": memory.confidence,
        "status": memory.status.value,
        "created_at": memory.created_at.isoformat(),
        "updated_at": memory.updated_at.isoformat(),
    }


def _memory_from_mapping(raw: Any, index: int) -> Memory:
    if not isinstance(raw, dict):
        raise MemoryStoreError(f"memories[{index}] must be a mapping")
    fields = {"id", "statement", "category", "scope", "evidence_ids", "confidence", "status", "created_at", "updated_at"}
    if set(raw) != fields:
        missing = sorted(fields - set(raw))
        unknown = sorted(set(raw) - fields)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise MemoryStoreError(f"invalid memories[{index}] fields: {'; '.join(details)}")
    try:
        if not isinstance(raw["scope"], list) or not all(isinstance(item, str) for item in raw["scope"]):
            raise TypeError("scope must be a list of strings")
        if not isinstance(raw["evidence_ids"], list) or not all(isinstance(item, str) for item in raw["evidence_ids"]):
            raise TypeError("evidence_ids must be a list of strings")
        if not isinstance(raw["confidence"], (int, float)) or isinstance(raw["confidence"], bool):
            raise TypeError("confidence must be a number")
        return Memory(
            id=_required_string(raw["id"], "id"),
            statement=_required_string(raw["statement"], "statement"),
            category=MemoryCategory(raw["category"]),
            scope=tuple(raw["scope"]),
            evidence_ids=tuple(raw["evidence_ids"]),
            confidence=float(raw["confidence"]),
            status=MemoryStatus(raw["status"]),
            created_at=datetime.fromisoformat(_required_string(raw["created_at"], "created_at")),
            updated_at=datetime.fromisoformat(_required_string(raw["updated_at"], "updated_at")),
        )
    except (TypeError, ValueError) as exc:
        raise MemoryStoreError(f"invalid memories[{index}]: {exc}") from exc


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value
