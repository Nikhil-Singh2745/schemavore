"""Deterministic candidate memory inference from repository evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath

from schemavore.domain import (
    EvidenceKind,
    EvidenceReference,
    Memory,
    MemoryCategory,
)
from schemavore.history import HistoryCommit
from schemavore.profiler import ConventionFeature, RepositoryProfile


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    """A candidate memory and the evidence needed to inspect it."""

    memory: Memory
    evidence: tuple[EvidenceReference, ...]
    file_support: int
    commit_support: int


_STATEMENTS: dict[tuple[str, str, str], str] = {
    ("naming", "function", "snake_case"): "Name functions using snake_case.",
    ("naming", "class", "pascal_case"): "Name classes using PascalCase.",
    ("typing", "function_annotations", "present"): "Add type annotations to functions.",
    ("imports", "form", "from_import"): "Prefer from imports.",
    ("imports", "form", "import"): "Prefer module imports.",
    ("exceptions", "except", "typed"): "Catch explicit exception types.",
    ("testing", "test_function", "pytest_style"): "Write tests as pytest-style functions.",
    ("testing", "test_class", "unittest_style"): "Write tests using unittest test classes.",
    ("structure", "class_decorator", "dataclass"): "Use dataclasses for data-focused classes.",
}


def infer_candidate_memories(
    profile: RepositoryProfile,
    history: tuple[HistoryCommit, ...],
    *,
    minimum_files: int = 2,
    minimum_commits: int = 2,
    minimum_consistency: float = 0.75,
) -> tuple[MemoryCandidate, ...]:
    """Infer scoped candidates supported repeatedly in files and history."""
    if minimum_files < 1 or minimum_commits < 1:
        raise ValueError("support thresholds must be positive")
    if not 0.0 <= minimum_consistency <= 1.0:
        raise ValueError("minimum_consistency must be between 0.0 and 1.0")

    all_features = tuple(feature for module in profile.modules for feature in module.features)
    recognized_dimensions = {(category, name) for category, name, _ in _STATEMENTS}
    features = tuple(
        feature
        for feature in all_features
        if (feature.category, feature.name, feature.value) in _STATEMENTS
    )
    totals = _observation_totals(
        tuple(
            feature
            for feature in all_features
            if (feature.category, feature.name) in recognized_dimensions
        )
    )
    candidates: list[MemoryCandidate] = []
    for key in sorted({(item.category, item.name, item.value) for item in features}):
        supporting = tuple(item for item in features if _feature_key(item) == key)
        paths = tuple(sorted({path for item in supporting for path in item.paths}))
        commits = _supporting_commits(history, paths)
        count = sum(item.count for item in supporting)
        consistency = count / totals[(key[0], key[1])]
        if (
            len(paths) < minimum_files
            or len(commits) < minimum_commits
            or consistency < minimum_consistency
        ):
            continue

        scope = _scope(paths)
        statement = _STATEMENTS[key]
        evidence = _evidence(key, supporting, commits)
        confidence = round(consistency * min(1.0, (len(paths) + len(commits)) / 6), 4)
        identity = f"{key[0]}:{key[1]}:{key[2]}:{','.join(scope)}"
        observed_at = max(commit.authored_at for commit in commits)
        memory = Memory(
            id=_stable_id("mem", identity),
            statement=statement,
            category=MemoryCategory(key[0]),
            scope=scope,
            evidence_ids=tuple(item.id for item in evidence),
            confidence=confidence,
            created_at=observed_at,
            updated_at=observed_at,
        )
        candidates.append(MemoryCandidate(memory, evidence, len(paths), len(commits)))
    return tuple(candidates)


def _observation_totals(
    features: tuple[ConventionFeature, ...],
) -> dict[tuple[str, str], int]:
    totals: dict[tuple[str, str], int] = {}
    for feature in features:
        key = (feature.category, feature.name)
        totals[key] = totals.get(key, 0) + feature.count
    return totals


def _feature_key(feature: ConventionFeature) -> tuple[str, str, str]:
    return feature.category, feature.name, feature.value


def _supporting_commits(
    history: tuple[HistoryCommit, ...], paths: tuple[str, ...]
) -> tuple[HistoryCommit, ...]:
    path_set = set(paths)
    return tuple(
        sorted(
            (
                commit
                for commit in history
                if any(region.path in path_set for region in commit.regions)
            ),
            key=lambda item: item.oid,
        )
    )


def _scope(paths: tuple[str, ...]) -> tuple[str, ...]:
    parent_parts = [PurePosixPath(path).parent.parts for path in paths]
    common: list[str] = []
    for parts in zip(*parent_parts):
        if len(set(parts)) != 1:
            break
        common.append(parts[0])
    return (PurePosixPath(*common).as_posix(),) if common else ("*",)


def _evidence(
    key: tuple[str, str, str],
    features: tuple[ConventionFeature, ...],
    commits: tuple[HistoryCommit, ...],
) -> tuple[EvidenceReference, ...]:
    summary = f"Observed {key[1]}={key[2]}."
    references = [
        EvidenceReference(
            id=_stable_id("ev", f"file:{path}:{':'.join(key)}"),
            kind=EvidenceKind.FILE,
            source=path,
            summary=summary,
            path=path,
        )
        for path in sorted({path for feature in features for path in feature.paths})
    ]
    references.extend(
        EvidenceReference(
            id=_stable_id("ev", f"commit:{commit.oid}:{':'.join(key)}"),
            kind=EvidenceKind.GIT_COMMIT,
            source=commit.oid,
            summary=f"Commit {commit.oid[:12]} supports {key[1]}={key[2]}.",
            commit=commit.oid,
        )
        for commit in commits
    )
    return tuple(references)


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
