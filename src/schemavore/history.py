"""Read-only Git history mining with Python symbol attribution."""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from schemavore.privacy import IgnoreMatcher, is_secret_filename
from schemavore.profiler import Symbol


class GitHistoryError(RuntimeError):
    """Raised when Git history cannot be read from the repository."""


@dataclass(frozen=True, slots=True)
class HistoryFilter:
    """Optional constraints applied before commit evidence is returned."""

    author: str | None = None
    paths: tuple[str, ...] = ()
    since: datetime | None = None
    until: datetime | None = None
    merge_status: bool | None = None


@dataclass(frozen=True, slots=True)
class ChangedRegion:
    """A changed Python line range and symbols overlapping that range."""

    path: str
    start_line: int
    end_line: int
    change_kind: str
    symbols: tuple[Symbol, ...] = ()


@dataclass(frozen=True, slots=True)
class HistoryCommit:
    """A selected commit and its attributable Python changes."""

    oid: str
    author_name: str
    author_email: str
    authored_at: datetime
    subject: str
    parents: tuple[str, ...]
    regions: tuple[ChangedRegion, ...]

    @property
    def is_merge(self) -> bool:
        return len(self.parents) > 1


class GitHistoryMiner:
    """Mines selected Git evidence without changing repository state."""

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root.resolve()
        self._ignore_matcher = IgnoreMatcher.load(self._repo_root)

    def mine(self, filters: HistoryFilter | None = None) -> tuple[HistoryCommit, ...]:
        """Return commits matching `filters`, newest first as reported by Git."""
        filters = filters or HistoryFilter()
        commit_ids = self._commit_ids(filters)
        return tuple(self._commit(commit_id) for commit_id in commit_ids)

    def _commit_ids(self, filters: HistoryFilter) -> tuple[str, ...]:
        arguments = ["log", "--format=%H"]
        if filters.author:
            arguments.append(f"--author={filters.author}")
        if filters.since:
            arguments.append(f"--since={_git_date(filters.since)}")
        if filters.until:
            arguments.append(f"--until={_git_date(filters.until)}")
        if filters.merge_status is True:
            arguments.append("--merges")
        elif filters.merge_status is False:
            arguments.append("--no-merges")
        if filters.paths:
            arguments.extend(("--", *filters.paths))
        return tuple(line for line in self._run(*arguments).splitlines() if line)

    def _commit(self, commit_id: str) -> HistoryCommit:
        metadata = self._run("show", "-s", "--format=%H%x1f%an%x1f%ae%x1f%aI%x1f%P%x1f%s", commit_id)
        oid, author_name, author_email, authored_at, parents, subject = metadata.rstrip("\n").split("\x1f")
        parent_ids = tuple(parents.split())
        regions = tuple(self._regions(commit_id, parent_ids))
        return HistoryCommit(
            oid=oid,
            author_name=author_name,
            author_email=author_email,
            authored_at=datetime.fromisoformat(authored_at),
            subject=subject,
            parents=parent_ids,
            regions=regions,
        )

    def _regions(self, commit_id: str, parents: tuple[str, ...]) -> Iterable[ChangedRegion]:
        parent = parents[0] if parents else None
        status = self._run("diff-tree", "--root", "--no-commit-id", "-r", "--name-status", "-z", commit_id)
        for change_kind, path in _name_status_entries(status):
            if not self._include_python_path(path):
                continue
            diff = self._diff(commit_id, parent, path)
            source_revision = parent if change_kind == "D" else commit_id
            symbols = self._symbols_at(source_revision, path)
            for start_line, end_line in _changed_ranges(diff, deletion=change_kind == "D"):
                yield ChangedRegion(
                    path=path,
                    start_line=start_line,
                    end_line=end_line,
                    change_kind=change_kind,
                    symbols=tuple(symbol for symbol in symbols if _overlaps(symbol, start_line, end_line)),
                )

    def _diff(self, commit_id: str, parent: str | None, path: str) -> str:
        if parent is None:
            return self._run("diff-tree", "--root", "--no-commit-id", "-U0", commit_id, "--", path)
        return self._run("diff", "--no-ext-diff", "-U0", parent, commit_id, "--", path)

    def _symbols_at(self, revision: str | None, path: str) -> tuple[Symbol, ...]:
        if revision is None:
            return ()
        result = self._run_result("show", f"{revision}:{path}")
        if result.returncode:
            return ()
        try:
            tree = ast.parse(result.stdout, filename=path)
        except SyntaxError:
            return ()
        return tuple(_symbols(tree, path))

    def _include_python_path(self, path: str) -> bool:
        return path.endswith(".py") and not is_secret_filename(path) and not self._ignore_matcher.is_ignored(path)

    def _run(self, *arguments: str) -> str:
        result = self._run_result(*arguments)
        if result.returncode:
            message = result.stderr.strip() or result.stdout.strip() or "unknown Git error"
            raise GitHistoryError(message)
        return result.stdout

    def _run_result(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ("git", *arguments),
                cwd=self._repo_root,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except OSError as exc:
            raise GitHistoryError(str(exc)) from exc


def mine_history(repo_root: Path, filters: HistoryFilter | None = None) -> tuple[HistoryCommit, ...]:
    """Return read-only Git evidence for `repo_root`."""
    return GitHistoryMiner(repo_root).mine(filters)


def _git_date(value: datetime) -> str:
    return value.isoformat()


def _name_status_entries(content: str) -> Iterable[tuple[str, str]]:
    fields = iter(content.split("\0"))
    for status in fields:
        if not status:
            continue
        path = next(fields, "")
        if status[0] in {"R", "C"}:
            path = next(fields, "")
        if path:
            yield status[0], path


def _changed_ranges(diff: str, *, deletion: bool) -> Iterable[tuple[int, int]]:
    for line in diff.splitlines():
        if not line.startswith("@@"):
            continue
        old_start, old_count, new_start, new_count = _hunk_ranges(line)
        start, count = (old_start, old_count) if deletion else (new_start, new_count)
        if count:
            yield start, start + count - 1


def _hunk_ranges(header: str) -> tuple[int, int, int, int]:
    parts = header.split("@@", 2)[1].strip().split()
    return (*_parse_hunk_range(parts[0]), *_parse_hunk_range(parts[1]))


def _parse_hunk_range(value: str) -> tuple[int, int]:
    start_and_count = value[1:].split(",", 1)
    return int(start_and_count[0]), int(start_and_count[1]) if len(start_and_count) == 2 else 1


def _symbols(tree: ast.Module, path: str) -> Iterable[Symbol]:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield Symbol(node.name, "function", path, node.lineno, node.end_lineno or node.lineno)
        elif isinstance(node, ast.ClassDef):
            yield Symbol(node.name, "class", path, node.lineno, node.end_lineno or node.lineno)


def _overlaps(symbol: Symbol, start_line: int, end_line: int) -> bool:
    return symbol.line <= end_line and start_line <= symbol.end_line
