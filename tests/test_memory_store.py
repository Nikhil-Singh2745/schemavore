from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

from schemavore.cli import main
from schemavore.domain import Memory, MemoryCategory, MemoryStatus
from schemavore.memory_store import MemoryConflictError, MemoryStore, MemoryStoreError


class MemoryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repository = Path(self.temporary_directory.name)
        self.store = MemoryStore(self.repository / ".schemavore" / "rules.yaml")

    def test_round_trips_human_readable_yaml(self) -> None:
        memory = self._memory("mem_typing", "Add type annotations.")

        self.store.save((memory,))

        self.assertEqual(self.store.load(), (memory,))
        content = self.store.path.read_text(encoding="utf-8")
        self.assertIn("statement: Add type annotations.", content)
        self.assertNotIn("!!python", content)

    def test_approval_is_explicit_and_only_approved_memories_are_active(self) -> None:
        candidate = self._memory("mem_typing", "Add type annotations.")
        rejected = self._memory("mem_imports", "Prefer module imports.", MemoryStatus.REJECTED)
        self.store.save((candidate, rejected))

        self.assertEqual(self.store.approved(), ())
        approved = self.store.approve("mem_typing", now=self._later())

        self.assertEqual(approved.status, MemoryStatus.APPROVED)
        self.assertEqual(self.store.approved(), (approved,))
        with self.assertRaisesRegex(MemoryStoreError, "cannot approve a rejected"):
            self.store.approve("mem_imports")

    def test_rejects_conflicting_approval_in_overlapping_scope(self) -> None:
        approved = self._memory(
            "mem_from_imports",
            "Prefer from imports.",
            MemoryStatus.APPROVED,
            category=MemoryCategory.IMPORTS,
            scope=("src",),
        )
        candidate = self._memory(
            "mem_module_imports",
            "Prefer module imports.",
            category=MemoryCategory.IMPORTS,
            scope=("src/pkg",),
        )
        self.store.save((approved, candidate))

        with self.assertRaisesRegex(MemoryConflictError, "mem_from_imports"):
            self.store.approve(candidate.id)

        self.assertEqual(self.store.get(candidate.id).status, MemoryStatus.CANDIDATE)

    def test_allows_same_category_in_disjoint_scopes(self) -> None:
        approved = self._memory(
            "mem_src",
            "Prefer from imports.",
            MemoryStatus.APPROVED,
            category=MemoryCategory.IMPORTS,
            scope=("src",),
        )
        candidate = self._memory(
            "mem_tests",
            "Prefer module imports.",
            category=MemoryCategory.IMPORTS,
            scope=("tests",),
        )
        self.store.save((approved, candidate))

        self.assertEqual(self.store.approve(candidate.id).status, MemoryStatus.APPROVED)

    def test_supports_rejection_and_superseding(self) -> None:
        candidate = self._memory("mem_candidate", "Add type annotations.")
        approved = self._memory("mem_approved", "Annotate public functions.", MemoryStatus.APPROVED)
        self.store.save((candidate, approved))

        self.assertEqual(self.store.reject(candidate.id).status, MemoryStatus.REJECTED)
        self.assertEqual(self.store.supersede(approved.id).status, MemoryStatus.SUPERSEDED)
        self.assertEqual(self.store.approved(), ())

    def test_rejects_unknown_yaml_fields(self) -> None:
        self.store.path.parent.mkdir(parents=True)
        self.store.path.write_text("version: 1\nmemories: []\nunexpected: true\n", encoding="utf-8")

        with self.assertRaisesRegex(MemoryStoreError, "version and memories"):
            self.store.load()

    def test_cli_lists_inspects_and_approves(self) -> None:
        candidate = self._memory("mem_typing", "Add type annotations.")
        self.store.save((candidate,))
        output = io.StringIO()

        with redirect_stdout(output):
            self.assertEqual(main(("--repository", str(self.repository), "memories", "list")), 0)
            self.assertEqual(main(("--repository", str(self.repository), "memories", "show", candidate.id)), 0)
            self.assertEqual(main(("--repository", str(self.repository), "memories", "approve", candidate.id)), 0)

        rendered = output.getvalue()
        self.assertIn("mem_typing\tcandidate", rendered)
        self.assertIn("Statement: Add type annotations.", rendered)
        self.assertIn("mem_typing is now approved", rendered)

    @staticmethod
    def _memory(
        memory_id: str,
        statement: str,
        status: MemoryStatus = MemoryStatus.CANDIDATE,
        *,
        category: MemoryCategory = MemoryCategory.TYPING,
        scope: tuple[str, ...] = ("src",),
    ) -> Memory:
        observed_at = datetime(2025, 1, 1, tzinfo=UTC)
        return Memory(
            id=memory_id,
            statement=statement,
            category=category,
            scope=scope,
            evidence_ids=("ev_001",),
            confidence=0.8,
            status=status,
            created_at=observed_at,
            updated_at=observed_at,
        )

    @staticmethod
    def _later() -> datetime:
        return datetime(2025, 1, 2, tzinfo=UTC)


if __name__ == "__main__":
    unittest.main()
