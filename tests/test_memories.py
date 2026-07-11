from __future__ import annotations

import unittest
from datetime import UTC, datetime

from schemavore.history import ChangedRegion, HistoryCommit
from schemavore.memories import infer_candidate_memories
from schemavore.profiler import ConventionFeature, ModuleProfile, RepositoryProfile


class CandidateMemoryInferenceTests(unittest.TestCase):
    def test_infers_scoped_candidate_with_traceable_confidence(self) -> None:
        profile = RepositoryProfile(
            modules=(
                self._module("src/pkg/one.py", "snake_case", 3),
                self._module("src/pkg/two.py", "snake_case", 2),
                self._module("src/pkg/legacy.py", "other", 1),
            ),
            tooling=(),
        )
        history = (
            self._commit("a" * 40, "src/pkg/one.py"),
            self._commit("b" * 40, "src/pkg/two.py"),
        )

        first = infer_candidate_memories(profile, history)
        second = infer_candidate_memories(profile, history)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        candidate = first[0]
        self.assertEqual(candidate.memory.statement, "Name functions using snake_case.")
        self.assertEqual(candidate.memory.scope, ("src/pkg",))
        self.assertEqual(candidate.memory.confidence, 0.5556)
        self.assertEqual((candidate.file_support, candidate.commit_support), (2, 2))
        self.assertEqual(candidate.memory.evidence_ids, tuple(item.id for item in candidate.evidence))
        self.assertEqual(len(candidate.evidence), 4)

    def test_rejects_single_file_and_single_commit_observations(self) -> None:
        profile = RepositoryProfile(
            modules=(self._module("src/feature.py", "snake_case", 8),),
            tooling=(),
        )
        history = (self._commit("a" * 40, "src/feature.py"),)

        self.assertEqual(infer_candidate_memories(profile, history), ())

    def test_rejects_inconsistent_and_unrecognized_facts(self) -> None:
        modules = (
            self._module("one.py", "snake_case", 1),
            self._module("two.py", "snake_case", 1),
            self._module("three.py", "other", 2),
            ModuleProfile(
                path="task.py",
                features=(ConventionFeature("structure", "ticket_123", "special_case", 10, ("task.py",)),),
            ),
        )
        profile = RepositoryProfile(modules=modules, tooling=())
        history = tuple(self._commit(character * 40, path) for character, path in zip("abcd", ("one.py", "two.py", "three.py", "task.py")))

        self.assertEqual(infer_candidate_memories(profile, history), ())

    @staticmethod
    def _module(path: str, value: str, count: int) -> ModuleProfile:
        return ModuleProfile(
            path=path,
            features=(ConventionFeature("naming", "function", value, count, (path,)),),
        )

    @staticmethod
    def _commit(oid: str, path: str) -> HistoryCommit:
        return HistoryCommit(
            oid=oid,
            author_name="Developer",
            author_email="developer@example.com",
            authored_at=datetime(2024, 1, 1, tzinfo=UTC),
            subject="convention evidence",
            parents=(),
            regions=(ChangedRegion(path, 1, 10, "M"),),
        )


if __name__ == "__main__":
    unittest.main()
