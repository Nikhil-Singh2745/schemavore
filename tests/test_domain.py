from __future__ import annotations

import unittest

from schemavore.domain import (
    EditKind,
    EditOperation,
    EvidenceKind,
    EvidenceReference,
    Memory,
    MemoryCategory,
    VerificationCommandResult,
    VerificationResult,
    VerificationStatus,
)


class DomainContractTests(unittest.TestCase):
    def test_memory_accepts_traceable_evidence(self) -> None:
        evidence = EvidenceReference(
            id="ev_001",
            kind=EvidenceKind.FILE,
            source="src/example.py",
            summary="Tests use unittest assertions.",
            path="src/example.py",
            line_start=10,
            line_end=18,
        )
        memory = Memory(
            id="mem_001",
            statement="Use unittest assertions in tests.",
            category=MemoryCategory.TESTING,
            scope=("tests",),
            evidence_ids=(evidence.id,),
            confidence=0.75,
        )

        self.assertEqual(memory.evidence_ids, ("ev_001",))

    def test_memory_rejects_invalid_confidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "confidence"):
            Memory(
                id="mem_001",
                statement="Use unittest.",
                category=MemoryCategory.TESTING,
                scope=("*",),
                evidence_ids=("ev_001",),
                confidence=1.5,
            )

    def test_edit_contracts_validate_required_fields(self) -> None:
        EditOperation(
            id="edit_001",
            kind=EditKind.CREATE,
            path="src/new_file.py",
            content="VALUE = 1\n",
        )

        with self.assertRaisesRegex(ValueError, "replace edits require"):
            EditOperation(
                id="edit_002",
                kind=EditKind.REPLACE,
                path="src/existing.py",
            )

    def test_repository_paths_must_be_relative(self) -> None:
        with self.assertRaisesRegex(ValueError, "relative path"):
            EvidenceReference(
                id="ev_001",
                kind=EvidenceKind.FILE,
                source="outside",
                summary="Invalid path.",
                path="../outside.py",
            )

    def test_verification_result_contracts(self) -> None:
        result = VerificationResult(
            status=VerificationStatus.PASSED,
            commands=(
                VerificationCommandResult(
                    command=("python3", "-m", "unittest"),
                    exit_code=0,
                    duration_seconds=0.5,
                ),
            ),
        )

        self.assertEqual(result.status, VerificationStatus.PASSED)

        with self.assertRaisesRegex(ValueError, "must not include"):
            VerificationResult(
                status=VerificationStatus.SKIPPED,
                commands=(
                    VerificationCommandResult(
                        command=("python3", "-m", "unittest"),
                        exit_code=0,
                        duration_seconds=0.5,
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
