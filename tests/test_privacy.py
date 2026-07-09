from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from schemavore.config import LimitsConfig
from schemavore.privacy import (
    ContextEntry,
    ExclusionReason,
    IgnoreMatcher,
    PathTraversalError,
    RepositoryPrivacyGuard,
    SymlinkEscapeError,
    is_secret_filename,
    looks_binary,
    resolve_repository_path,
)


class IgnoreMatcherTests(unittest.TestCase):
    def test_default_patterns_ignore_common_generated_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            matcher = IgnoreMatcher.load(Path(directory))

        self.assertTrue(matcher.is_ignored(".git/HEAD"))
        self.assertTrue(matcher.is_ignored(".venv/bin/python"))
        self.assertTrue(matcher.is_ignored("src/__pycache__/module.cpython-312.pyc"))
        self.assertTrue(matcher.is_ignored("state.sqlite"))
        self.assertFalse(matcher.is_ignored("src/schemavore/domain.py"))

    def test_ignore_file_extends_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".schemavoreignore").write_text("fixtures/**/large.bin\n*.log\n", encoding="utf-8")
            matcher = IgnoreMatcher.load(root)

        self.assertTrue(matcher.is_ignored("fixtures/a/b/large.bin"))
        self.assertTrue(matcher.is_ignored("debug.log"))
        self.assertFalse(matcher.is_ignored("fixtures/a/b/small.txt"))

    def test_negation_overrides_earlier_pattern(self) -> None:
        matcher = IgnoreMatcher(["*.md", "!README.md"])

        self.assertTrue(matcher.is_ignored("notes.md"))
        self.assertFalse(matcher.is_ignored("README.md"))

    def test_directory_pattern_ignores_descendants(self) -> None:
        matcher = IgnoreMatcher(["build/"])

        self.assertTrue(matcher.is_ignored("build"))
        self.assertTrue(matcher.is_ignored("build/output.txt"))
        self.assertFalse(matcher.is_ignored("rebuild/output.txt"))

    def test_anchored_pattern_only_matches_at_root(self) -> None:
        matcher = IgnoreMatcher(["/config.toml"])

        self.assertTrue(matcher.is_ignored("config.toml"))
        self.assertFalse(matcher.is_ignored("nested/config.toml"))

    def test_comments_and_blank_lines_are_ignored(self) -> None:
        matcher = IgnoreMatcher(["# a comment", "", "   ", "*.tmp"])

        self.assertTrue(matcher.is_ignored("scratch.tmp"))


class SecretAndBinaryDetectionTests(unittest.TestCase):
    def test_common_secret_filenames_are_detected(self) -> None:
        self.assertTrue(is_secret_filename(".env"))
        self.assertTrue(is_secret_filename(".env.production"))
        self.assertTrue(is_secret_filename("id_rsa"))
        self.assertTrue(is_secret_filename("service-account-credentials.json"))
        self.assertTrue(is_secret_filename("private.pem"))
        self.assertFalse(is_secret_filename("src/schemavore/domain.py"))

    def test_binary_detection_via_null_byte(self) -> None:
        self.assertTrue(looks_binary(b"\x00\x01\x02binary"))
        self.assertFalse(looks_binary(b"def foo():\n    return 1\n"))

    def test_binary_detection_via_nontext_ratio(self) -> None:
        noisy = bytes(range(200, 256)) * 4
        self.assertTrue(looks_binary(noisy))

    def test_empty_content_is_not_binary(self) -> None:
        self.assertFalse(looks_binary(b""))


class SafePathResolutionTests(unittest.TestCase):
    def test_rejects_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(PathTraversalError):
                resolve_repository_path(Path(directory), "/etc/passwd")

    def test_rejects_traversal_segments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(PathTraversalError):
                resolve_repository_path(Path(directory), "../outside.py")

    def test_rejects_symlink_escaping_repository_root(self) -> None:
        with tempfile.TemporaryDirectory() as outer:
            outer_path = Path(outer)
            repo_root = outer_path / "repo"
            repo_root.mkdir()
            secret_outside = outer_path / "secret.txt"
            secret_outside.write_text("outside", encoding="utf-8")
            (repo_root / "link.txt").symlink_to(secret_outside)

            with self.assertRaises(SymlinkEscapeError):
                resolve_repository_path(repo_root, "link.txt")

    def test_resolves_valid_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pkg").mkdir()
            file_path = root / "pkg" / "module.py"
            file_path.write_text("VALUE = 1\n", encoding="utf-8")

            resolved = resolve_repository_path(root, "pkg/module.py")

        self.assertEqual(resolved, file_path.resolve())


class RepositoryPrivacyGuardTests(unittest.TestCase):
    def _guard(self, root: Path, limits: LimitsConfig | None = None) -> RepositoryPrivacyGuard:
        return RepositoryPrivacyGuard.load(root, limits or LimitsConfig())

    def test_includes_ordinary_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            guard = self._guard(root)

            entry = guard.evaluate("module.py")

        self.assertEqual(
            entry, ContextEntry(path="module.py", included=True, size_bytes=len(b"VALUE = 1\n"))
        )

    def test_excludes_secret_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
            guard = self._guard(root)

            entry = guard.evaluate(".env")

        self.assertFalse(entry.included)
        self.assertEqual(entry.reason, ExclusionReason.SECRET)

    def test_excludes_ignored_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
            guard = self._guard(root)

            entry = guard.evaluate(".git/HEAD")

        self.assertFalse(entry.included)
        self.assertEqual(entry.reason, ExclusionReason.IGNORED)

    def test_excludes_oversized_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "big.py").write_text("x = 1\n" * 100, encoding="utf-8")
            guard = self._guard(root, LimitsConfig(max_file_bytes=10))

            entry = guard.evaluate("big.py")

        self.assertFalse(entry.included)
        self.assertEqual(entry.reason, ExclusionReason.OVERSIZED)

    def test_excludes_binary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "asset.bin").write_bytes(b"\x00\x01\x02\x03")
            guard = self._guard(root)

            entry = guard.evaluate("asset.bin")

        self.assertFalse(entry.included)
        self.assertEqual(entry.reason, ExclusionReason.BINARY)

    def test_excludes_missing_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            guard = self._guard(Path(directory))

            entry = guard.evaluate("does_not_exist.py")

        self.assertFalse(entry.included)
        self.assertEqual(entry.reason, ExclusionReason.MISSING)

    def test_excludes_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as outer:
            outer_path = Path(outer)
            repo_root = outer_path / "repo"
            repo_root.mkdir()
            secret_outside = outer_path / "secret.txt"
            secret_outside.write_text("outside", encoding="utf-8")
            (repo_root / "link.txt").symlink_to(secret_outside)
            guard = self._guard(repo_root)

            entry = guard.evaluate("link.txt")

        self.assertFalse(entry.included)
        self.assertEqual(entry.reason, ExclusionReason.SYMLINK_ESCAPE)

    def test_build_manifest_reports_audit_trail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
            guard = self._guard(root)

            manifest = guard.build_manifest(["module.py", ".env"])

        self.assertEqual(manifest.included_paths, ("module.py",))
        self.assertEqual(len(manifest.excluded_entries), 1)
        self.assertIn("include  module.py", manifest.to_report())
        self.assertIn("exclude  .env [secret]", manifest.to_report())


if __name__ == "__main__":
    unittest.main()
