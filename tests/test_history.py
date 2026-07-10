from __future__ import annotations

import subprocess
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from schemavore.history import GitHistoryMiner, HistoryFilter


class GitHistoryMinerTests(unittest.TestCase):
    def test_filters_commits_and_attributes_changed_python_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._init_repository(root)
            self._commit(root, "app.py", "def first() -> None:\n    pass\n", "Alice", "alice@example.com", "first")
            self._commit(root, "app.py", "def first() -> None:\n    pass\n\ndef second() -> None:\n    pass\n", "Bob", "bob@example.com", "second")
            self._commit(root, "notes.txt", "not Python\n", "Alice", "alice@example.com", "notes")

            miner = GitHistoryMiner(root)
            all_commits = miner.mine()
            alice_commits = miner.mine(HistoryFilter(author="Alice"))
            app_commits = miner.mine(HistoryFilter(paths=("app.py",)))
            dated_commits = miner.mine(HistoryFilter(since=datetime(2024, 1, 2, tzinfo=UTC)))

        self.assertEqual([commit.subject for commit in all_commits], ["notes", "second", "first"])
        self.assertEqual([commit.subject for commit in alice_commits], ["notes", "first"])
        self.assertEqual([commit.subject for commit in app_commits], ["second", "first"])
        self.assertEqual([commit.subject for commit in dated_commits], ["notes", "second"])
        second_region = app_commits[0].regions[0]
        self.assertEqual((second_region.path, second_region.start_line, second_region.end_line), ("app.py", 3, 5))
        self.assertEqual([symbol.name for symbol in second_region.symbols], ["second"])

    def test_filters_merge_status_and_does_not_change_repository_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._init_repository(root)
            self._commit(root, "app.py", "def main() -> None:\n    pass\n", "Alice", "alice@example.com", "base")
            self._git(root, "checkout", "-b", "feature")
            self._commit(root, "feature.py", "def feature() -> None:\n    pass\n", "Bob", "bob@example.com", "feature")
            self._git(root, "checkout", "master")
            self._commit(root, "app.py", "def main() -> None:\n    return None\n", "Alice", "alice@example.com", "main")
            self._git(root, "merge", "--no-ff", "feature", "-m", "merge feature")
            head_before = self._git(root, "rev-parse", "HEAD").strip()
            branch_before = self._git(root, "branch", "--show-current").strip()

            miner = GitHistoryMiner(root)
            merges = miner.mine(HistoryFilter(merge_status=True))
            non_merges = miner.mine(HistoryFilter(merge_status=False))

            head_after = self._git(root, "rev-parse", "HEAD").strip()
            branch_after = self._git(root, "branch", "--show-current").strip()

        self.assertEqual([commit.subject for commit in merges], ["merge feature"])
        self.assertTrue(all(not commit.is_merge for commit in non_merges))
        self.assertEqual((head_after, branch_after), (head_before, branch_before))

    @staticmethod
    def _init_repository(root: Path) -> None:
        GitHistoryMinerTests._git(root, "init", "-b", "master")
        GitHistoryMinerTests._git(root, "config", "user.name", "Test User")
        GitHistoryMinerTests._git(root, "config", "user.email", "test@example.com")

    @staticmethod
    def _commit(root: Path, path: str, content: str, author_name: str, author_email: str, subject: str) -> None:
        (root / path).write_text(content, encoding="utf-8")
        GitHistoryMinerTests._git(root, "add", path)
        subprocess.run(
            ("git", "commit", "-m", subject),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            env={"GIT_AUTHOR_NAME": author_name, "GIT_AUTHOR_EMAIL": author_email, "GIT_COMMITTER_NAME": author_name, "GIT_COMMITTER_EMAIL": author_email, "GIT_AUTHOR_DATE": "2024-01-01T12:00:00+00:00" if subject == "first" else "2024-01-02T12:00:00+00:00", "GIT_COMMITTER_DATE": "2024-01-01T12:00:00+00:00" if subject == "first" else "2024-01-02T12:00:00+00:00"},
        )

    @staticmethod
    def _git(root: Path, *arguments: str) -> str:
        return subprocess.run(("git", *arguments), cwd=root, check=True, capture_output=True, text=True).stdout


if __name__ == "__main__":
    unittest.main()
