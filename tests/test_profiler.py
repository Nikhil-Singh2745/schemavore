from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from schemavore.profiler import PythonRepositoryProfiler, profile_repository


class PythonRepositoryProfilerTests(unittest.TestCase):
    def test_profiles_symbols_imports_and_conventions_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pkg").mkdir()
            (root / "pkg" / "module.py").write_text(
                "from dataclasses import dataclass\n"
                "import pathlib\n\n"
                "ANSWER = 42\n\n"
                "@dataclass\n"
                "class ExampleThing:\n"
                "    name: str\n\n"
                "def make_thing(name: str) -> ExampleThing:\n"
                "    try:\n"
                "        return ExampleThing(name)\n"
                "    except ValueError:\n"
                "        raise\n",
                encoding="utf-8",
            )
            (root / "tests").mkdir()
            (root / "tests" / "test_module.py").write_text(
                "def test_example() -> None:\n    assert True\n", encoding="utf-8"
            )

            profile = profile_repository(root)

        self.assertEqual([module.path for module in profile.modules], ["pkg/module.py", "tests/test_module.py"])
        self.assertEqual(
            [(symbol.name, symbol.kind) for symbol in profile.symbols],
            [("ANSWER", "variable"), ("ExampleThing", "class"), ("make_thing", "function"), ("test_example", "function")],
        )
        self.assertEqual(
            [(import_.module, import_.names) for import_ in profile.imports],
            [("dataclasses", ("dataclass",)), ("", ("pathlib",))],
        )
        features = {(item.category, item.name, item.value) for module in profile.modules for item in module.features}
        self.assertIn(("naming", "function", "snake_case"), features)
        self.assertIn(("typing", "function_annotations", "present"), features)
        self.assertIn(("structure", "class_decorator", "dataclass"), features)
        self.assertIn(("exceptions", "except", "typed"), features)
        self.assertIn(("testing", "test_function", "pytest_style"), features)

    def test_extracts_common_tooling_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text(
                "[build-system]\nrequires = [\"hatchling\"]\n"
                "[tool.ruff]\nline-length = 88\n"
                "[tool.pytest.ini_options]\naddopts = \"-q\"\n",
                encoding="utf-8",
            )
            (root / "mypy.ini").write_text("[mypy]\nstrict = True\n", encoding="utf-8")

            profile = PythonRepositoryProfiler(root).profile()

        configurations = {(item.tool, item.path): item.settings for item in profile.tooling}
        self.assertEqual(configurations[("ruff", "pyproject.toml")], (("line-length", "88"),))
        self.assertEqual(configurations[("pytest", "pyproject.toml")], (("ini_options.addopts", "-q"),))
        self.assertEqual(configurations[("mypy", "mypy.ini")], (("strict", "True"),))
        self.assertIn(("build-system", "pyproject.toml"), configurations)

    def test_reports_syntax_errors_without_aborting_the_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "broken.py").write_text("def incomplete(:\n", encoding="utf-8")
            (root / "valid.py").write_text("def valid() -> None:\n    pass\n", encoding="utf-8")

            profile = profile_repository(root)

        self.assertEqual([module.path for module in profile.modules], ["broken.py", "valid.py"])
        self.assertEqual(profile.parse_errors[0].path, "broken.py")
        self.assertIn("line 1", profile.parse_errors[0].parse_error or "")
        self.assertEqual([symbol.name for symbol in profile.symbols], ["valid"])

    def test_ignores_generated_and_oversized_python_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "build").mkdir()
            (root / "build" / "generated.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "kept.py").write_text("VALUE = 2\n", encoding="utf-8")

            profile = profile_repository(root)

        self.assertEqual([module.path for module in profile.modules], ["kept.py"])


if __name__ == "__main__":
    unittest.main()
