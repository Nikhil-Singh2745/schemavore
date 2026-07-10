"""Static, deterministic profiling for Python repositories."""

from __future__ import annotations

import ast
import configparser
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from schemavore.config import LimitsConfig
from schemavore.privacy import RepositoryPrivacyGuard


@dataclass(frozen=True, slots=True)
class Symbol:
    """A top-level Python symbol discovered without importing its module."""

    name: str
    kind: str
    path: str
    line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class Import:
    """An import declaration and its resolved textual target."""

    path: str
    line: int
    module: str
    names: tuple[str, ...]
    level: int = 0


@dataclass(frozen=True, slots=True)
class ConventionFeature:
    """An observed convention with the files that support it."""

    category: str
    name: str
    value: str
    count: int
    paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModuleProfile:
    """Static information extracted from a single Python module."""

    path: str
    symbols: tuple[Symbol, ...] = ()
    imports: tuple[Import, ...] = ()
    features: tuple[ConventionFeature, ...] = ()
    parse_error: str | None = None


@dataclass(frozen=True, slots=True)
class ToolConfiguration:
    """Explicit project tooling configuration normalized into scalar settings."""

    tool: str
    path: str
    settings: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class RepositoryProfile:
    """A deterministic static profile of a Python repository."""

    modules: tuple[ModuleProfile, ...]
    tooling: tuple[ToolConfiguration, ...]

    @property
    def symbols(self) -> tuple[Symbol, ...]:
        return tuple(symbol for module in self.modules for symbol in module.symbols)

    @property
    def imports(self) -> tuple[Import, ...]:
        return tuple(import_ for module in self.modules for import_ in module.imports)

    @property
    def parse_errors(self) -> tuple[ModuleProfile, ...]:
        return tuple(module for module in self.modules if module.parse_error is not None)


class PythonRepositoryProfiler:
    """Profiles Python files and explicit tooling configuration without execution."""

    def __init__(self, repo_root: Path, *, limits: LimitsConfig | None = None) -> None:
        self._repo_root = repo_root.resolve()
        self._privacy_guard = RepositoryPrivacyGuard.load(
            self._repo_root, limits or LimitsConfig()
        )

    def profile(self) -> RepositoryProfile:
        modules = tuple(self._profile_module(path) for path in self._python_paths())
        return RepositoryProfile(modules=modules, tooling=self._tooling_configurations())

    def _python_paths(self) -> tuple[str, ...]:
        paths = []
        for candidate in self._repo_root.rglob("*.py"):
            if not candidate.is_file():
                continue
            relative_path = candidate.relative_to(self._repo_root).as_posix()
            if self._privacy_guard.evaluate(relative_path).included:
                paths.append(relative_path)
        return tuple(sorted(paths))

    def _profile_module(self, path: str) -> ModuleProfile:
        source_path = self._repo_root / path
        try:
            source = source_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=path)
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            return ModuleProfile(path=path, parse_error=_format_parse_error(exc))

        symbols = tuple(_extract_symbols(tree, path))
        imports = tuple(_extract_imports(tree, path))
        features = tuple(_extract_features(tree, path))
        return ModuleProfile(path=path, symbols=symbols, imports=imports, features=features)

    def _tooling_configurations(self) -> tuple[ToolConfiguration, ...]:
        configurations: list[ToolConfiguration] = []
        pyproject = self._load_toml("pyproject.toml")
        if pyproject is not None:
            configurations.extend(_pyproject_tooling(pyproject))
        for filename, tool in (
            ("pytest.ini", "pytest"),
            ("tox.ini", "tox"),
            ("mypy.ini", "mypy"),
            (".flake8", "flake8"),
            ("setup.cfg", "setup"),
        ):
            configurations.extend(self._load_ini_tooling(filename, tool))
        ruff = self._load_toml("ruff.toml")
        if ruff is not None:
            configurations.append(_configuration("ruff", "ruff.toml", ruff))
        return tuple(sorted(configurations, key=lambda item: (item.tool, item.path)))

    def _load_toml(self, path: str) -> dict[str, Any] | None:
        if not self._privacy_guard.evaluate(path).included:
            return None
        try:
            with (self._repo_root / path).open("rb") as file:
                content = tomllib.load(file)
        except (OSError, tomllib.TOMLDecodeError):
            return None
        return content

    def _load_ini_tooling(self, path: str, tool: str) -> tuple[ToolConfiguration, ...]:
        if not self._privacy_guard.evaluate(path).included:
            return ()
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read(self._repo_root / path, encoding="utf-8")
        except (OSError, configparser.Error, UnicodeDecodeError):
            return ()
        configurations = []
        for section in parser.sections():
            section_tool = _ini_section_tool(section, tool)
            if section_tool is None:
                continue
            settings = tuple(sorted(parser.items(section)))
            configurations.append(ToolConfiguration(section_tool, path, settings))
        return tuple(configurations)


def profile_repository(repo_root: Path, *, limits: LimitsConfig | None = None) -> RepositoryProfile:
    """Return a static profile for `repo_root`."""
    return PythonRepositoryProfiler(repo_root, limits=limits).profile()


def _extract_symbols(tree: ast.Module, path: str) -> Iterable[Symbol]:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield Symbol(node.name, "function", path, node.lineno, node.end_lineno or node.lineno)
        elif isinstance(node, ast.ClassDef):
            yield Symbol(node.name, "class", path, node.lineno, node.end_lineno or node.lineno)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for name in _assigned_names(node):
                yield Symbol(name, "variable", path, node.lineno, node.end_lineno or node.lineno)


def _extract_imports(tree: ast.Module, path: str) -> Iterable[Import]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield Import(path, node.lineno, "", tuple(alias.name for alias in node.names))
        elif isinstance(node, ast.ImportFrom):
            yield Import(
                path,
                node.lineno,
                node.module or "",
                tuple(alias.name for alias in node.names),
                node.level,
            )


def _extract_features(tree: ast.Module, path: str) -> Iterable[ConventionFeature]:
    observations: dict[tuple[str, str, str], int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _observe(observations, "naming", "function", _name_style(node.name))
            if node.name.startswith("test_"):
                _observe(observations, "testing", "test_function", "pytest_style")
            if _function_has_annotations(node):
                _observe(observations, "typing", "function_annotations", "present")
        elif isinstance(node, ast.ClassDef):
            _observe(observations, "naming", "class", _name_style(node.name))
            if any(_decorator_name(item) == "dataclass" for item in node.decorator_list):
                _observe(observations, "structure", "class_decorator", "dataclass")
            if any(base.id == "TestCase" for base in node.bases if isinstance(base, ast.Name)):
                _observe(observations, "testing", "test_class", "unittest_style")
        elif isinstance(node, ast.Import):
            _observe(observations, "imports", "form", "import")
        elif isinstance(node, ast.ImportFrom):
            _observe(observations, "imports", "form", "from_import")
        elif isinstance(node, ast.Raise):
            _observe(observations, "exceptions", "raise", "present")
        elif isinstance(node, ast.ExceptHandler):
            _observe(observations, "exceptions", "except", "bare" if node.type is None else "typed")
        elif isinstance(node, ast.Dict):
            _observe(observations, "structure", "data_structure", "dict")
        elif isinstance(node, (ast.List, ast.ListComp)):
            _observe(observations, "structure", "data_structure", "list")
        elif isinstance(node, (ast.Set, ast.SetComp)):
            _observe(observations, "structure", "data_structure", "set")
        elif isinstance(node, (ast.Tuple, ast.GeneratorExp)):
            _observe(observations, "structure", "data_structure", "tuple")
    for (category, name, value), count in sorted(observations.items()):
        yield ConventionFeature(category, name, value, count, (path,))


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> Iterable[str]:
    targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
    for target in targets:
        if isinstance(target, ast.Name):
            yield target.id


def _function_has_annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
    if node.args.vararg is not None:
        arguments += (node.args.vararg,)
    if node.args.kwarg is not None:
        arguments += (node.args.kwarg,)
    return node.returns is not None or any(argument.annotation is not None for argument in arguments)


def _name_style(name: str) -> str:
    if name.isupper() and "_" in name:
        return "upper_snake_case"
    if re.fullmatch(r"[a-z][a-z0-9_]*", name):
        return "snake_case"
    if re.fullmatch(r"[A-Z][A-Za-z0-9]*", name):
        return "pascal_case"
    return "other"


def _decorator_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return None


def _observe(observations: dict[tuple[str, str, str], int], category: str, name: str, value: str) -> None:
    key = (category, name, value)
    observations[key] = observations.get(key, 0) + 1


def _format_parse_error(error: OSError | SyntaxError | UnicodeDecodeError) -> str:
    if isinstance(error, SyntaxError):
        location = f"line {error.lineno}" if error.lineno is not None else "unknown line"
        return f"{location}: {error.msg}"
    return str(error)


def _pyproject_tooling(content: dict[str, Any]) -> Iterable[ToolConfiguration]:
    tool_config = content.get("tool")
    if isinstance(tool_config, dict):
        for tool, settings in tool_config.items():
            if isinstance(settings, dict):
                yield _configuration(str(tool), "pyproject.toml", settings)
    for project_tool in ("build-system", "project"):
        settings = content.get(project_tool)
        if isinstance(settings, dict):
            yield _configuration(project_tool, "pyproject.toml", settings)


def _configuration(tool: str, path: str, settings: dict[str, Any]) -> ToolConfiguration:
    return ToolConfiguration(tool, path, tuple(_flatten_settings(settings)))


def _flatten_settings(settings: dict[str, Any], prefix: str = "") -> Iterable[tuple[str, str]]:
    for key in sorted(settings):
        value = settings[key]
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            yield from _flatten_settings(value, full_key)
        else:
            yield full_key, _value_text(value)


def _value_text(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return ",".join(_value_text(item) for item in value)
    return str(value)


def _ini_section_tool(section: str, default_tool: str) -> str | None:
    normalized = section.lower()
    aliases = {
        "pytest": "pytest",
        "tool:pytest": "pytest",
        "mypy": "mypy",
        "flake8": "flake8",
        "isort": "isort",
        "coverage:run": "coverage",
        "tool:coverage": "coverage",
        "tox": "tox",
    }
    if normalized in aliases:
        return aliases[normalized]
    if default_tool == "setup" and normalized in {"metadata", "options"}:
        return "setuptools"
    return default_tool if default_tool != "setup" else None
