"""Command-line interface for reviewable memory management."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from schemavore.domain import Memory, MemoryStatus
from schemavore.memory_store import MemoryStore, MemoryStoreError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="schemavore")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    memories = commands.add_parser("memories", help="inspect and review repository memories")
    memory_commands = memories.add_subparsers(dest="memory_command", required=True)
    list_parser = memory_commands.add_parser("list", help="list memories")
    list_parser.add_argument("--status", choices=[status.value for status in MemoryStatus])
    show_parser = memory_commands.add_parser("show", help="show one memory")
    show_parser.add_argument("id")
    for command in ("approve", "reject", "supersede"):
        transition_parser = memory_commands.add_parser(command, help=f"{command} a memory")
        transition_parser.add_argument("id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    store = MemoryStore(args.repository.resolve() / ".schemavore" / "rules.yaml")
    try:
        if args.memory_command == "list":
            memories = store.load()
            if args.status:
                memories = tuple(item for item in memories if item.status.value == args.status)
            for memory in memories:
                print(f"{memory.id}\t{memory.status.value}\t{memory.confidence:.4f}\t{memory.statement}")
            return 0
        if args.memory_command == "show":
            print(_format_memory(store.get(args.id)))
            return 0
        memory = getattr(store, args.memory_command)(args.id)
        print(f"{memory.id} is now {memory.status.value}")
        return 0
    except MemoryStoreError as exc:
        parser.exit(2, f"error: {exc}\n")


def _format_memory(memory: Memory) -> str:
    return "\n".join(
        (
            f"ID: {memory.id}",
            f"Status: {memory.status.value}",
            f"Category: {memory.category.value}",
            f"Confidence: {memory.confidence:.4f}",
            f"Scope: {', '.join(memory.scope)}",
            f"Statement: {memory.statement}",
            f"Evidence: {', '.join(memory.evidence_ids)}",
            f"Created: {memory.created_at.isoformat()}",
            f"Updated: {memory.updated_at.isoformat()}",
        )
    )
