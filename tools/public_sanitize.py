#!/usr/bin/env python3
import argparse
import fnmatch
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

CODE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cuh", ".cu", ".h", ".hpp", ".ini", ".json",
    ".py", ".sh", ".toml", ".yaml", ".yml",
}
SCAN_ROOTS = {".github", "cpp", "qgis_plugin", "scripts", "swe2d", "tests", "tools"}
# Files that intentionally reference private paths for self-testing only.
SELF_TEST_ALLOWLIST = {
    "tools/public_sanitize.py",
    "tests/test_public_sanitize.py",
}
PRIVATE_PATH_RE = re.compile(r"(?:/home/[a-z0-9_-]+|/Users/[a-z0-9_-]+|/home/developer|private-repo-hydra2dgpu)")
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|password|secret|token)\s*[:=]\s*['\"][^'\"\\s][^'\"]*['\"]"
)
BINARY_SUFFIXES = {".dll", ".dylib", ".pyd", ".so"}
# Files that ALWAYS stay private (never sync to public), regardless of the
# manifest. Keeping the manifest itself out of public is non-negotiable.
ALWAYS_PRIVATE = {".publicsync-ignore"}


def tracked_paths(repo: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "-z", "--name-only", "HEAD"],
        check=True, capture_output=True,
    )
    return [entry.decode("utf-8", "replace") for entry in result.stdout.split(b"\0") if entry]


def read_exclusions(repo: Path) -> list[str]:
    manifest = repo / ".publicsync-ignore"
    if not manifest.exists():
        return []
    return [
        line.strip() for line in manifest.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def is_excluded(path: str, patterns: list[str]) -> bool:
    return any(
        fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, pattern.rstrip("/") + "/**")
        for pattern in patterns
    )


def export_working_tree(repo: Path, export_dir: Path) -> None:
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    for path in repo.rglob("*"):
        relative = path.relative_to(repo).as_posix()
        first_component = relative.split("/", 1)[0]
        if first_component == ".git":
            continue
        if path.is_symlink() or path.is_dir():
            continue
        destination = export_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def export_tracked(repo: Path, export_dir: Path, exclusions: list[str], use_working_tree: bool = False) -> None:
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    paths = [(p, repo / p) for p in tracked_paths(repo)]
    for path, source in paths:
        if path in ALWAYS_PRIVATE:
            continue
        if is_excluded(path, exclusions):
            continue
        destination = export_dir / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if use_working_tree and source.exists():
            blob = source.read_bytes()
        else:
            blob = subprocess.run(
                ["git", "-C", str(repo), "cat-file", "blob", f"HEAD:{path}"],
                check=True, capture_output=True,
            ).stdout
        destination.write_bytes(blob)


def scan_tree(root: Path) -> list[str]:
    violations: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        first_component = relative.split("/", 1)[0]
        if first_component not in SCAN_ROOTS:
            continue
        if relative in SELF_TEST_ALLOWLIST:
            continue
        if path.is_symlink():
            violations.append(f"{relative}: symlink is not allowed")
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            violations.append(f"{relative}: native binary is not allowed")
        if path.suffix.lower() not in CODE_SUFFIXES:
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError as exc:
            violations.append(f"{relative}: cannot read: {exc}")
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            if PRIVATE_PATH_RE.search(line):
                violations.append(f"{relative}:{line_number}: private absolute path")
            if SECRET_ASSIGNMENT_RE.search(line):
                violations.append(f"{relative}:{line_number}: credential-like literal")
    return violations


def verify_exclusions(export_dir: Path, exclusions: list[str]) -> list[str]:
    violations = []
    exported_paths = {
        path.relative_to(export_dir).as_posix()
        for path in export_dir.rglob("*")
    }
    for pattern in exclusions:
        if any(fnmatch.fnmatch(path, pattern) for path in exported_paths):
            violations.append(f"excluded path present: {pattern}")
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-repo", type=Path, required=True)
    parser.add_argument("--public-repo", type=Path)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument(
        "--source",
        choices=("head", "working-tree"),
        default="head",
        help="Use committed HEAD (default) or the working tree (with --precommit).",
    )
    args = parser.parse_args(argv)
    exclusions = read_exclusions(args.private_repo)
    if args.source == "working-tree":
        export_working_tree_with_exclusions(args.private_repo, args.export_dir, exclusions)
    else:
        export_tracked(args.private_repo, args.export_dir, exclusions)
    violations = verify_exclusions(args.export_dir, exclusions)
    violations.extend(scan_tree(args.export_dir))
    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        return 1
    if args.public_repo:
        print(f"export ready: {args.export_dir}")
    return 0


def export_working_tree_with_exclusions(repo: Path, export_dir: Path, exclusions: list[str]) -> None:
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    for path in repo.rglob("*"):
        relative = path.relative_to(repo).as_posix()
        if relative in ALWAYS_PRIVATE:
            continue
        first_component = relative.split("/", 1)[0]
        if first_component == ".git":
            continue
        if is_excluded(relative, exclusions):
            continue
        if path.is_symlink() or path.is_dir():
            continue
        if path.stat().st_size > 32 * 1024 * 1024:
            continue
        destination = export_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


if __name__ == "__main__":
    raise SystemExit(main())
