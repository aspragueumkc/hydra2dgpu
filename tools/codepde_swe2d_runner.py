#!/usr/bin/env python3
"""Minimal local CodePDE-style runner for SWE2D optimization tasks.

This script is intentionally lightweight: it gathers a small bundle of repo
context, sends a prompt to a local Ollama model, and writes the response to
stdout and optionally to a file. It is meant as a quick way to test a
CodePDE-like workflow before building any VS Code integration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence
from urllib import error, request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTEXT_FILES = [
    ROOT / ".copilot" / "context.md",
    ROOT / ".copilot" / "codepde-agent.prompt.md",
    ROOT / "AGENTS.md",
]
DEFAULT_TARGET_FILES = [
    ROOT / "cpp" / "src" / "swe2d_gpu.cu",
    ROOT / "cpp" / "src" / "swe2d_gpu.cuh",
    ROOT / "tests" / "test_swe2d_gpu_validation_perf.py",
    ROOT / "tests" / "test_swe2d_gpu_unstructured.py",
]


def read_text(path: Path, limit: int | None = None) -> str:
    text = path.read_text(encoding="utf-8")
    if limit is not None and len(text) > limit:
        return text[:limit] + "\n\n[...truncated...]\n"
    return text


def format_file_bundle(paths: Sequence[Path], limit: int) -> str:
    chunks: List[str] = []
    for path in paths:
        if not path.exists():
            continue
        rel = path.relative_to(ROOT)
        chunks.append(f"\n### {rel}\n\n```text\n{read_text(path, limit=limit)}\n```\n")
    return "\n".join(chunks)


def build_prompt(goal: str, targets: Sequence[Path], context_files: Sequence[Path], snippet_limit: int) -> str:
    context_block = format_file_bundle(context_files, limit=snippet_limit)
    target_block = format_file_bundle(targets, limit=snippet_limit)
    return (
        "You are CodePDE operating inside this repository. "
        "This is an existing C++/CUDA QGIS plugin; do not suggest rewriting the solver in Python, CuPy, Numba, or any new framework. "
        "Produce a focused optimization plan or patch proposal for the requested task using the code that is already here.\n\n"
        "Constraints:\n"
        "- Treat SWE2D as GPU-first and GPU-only-in-practice.\n"
        "- Do not chase CPU/GPU parity unless explicitly required.\n"
        "- Preserve wetting/drying, positivity, and numerical stability.\n"
        "- Prefer small, reviewable changes over broad refactors.\n\n"
        "Important guidance:\n"
        "- Anchor all suggestions in the provided repository files.\n"
        "- Prefer edits to cpp/src/swe2d_gpu.cu, cpp/src/swe2d_gpu.cuh, and nearby tests when relevant.\n"
        "- If you propose code, keep it compatible with the current build and kernel structure.\n"
        "- If the task can be solved by changing a few functions, do that instead of proposing a new architecture.\n\n"
        f"Task:\n{goal}\n\n"
        f"Repository context:\n{context_block}\n\n"
        f"Relevant code:\n{target_block}\n\n"
        "Return:\n"
        "1. A concise implementation plan tied to the repo files.\n"
        "2. Any concrete code changes or patch guidance, preferably specific functions or kernels.\n"
        "3. Validation steps to run next.\n"
    )


def build_system_prompt() -> str:
    return (
        "You are a senior numerical PDE and CUDA code reviewer working only within this repository. "
        "Stay grounded in the provided SWE2D code and outputs. "
        "Do not suggest rewriting the solver in Python, CuPy, Numba, PyTorch, or any unrelated framework. "
        "Do not propose a new architecture when a targeted change to existing C++/CUDA kernels is enough. "
        "Your task is to produce specific, repository-local optimization guidance or patch text for SWE2D. "
        "Prefer concrete edits to swe2d_gpu.cu, swe2d_gpu.cuh, or nearby tests."
    )


def call_ollama(model: str, prompt: str, host: str, timeout: float) -> str:
    url = host.rstrip("/") + "/api/generate"
    payload = json.dumps(
        {
            "model": model,
            "system": build_system_prompt(),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
            },
        }
    ).encode("utf-8")
    req = request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"Failed to contact Ollama at {url}: {exc}") from exc
    if "response" in data and isinstance(data["response"], str):
        return data["response"]
    return json.dumps(data, indent=2, sort_keys=True)


def write_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_paths(values: Iterable[str]) -> List[Path]:
    return [Path(value).expanduser().resolve() for value in values]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local CodePDE-style SWE2D optimization prompt.")
    parser.add_argument("goal", help="Optimization goal for SWE2D, e.g. 'reduce wet/dry overhead in swe2d_gpu.cu'.")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "llama3.1:8b-instruct"), help="Local Ollama model name.")
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"), help="Ollama host URL.")
    parser.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout in seconds.")
    parser.add_argument("--snippet-limit", type=int, default=12_000, help="Maximum characters to include from each file.")
    parser.add_argument("--context-file", action="append", default=[], help="Additional context file to include.")
    parser.add_argument("--target-file", action="append", default=[], help="Additional code file to include.")
    parser.add_argument("--output", type=Path, help="Optional file to write the model response to.")
    parser.add_argument("--prompt-only", action="store_true", help="Print the prompt and exit without calling the model.")
    args = parser.parse_args()

    context_files = DEFAULT_CONTEXT_FILES + parse_paths(args.context_file)
    target_files = DEFAULT_TARGET_FILES + parse_paths(args.target_file)
    prompt = build_prompt(args.goal, target_files, context_files, args.snippet_limit)

    if args.prompt_only:
        print(prompt)
        return 0

    response = call_ollama(args.model, prompt, args.host, args.timeout)
    print(response)
    if args.output:
        write_output(args.output, response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())