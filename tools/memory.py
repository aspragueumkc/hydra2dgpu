"""HYDRA agent-memory CLI. The single writer of docs/memory/."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util as _ilu
import json
import logging
import math
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger("hydra-memory")

REPO_ROOT = Path(__file__).resolve().parent.parent

_ENC_PATH = Path(__file__).resolve().parent / "encoder.py"
_spec = _ilu.spec_from_file_location("hydra_memory_encoder", _ENC_PATH)
_encoder_mod = _ilu.module_from_spec(_spec)  # type: ignore
sys.modules["hydra_memory_encoder"] = _encoder_mod
_spec.loader.exec_module(_encoder_mod)  # type: ignore

VALID_STATUSES = {"active", "review-pending", "superseded", "complete"}
TOPIC_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
EVIDENCE_RE = re.compile(r"^[\w./-]+:\d+(-\d+)?([,;]\s*[\w./-]+:\d+(-\d+)?)*$")
SECRET_RE = re.compile(
    r"(?i)\b(password|api[_-]?key|aws_access|aws_secret|private_key)\b"
    r"|(?<![A-Za-z0-9])(token|secret)\s*[:=]\s*[^A-Za-z0-9\s]"
)
DEFAULT_TAG_ALLOWLIST = (
    "cpp cuda units kernel swe2d gui qgis mcp test infra doc "
    "plan session lesson decision hazard python"
)


def _active_dir() -> Path:
    return Path(os.environ.get("HYDRA_MEMORY_ACTIVE_DIR", "docs/memory/active")).resolve()


def _parse_frontmatter(text: str) -> tuple:
    if not text.startswith("---"):
        raise ValueError("missing frontmatter delimiter at start of file")
    end = text.find("\n---", 3)
    if end < 0:
        raise ValueError("missing closing frontmatter delimiter")
    block = text[3:end].strip()
    body = text[end + 4 :].lstrip("\n")
    fm: dict = {}
    current_list_key: str | None = None
    for raw in block.splitlines():
        line = raw.rstrip()
        if not line or line.startswith("#"):
            continue
        # Indented "- value" line — continuation of a list under the last key.
        if line.lstrip().startswith("- ") and current_list_key is not None:
            fm[current_list_key].append(line.lstrip()[2:].strip().strip('"').strip("'"))
            continue
        if ":" not in line:
            raise ValueError(f"malformed frontmatter line: {line!r}")
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            items = [x.strip().strip('"').strip("'") for x in val[1:-1].split(",") if x.strip()]
            fm[key] = items
            current_list_key = key
        elif val == "":
            # key with no inline value: expect a list on subsequent lines.
            fm[key] = []
            current_list_key = key
        else:
            fm[key] = val.strip('"').strip("'")
            current_list_key = None
    return fm, body


def _load_tag_allowlist(active: Path) -> set:
    f = active / "tag-allowlist.md"
    if not f.exists():
        return set(DEFAULT_TAG_ALLOWLIST.split())
    text = f.read_text(encoding="utf8")
    try:
        fm, body = _parse_frontmatter(text)
    except ValueError:
        return set(DEFAULT_TAG_ALLOWLIST.split())
    for line in body.splitlines():
        if line.lower().startswith("## decision"):
            tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", body)
            return {t for t in tokens if t.islower()}
    return set(DEFAULT_TAG_ALLOWLIST.split())


def _strip_fenced(body: str) -> str:
    return re.sub(r"```.*?```", "", body, flags=re.S)


def _today() -> str:
    return dt.date.today().isoformat()


def _utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _review_dir() -> Path:
    return Path(
        os.environ.get("HYDRA_MEMORY_REVIEW_PENDING_DIR", "docs/memory/review-pending")
    ).resolve()


def _superseded_dir() -> Path:
    return Path(
        os.environ.get("HYDRA_MEMORY_SUPERSEDED_DIR", "docs/memory/superseded")
    ).resolve()


def _render(fm: dict, body: str) -> str:
    lines = ["---"]
    for k, v in fm.items():
        if isinstance(v, list):
            lines.append(f"{k}: [{', '.join(str(x) for x in v)}]")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines) + body.lstrip("\n")


def _set_status(text: str, new_status: str, **extra) -> str:
    fm, body = _parse_frontmatter(text)
    fm["status"] = new_status
    fm.update(extra)
    return _render(fm, body)


def draft(text: str, topic: str, tags: list) -> int:
    if not TOPIC_RE.match(topic):
        print(f"draft: topic {topic!r} is not a valid slug")
        return 1
    rp = _review_dir()
    rp.mkdir(parents=True, exist_ok=True)
    name = f"{_utc_stamp()}-{topic}.md"
    path = rp / name
    fm = {
        "type": "memory",
        "status": "review-pending",
        "created": _today(),
        "topic": topic,
        "tags": tags,
        "evidence": "PENDING: add citation before approve",
    }
    body = (
        f"# {topic}\n\n"
        "## Context\n\n"
        f"{text}\n\n"
        "## Decision\n\n"
        f"{text}\n\n"
        "## Open questions\n\n"
    )
    path.write_text(_render(fm, body) + "\n", encoding="utf8")
    print(f"draft: wrote {path}")
    return 0


def approve(path: Path) -> int:
    if validate(path) != 0:
        print("approve: validation failed; not moved")
        return 1
    text = path.read_text(encoding="utf8")
    fm, _ = _parse_frontmatter(text)
    topic = str(fm["topic"])
    active = _active_dir()
    target = active / f"{topic}.md"
    if target.exists() and target.resolve() != path.resolve():
        # Topic collision: merge into Open questions and archive the draft.
        active_text = target.read_text(encoding="utf8")
        afm, abody = _parse_frontmatter(active_text)
        new_block = (
            f"\n\n- {dt.datetime.now(dt.timezone.utc).isoformat()} — merge from "
            f"{path.name}: {text.split('## Decision', 1)[-1].strip().splitlines()[0]}\n"
        )
        if "## Open questions" in abody:
            abody = abody.replace(
                "## Open questions\n", f"## Open questions\n{new_block}", 1
            )
        else:
            abody = abody.rstrip() + f"\n\n## Open questions\n{new_block}"
        # Re-render the active file with the merged body (preserve its frontmatter).
        target.write_text(_render(afm, abody) + "\n", encoding="utf8")
        sup = _superseded_dir() / path.name
        sup.parent.mkdir(parents=True, exist_ok=True)
        archived = _set_status(
            text, "superseded",
            superseded_by=f"docs/memory/active/{topic}.md",
            completed=_today(),
        )
        sup.write_text(archived, encoding="utf8")
        path.unlink()
        print(f"approve: merged into {target}; archived draft to {sup}")
        return 0
    active.mkdir(parents=True, exist_ok=True)
    new_text = _set_status(text, "active", created=_today())
    target.write_text(new_text + "\n", encoding="utf8")
    path.unlink()
    print(f"approve: moved to {target}")
    return 0


def list_active() -> int:
    active = _active_dir()
    for p in sorted(active.glob("*.md")):
        if p.stem == "tag-allowlist":
            continue
        print(p.stem)
    return 0


def archive(topic: str, reason: str, by: str) -> int:
    active = _active_dir()
    src = active / f"{topic}.md"
    if not src.exists():
        print(f"archive: {src} not found")
        return 1
    if reason != "superseded":
        print(f"archive: only reason=superseded supported in v1, got {reason!r}")
        return 1
    target_path = f"docs/memory/active/{by}.md"
    if not (active / f"{by}.md").exists():
        print(f"archive: target {target_path} not found")
        return 1
    sup_dir = _superseded_dir()
    sup_dir.mkdir(parents=True, exist_ok=True)
    new_text = _set_status(
        src.read_text(encoding="utf8"),
        "superseded",
        superseded_by=target_path,
        completed=_today(),
    )
    (sup_dir / f"{topic}.md").write_text(new_text, encoding="utf8")
    src.unlink()
    print(f"archive: moved {topic} → superseded/")
    return 0


def _index_dir() -> Path:
    return Path(os.environ.get("HYDRA_MEMORY_INDEX_DIR", ".memory")).resolve()


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _chunk_text(body: str, *, target: int = 256, max_tokens: int = 400, overlap: int = 32) -> list:
    words = body.split()
    if not words:
        return []
    approx = max(1, max_tokens)
    chunks: list = []
    i = 0
    step = max(1, approx - overlap)
    while i < len(words):
        chunk_words = words[i : i + approx]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if i + approx >= len(words):
            break
        i += step
    return chunks


def _store_path() -> Path:
    return _index_dir() / "index.sqlite"


def _connect() -> sqlite3.Connection:
    p = _store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(p)


def reindex() -> int:
    enc = _encoder_mod.load_encoder()
    backend = _encoder_mod.backend_name(enc)
    conn = _connect()
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute(
        "CREATE TABLE chunks ("
        "  path TEXT, topic TEXT, tags TEXT, chunk_id INTEGER, chunk_text TEXT,"
        "  embedding BLOB, evidence TEXT, body_sha TEXT"
        ")"
    )
    chunk_count = 0
    for entry in sorted(_active_dir().glob("*.md")):
        if entry.stem == "tag-allowlist":
            continue
        try:
            text = entry.read_text(encoding="utf8")
            fm, body = _parse_frontmatter(text)
        except (OSError, ValueError) as e:
            print(f"reindex: skipping {entry}: {e}", file=sys.stderr)
            continue
        chunks = _chunk_text(body)
        if not chunks:
            continue
        vecs = enc.encode(chunks)
        evidence = str(fm.get("evidence", ""))
        tags = ",".join(fm.get("tags") or [])  # type: ignore[arg-type]
        for cid, (chunk, vec) in enumerate(zip(chunks, vecs)):
            sha = hashlib.sha1(chunk.encode("utf8")).hexdigest()
            conn.execute(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)",
                (
                    str(entry), str(fm.get("topic", "")), tags, cid, chunk,
                    json.dumps(vec).encode("utf8"), evidence, sha,
                ),
            )
            chunk_count += 1
    conn.commit()
    conn.close()
    (_index_dir() / "index_source.txt").write_text("sqlite+numpy\n", encoding="utf8")
    meta = {
        "commit": _git_head(),
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "chunk_count": chunk_count,
        "backend": backend,
    }
    (_index_dir() / "last_indexed_at.json").write_text(
        json.dumps(meta, indent=2), encoding="utf8"
    )
    print(f"reindex: {chunk_count} chunks via {backend}")
    if backend == "cpu":
        print(
            "WARNING: CPU embedding fallback in use. Reindex will be ~15x slower; "
            "install a CUDA torch build to restore speed."
        )
    return 0


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(y * y for y in b)) or 1e-12
    return dot / (na * nb)


def search(query: str, k: int = 5, max_tokens: int = 4000, unbounded: bool = False,
           tags: list | None = None) -> int:
    enc = _encoder_mod.load_encoder()
    qv = enc.encode([query])[0]
    conn = _connect()
    rows = list(
        conn.execute(
            "SELECT path, topic, chunk_id, chunk_text, embedding, tags, evidence "
            "FROM chunks"
        )
    )
    conn.close()
    scored: list = []
    for path, topic, cid, chunk_text, emb_blob, tags_csv, evidence in rows:
        if tags and not all(t in (tags_csv or "").split(",") for t in tags):
            continue
        vec = json.loads(emb_blob.decode("utf8"))
        s = _cosine(qv, vec)
        scored.append((s, path, cid, chunk_text, tags_csv or "", evidence or ""))
    scored.sort(key=lambda r: r[0], reverse=True)
    top = scored[:k]

    out: list = []
    used = 0
    truncated = 0
    for s, path, cid, chunk_text, tags_csv, evidence in top:
        line = (
            f"  [score {s:.2f}] {path} :: {tags_csv}\n"
            f"    L{cid}  \"{chunk_text[:120]}\"  ({evidence})"
        )
        cost = len(line.split())
        if not unbounded and used + cost > max_tokens and out:
            truncated = sum(1 for _ in scored) - len(out)
            break
        out.append(line)
        used += cost

    if out:
        print("\n".join(out))
    if truncated and not unbounded:
        print(f"… +{truncated} more chunks, rerun with --max-tokens higher")
    if unbounded:
        print(
            "WARNING: --unbounded may inject >N tokens into the next message",
            file=sys.stderr,
        )
    return 0


def status() -> int:
    p = _index_dir() / "last_indexed_at.json"
    blocking: list = []
    advisory: list = []
    if not p.exists():
        blocking.append("never_indexed")
    else:
        meta = json.loads(p.read_text(encoding="utf8"))
        head = _git_head()
        if meta.get("commit") != head:
            blocking.append("stale_commit")
        ts = meta.get("ts", "")
        if ts:
            try:
                when = dt.datetime.fromisoformat(ts)
                if (dt.datetime.now(dt.timezone.utc) - when).total_seconds() > 86400:
                    blocking.append("stale_age")
            except ValueError:
                blocking.append("malformed_ts")
    enc = _encoder_mod.load_encoder()
    backend = _encoder_mod.backend_name(enc)
    if backend != "cuda":
        # Non-blocking per spec §9.2 — surfaced for visibility only.
        advisory.append("cpu_fallback_active")
    state = "ok" if not blocking else "stale"
    print(
        f"status: {state} reasons_blocking={blocking} reasons_advisory={advisory} "
        f"backend={backend}"
    )
    return 0 if state == "ok" else 1


HOOK_PRE_COMMIT = """#!/usr/bin/env bash
# Installed by tools/memory.py install-hook. Validates staged memory entries.
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
fail=0
for f in $(git diff --cached --name-only -- 'docs/memory/active/*.md' 'docs/memory/review-pending/*.md' 'docs/memory/superseded/*.md'); do
  if ! "$REPO_ROOT"/tools/memory.py validate "$f" >/dev/null; then
    echo "memory: $f failed validate; refusing commit" >&2
    "$REPO_ROOT"/tools/memory.py validate "$f" || true
    fail=1
  fi
done
if [ "$fail" -ne 0 ]; then exit 1; fi
status_out=$("$REPO_ROOT"/tools/memory.py status || true)
if echo "$status_out" | grep -q stale; then
  echo "memory: index is stale; re-run tools/memory.py reindex" >&2
fi
exit 0
"""

HOOK_POST_MERGE = """#!/usr/bin/env bash
# Installed by tools/memory.py install-hook. Rebuilds the index on merge.
set -euo pipefail
if [ "${HYDRA_MEMORY_NO_AUTOINDEX:-0}" = "1" ]; then exit 0; fi
REPO_ROOT="$(git rev-parse --show-toplevel)"
"$REPO_ROOT"/tools/memory.py reindex || true
exit 0
"""


def install_hook() -> int:
    root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )
    hooks = root / ".git" / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    pc = hooks / "pre-commit"
    pm = hooks / "post-merge"
    pc.write_text(HOOK_PRE_COMMIT)
    pm.write_text(HOOK_POST_MERGE)
    pc.chmod(0o755)
    pm.chmod(0o755)
    print(f"install-hook: wrote {pc} and {pm}")
    return 0


def validate(path: Path) -> int:
    active = _active_dir()
    allowlist = _load_tag_allowlist(active)
    try:
        text = path.read_text(encoding="utf8")
        fm, body = _parse_frontmatter(text)
    except (OSError, ValueError) as e:
        print(f"validate: {path}: ERROR: {e}")
        return 1

    errors: list = []
    if fm.get("type") != "memory":
        errors.append("type must be 'memory'")
    status = fm.get("status")
    if status not in VALID_STATUSES:
        errors.append(f"status must be one of {sorted(VALID_STATUSES)}; got {status!r}")
    topic = str(fm.get("topic", ""))
    if not TOPIC_RE.match(topic):
        errors.append("topic must be kebab-case, <= 64 chars, starting with a letter or digit")
    if topic and path.stem != topic and status != "review-pending":
        # review-pending files carry a timestamp prefix by design; the check
        # only applies once the entry has been moved to active/ (where the
        # prefix is stripped by `approve`).
        errors.append(f"topic {topic!r} does not match filename stem {path.stem!r}")
    tags = fm.get("tags") or []
    if not isinstance(tags, list) or not tags:
        errors.append("tags must be a non-empty list")
    else:
        bad = [t for t in tags if t not in allowlist]
        if bad:
            errors.append(f"tags not in allowlist: {bad}")
    evidence = fm.get("evidence")
    if not evidence or not EVIDENCE_RE.match(str(evidence)):
        errors.append("evidence must be 'path/to/file:line'")
    if status == "superseded" and not fm.get("superseded_by"):
        errors.append("status=superseded requires superseded_by")
    if status in {"complete", "superseded"} and not fm.get("completed"):
        errors.append(f"status={status} requires completed date")
    body_clean = body.strip()
    if not body_clean:
        errors.append("body is empty")
    if len(body_clean) > 4000:
        errors.append("body exceeds 4000 chars")
    headings = [h.lower() for h in re.findall(r"^##\s+(.+)$", body, flags=re.M)]
    if not any(h.startswith("decision") or h.startswith("lesson") for h in headings):
        errors.append("body must include a '## Decision' or '## Lesson' heading")
    if SECRET_RE.search(_strip_fenced(body)):
        errors.append("body contains a secret-like token; redact it")

    if errors:
        print(f"validate: {path}: FAIL")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"validate: {path}: ok")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="memory", description="HYDRA agent-memory CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    val = sub.add_parser("validate", help="validate a memory file")
    val.add_argument("path")

    drf = sub.add_parser("draft", help="write a draft to review-pending/")
    drf.add_argument("--text", required=True)
    drf.add_argument("--topic", required=True)
    drf.add_argument("--tags", required=True, help="comma-separated tag list")

    app = sub.add_parser("approve", help="move a review-pending draft to active/")
    app.add_argument("path")

    sub.add_parser("list", help="list active topics")

    arc = sub.add_parser("archive", help="move an active topic to superseded/")
    arc.add_argument("topic")
    arc.add_argument("--reason", required=True, choices=["superseded"])
    arc.add_argument("--by", required=True)

    sub.add_parser("reindex", help="rebuild the local vector index")
    s = sub.add_parser("search", help="semantic search over memory")
    s.add_argument("query")
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--max-tokens", type=int, default=4000)
    s.add_argument("--unbounded", action="store_true")
    s.add_argument("--tag", action="append", default=[])
    sub.add_parser("status", help="report index freshness and backend")

    sub.add_parser("install-hook", help="install git pre-commit and post-merge hooks")

    args = p.parse_args(argv)
    if args.cmd == "validate":
        return validate(Path(args.path).resolve())
    if args.cmd == "draft":
        return draft(args.text, args.topic, [t.strip() for t in args.tags.split(",") if t.strip()])
    if args.cmd == "approve":
        return approve(Path(args.path).resolve())
    if args.cmd == "list":
        return list_active()
    if args.cmd == "archive":
        return archive(args.topic, args.reason, args.by)
    if args.cmd == "reindex":
        return reindex()
    if args.cmd == "search":
        return search(
            args.query, k=args.k, max_tokens=args.max_tokens,
            unbounded=args.unbounded, tags=args.tag or None,
        )
    if args.cmd == "status":
        return status()
    if args.cmd == "install-hook":
        return install_hook()
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
