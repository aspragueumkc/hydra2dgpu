# Memory Index

Curated agent memory for the HYDRA2DGPU project. The CLI (`tools/memory.py`)
is the only writer. See
`docs/specs/2026-07-26-agent-memory-architecture-design.md` for the schema
and the `.agents/skills/hydra-agent-memory/` skill for capture/recall.

## Active topics

- `cpp-culvert-units.md` — C++ culvert path returns CFS; gravity is
  CRS-derived; coupling converts to model units.
- `computation-source-truth.md` — GPU kernel output is the source of truth;
  never re-compute in Python.
- `no-premature-backwards-compat.md` — Don't add backwards-compat shims
  during active development.
- `tag-allowlist.md` — Allowlist for `tags:` in memory frontmatter.
