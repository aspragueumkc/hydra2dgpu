---
type: memory
status: active
created: 2026-07-28
topic: encoder-model-name-coupling
tags: [infra, lesson]
evidence: tools/encoder.py:60-62, tools/encoder.py:83
related:
  - docs/plans/2026-07-26-agent-memory-architecture.md
---

# Encoder Model Name Lives in Two Places

## Context

`tools/encoder.py` hard-codes the sentence-transformers repo id in two
adjacent spots:

1. The auto-detect that flips `HF_HUB_OFFLINE=1` when a local snapshot exists
   (`_local_snapshot_exists("sentence-transformers/all-MiniLM-L6-v2")`,
   line 60-62).
2. The `SentenceTransformer(...)` call inside `_try_real()` (line 83).

If the spec ever swaps the embedding model, both call sites have to be
updated together. The offline auto-detect only fires for that exact repo id,
so leaving site (1) stale means the CLI silently regresses to a slow network
refresh on every command.

## Decision

When the canonical embedding model is changed, update `tools/encoder.py` in
one edit by replacing both occurrences of `all-MiniLM-L6-v2` (and the
`sentence-transformers/` prefix in the snapshot path). Do not split this
across commits.

## Open questions

- None. The fix is mechanical; the lesson is "these are coupled".
