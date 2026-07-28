---
type: memory
status: active
created: 2026-07-26
topic: no-premature-backwards-compat
tags: [infra, doc]
evidence: .opencode/rules/NO_PREMATURE_BACKWARDS_COMPAT.md:1
related:
  - .opencode/rules/NO_PREMATURE_BACKWARDS_COMPAT.md
---

# No Premature Backwards Compatibility

## Context

Backwards-compat shims and fallback paths are dead code during active
development. They add maintenance burden, testing surface, and cognitive
load for zero benefit, and they create silent fallback paths — the single
worst failure mode in this repo.

## Decision

- Add backwards-compat only when **already-shipped production code** depends
  on the old behaviour.
- Intermediate API shapes, signatures adjusted mid-feature, and corrected
  behaviour are not contracts. Fix the callers; delete the shim.
- Default action when tempted to add a fallback: **don't**.

## Open questions

- None.
