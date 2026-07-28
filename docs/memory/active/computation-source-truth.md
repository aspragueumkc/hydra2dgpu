---
type: memory
status: active
created: 2026-07-26
topic: computation-source-truth
tags: [infra, lesson]
evidence: .agents/computation-source-truth.md:1
related:
  - .agents/computation-source-truth.md
---

# Computation Source of Truth

## Context

If a GPU kernel computes a value internally, that value must come from the
kernel — never re-compute it in Python. Re-computing drifts the answer,
introduces silent disagreement, and makes the GPU the wrong target for
debugging.

## Decision

- Add a device buffer + D2H readback whenever the kernel is the source of
  truth for a quantity.
- Never re-implement kernel arithmetic in Python and trust it.
- The Python side reads, renders, and validates — it does not derive.

## Open questions

- None.
