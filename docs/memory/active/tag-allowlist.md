---
type: memory
status: active
created: 2026-07-26
topic: tag-allowlist
tags: [infra]
evidence: docs/specs/2026-07-26-agent-memory-architecture-design.md:171-175
related:
  - docs/specs/2026-07-26-agent-memory-architecture-design.md
---

# Memory Tag Allowlist

## Context

`tools/memory.py validate` reads this entry to determine which tags are
permitted in the `tags:` frontmatter field. Adding a tag requires the same
review-gated capture flow (draft → validate → approve) as any other memory
entry.

## Decision

The current allowlist is the lowercase tokens below. New tags should be added
only with a clear use case; do not pre-seed speculative domains.

```
cpp cuda units kernel swe2d gui qgis mcp test infra doc plan session
lesson decision hazard python
```

## Open questions

- Should the `infra` tag be split into `infra`, `ci`, `release` once we have
  more than three infra entries? (Defer until drift appears.)
