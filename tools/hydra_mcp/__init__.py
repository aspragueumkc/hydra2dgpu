"""HYDRA MCP server package.

Stdio MCP server exposing 39 tools over the existing HYDRA core modules,
across four tiers: Tier A modeling (Phase 0 read-only inspect/query tools
plus Phase 1 build/run/results tools, mostly mutating), Tier B live-GUI
tools (Phase 2 introspection — read-only except ``gui_set_value`` — and
Phase 3 mutating behavioral tools), and Tier C design tools (Phase 4;
``design_apply_patch`` edits source and is disabled by default). See
``server.py``'s module docstring for the full read-only/mutating breakdown
— it matters for client approval rules (plan §6).

Design principle: thin adapters only — no modeling logic is re-implemented
here (see ``docs/HYDRA_MCP_SERVER_PLAN.md``).
"""
