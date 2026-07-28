"""swe2d/workbench/services/drainage_graph_service.py

Pure-Python abstraction over the drainage network: builds a graph from
swe2d_drainage_links + swe2d_drainage_nodes tables and finds shortest
paths between nodes via BFS. Zero Qt, zero numpy, fully testable.
"""

from __future__ import annotations

import os
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DrainageGraph:
    node_ids: List[str]
    link_ids: List[str]
    from_node: Dict[str, str] = field(default_factory=dict)
    to_node: Dict[str, str] = field(default_factory=dict)
    outgoing: Dict[str, List[str]] = field(default_factory=dict)
    incoming: Dict[str, List[str]] = field(default_factory=dict)
    both: Dict[str, List[str]] = field(default_factory=dict)


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def load_drainage_graph(gpkg_path: str) -> DrainageGraph:
    """Load drainage network topology into a DrainageGraph.

    Returns an empty DrainageGraph if the file does not exist, the
    drainage tables are missing, or there are no rows.
    """
    if not gpkg_path or not os.path.exists(gpkg_path):
        return DrainageGraph(node_ids=[], link_ids=[])

    try:
        conn = sqlite3.connect(gpkg_path)
    except sqlite3.Error:
        return DrainageGraph(node_ids=[], link_ids=[])
    try:
        cur = conn.cursor()

        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_drainage_links'"
        )
        if cur.fetchone() is None:
            return DrainageGraph(node_ids=[], link_ids=[])

        cur.execute(
            f"SELECT link_id, from_node, to_node FROM {_quote_ident('swe2d_drainage_links')}"
        )
        link_rows = list(cur.fetchall())

        node_table_exists = False
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_drainage_nodes'"
        )
        node_table_exists = cur.fetchone() is not None
        node_rows = []
        if node_table_exists:
            cur.execute(f"SELECT node_id FROM {_quote_ident('swe2d_drainage_nodes')}")
            node_rows = [str(r[0]) for r in cur.fetchall() if r and r[0] is not None]

        link_ids: List[str] = []
        from_node: Dict[str, str] = {}
        to_node: Dict[str, str] = {}
        outgoing: Dict[str, List[str]] = {}
        incoming: Dict[str, List[str]] = {}
        both: Dict[str, List[str]] = {}
        node_set: set = set()

        for row in link_rows:
            if not row:
                continue
            lid = str(row[0]) if row[0] is not None else ""
            fn = str(row[1]) if row[1] is not None else ""
            tn = str(row[2]) if row[2] is not None else ""
            if not lid or not fn or not tn:
                continue
            link_ids.append(lid)
            from_node[lid] = fn
            to_node[lid] = tn
            node_set.add(fn)
            node_set.add(tn)
            outgoing.setdefault(fn, []).append(lid)
            incoming.setdefault(tn, []).append(lid)
            both.setdefault(fn, []).append(lid)
            both.setdefault(tn, []).append(lid)
    finally:
        conn.close()

    for n in node_rows:
        node_set.add(n)
    return DrainageGraph(
        node_ids=sorted(node_set),
        link_ids=link_ids,
        from_node=from_node,
        to_node=to_node,
        outgoing=outgoing,
        incoming=incoming,
        both=both,
    )


def find_chain(
    graph: DrainageGraph,
    start_node: str,
    end_node: str,
) -> List[str]:
    """Find a shortest (unweighted) path of links from start_node to end_node.

    Returns an ordered list of link_ids walking the path, or [] if no path
    exists or start_node == end_node. The traversal treats links as
    undirected (any node can flow to any neighbour).
    """
    if not graph.link_ids:
        return []
    if start_node == end_node:
        return []
    if start_node not in graph.both or end_node not in graph.both:
        return []

    parent: Dict[str, Optional[str]] = {start_node: None}
    parent_link: Dict[str, str] = {}
    queue = deque([start_node])
    found = False

    while queue:
        node = queue.popleft()
        if node == end_node:
            found = True
            break
        for link_id in graph.both.get(node, []):
            other = graph.to_node[link_id] if graph.from_node[link_id] == node else graph.from_node[link_id]
            if other not in parent:
                parent[other] = node
                parent_link[other] = link_id
                queue.append(other)
        if found:
            break

    if not found:
        return []

    chain_links: List[str] = []
    cur = end_node
    while parent.get(cur) is not None:
        chain_links.append(parent_link[cur])
        cur = parent[cur]
    chain_links.reverse()
    return chain_links


def link_orientation(graph: DrainageGraph, link_id: str, expected_upstream: str) -> bool:
    """Return True if the link should be traversed from -> to.

    If expected_upstream is neither endpoint, defaults to True (forward).
    """
    if link_id not in graph.from_node:
        return True
    fn = graph.from_node[link_id]
    tn = graph.to_node[link_id]
    if expected_upstream not in (fn, tn):
        return True
    return fn == expected_upstream


def supplement_graph_from_coupling(
    graph: DrainageGraph,
    results_gpkg_path: str,
    run_id: str,
) -> DrainageGraph:
    """Build a DrainageGraph from coupling results alone.

    Parses link-node connectivity from the ``object_name`` column of
    ``swe2d_baked_coupling``.  For every ``drainage_link`` record the
    ``object_name`` stores ``"{from_node_id} -> {to_node_id}"`` (set at
    coupling-build time in ``non_gui_runtime_service.build_coupling_keys()``).
    Also collects all distinct ``drainage_node`` IDs so orphan nodes appear
    in the picker.

    If the coupling table is missing or empty the returned graph is identical
    to the input *graph* (which is typically empty when called from the
    ``NetworkProfileDialog`` — no coupling data means no results).
    """
    import re as _re
    import sqlite3
    if not results_gpkg_path or not run_id:
        return graph
    try:
        conn = sqlite3.connect(f"file:{results_gpkg_path}?mode=ro", uri=True)
    except Exception:
        return graph
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_baked_coupling'"
        )
        if cur.fetchone() is None:
            return graph

        out_link_ids: list[str] = []
        out_from_node: dict[str, str] = dict(graph.from_node)
        out_to_node: dict[str, str] = dict(graph.to_node)
        out_outgoing: dict[str, list[str]] = dict(graph.outgoing)
        out_incoming: dict[str, list[str]] = dict(graph.incoming)
        out_both: dict[str, list[str]] = dict(graph.both)
        node_set: set[str] = set(graph.node_ids)
        link_set: set[str] = set(graph.link_ids)
        changed = False

        rows = cur.execute(
            "SELECT object_id, object_name FROM swe2d_baked_coupling "
            "WHERE run_id=? AND component='drainage_link' AND metric='flow'",
            (run_id,),
        ).fetchall()

        _ARROW = _re.compile(r"\s*->\s*")

        for oid, oname in rows:
            lid = str(oid) if oid else ""
            if not lid or lid in link_set:
                continue
            link_set.add(lid)
            out_link_ids.append(lid)
            changed = True

            fn = tn = ""
            name_str = str(oname or "")
            if "->" in name_str:
                parts = _ARROW.split(name_str, maxsplit=1)
                if len(parts) == 2:
                    fn = parts[0].strip()
                    tn = parts[1].strip()
            out_from_node[lid] = fn
            out_to_node[lid] = tn
            if fn:
                node_set.add(fn)
                out_outgoing.setdefault(fn, []).append(lid)
                out_both.setdefault(fn, []).append(lid)
            if tn:
                node_set.add(tn)
                out_incoming.setdefault(tn, []).append(lid)
                out_both.setdefault(tn, []).append(lid)

        rows = cur.execute(
            "SELECT DISTINCT object_id FROM swe2d_baked_coupling "
            "WHERE run_id=? AND component='drainage_node'",
            (run_id,),
        ).fetchall()
        for (oid,) in rows:
            nid = str(oid) if oid else ""
            if nid and nid not in node_set:
                node_set.add(nid)
                out_both.setdefault(nid, [])
                changed = True

        if not changed:
            return graph
        return DrainageGraph(
            node_ids=sorted(node_set),
            link_ids=sorted(link_set),
            from_node=out_from_node,
            to_node=out_to_node,
            outgoing=out_outgoing,
            incoming=out_incoming,
            both=out_both,
        )
    finally:
        conn.close()
