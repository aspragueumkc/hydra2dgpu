"""End-to-end MCP widget-walk integration test.

Spec: docs/specs/2026-07-26-test-discipline-design.md §3.3 (adapted)

This is the integration boundary test: it catches Qt-touching changes that
break the connection between the test client and a running QGIS bridge.

Skipped unless ``HYDRA_MCP_INTEGRATION=1`` is set.  Requires a running QGIS
bridge (the ``qgis-smoke-test`` CI job provides this; locally, run QGIS via
the MCP server first).

The spec assumed ``McpClient.from_env()`` and ``gui_*``-prefixed methods.  The
actual client is ``BridgeClient()`` with ``get_widget_tree``, ``get_value``,
``set_value``, ``read_log``, and ``close`` methods.
"""

from __future__ import annotations

import os
import unittest
from typing import Any


_SUPPORTED_VALUE_WIDGETS = frozenset(
    {
        "QSpinBox",
        "QDoubleSpinBox",
        "QCheckBox",
        "QComboBox",
        "QLineEdit",
        "QTextEdit",
    }
)


def _widget_path(
    node: dict[str, Any], nodes_by_id: dict[int, dict[str, Any]]
) -> str | None:
    """Build the bridge path for a node, excluding the root widget name."""
    parts: list[str] = []
    current = node
    while current["parent_id"] is not None:
        object_name = current["object_name"]
        if not object_name:
            return None
        parts.append(object_name)
        current = nodes_by_id.get(current["parent_id"])
        if current is None:
            return None
    return ".".join(reversed(parts)) or None


@unittest.skipUnless(
    os.environ.get("HYDRA_MCP_INTEGRATION") == "1",
    "set HYDRA_MCP_INTEGRATION=1 to run MCP widget-walk integration test",
)
class TestMcpWidgetWalk(unittest.TestCase):
    def test_walk_widgets_clean_log(self) -> None:
        from tools.hydra_mcp.bridge_client import BridgeClient

        client = BridgeClient()
        try:
            tree = client.get_widget_tree()
            self.assertIsInstance(tree, list)
            self.assertTrue(tree, "get_widget_tree returned no widgets")
            required_keys = {
                "object_name", "class_name", "widget_id",
                "parent_id", "text", "depth",
            }
            self.assertTrue(
                all(isinstance(node, dict) and required_keys <= node for node in tree),
                f"get_widget_tree returned an unexpected shape: {tree!r}",
            )

            nodes_by_id = {node["widget_id"]: node for node in tree}
            parent_ids = {
                node["parent_id"] for node in tree if node["parent_id"] is not None
            }
            leaves = [
                node
                for node in tree
                if node["widget_id"] not in parent_ids
                and node["class_name"] in _SUPPORTED_VALUE_WIDGETS
            ]
            self.assertGreater(len(leaves), 0, "widget tree has no settable leaves")

            # Round-trip five leaves without changing model configuration.
            for node in leaves[:5]:
                path = _widget_path(node, nodes_by_id)
                if path is None:
                    continue
                try:
                    current = client.get_value(path)
                    if (
                        isinstance(current, dict)
                        and current.get("ok")
                        and "value" in current
                    ):
                        client.set_value(path, current["value"])
                except Exception:
                    # The final log assertion reports bridge-side failures.
                    continue

            log = client.read_log(max_lines=200)
            self.assertIsInstance(log, dict)
            self.assertIn("lines", log)
            log_text = "\n".join(str(line) for line in log["lines"])
            self.assertNotIn(
                "Traceback",
                log_text,
                f"Runtime log contains a traceback:\n{log_text[-2000:]}",
            )
            self.assertNotIn(
                "[ERROR]",
                log_text,
                f"Runtime log contains [ERROR] entries:\n{log_text[-2000:]}",
            )
        finally:
            try:
                client.close()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
