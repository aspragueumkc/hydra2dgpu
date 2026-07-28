"""Smoke test for the MCP client wrapper without launching a real QGIS.

Spec: docs/specs/2026-07-26-test-discipline-design.md §6.4

Runs only when HYDRA_MCP_SMOKE=1 (or HYDRA_MCP_INTEGRATION=1) is set. The
smoke test verifies that the test-side client wrapper can be imported and
that the API surface the widget-walk test depends on exists.

This is the cheap "the test plumbing works" check. The full walk is in
tests/test_mcp_widget_walk.py and requires a real QGIS via
HYDRA_MCP_INTEGRATION=1.

Note on the API: the spec assumed ``tools.hydra_mcp.bridge_client.McpClient``
with a ``from_env`` classmethod and ``gui_*``-prefixed methods. The
actual class is ``BridgeClient``: it has a single ``__init__`` that
auto-discovers the bridge socket + token from ``HYDRA_MCP_BRIDGE_SOCKET``
/ ``HYDRA_MCP_BRIDGE_TOKEN`` (or from a token file in ``$XDG_RUNTIME_DIR`` /
``/tmp``), and exposes the underlying JSON-RPC methods without the
``gui_`` prefix. ``gui_launch`` lives server-side in ``tools_gui.py``
and is invoked by the MCP server, not the test-side client. The
widget-walk integration test (3.2) will need to be adapted to call
``BridgeClient`` directly; the smoke test is written against the real
API so the plumbing check is meaningful.
"""
import os
import unittest


@unittest.skipUnless(
    os.environ.get("HYDRA_MCP_SMOKE") == "1"
    or os.environ.get("HYDRA_MCP_INTEGRATION") == "1",
    "set HYDRA_MCP_SMOKE=1 to run MCP client smoke test",
)
class TestMcpClientSmoke(unittest.TestCase):
    def test_client_imports(self):
        # The spec assumed McpClient.from_env(); the actual class is
        # BridgeClient with no from_env classmethod — the constructor
        # auto-discovers socket + token from env / token-file discovery.
        from tools.hydra_mcp.bridge_client import BridgeClient
        self.assertTrue(callable(BridgeClient))

    def test_client_has_required_api(self):
        # The widget-walk test (3.2) needs the underlying JSON-RPC methods
        # that mirror the server-side MCP tools. ``gui_launch`` is a
        # server-side tool (tools.hydra_mcp.tools_gui), not a client method,
        # so it is not asserted here.
        from tools.hydra_mcp.bridge_client import BridgeClient
        for method in (
            "get_widget_tree",  # gui_widget_tree
            "set_value",        # gui_set_value
            "read_log",         # gui_read_log
            "close",            # gui_close
        ):
            with self.subTest(method=method):
                self.assertTrue(
                    hasattr(BridgeClient, method),
                    f"BridgeClient missing {method!r}; the widget-walk "
                    "test depends on it.",
                )
                self.assertTrue(
                    callable(getattr(BridgeClient, method)),
                    f"BridgeClient.{method} is not callable.",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
