"""Promote the 4 bash one-liners in MVP_ARCHITECTURE.md §Enforcement to actual greps
that fail CI when violated.

Spec: docs/specs/2026-07-26-test-discipline-design.md §3.2
Catches audit §4 row #2 (Stale-spec-from-training-data).
"""
import re
import subprocess
import unittest

MVP_RULES_PATH = ".opencode/rules/MVP_ARCHITECTURE.md"
RULES_HEADER = "## Enforcement"


class TestMvpGrepRules(unittest.TestCase):
    def _extract_greps(self):
        with open(MVP_RULES_PATH) as f:
            content = f.read()
        start = content.find(RULES_HEADER)
        self.assertNotEqual(
            start, -1,
            f"Section {RULES_HEADER!r} not found in {MVP_RULES_PATH}. "
            "If the spec moved, update MVP_RULES_PATH and RULES_HEADER.",
        )
        # Find the next `## ` heading after Enforcement; everything from start to
        # there is the section.
        after = content[start + len(RULES_HEADER):]
        m = re.search(r"^##\s", after, re.MULTILINE)
        end = (start + len(RULES_HEADER) + m.start()) if m else len(content)
        section = content[start:end]
        # The Enforcement section typically contains one or more ```bash fenced
        # blocks; each block's body lists `! grep -q ...` rules (one per comment).
        # Split each block on its grep lines so each rule runs independently.
        bash_blocks = re.findall(r"```bash\n(.*?)\n```", section, re.DOTALL)
        greps = []
        for block in bash_blocks:
            for line in block.split("\n"):
                stripped = line.strip()
                if stripped.startswith("! grep"):
                    greps.append(stripped)
        return greps

    def test_mvp_grep_rules_pass(self):
        greps = self._extract_greps()
        self.assertGreaterEqual(
            len(greps), 4,
            f"Expected ≥4 bash greps in {MVP_RULES_PATH} Enforcement section; "
            f"found {len(greps)}. If MVP rules were removed, update the count.",
        )
        for i, grep in enumerate(greps, 1):
            result = subprocess.run(
                ["bash", "-c", grep],
                capture_output=True, text=True,
            )
            self.assertEqual(
                result.returncode, 0,
                f"MVP grep rule #{i} FAILED. The rule:\n{grep}\n"
                f"stderr:\n{result.stderr[-1000:]}\n\n"
                f"This means an MVP rule is currently violated. "
                f"Read the rule, find the violation, refactor.",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)