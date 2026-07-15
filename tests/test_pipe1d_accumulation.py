"""Test the pipe1d accumulation kernel's HEC-22 loss logic.

Validates that the loss term:
  - Reduces |Q| towards zero but never reverses flow direction
  - Handles flow reversal (Q < 0) correctly
  - Skips loss when pipe is dry (A = 0) or no loss coefficient (k = 0)
  - Does not blow up when A is very small (near-dry pipe)
"""

from __future__ import annotations

import math
import unittest


def hec22_loss_qeff(
    Q: float,
    A: float,
    k: float,
    g: float = 9.81,
) -> float:
    """Compute Q_eff after HEC-22 loss (mirrors C++ accumulation kernel).

    Loss magnitude:  loss_mag = k * Q^2 / (2 * g * A^2 + 1e-12)
    Loss never reverses flow:  Q_eff = max(0, Q - loss) for Q>0
                               Q_eff = min(0, Q + loss) for Q<0
    """
    if k <= 0.0 or A <= 0.0:
        return Q
    denom = 2.0 * g * A * A + 1e-12
    loss_mag = k * Q * Q / denom
    if Q > 0.0:
        return max(0.0, Q - loss_mag)
    else:
        return min(0.0, Q + loss_mag)


def hec22_loss_qeff_OLD(
    Q: float,
    A: float,
    k: float,
    g: float = 9.81,
) -> float:
    """OLD buggy formula: loss_Q = k * |Q| * Q / denom; Q_eff = Q - loss_Q."""
    if k <= 0.0 or A <= 0.0:
        return Q
    denom = 2.0 * g * A * A + 1e-12
    loss_Q = k * abs(Q) * Q / denom
    return Q - loss_Q


class TestHEC22LossForwardFlow(unittest.TestCase):
    """Forward flow (Q > 0): loss reduces Q towards 0."""

    def test_normal_forward_reduces(self):
        Q_eff = hec22_loss_qeff(Q=1.0, A=1.0, k=0.5)
        self.assertGreater(Q_eff, 0.0)
        self.assertLess(Q_eff, 1.0)

    def test_normal_forward_old_vs_new(self):
        """Old and new formula agree when loss < Q."""
        Q_eff_new = hec22_loss_qeff(Q=1.0, A=1.0, k=0.5)
        Q_eff_old = hec22_loss_qeff_OLD(Q=1.0, A=1.0, k=0.5)
        self.assertAlmostEqual(Q_eff_new, Q_eff_old, places=10)

    def test_loss_dominated_forward_does_not_reverse(self):
        """When loss >> Q (tiny A), Q_eff clamps to 0, never reverses."""
        Q_eff = hec22_loss_qeff(Q=1.0, A=0.001, k=0.5)
        self.assertGreaterEqual(Q_eff, 0.0)
        self.assertLess(Q_eff, 1.0)

    def test_loss_dominated_old_reverses(self):
        """OLD buggy formula reverses flow when loss > Q."""
        Q_eff_old = hec22_loss_qeff_OLD(Q=1.0, A=0.001, k=0.5)
        self.assertLess(Q_eff_old, 0.0)

    def test_dry_pipe_skips_loss(self):
        """A = 0: no loss applied."""
        Q_eff = hec22_loss_qeff(Q=1.0, A=0.0, k=0.5)
        self.assertEqual(Q_eff, 1.0)

    def test_zero_k_skips_loss(self):
        """k = 0: no loss applied."""
        Q_eff = hec22_loss_qeff(Q=1.0, A=1.0, k=0.0)
        self.assertEqual(Q_eff, 1.0)


class TestHEC22LossReverseFlow(unittest.TestCase):
    """Reverse flow (Q < 0): loss moves Q towards 0 (increases magnitude)."""

    def test_normal_reverse_reduces(self):
        Q_eff = hec22_loss_qeff(Q=-1.0, A=1.0, k=0.5)
        self.assertLess(Q_eff, 0.0)
        self.assertGreater(Q_eff, -1.0)

    def test_normal_reverse_old_vs_new(self):
        Q_eff_new = hec22_loss_qeff(Q=-1.0, A=1.0, k=0.5)
        Q_eff_old = hec22_loss_qeff_OLD(Q=-1.0, A=1.0, k=0.5)
        self.assertAlmostEqual(Q_eff_new, Q_eff_old, places=10)

    def test_loss_dominated_reverse_does_not_reverse(self):
        """When loss >> |Q|, Q_eff clamps to 0, never crosses to positive."""
        Q_eff = hec22_loss_qeff(Q=-1.0, A=0.001, k=0.5)
        self.assertLessEqual(Q_eff, 0.0)
        self.assertGreater(Q_eff, -1.0)

    def test_loss_dominated_old_reverses(self):
        """OLD formula: Q=-1, A=0.001 → Q_eff becomes positive."""
        Q_eff_old = hec22_loss_qeff_OLD(Q=-1.0, A=0.001, k=0.5)
        self.assertGreater(Q_eff_old, 0.0)


class TestHEC22LossZeroFlow(unittest.TestCase):
    """No flow (Q = 0): no change."""

    def test_zero_q(self):
        self.assertEqual(hec22_loss_qeff(Q=0.0, A=1.0, k=0.5), 0.0)

    def test_zero_q_dry(self):
        self.assertEqual(hec22_loss_qeff(Q=0.0, A=0.0, k=0.5), 0.0)

    def test_zero_q_old(self):
        self.assertEqual(hec22_loss_qeff_OLD(Q=0.0, A=1.0, k=0.5), 0.0)


class TestHEC22LossRapidReversal(unittest.TestCase):
    """Flow reversal: direction changes between steps."""

    def test_forward_after_reverse(self):
        """Forward flow after reverse should still work correctly."""
        for Q in [-1.0, -0.5, 0.5, 1.0]:
            with self.subTest(Q=Q):
                Q_eff = hec22_loss_qeff(Q=Q, A=0.01, k=0.5)
                if Q > 0:
                    self.assertGreaterEqual(Q_eff, 0.0)
                elif Q < 0:
                    self.assertLessEqual(Q_eff, 0.0)
                self.assertLessEqual(abs(Q_eff), abs(Q))

    def test_tiny_a_range(self):
        """A sweep from very small to full pipe area — no reversals."""
        for A in [1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0]:
            for Q in [-10.0, -1.0, -0.01, 0.01, 1.0, 10.0]:
                with self.subTest(A=A, Q=Q):
                    Q_eff = hec22_loss_qeff(Q=Q, A=A, k=0.5)
                    # Magnitude must not increase
                    self.assertLessEqual(abs(Q_eff), abs(Q) + 1e-12,
                        f"A={A} Q={Q} Q_eff={Q_eff}")
                    # Sign must not flip
                    if Q > 1e-12:
                        self.assertGreaterEqual(Q_eff, 0.0,
                            f"A={A} Q={Q} Q_eff={Q_eff}")
                    elif Q < -1e-12:
                        self.assertLessEqual(Q_eff, 0.0,
                            f"A={A} Q={Q} Q_eff={Q_eff}")
