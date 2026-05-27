from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BridgeLossLaw:
    """Empirical deck-entry/deck-exit loss law for the toy bridge coupling.

    The law is intentionally simple so it can be reused in the toy prototype,
    then ported to the compiled solver and later SWE2D coupling.
    """

    upstream_k: float = 1.0
    downstream_k: float = 1.0

    def damping_factor(self, velocity_ft_s: float, dt_s: float, dx_ft: float, *, upstream: bool) -> float:
        k = self.upstream_k if upstream else self.downstream_k
        if k <= 0.0:
            return 1.0
        return 1.0 + 16.0 * k * dt_s * abs(velocity_ft_s) * abs(velocity_ft_s) / max(dx_ft, 1e-9)

    def apply_face_velocity(
        self,
        velocity_ft_s: float,
        dt_s: float,
        dx_ft: float,
        *,
        upstream: bool,
    ) -> float:
        return velocity_ft_s / self.damping_factor(velocity_ft_s, dt_s, dx_ft, upstream=upstream)


def apply_bridge_loss_to_faces(
    face_velocities_ft_s: np.ndarray,
    *,
    law: BridgeLossLaw,
    dt_s: float,
    dx_ft: float,
    upstream_mask: np.ndarray,
    downstream_mask: np.ndarray,
) -> np.ndarray:
    """Return a damped copy of face velocities near the bridge openings.

    Parameters
    ----------
    face_velocities_ft_s:
        Face-normal velocities on the x-oriented faces.
    upstream_mask / downstream_mask:
        Boolean masks identifying the bridge entry and exit faces.
    """

    damped = np.array(face_velocities_ft_s, copy=True, dtype=float)
    damped[upstream_mask] = [
        law.apply_face_velocity(v, dt_s, dx_ft, upstream=True) for v in damped[upstream_mask]
    ]
    damped[downstream_mask] = [
        law.apply_face_velocity(v, dt_s, dx_ft, upstream=False) for v in damped[downstream_mask]
    ]
    return damped