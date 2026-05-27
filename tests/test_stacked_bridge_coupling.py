from stacked_bridge_coupling import BridgeLossLaw


def test_bridge_loss_law_monotonic_with_velocity():
    law = BridgeLossLaw(upstream_k=1.0, downstream_k=1.0)
    a = law.apply_face_velocity(1.0, 0.05, 2.0, upstream=True)
    b = law.apply_face_velocity(4.0, 0.05, 2.0, upstream=True)

    assert b < a


def test_bridge_loss_law_respects_directional_coefficients():
    law = BridgeLossLaw(upstream_k=2.0, downstream_k=0.5)
    up = law.apply_face_velocity(3.0, 0.05, 2.0, upstream=True)
    dn = law.apply_face_velocity(3.0, 0.05, 2.0, upstream=False)

    assert up < dn