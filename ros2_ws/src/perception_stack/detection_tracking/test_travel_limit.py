"""Tests for travel_limit.py. Run from the repo root:

    python -m pytest ros2_ws/src/perception_stack/detection_tracking/test_travel_limit.py -v
"""

import pytest

from travel_limit import LimitState, TravelLimitHandler, TravelLimits, wrap180


@pytest.fixture
def handler():
    # Defaults: pan -170..170, margin 2, trigger 3 -> parking points at +168 / -168.
    return TravelLimitHandler(TravelLimits())


def test_tracks_target_inside_travel(handler):
    d = handler.update(gimbal_pan_deg=10, target_pan_deg=40, target_tilt_deg=5, target_pan_rate_dps=20)
    assert d.state is LimitState.TRACKING
    assert (d.pan_deg, d.tilt_deg) == (40, 5)


def test_target_at_edge_is_followed_but_flagged(handler):
    d = handler.update(gimbal_pan_deg=165, target_pan_deg=169, target_tilt_deg=0, target_pan_rate_dps=20)
    assert d.state is LimitState.AT_LIMIT
    assert d.pan_deg == 169


def test_just_past_edge_holds_instead_of_reversing(handler):
    # 2 deg past the edge is inside the 3 deg trigger band: no 340 deg swing for jitter at the edge.
    d = handler.update(gimbal_pan_deg=169, target_pan_deg=172, target_tilt_deg=0, target_pan_rate_dps=20)
    assert d.state is LimitState.AT_LIMIT
    assert d.pan_deg == 168


def test_reverses_when_target_keeps_going_clockwise(handler):
    d = handler.update(gimbal_pan_deg=169, target_pan_deg=175, target_tilt_deg=0, target_pan_rate_dps=30)
    assert d.state is LimitState.REVERSING
    assert d.pan_deg == -168  # opposite edge, where a clockwise target comes back out


def test_reverses_when_target_keeps_going_counterclockwise(handler):
    d = handler.update(gimbal_pan_deg=-169, target_pan_deg=-176, target_tilt_deg=0, target_pan_rate_dps=-30)
    assert d.state is LimitState.REVERSING
    assert d.pan_deg == 168


def test_same_direction_given_as_wrapped_angle(handler):
    # -185 and 175 are the same direction.
    d = handler.update(gimbal_pan_deg=169, target_pan_deg=-185, target_tilt_deg=0, target_pan_rate_dps=30)
    assert d.state is LimitState.REVERSING
    assert d.pan_deg == -168


def test_full_reversal_cycle(handler):
    handler.update(gimbal_pan_deg=169, target_pan_deg=175, target_tilt_deg=0, target_pan_rate_dps=30)
    # Mid-swing: the target is still in the dead zone, keep going.
    d = handler.update(gimbal_pan_deg=0, target_pan_deg=180, target_tilt_deg=0, target_pan_rate_dps=30)
    assert d.state is LimitState.REVERSING
    assert d.pan_deg == -168
    # Arrived at the parking point.
    d = handler.update(gimbal_pan_deg=-167.5, target_pan_deg=185, target_tilt_deg=0, target_pan_rate_dps=30)
    assert d.state is LimitState.WAITING
    # Target comes out of the dead zone at -169 (= 191): tracking resumes.
    d = handler.update(gimbal_pan_deg=-168, target_pan_deg=191, target_tilt_deg=0, target_pan_rate_dps=30)
    assert d.state is LimitState.AT_LIMIT  # still within the edge margin, but followed
    assert d.pan_deg == pytest.approx(-169)
    d = handler.update(gimbal_pan_deg=-169, target_pan_deg=200, target_tilt_deg=0, target_pan_rate_dps=30)
    assert d.state is LimitState.TRACKING
    assert d.pan_deg == pytest.approx(-160)


def test_target_turning_back_mid_reversal_is_followed(handler):
    handler.update(gimbal_pan_deg=169, target_pan_deg=175, target_tilt_deg=0, target_pan_rate_dps=30)
    # Target reverses and re-enters on the original side while the gimbal is swinging.
    d = handler.update(gimbal_pan_deg=50, target_pan_deg=160, target_tilt_deg=0, target_pan_rate_dps=-30)
    assert d.state is LimitState.TRACKING
    assert d.pan_deg == 160


def test_target_heading_back_holds_at_edge(handler):
    # Past the edge but moving counter-clockwise: it will come back out at +170, so wait there.
    d = handler.update(gimbal_pan_deg=169, target_pan_deg=176, target_tilt_deg=0, target_pan_rate_dps=-30)
    assert d.state is LimitState.AT_LIMIT
    assert d.pan_deg == 168


def test_stationary_target_in_dead_zone_holds(handler):
    d = handler.update(gimbal_pan_deg=169, target_pan_deg=178, target_tilt_deg=0, target_pan_rate_dps=0.5)
    assert d.state is LimitState.AT_LIMIT
    assert d.pan_deg == 168


def test_hold_policy_never_reverses():
    h = TravelLimitHandler(TravelLimits(), policy="hold")
    d = h.update(gimbal_pan_deg=169, target_pan_deg=178, target_tilt_deg=0, target_pan_rate_dps=30)
    assert d.state is LimitState.AT_LIMIT
    assert d.pan_deg == 168


def test_tilt_is_clamped(handler):
    d = handler.update(gimbal_pan_deg=0, target_pan_deg=0, target_tilt_deg=120, target_pan_rate_dps=0)
    assert d.tilt_deg == 90
    d = handler.update(gimbal_pan_deg=0, target_pan_deg=0, target_tilt_deg=-60, target_pan_rate_dps=0)
    assert d.tilt_deg == -45


def test_continuous_rotation_takes_short_way():
    h = TravelLimitHandler(TravelLimits(pan_min_deg=-360, pan_max_deg=360))
    d = h.update(gimbal_pan_deg=170, target_pan_deg=-170, target_tilt_deg=0, target_pan_rate_dps=30)
    assert d.state is LimitState.TRACKING
    assert d.pan_deg == 190  # 20 deg clockwise, not 340 deg back


def test_reversal_time_estimate(handler):
    # (168 - -168) / 100 deg/s
    assert handler.reversal_time_s() == pytest.approx(3.36)


def test_wrap180():
    assert wrap180(190) == -170
    assert wrap180(-190) == 170
    assert wrap180(180) == -180


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        TravelLimits(pan_min_deg=10, pan_max_deg=-10)
    with pytest.raises(ValueError):
        TravelLimitHandler(policy="spin")
