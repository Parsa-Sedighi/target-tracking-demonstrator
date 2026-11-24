"""Acceptance test for UC-15: manual jog within travel limits (ES-10.1, ES-04.3, ES-11.2).

UC-15 acceptance test (PRD v2, section 7):
    Each jog step moves its axis by the set step within +/-0.25 deg, and no
    jog command drives an axis past its travel limit.

The simulated gimbal settles with up to 0.05 deg of random pointing error and
0.02 deg encoder quantization, so measured steps are not exact.
"""
import random

import pytest

from aimbot.uc15_manual_jog import ALLOWED_STEPS_DEG, Axis, JogController, TravelLimits


class SimGimbal:
    """Gimbal that refuses to move past its hard stops and reports encoder angles."""

    HARD_PAN, HARD_TILT = (-325.0, 325.0), (-120.0, 120.0)

    def __init__(self, seed=15):
        self.rng = random.Random(seed)
        self.pan, self.tilt = 0.0, 0.0
        self.commands = []

    def position(self):
        q = 0.02
        return round(self.pan / q) * q, round(self.tilt / q) * q

    def move_to(self, pan, tilt):
        self.commands.append((pan, tilt))
        assert self.HARD_PAN[0] <= pan <= self.HARD_PAN[1], "pan command past hard stop"
        assert self.HARD_TILT[0] <= tilt <= self.HARD_TILT[1], "tilt command past hard stop"
        self.pan = pan + self.rng.uniform(-0.05, 0.05)
        self.tilt = tilt + self.rng.uniform(-0.05, 0.05)


@pytest.mark.parametrize("axis", [Axis.PAN, Axis.TILT])
@pytest.mark.parametrize("step", ALLOWED_STEPS_DEG)
def test_each_jog_step_moves_by_the_set_step_within_quarter_degree(axis, step):
    gimbal = SimGimbal()
    jog = JogController(gimbal)
    index = 0 if axis == Axis.PAN else 1
    for direction in (1, 1, -1, 1, -1, -1):
        before = gimbal.position()[index]
        jog.jog(axis, direction, step)
        moved = gimbal.position()[index] - before
        assert abs(moved - direction * step) <= 0.25


def test_no_jog_command_drives_an_axis_past_its_travel_limit():
    gimbal = SimGimbal()
    limits = TravelLimits()
    jog = JogController(gimbal, limits)
    for _ in range(80):
        jog.jog(Axis.PAN, 1, 10.0)
        jog.jog(Axis.TILT, 1, 10.0)
    for _ in range(160):
        jog.jog(Axis.PAN, -1, 10.0)
        jog.jog(Axis.TILT, -1, 10.0)
    pans = [p for p, _ in gimbal.commands]
    tilts = [t for _, t in gimbal.commands]
    assert max(pans) == limits.pan_max and min(pans) == limits.pan_min
    assert max(tilts) == limits.tilt_max and min(tilts) == limits.tilt_min


def test_jog_reports_when_a_step_was_clamped_at_the_limit():
    gimbal = SimGimbal()
    gimbal.tilt = 89.5
    result = JogController(gimbal).jog(Axis.TILT, 1, 5.0)
    assert result.clamped and result.commanded[1] == 90.0


def test_unsupported_step_size_is_rejected():
    with pytest.raises(ValueError):
        JogController(SimGimbal()).jog(Axis.PAN, 1, 2.5)
