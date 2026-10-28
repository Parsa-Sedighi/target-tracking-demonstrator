# ros2_ws/src/integration_stack/sensor_interface/test_target_simulator.py
# These tests do not need ROS 2 installed. Run with: pytest test_target_simulator.py
import pytest
from target_simulator import SimulatedTarget

DT = 1 / 30  # one step at 30 Hz


def test_starts_visible_at_frame_center():
    sim = SimulatedTarget(seed=1)
    assert sim.in_frame()
    assert sim.x == sim.frame_width / 2
    assert sim.y == sim.frame_height / 2


def test_error_is_offset_from_frame_center():
    sim = SimulatedTarget(seed=1)
    obs = sim.step(DT)
    assert obs.error_x == pytest.approx(obs.x - sim.frame_width / 2)
    assert obs.error_y == pytest.approx(obs.y - sim.frame_height / 2)


def test_moves_by_velocity_times_dt():
    sim = SimulatedTarget(seed=1, speed_px_s=300.0)
    x0, y0, vx, vy = sim.x, sim.y, sim.vx, sim.vy
    obs = sim.step(0.1)
    assert obs.x == pytest.approx(x0 + vx * 0.1)
    assert obs.y == pytest.approx(y0 + vy * 0.1)


def test_target_leaves_and_reenters_frame():
    sim = SimulatedTarget(seed=1, speed_px_s=800.0)
    seen = [sim.step(DT).visible for _ in range(30 * 120)]  # 2 minutes
    assert True in seen and False in seen
    # It must come back into view after being lost at least once.
    first_lost = seen.index(False)
    assert True in seen[first_lost:]


def test_stays_inside_arena():
    sim = SimulatedTarget(seed=3, speed_px_s=2000.0)
    for _ in range(5000):
        obs = sim.step(DT)
        assert sim.arena_left <= obs.x <= sim.arena_right
        assert sim.arena_top <= obs.y <= sim.arena_bottom


def test_same_seed_gives_same_motion():
    a = SimulatedTarget(seed=7, dropout_prob=0.2)
    b = SimulatedTarget(seed=7, dropout_prob=0.2)
    for _ in range(200):
        assert a.step(DT) == b.step(DT)


def test_full_dropout_never_visible():
    sim = SimulatedTarget(seed=1, dropout_prob=1.0)
    assert not any(sim.step(DT).visible for _ in range(100))


def test_range_stays_in_expected_band():
    sim = SimulatedTarget(seed=1)
    for _ in range(1000):
        assert 5.0 <= sim.step(DT).range_m <= 25.0


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        SimulatedTarget(frame_width=0)
    with pytest.raises(ValueError):
        SimulatedTarget(dropout_prob=1.5)
    with pytest.raises(ValueError):
        SimulatedTarget(seed=1).step(0)
