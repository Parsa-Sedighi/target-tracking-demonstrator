"""Acceptance test for UC-13: reacquire after a brief blockage (FR-04, ES-03.1, ES-03.2).

UC-13 acceptance test (PRD v2, section 7):
    In 20 trials with a 1-3 s blockage, tracking resumes within 2 s of
    reappearance in >= 90%.

Simulation: a drone moves across the frame at a random speed and heading, is
blocked for 1-3 s, then reappears. The detector runs at 15 Hz with 90%
recall. A hand (other class) and a second drone far away are always in view
as distractors. Time is simulated, so the test is deterministic and fast.
"""
import math
import random

from aimbot.uc13_reacquisition import Candidate, ReacquisitionManager, State, TargetSignature

DETECTOR_HZ = 15
DT = 1 / DETECTOR_HZ


def _trial(seed: int, block_s: float | None = None):
    rng = random.Random(seed)
    speed = rng.uniform(30, 90)                      # px/s
    heading = rng.uniform(0, 2 * math.pi)
    vx, vy = speed * math.cos(heading), speed * math.sin(heading)
    x0, y0 = 320.0, 240.0
    block_start = 2.0
    block_len = block_s if block_s is not None else rng.uniform(1.0, 3.0)
    block_end = block_start + block_len

    manager = ReacquisitionManager()
    reacquired_at, lost_message_at, held_while_searching = None, None, True
    t = 0.0
    while t < block_end + 6.0:
        x, y = x0 + vx * t, y0 + vy * t
        blocked = block_start <= t < block_end
        candidates = [
            Candidate(x + 60, y + 50, 40, 60, "hand", 0.9),          # other class, close by
            Candidate(x0 + 450, y0 - 200, 30, 30, "drone", 0.9),    # same class, far away
        ]
        if not blocked and rng.random() < 0.9:
            candidates.append(Candidate(x + rng.gauss(0, 2), y + rng.gauss(0, 2), 30, 30, "drone", 0.85))

        if manager.state == State.TRACKING and not blocked:
            manager.on_track(TargetSignature(x, y, vx, vy, 30, 30, "drone"))
        elif manager.state == State.TRACKING and blocked:
            decision = manager.on_loss(t)
            held_while_searching &= decision.hold_gimbal
        else:
            decision = manager.on_detections(t, candidates)
            if decision.state == State.SEARCHING:
                held_while_searching &= decision.hold_gimbal
            if decision.state == State.TRACKING and reacquired_at is None:
                reacquired_at = t
                box = decision.reinit_box
                assert box.label == "drone" and math.hypot(box.cx - x, box.cy - y) < 10
            if decision.operator_message == "Target lost" and lost_message_at is None:
                lost_message_at = t
            if decision.state == State.LOST:
                break
        t += DT
    return block_start, block_end, reacquired_at, lost_message_at, held_while_searching


def test_tracking_resumes_within_2_s_in_at_least_90_percent_of_20_blockages():
    resumed = 0
    for seed in range(20):
        _, block_end, reacquired_at, _, held = _trial(seed)
        assert held, "gimbal must hold while searching"
        if reacquired_at is not None and reacquired_at - block_end <= 2.0:
            resumed += 1
    print(f"UC-13 metric: resumed within 2 s in {resumed}/20 trials ({resumed / 20:.0%})")
    assert resumed / 20 >= 0.90


def test_target_lost_shown_within_1_s_after_the_5_s_search_window():
    block_start, _, reacquired_at, lost_message_at, held = _trial(seed=7, block_s=8.0)
    assert reacquired_at is None
    assert held
    window_end = block_start + 5.0
    assert lost_message_at is not None
    assert window_end <= lost_message_at <= window_end + 1.0


def test_lost_is_declared_by_the_timer_even_with_no_detections():
    manager = ReacquisitionManager()
    manager.on_track(TargetSignature(100, 100, 0, 0, 30, 30, "drone"))
    manager.on_loss(10.0)
    assert manager.tick(14.9).state == State.SEARCHING
    decision = manager.tick(15.1)
    assert decision.state == State.LOST and decision.operator_message == "Target lost"
