"""Acceptance test for UC-12: keep the chosen target when paths cross (FR-02, ES-01.4).

UC-12 acceptance test (PRD v2, section 7):
    In 20 trials with two targets crossing paths, the selected target is kept
    in >= 95%.

Worst case: two targets of the same class and size cross head-on through the
same point. While they overlap, the detector reports a single merged box, and
it randomly misses 5% of detections throughout.
"""
import math
import random

from aimbot.uc12_target_retention import Detection, SelectedTargetTracker

DETECTOR_HZ = 15
TRIALS = 20


def _run_crossing_trial(seed: int, distractor_label: str = "drone") -> bool:
    rng = random.Random(seed)
    steps = 90  # 6 s of detector output
    speed = rng.uniform(4.0, 8.0)            # px per detector frame
    slope = rng.uniform(0.1, 0.5)            # crossing angle
    cross_x, cross_y = 320 + rng.uniform(-40, 40), 240 + rng.uniform(-40, 40)

    def position(name: str, k: int) -> tuple[float, float]:
        dt = k - steps / 2
        if name == "chosen":
            return cross_x + speed * dt, cross_y + slope * speed * dt
        return cross_x - speed * dt, cross_y + slope * speed * dt * 0.5

    tracker = SelectedTargetTracker()
    for k in range(steps):
        a, b = position("chosen", k), position("other", k)
        dets = []
        overlapping = math.hypot(a[0] - b[0], a[1] - b[1]) < 12
        visible = [("chosen", a)] if overlapping else [("chosen", a), ("other", b)]
        for name, (x, y) in visible:
            if rng.random() < 0.05:
                continue  # detector miss
            label = "drone" if name == "chosen" else distractor_label
            dets.append(Detection(x + rng.gauss(0, 1.5), y + rng.gauss(0, 1.5), 30, 30, label))
        tracker.update(dets)
        if k == 3 and tracker.selected_id is None:
            tracker.select_nearest(*position("chosen", k))

    selected = tracker.selected
    if selected is None:
        return False
    end_chosen = position("chosen", steps - 1)
    end_other = position("other", steps - 1)
    d_chosen = math.hypot(selected.cx - end_chosen[0], selected.cy - end_chosen[1])
    d_other = math.hypot(selected.cx - end_other[0], selected.cy - end_other[1])
    return d_chosen < 10 and d_chosen < d_other


def test_selected_target_kept_in_at_least_95_percent_of_20_crossings():
    kept = sum(_run_crossing_trial(seed) for seed in range(TRIALS))
    print(f"UC-12 metric: selected target kept in {kept}/{TRIALS} crossing trials ({kept / TRIALS:.0%})")
    assert kept / TRIALS >= 0.95


def test_target_of_a_different_class_never_steals_the_lock():
    kept = sum(_run_crossing_trial(seed, distractor_label="hand") for seed in range(100, 100 + TRIALS))
    assert kept == TRIALS


def test_operator_can_only_select_an_existing_track():
    tracker = SelectedTargetTracker()
    tracker.update([Detection(100, 100, 30, 30, "drone")])
    track_id = tracker.select_nearest(102, 98)
    assert tracker.selected.track_id == track_id
    try:
        tracker.select(999)
    except KeyError:
        pass
    else:
        raise AssertionError("selecting a missing track should fail")
