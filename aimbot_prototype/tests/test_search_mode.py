"""PROJECT23-13 / UC-09 preliminary mock tests. ROS DDS and physical hardware are untested."""

import importlib.util
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from aimbot import clock
from aimbot.bus import Topics
from aimbot.config import load_config
from aimbot.engine import Engine
from aimbot.ros2_bridge import Ros2Bridge
from aimbot.sim.world import Occluder, build_scenario
from aimbot.state_machine import SEARCHING, TRACKING
from aimbot.types import GimbalCommand
from ros2_mock import String, install_ros2_mock


def wait_for(predicate, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail("timed out waiting for mock target/state/gimbal")


@pytest.fixture
def engine(tmp_path):
    cfg = load_config("search_mock", [f"calibration.file={tmp_path / 'calibration.json'}",
                                     f"logging.dir={tmp_path / 'logs'}", "safety.home_on_shutdown=false"])
    e = Engine(cfg)
    yield e
    e.shutdown()


@pytest.fixture
def bridge(engine, monkeypatch):
    install_ros2_mock(monkeypatch)
    b = Ros2Bridge(engine, {})
    unsubscribe = engine.bus.subscribe(Topics.STATE, b._on_bus)
    yield b
    unsubscribe()


def published_states(bridge):
    return [(t, json.loads(message.data)) for t, message in bridge.p_state.messages]


@pytest.fixture
def ros_node(monkeypatch):
    install_ros2_mock(monkeypatch)
    path = Path(__file__).resolve().parents[2] / "ros2_ws/src/integration_stack/state_machine_node.py"
    spec = importlib.util.spec_from_file_location("uc09_state_machine_node", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SystemStateMachine()


@pytest.mark.parametrize("status, expected", [("LOST", "SEARCHING"), ("ACQUIRED", "TRACKING")])
def test_ros_node_publishes_transition_in_callback_without_timer(ros_node, status, expected):
    msg = String()
    msg.data = "ACQUIRED"
    ros_node.target_callback(msg)
    if status == "ACQUIRED":
        msg.data = "LOST"
        ros_node.target_callback(msg)
    ros_node.state_publisher.messages.clear()
    msg.data = status
    ros_node.target_callback(msg)
    assert [m.data for _, m in ros_node.state_publisher.messages] == [expected]
    assert ros_node.timers[0][0] == 0.01  # heartbeat retained, never invoked by this test


def test_ros_node_repeated_loss_does_not_restart_transition(ros_node):
    msg = String()
    msg.data = "ACQUIRED"
    ros_node.target_callback(msg)
    msg.data = "LOST"
    ros_node.target_callback(msg)
    before = len(ros_node.state_publisher.messages)
    ros_node.target_callback(msg)
    assert len(ros_node.state_publisher.messages) == before
    ros_node.publish_state()
    assert ros_node.state_publisher.messages[-1][1].data == "SEARCHING"


def test_bridge_does_not_throttle_changed_state(bridge, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(clock, "now", lambda: now[0])
    bridge._on_bus(Topics.STATE, {"state": TRACKING})
    bridge._on_bus(Topics.STATE, {"state": SEARCHING})
    bridge._on_bus(Topics.STATE, {"state": SEARCHING})
    assert [s["state"] for _, s in published_states(bridge)] == [TRACKING, SEARCHING]
    now[0] += 0.02
    bridge._on_bus(Topics.STATE, {"state": SEARCHING})
    assert len(published_states(bridge)) == 3
    assert bridge.p_state.topic == "aimbot/state"


@pytest.mark.parametrize("now", [0.0, 100.0])
def test_engine_publishes_and_holds_before_loss_callback_returns(engine, bridge, monkeypatch, now):
    monkeypatch.setattr(clock, "now", lambda: now)
    assert engine.cfg.search.coast_s == 0.0
    engine.sm.fire("start")
    engine.sm.fire("acquired")
    engine.gimbal.command(GimbalCommand("rate", 0.3, 0.1))
    bridge.p_state.messages.clear()
    engine.sm.fire("lost", "UC-09 mock removal")
    assert engine.gimbal.last_cmd.mode == "hold"
    assert engine.bus.latest(Topics.GIMBAL_CMD).mode == "hold"
    assert published_states(bridge)[-1][1]["state"] == SEARCHING
    assert published_states(bridge)[-1][1]["reason"] == "UC-09 mock removal"
    assert engine.sm.reacquire_remaining() > 0


def test_delayed_tracking_heartbeat_cannot_overwrite_search_state(engine, bridge):
    engine.sm.fire("start")
    engine.sm.fire("acquired")
    engine.sm.fire("lost")
    # The control loop captured TRACKING before the vision thread detected a loss.
    engine._publish_heartbeat(clock.now(), TRACKING, engine.gimbal.state(), None, SimpleNamespace(dt_avg=0.01))
    assert engine.bus.latest(Topics.STATE)["state"] == SEARCHING
    assert published_states(bridge)[-1][1]["state"] == SEARCHING


@pytest.mark.parametrize("event", ["lost", "stop", "estop", "fault"])
def test_precomputed_tracking_command_is_rejected_after_transition(engine, event):
    engine.sm.fire("start")
    engine.sm.fire("acquired")
    stale = GimbalCommand("rate", 0.8, 0.2)
    engine.sm.fire(event)
    last = engine.gimbal.last_cmd
    assert engine._command_for_state(stale, TRACKING) is None
    assert engine.gimbal.last_cmd is last


def test_coasting_remains_explicitly_opt_in(engine):
    engine.cfg.search.coast_s = 0.4
    engine.sm.fire("start")
    engine.sm.fire("acquired")
    engine.gimbal.command(GimbalCommand("rate", 0.3, 0))
    last = engine.gimbal.last_cmd
    engine.sm.fire("lost")
    assert engine.gimbal.last_cmd is last


def test_simulator_demo_actually_removes_and_restores_target():
    targets, _ = build_scenario("target_loss")
    assert targets[0].position(4.99) is not None
    assert targets[0].position(5.0) is None
    assert targets[0].position(7.99) is None
    assert targets[0].position(8.0) is not None


@pytest.fixture
def tracking(engine, bridge):
    e = engine
    e.start()
    assert e.camera.kind == e.gimbal.kind == e.lidar.kind == "sim"
    # Keep the target visible until the test removes it, independent of wall-clock startup time.
    e.world.targets[0].path = lambda t: np.array([3.048, 0.4 * math.sin(0.7 * t), 0.0])
    e.command({"cmd": "start"})
    wait_for(lambda: e.sm.state == TRACKING and e.kf.updates >= 3)
    return e, bridge


def hide_target(e, failure):
    if failure == "occluded":
        e.world.occluders = [Occluder(1.0, 0.0, 0.8, z0=-1.0, z1=2.0)]
    elif failure == "outside_frame":
        e.world.targets[0].path = lambda t: np.array([3.048, 100.0, 0.0])
    else:
        e.world.targets[0].path = lambda t: None


@pytest.mark.slow
@pytest.mark.parametrize("failure", ["removed", "outside_frame", "occluded"])
@pytest.mark.parametrize("mode", ["angle", "rate"])
def test_camera_loss_publishes_search_and_holds_gimbal(engine, bridge, failure, mode, record_property):
    e = engine
    e.cfg.control.mode = mode
    e.gimbal.cfg["command_mode"] = mode
    e.controller.mode = mode
    e.start()
    assert e.camera.kind == e.gimbal.kind == e.lidar.kind == "sim"
    e.world.targets[0].path = lambda t: np.array([3.048, 0.4 * math.sin(0.7 * t), 0.0])
    e.command({"cmd": "start"})
    wait_for(lambda: e.sm.state == TRACKING and e.kf.updates >= 3)
    observed = []
    e.sm.listeners.append(lambda old, new, reason, t: observed.append((t, old, new)))
    sent = []
    e.bus.subscribe(Topics.GIMBAL_CMD, lambda topic, cmd: sent.append((clock.now(), cmd)))
    removed_at = clock.now()
    hide_target(e, failure)
    wait_for(lambda: e.framebuf.latest().t_capture > removed_at and
             not any(t["id"] == 9 and t["visible"] > 0.5 for t in e.framebuf.latest().meta["truth"]))
    wait_for(lambda: any(old == TRACKING and new == SEARCHING for _, old, new in observed))
    lost_at = next(t for t, old, new in observed if old == TRACKING and new == SEARCHING)
    wait_for(lambda: any(t >= lost_at and s["state"] == SEARCHING for t, s in published_states(bridge)))
    published_at = next(t for t, s in published_states(bridge) if t >= lost_at and s["state"] == SEARCHING)
    hold_at = next(t for t, c in sent if t >= lost_at and c.mode == "hold")
    # Mock regression budget includes 250 ms detector confirmation + camera / OS scheduling.
    # Synchronous callback tests above prove no timer wait; this is not a hardware timing claim.
    assert 0 <= lost_at - removed_at < 0.75
    assert 0 <= published_at - lost_at < 0.05
    assert 0 <= hold_at - lost_at < 0.05
    time.sleep(0.3)  # simulated actuator deceleration / transport settling
    a = e.gimbal.state()
    time.sleep(0.25)
    b = e.gimbal.state()
    drift = math.degrees(math.hypot(b.pan - a.pan, b.tilt - a.tilt))
    assert e.sm.state == SEARCHING
    assert drift < 0.2
    assert abs(math.degrees(b.pan_rate)) < 1 and abs(math.degrees(b.tilt_rate)) < 1
    assert e.gimbal.last_cmd.mode == "hold"
    assert all(c.mode == "hold" for t, c in sent if t >= lost_at)
    states = [(t, s) for t, s in published_states(bridge) if t >= lost_at]
    assert len(states) >= 10 and all(s["state"] == SEARCHING for _, s in states)
    report = dict(status="PRELIMINARY MOCK - ROS DDS and hardware untested", failure=failure, mode=mode,
                  loss_detection_ms=(lost_at - removed_at) * 1000,
                  search_publish_ms=(published_at - lost_at) * 1000, hold_command_ms=(hold_at - lost_at) * 1000,
                  settled_drift_deg=drift, state_messages=len(states))
    record_property("uc09_mock_measurements", json.dumps(report))
    print("\nUC-09 preliminary mock: " + json.dumps(report))


@pytest.mark.slow
def test_reacquisition_resumes_tracking_without_restart(tracking):
    e, bridge = tracking
    hide_target(e, "removed")
    wait_for(lambda: e.sm.state == SEARCHING)
    pan, tilt = e.gimbal.true_angles(clock.now())
    e.world.targets[0].path = lambda t: np.array([3.048, 3.048 * math.tan(pan), 3.048 * math.tan(tilt)])
    wait_for(lambda: e.sm.state == TRACKING)
    assert e.stats["reacquired"] >= 1
    assert published_states(bridge)[-1][1]["state"] == TRACKING


@pytest.mark.slow
def test_search_timeout_holds_and_estop_stays_latched_when_target_returns(tracking):
    e, bridge = tracking
    e.sm.reacquire_window = 0.3
    hide_target(e, "removed")
    wait_for(lambda: e.sm.state == "LOST")
    assert e.gimbal.last_cmd.mode == "hold"
    assert published_states(bridge)[-1][1]["state"] == "LOST"
    e.command({"cmd": "estop"})
    e.world.targets[0].path = lambda t: np.array([3.048, 0.0, 0.0])
    time.sleep(0.3)
    assert e.sm.state == "ESTOP"
    assert e.gimbal.estopped
    assert published_states(bridge)[-1][1]["state"] == "ESTOP"
