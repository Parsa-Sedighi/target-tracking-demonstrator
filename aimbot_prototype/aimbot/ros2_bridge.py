"""Optional ROS 2 bridge (FR-04 / FR-08): republishes the engine's bus topics with rclpy.

Enable with ``ros2.enabled: true`` on a machine with ROS 2 (Jetson / JetPack 6 =
Ubuntu 22.04 -> ROS 2 Humble): ``source /opt/ros/humble/setup.bash`` first.
Without rclpy the engine runs normally and this module is skipped.

Published (all stamps = frame capture / measurement time, mapped to ROS time)
  /aimbot/state                  std_msgs/String (JSON)          100 Hz heartbeat
  /aimbot/target/offset          geometry_msgs/Vector3Stamped    x=err_az, y=err_el [rad], z=match score
  /aimbot/target/offset_norm     geometry_msgs/Vector3Stamped    x=dx, y=dy  (FR-08, [-1,1], +right/+up)
  /aimbot/target/estimate        geometry_msgs/Vector3Stamped    x=az, y=el [rad], z=range [m] (NaN if none)
  /aimbot/gimbal/joint_states    sensor_msgs/JointState          pan, tilt position [rad] / velocity [rad/s]
  /aimbot/lidar/range            sensor_msgs/Range               TF02-Pro distance
  /aimbot/detections, /aimbot/events, /aimbot/ocr   std_msgs/String (JSON)
Subscribed
  /aimbot/gimbal/cmd_rate_external  geometry_msgs/Vector3Stamped x=pan rate, y=tilt rate [rad/s]
                                    (used when control.source = external: the ECE controller drives the gimbal)
  /aimbot/operator/command          std_msgs/String JSON, same commands as the GUI ({"cmd": "start"})

Written against the standard rclpy API; not yet run on a ROS 2 install.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from typing import Optional

from . import clock
from .bus import Topics

log = logging.getLogger(__name__)


def start_ros2_bridge(engine, cfg: dict) -> Optional["Ros2Bridge"]:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        log.warning("ROS 2 bridge requested but rclpy is not importable (source /opt/ros/<distro>/setup.bash)")
        return None
    try:
        b = Ros2Bridge(engine, cfg or {})
        b.start()
        return b
    except Exception as exc:
        log.error("ROS 2 bridge failed to start: %s", exc)
        return None


class Ros2Bridge:
    def __init__(self, engine, cfg: dict):
        import rclpy
        from geometry_msgs.msg import Vector3Stamped
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.time import Time
        from sensor_msgs.msg import JointState, Range
        from std_msgs.msg import String

        self.rclpy = rclpy
        self.Time = Time
        self.engine = engine
        self.cfg = cfg
        self.M = {"String": String, "V3": Vector3Stamped, "JointState": JointState, "Range": Range}
        if not rclpy.ok():
            rclpy.init(args=None)
        ns = str(cfg.get("namespace", "") or "")
        self.node = rclpy.create_node("aimbot", namespace=ns) if ns else rclpy.create_node("aimbot")
        fast = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        rel = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        P = self.node.create_publisher
        self.p_state = P(String, "aimbot/state", rel)
        self.p_off = P(Vector3Stamped, "aimbot/target/offset", fast)
        self.p_offn = P(Vector3Stamped, "aimbot/target/offset_norm", fast)
        self.p_est = P(Vector3Stamped, "aimbot/target/estimate", fast)
        self.p_js = P(JointState, "aimbot/gimbal/joint_states", fast)
        self.p_rng = P(Range, "aimbot/lidar/range", fast)
        self.p_det = P(String, "aimbot/detections", rel)
        self.p_evt = P(String, "aimbot/events", rel)
        self.p_ocr = P(String, "aimbot/ocr", rel)
        self.node.create_subscription(Vector3Stamped, "aimbot/gimbal/cmd_rate_external", self._on_ext_rate, fast)
        self.node.create_subscription(String, "aimbot/operator/command", self._on_cmd, rel)
        self._stop = threading.Event()
        self._unsubs = []
        self._last_state_pub = 0.0
        self.state_period = 1.0 / float(cfg.get("state_hz", 100))

    # ------------------------------------------------------------------ helpers
    def _stamp(self, t_mono: float):
        """Engine monotonic time -> ROS time message (same instant)."""
        now_ns = self.node.get_clock().now().nanoseconds
        ns = now_ns - int((clock.now() - t_mono) * 1e9)
        return self.Time(nanoseconds=max(0, ns)).to_msg()

    def _v3(self, t, x, y, z, frame_id="aimbot_aim"):
        m = self.M["V3"]()
        m.header.stamp = self._stamp(t)
        m.header.frame_id = frame_id
        m.vector.x, m.vector.y, m.vector.z = float(x), float(y), float(z)
        return m

    def _string(self, obj) -> object:
        m = self.M["String"]()
        m.data = json.dumps(obj, default=str)
        return m

    # ------------------------------------------------------------------ bus -> ROS
    def _on_bus(self, topic: str, msg) -> None:
        try:
            if topic == Topics.STATE:
                t = clock.now()
                if t - self._last_state_pub >= self.state_period * 0.95:
                    self._last_state_pub = t
                    self.p_state.publish(self._string(msg))
            elif topic == Topics.TARGET_OFFSET:
                self.p_off.publish(self._v3(msg["t_capture"], msg["err_az_rad"], msg["err_el_rad"], msg.get("score", 0.0)))
                self.p_offn.publish(self._v3(msg["t_capture"], msg["dx"], msg["dy"], 0.0))
            elif topic == Topics.TARGET_ESTIMATE:
                r = msg.get("range_m")
                self.p_est.publish(self._v3(msg["t"], msg["az_rad"], msg["el_rad"], math.nan if r is None else r, "aimbot_base"))
            elif topic == Topics.GIMBAL_STATE:
                js = self.M["JointState"]()
                js.header.stamp = self._stamp(msg.t if msg.t > 0 else clock.now())
                js.name = ["pan", "tilt"]
                js.position = [float(msg.pan), float(msg.tilt)]
                js.velocity = [float(msg.pan_rate), float(msg.tilt_rate)]
                self.p_js.publish(js)
            elif topic == Topics.LIDAR:
                R = self.M["Range"]
                rg = R()
                rg.header.stamp = self._stamp(msg.t)
                rg.header.frame_id = "aimbot_lidar"
                rg.radiation_type = R.INFRARED
                rg.field_of_view = math.radians(3.0)
                rg.min_range, rg.max_range = 0.1, 40.0
                rg.range = float(msg.distance_m) if msg.valid else math.inf
                self.p_rng.publish(rg)
            elif topic == Topics.DETECTIONS:
                self.p_det.publish(self._string(msg))
            elif topic == Topics.EVENTS:
                self.p_evt.publish(self._string(msg))
            elif topic == Topics.OCR:
                self.p_ocr.publish(self._string(msg))
        except Exception:
            log.exception("ros2 publish failed on %s", topic)

    # ------------------------------------------------------------------ ROS -> engine
    def _on_ext_rate(self, msg) -> None:
        self.engine.command({"cmd": "external_rate", "pan_dps": math.degrees(msg.vector.x),
                             "tilt_dps": math.degrees(msg.vector.y)})

    def _on_cmd(self, msg) -> None:
        try:
            self.engine.command(json.loads(msg.data))
        except Exception as exc:
            log.warning("bad operator command on ROS topic: %s", exc)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        for tp in (Topics.STATE, Topics.TARGET_OFFSET, Topics.TARGET_ESTIMATE, Topics.GIMBAL_STATE, Topics.LIDAR,
                   Topics.DETECTIONS, Topics.EVENTS, Topics.OCR):
            self._unsubs.append(self.engine.bus.subscribe(tp, self._on_bus))
        from rclpy.executors import SingleThreadedExecutor

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        def spin():
            while not self._stop.is_set() and self.rclpy.ok():
                self.executor.spin_once(timeout_sec=0.05)

        threading.Thread(target=spin, name="ros2-spin", daemon=True).start()
        log.info("ROS 2 bridge up (node /aimbot)")

    def stop(self) -> None:
        self._stop.set()
        for u in self._unsubs:
            u()
        try:
            self.node.destroy_node()
            if self.rclpy.ok():
                self.rclpy.shutdown()
        except Exception:
            pass
