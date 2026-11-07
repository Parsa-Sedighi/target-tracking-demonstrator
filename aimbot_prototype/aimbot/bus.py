"""In-process message bus that mirrors the ROS 2 topic layout.

On a PC without ROS 2 this *is* the communication layer. On the Jetson the
optional ``ros2_bridge`` republishes the same topics with rclpy so the ECE
team's C++ nodes (Kalman filter, motor control) can subscribe to them.

Semantics: publish() stores the latest message per topic and synchronously
calls subscribers in the publisher's thread, so callbacks must be quick
(copy/enqueue only). Topic names are listed in ``Topics`` and documented in
docs/ICD.md.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, Callable

from . import clock

log = logging.getLogger(__name__)


class Topics:
    STATE = "/aimbot/state"                      # 100 Hz heartbeat: system state machine
    TARGET_OFFSET = "/aimbot/target/offset"      # FR-08: normalised (dx, dy) + angular error, per frame
    TARGET_ESTIMATE = "/aimbot/target/estimate"  # Kalman estimate (az, el, rates, range, xyz)
    DETECTIONS = "/aimbot/detections"            # macro (YOLO) detections with track IDs
    GIMBAL_STATE = "/aimbot/gimbal/state"        # measured pan/tilt (encoder feedback)
    GIMBAL_CMD = "/aimbot/gimbal/command"        # command sent to gimbal (angle or rate)
    GIMBAL_RATE_CMD_EXT = "/aimbot/gimbal/cmd_rate_external"  # rate command from an external (ECE) controller
    LIDAR = "/aimbot/lidar/range"                # TF02-Pro range
    EVENTS = "/aimbot/events"                    # state changes, operator commands, faults
    OCR = "/aimbot/ocr"                          # OCR results
    OPERATOR_CMD = "/aimbot/operator/command"    # commands from GUI / gateway / ROS


Callback = Callable[[str, Any], None]


class Bus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, tuple[float, Any]] = {}
        self._subs: dict[str, list[Callback]] = defaultdict(list)
        self.counts: dict[str, int] = defaultdict(int)

    def publish(self, topic: str, msg: Any) -> None:
        t = clock.now()
        with self._lock:
            self._latest[topic] = (t, msg)
            self.counts[topic] += 1
            subs = list(self._subs.get(topic, ())) + list(self._subs.get("*", ()))
        for cb in subs:
            try:
                cb(topic, msg)
            except Exception:  # a bad subscriber must never break the publisher
                log.exception("bus subscriber failed on %s", topic)

    def latest(self, topic: str, default: Any = None) -> Any:
        item = self._latest.get(topic)
        return item[1] if item else default

    def latest_with_time(self, topic: str) -> tuple[float, Any] | None:
        return self._latest.get(topic)

    def subscribe(self, topic: str, cb: Callback) -> Callable[[], None]:
        with self._lock:
            self._subs[topic].append(cb)

        def unsubscribe() -> None:
            with self._lock:
                if cb in self._subs.get(topic, []):
                    self._subs[topic].remove(cb)

        return unsubscribe
