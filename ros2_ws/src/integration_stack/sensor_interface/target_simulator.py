# ros2_ws/src/integration_stack/sensor_interface/target_simulator.py
"""
Hardware-free target simulator.

Pretends to be "the camera + LiDAR saw a target" so the rest of the software
(state machine, tracking, gimbal control, logging) can be developed and tested
before real hardware arrives.

This file is plain Python on purpose: it does NOT import ROS 2, so it can be
unit-tested on any laptop. The ROS 2 wrapper lives in sim_camera_node.py.

How it works
------------
A virtual target moves around an "arena" that is larger than the camera frame.
When the target is inside the frame it is VISIBLE; when it drifts outside, it
is LOST. It bounces off the arena edges, so it leaves and re-enters view
repeatedly. That exercises the IDLE/SEARCHING/TRACKING transitions.

Coordinates are pixels, origin at the top-left of the frame (x right, y down),
the same convention OpenCV uses.
"""
import math
import random
from dataclasses import dataclass
from typing import Optional

# Default frame size assumes the FLIR Blackfly S BFS-U3-28S5C (2.8 MP).
# CONFIRM against the camera datasheet / Spinnaker before relying on it.
DEFAULT_FRAME_WIDTH = 1936
DEFAULT_FRAME_HEIGHT = 1464


@dataclass(frozen=True)
class Observation:
    """What the simulated sensors report at one instant."""
    t: float            # simulated time in seconds
    x: float            # target x position in pixels
    y: float            # target y position in pixels
    visible: bool       # is the target inside the camera frame (and not dropped)?
    error_x: float      # x offset from frame center (px). Gimbal control input.
    error_y: float      # y offset from frame center (px). Gimbal control input.
    range_m: float      # simulated LiDAR distance to target, meters


class SimulatedTarget:
    def __init__(
        self,
        frame_width: int = DEFAULT_FRAME_WIDTH,
        frame_height: int = DEFAULT_FRAME_HEIGHT,
        arena_margin: float = 0.5,
        speed_px_s: float = 400.0,
        dropout_prob: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        arena_margin: how far past each frame edge the target can roam, as a
            fraction of the frame size. 0.5 means the arena is 2x the frame in
            each direction, so the target spends a while out of view.
        speed_px_s: target speed in pixels per second.
        dropout_prob: chance (0..1) per step that a visible target is missed,
            like a failed detection. 0 means never.
        seed: set to get the exact same motion every run (good for tests).
        """
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame size must be positive")
        if not 0.0 <= dropout_prob <= 1.0:
            raise ValueError("dropout_prob must be between 0 and 1")
        if arena_margin < 0:
            raise ValueError("arena_margin must be >= 0")

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.dropout_prob = dropout_prob
        self._rng = random.Random(seed)

        # Arena = frame plus a margin on every side.
        mx = frame_width * arena_margin
        my = frame_height * arena_margin
        self.arena_left, self.arena_right = -mx, frame_width + mx
        self.arena_top, self.arena_bottom = -my, frame_height + my

        # Start at the frame center, heading off in a random direction.
        self.x = frame_width / 2
        self.y = frame_height / 2
        angle = self._rng.uniform(0, 2 * math.pi)
        self.vx = speed_px_s * math.cos(angle)
        self.vy = speed_px_s * math.sin(angle)
        self.t = 0.0

    def in_frame(self) -> bool:
        return 0 <= self.x < self.frame_width and 0 <= self.y < self.frame_height

    def _range_m(self) -> float:
        # Fake LiDAR: distance slowly oscillates between 5 m and 25 m.
        return 15.0 + 10.0 * math.sin(self.t / 10.0)

    def step(self, dt: float) -> Observation:
        """Advance the simulation by dt seconds and return what sensors see."""
        if dt <= 0:
            raise ValueError("dt must be positive")

        self.x += self.vx * dt
        self.y += self.vy * dt
        self.t += dt

        # Bounce off the arena walls.
        if self.x < self.arena_left:
            self.x = self.arena_left + (self.arena_left - self.x)
            self.vx = abs(self.vx)
        elif self.x > self.arena_right:
            self.x = self.arena_right - (self.x - self.arena_right)
            self.vx = -abs(self.vx)
        if self.y < self.arena_top:
            self.y = self.arena_top + (self.arena_top - self.y)
            self.vy = abs(self.vy)
        elif self.y > self.arena_bottom:
            self.y = self.arena_bottom - (self.y - self.arena_bottom)
            self.vy = -abs(self.vy)

        visible = self.in_frame()
        if visible and self.dropout_prob > 0 and self._rng.random() < self.dropout_prob:
            visible = False  # simulated missed detection

        return Observation(
            t=self.t,
            x=self.x,
            y=self.y,
            visible=visible,
            error_x=self.x - self.frame_width / 2,
            error_y=self.y - self.frame_height / 2,
            range_m=self._range_m(),
        )
