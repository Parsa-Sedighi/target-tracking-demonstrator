"""Gimbal travel-limit handling: what to do when a tracked target moves past the end of pan travel.

The gimbal cannot spin all the way round. With the default soft limits (pan -170..+170 deg)
there is a 20 deg "dead zone" behind it. A target that keeps moving clockwise past +170 deg
disappears into the dead zone and comes back out at -170 deg (the same direction as +190 deg).

Without special handling, a controller that steers along the shortest angle would push the
gimbal into its +170 deg stop and leave it there. This module implements the team's first
idea instead: **reverse direction**. When the target goes past the limit, the gimbal swings
back the long way round (through 0 deg) and waits at the opposite edge for the target to
come back out of the dead zone.

Policies
--------
``reverse`` (default)
    Target past a pan limit and moving away -> swing the long way to the opposite edge and
    wait there. Target past the limit but stopped or turning back -> hold at the near edge.
``hold``
    Always stop at the nearest edge and wait (the prototype's current behaviour), so the two
    policies can be compared.

Trade-off to keep in mind: the reversal is a long move. Swinging 340 deg at 100 deg/s takes
about 3.4 s, and the target is out of view for that time (``reversal_time_s()`` gives the
number for the configured limits). ``reverse`` wins when the target keeps circling;
``hold`` wins when it tends to turn back.

Tilt does not wrap (-45..+90 deg), so tilt is only clamped to its limits.

Angles are in degrees, + pan = clockwise seen from above (same convention as the prototype).
The module has no dependencies, so it can be used from the AimBot prototype or a ROS 2 node.

Example::

    handler = TravelLimitHandler(TravelLimits(pan_min_deg=-170, pan_max_deg=170))
    decision = handler.update(gimbal_pan_deg=169.0, target_pan_deg=175.0,
                              target_tilt_deg=10.0, target_pan_rate_dps=30.0)
    decision.state      # LimitState.REVERSING
    decision.pan_deg    # -168.0: command the gimbal here (angle mode)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

POLICIES = ("reverse", "hold")


class LimitState(Enum):
    TRACKING = "TRACKING"    # target is inside travel; follow it
    AT_LIMIT = "AT_LIMIT"    # target is at or past an edge; the gimbal holds at that edge
    REVERSING = "REVERSING"  # swinging the long way to the opposite edge
    WAITING = "WAITING"      # parked at the opposite edge, waiting for the target to come out


@dataclass
class TravelLimits:
    pan_min_deg: float = -170.0
    pan_max_deg: float = 170.0
    tilt_min_deg: float = -45.0
    tilt_max_deg: float = 90.0
    margin_deg: float = 2.0         # park this far inside an edge, never on the hard stop
    trigger_deg: float = 3.0        # target must be this far past an edge before reversing (hysteresis)
    min_rate_dps: float = 2.0       # slower than this counts as "not moving" (no reversal)
    arrive_tol_deg: float = 1.0     # how close to the parking point counts as arrived
    max_rate_dps: float = 100.0     # gimbal slew rate, only used for reversal_time_s()

    def __post_init__(self) -> None:
        if self.pan_max_deg <= self.pan_min_deg:
            raise ValueError("pan_max_deg must be greater than pan_min_deg")
        if self.tilt_max_deg <= self.tilt_min_deg:
            raise ValueError("tilt_max_deg must be greater than tilt_min_deg")
        if self.margin_deg < 0 or 2 * self.margin_deg >= self.pan_max_deg - self.pan_min_deg:
            raise ValueError("margin_deg must be >= 0 and smaller than half the pan range")

    @property
    def continuous(self) -> bool:
        """True if pan covers a full turn (slip ring): there is no dead zone to handle."""
        return self.pan_max_deg - self.pan_min_deg >= 360.0


@dataclass
class LimitDecision:
    pan_deg: float           # pan angle to command (angle mode), inside the travel limits
    tilt_deg: float          # tilt angle to command, inside the travel limits
    state: LimitState
    reason: str              # short text for the operator display / logs


def wrap180(angle_deg: float) -> float:
    """Wrap any angle into [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


class TravelLimitHandler:
    """Turns a target direction into a pan/tilt setpoint that respects the gimbal's travel limits.

    Call ``update`` once per control cycle while tracking, and ``reset`` when tracking stops or a
    new target is selected.
    """

    def __init__(self, limits: TravelLimits | None = None, policy: str = "reverse") -> None:
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
        self.limits = limits or TravelLimits()
        self.policy = policy
        self.state = LimitState.TRACKING
        self._goal_pan: float | None = None  # parking point while REVERSING / WAITING

    def reset(self) -> None:
        self.state = LimitState.TRACKING
        self._goal_pan = None

    def reversal_time_s(self) -> float:
        """Time for a full swing from one parking point to the other at the gimbal's slew rate."""
        lim = self.limits
        travel = (lim.pan_max_deg - lim.margin_deg) - (lim.pan_min_deg + lim.margin_deg)
        return travel / lim.max_rate_dps

    def update(
        self,
        gimbal_pan_deg: float,
        target_pan_deg: float,
        target_tilt_deg: float,
        target_pan_rate_dps: float = 0.0,
    ) -> LimitDecision:
        """Decide where to point the gimbal this cycle.

        Args:
            gimbal_pan_deg: current pan from the encoders (inside the travel limits).
            target_pan_deg: direction of the target, any value (e.g. 190 and -170 are the same).
            target_tilt_deg: elevation of the target.
            target_pan_rate_dps: how fast the target direction is changing, + = clockwise
                (e.g. from the Kalman filter). Used to tell which edge it will come back out of.
        """
        lim = self.limits
        tilt = min(lim.tilt_max_deg, max(lim.tilt_min_deg, target_tilt_deg))
        tilt_note = "; tilt at limit" if tilt != target_tilt_deg else ""

        pan = self._reachable(target_pan_deg, gimbal_pan_deg)
        if pan is not None:
            # Target is inside travel. Commanding this angle never crosses the dead zone, so if the
            # target came back on the other side the gimbal simply swings the long way round to it.
            self.reset()
            near_edge = pan >= lim.pan_max_deg - lim.margin_deg or pan <= lim.pan_min_deg + lim.margin_deg
            if near_edge:
                self.state = LimitState.AT_LIMIT
                return LimitDecision(pan, tilt, self.state, "target at edge of pan travel" + tilt_note)
            return LimitDecision(pan, tilt, self.state, "tracking" + tilt_note)

        # Target is in the dead zone behind the gimbal.
        near_edge, past_by = self._nearest_edge(target_pan_deg)
        moving = abs(target_pan_rate_dps) >= lim.min_rate_dps

        if self.state in (LimitState.REVERSING, LimitState.WAITING) and self._goal_pan is not None:
            return self._continue_reversal(gimbal_pan_deg, tilt, tilt_note)

        if self.policy == "hold" or not moving or past_by < lim.trigger_deg:
            self.state = LimitState.AT_LIMIT
            why = ("hold policy" if self.policy == "hold"
                   else "target stopped" if not moving else "target just past edge")
            return LimitDecision(self._park(near_edge), tilt, self.state,
                                 f"target outside pan travel ({why}); holding at edge" + tilt_note)

        # Moving clockwise (+) through the dead zone means it comes out at pan_min, and vice versa.
        exit_edge = lim.pan_min_deg if target_pan_rate_dps > 0 else lim.pan_max_deg
        if exit_edge == near_edge:
            # Target is heading back the way it came: wait at this edge, no reversal needed.
            self.state = LimitState.AT_LIMIT
            return LimitDecision(self._park(near_edge), tilt, self.state,
                                 "target turning back; holding at edge" + tilt_note)

        self._goal_pan = self._park(exit_edge)
        self.state = LimitState.REVERSING
        return self._continue_reversal(gimbal_pan_deg, tilt, tilt_note)

    # ------------------------------------------------------------------ helpers
    def _continue_reversal(self, gimbal_pan_deg: float, tilt: float, tilt_note: str) -> LimitDecision:
        assert self._goal_pan is not None
        if abs(gimbal_pan_deg - self._goal_pan) <= self.limits.arrive_tol_deg:
            self.state = LimitState.WAITING
            reason = "waiting at opposite edge for target to come back"
        else:
            self.state = LimitState.REVERSING
            reason = f"reversing to opposite edge (~{self.reversal_time_s():.1f} s for a full swing)"
        return LimitDecision(self._goal_pan, tilt, self.state, reason + tilt_note)

    def _reachable(self, target_pan_deg: float, gimbal_pan_deg: float) -> float | None:
        """The pan angle (inside travel) pointing at the target, or None if it is in the dead zone.

        With more than 360 deg of travel several angles point the same way; the closest to the
        current pan is used so the gimbal never takes the long way unnecessarily.
        """
        lim = self.limits
        base = wrap180(target_pan_deg)
        candidates = [base + 360.0 * k for k in range(-2, 3)]
        inside = [c for c in candidates if lim.pan_min_deg <= c <= lim.pan_max_deg]
        if not inside:
            return None
        return min(inside, key=lambda c: abs(c - gimbal_pan_deg))

    def _nearest_edge(self, target_pan_deg: float) -> tuple[float, float]:
        """(edge, degrees past it) for a target in the dead zone."""
        lim = self.limits
        past_max = (wrap180(target_pan_deg) - lim.pan_max_deg) % 360.0
        past_min = (lim.pan_min_deg - wrap180(target_pan_deg)) % 360.0
        return (lim.pan_max_deg, past_max) if past_max <= past_min else (lim.pan_min_deg, past_min)

    def _park(self, edge: float) -> float:
        lim = self.limits
        return edge - lim.margin_deg if edge == lim.pan_max_deg else edge + lim.margin_deg
