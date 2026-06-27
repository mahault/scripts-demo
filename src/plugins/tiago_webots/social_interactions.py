"""Social interaction primitives for the retail script-learning demo.

Turns worker<->customer encounters into structured, *observable* social acts
(yield, greet, hand-over, wayfinding) instead of physical collisions.  Both the
deadlock fix and the learnable social script live here:

* The worker (store staff) gives way to the customer (shopper) — staff yield to
  shoppers.  The yield is performed with :meth:`TiagoDriver.creep`, which is
  allowed to reverse, so the two robots can break a face-to-face wedge that the
  potential-field navigator (which never reverses) cannot.

* Each social act is emitted as a discrete ``action`` label in the worker's
  ``customData``.  The learner's ``TeacherObserver`` segments on those labels,
  so the acts become primitives in the crystallised social script.

The manager is deliberately self-contained: ``WorkerEncounterManager.update``
takes over the base/arm/head while an interaction is active and tells the
restock FSM to pause.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


# Right-of-way.  Higher priority is *not* expected to yield; lower priority
# gives way.  Store staff (workers) yield to shoppers (customers).
PRIORITY = {"Worker_T": 0, "Learner_L": 0, "Customer_1": 1, "Customer_2": 1}

# Product -> (shelf label, approach point) map used to answer wayfinding
# queries ("where's the milk?").  Aligned with the shelves in the world file.
PRODUCT_SHELF = {
    "milk":   ("shelf_A", (-2.0, -5.0)),
    "apple":  ("shelf_A", (-2.0, -5.0)),
    "bread":  ("shelf_A", (-2.0, -5.0)),
    "orange": ("shelf_B", (1.5, -5.0)),
    "can":    ("shelf_B", (1.5, -5.0)),
    "soda":   ("shelf_B", (1.5, -5.0)),
}


def _norm_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def agent_pose(pb, agent_id: str) -> Optional[Tuple[float, float]]:
    """Return the (x, y) pose of a named agent from perception, or None."""
    for a in pb.world.get("agents", []):
        if a.get("id") == agent_id:
            p = a.get("pose")
            if p is not None:
                return (p[0], p[1])
    return None


def relative_bearing(self_pose, other_xy) -> float:
    """Bearing of *other* relative to self heading, in [-pi, pi]."""
    cx, cy = self_pose[0], self_pose[1]
    heading = self_pose[3] if len(self_pose) > 3 else 0.0
    return _norm_angle(math.atan2(other_xy[1] - cy, other_xy[0] - cx) - heading)


def in_forward_cone(self_pose, other_xy,
                    half_angle: float = 1.0, max_range: float = 2.2) -> bool:
    """True if *other* is within range and roughly ahead of self."""
    cx, cy = self_pose[0], self_pose[1]
    d = math.hypot(other_xy[0] - cx, other_xy[1] - cy)
    if d > max_range or d < 1e-3:
        return False
    return abs(relative_bearing(self_pose, other_xy)) < half_angle


# Action labels emitted to customData (the learner's segmentation vocabulary).
ACT_YIELD = "yield"
ACT_GREET = "greet"


@dataclass
class EncounterResult:
    active: bool          # True while the manager owns the robot
    action: str           # action label to publish (when active)
    just_finished: bool   # True on the tick the interaction completes


class WorkerEncounterManager:
    """Drives the worker's give-way + greet response to an approaching shopper.

    State machine (only runs when a customer is ahead and close):

        IDLE -> BACK_OFF -> GREET_WAIT -> (resume) -> IDLE

    BACK_OFF    reverse a short distance to open a gap and break any contact,
                while turning the head toward the shopper.
    GREET_WAIT  hold position, raise a hand and keep gaze on the shopper
                (acknowledgement), until the shopper has clearly passed.
    """

    # Tunables (metres unless noted).
    ENGAGE_RANGE = 1.6        # start yielding when shopper is this close & ahead
    CLEAR_RANGE = 1.7         # shopper considered "passing" beyond this
    RESUME_RANGE = 2.0        # resume the task once shopper is this far
    MAX_BACKOFF = 0.7         # cap on how far we reverse
    BACKOFF_SPEED = 0.3       # m/s reverse creep
    MAX_GREET_TICKS = 200     # safety cap so a lingering shopper can't hang us

    def __init__(self, name: str, customer_id: str = "Customer_1") -> None:
        self.name = name
        self.customer_id = customer_id
        self.state = "IDLE"
        self._backoff_origin: Optional[Tuple[float, float]] = None
        self._prev_dist: float = 1e9
        self._greet_ticks: int = 0

    def _customer_xy(self, pb):
        return agent_pose(pb, self.customer_id)

    def update(self, pb, pose, driver) -> EncounterResult:
        cust = self._customer_xy(pb)
        cx, cy = pose[0], pose[1]

        if cust is None:
            if self.state != "IDLE":
                return self._finish(driver)
            return EncounterResult(False, "", False)

        dist = math.hypot(cust[0] - cx, cust[1] - cy)

        if self.state == "IDLE":
            # Only yield to a shopper that is ahead of us and approaching.
            if dist < self.ENGAGE_RANGE and in_forward_cone(pose, cust):
                self.state = "BACK_OFF"
                self._backoff_origin = (cx, cy)
                self._prev_dist = dist
                print(f"{self.name}: YIELD — shopper ahead ({dist:.2f} m), giving way")
            else:
                return EncounterResult(False, "", False)

        # --- gaze stays on the shopper throughout the interaction ---
        bearing = relative_bearing(pose, cust)
        driver.set_head_pan_tilt(max(-1.3, min(1.3, bearing)), 0.0)

        if self.state == "BACK_OFF":
            backed = (math.hypot(cx - self._backoff_origin[0],
                                 cy - self._backoff_origin[1])
                      if self._backoff_origin else 0.0)
            # Step aside until we have opened a gap or hit the backoff cap
            # (humans strafe; the wheeled base falls back to reversing).
            if dist < self.CLEAR_RANGE and backed < self.MAX_BACKOFF:
                driver.yield_aside(cust[0], cust[1], self.BACKOFF_SPEED)
                return EncounterResult(True, ACT_YIELD, False)
            # Gap opened (or cannot back up further): hold and acknowledge.
            driver.stop()
            driver.extend_arm()          # raise hand — acknowledgement gesture
            self.state = "GREET_WAIT"
            self._greet_ticks = 0
            print(f"{self.name}: GREET — acknowledging shopper")
            return EncounterResult(True, ACT_GREET, False)

        if self.state == "GREET_WAIT":
            driver.stop()
            self._greet_ticks += 1
            # Resume once the shopper has cleared our path — either it moved
            # out of the way ahead (passed to the side / behind), got far away,
            # or it is simply lingering (browsing) past a safety cap.  We do not
            # require it to keep receding, so a shopper that stops to shop does
            # not pin us in place forever.
            passed = dist > self.CLEAR_RANGE and not in_forward_cone(pose, cust)
            if (passed or dist > self.RESUME_RANGE
                    or self._greet_ticks > self.MAX_GREET_TICKS):
                return self._finish(driver)
            return EncounterResult(True, ACT_GREET, False)

        return EncounterResult(False, "", False)

    def _finish(self, driver) -> EncounterResult:
        driver.retract_arm()
        driver.set_head_pan_tilt(0.0, 0.0)
        self.state = "IDLE"
        self._backoff_origin = None
        self._prev_dist = 1e9
        print(f"{self.name}: RESUME — shopper passed, back to restock")
        return EncounterResult(False, "", True)


class StallRecovery:
    """Breaks a wedged base — commanding motion but not actually moving.

    The potential-field navigator never reverses (by design, to avoid backing
    into things), so once a robot high-centres on a shelf edge or corner it
    spins its wheels forever.  This watches for lack of progress while
    navigating and, when detected, drives a short reverse-and-turn escape with
    :meth:`TiagoDriver.creep` (which *is* allowed to reverse) before handing
    control back to the navigator.
    """

    STALL_TICKS = 120       # ~2 s of no progress while navigating
    MOVE_EPS = 0.10         # metres that count as "made progress"
    RECOVER_TICKS = 45      # duration of the escape maneuver
    REVERSE_SPEED = 0.3     # m/s
    TURN_SPEED = 1.0        # rad/s (turn while reversing to change approach)
    PROGRESS_EPS = 0.5      # real escape progress between recoveries (m)
    MAX_CONSEC = 6          # consider the goal unreachable after this many fails

    def __init__(self) -> None:
        self._ref: Optional[Tuple[float, float]] = None
        self._stall = 0
        self._recover = 0
        self._consec = 0                       # escapes without real progress
        self._progress_ref: Optional[Tuple[float, float]] = None

    @property
    def recovering(self) -> bool:
        return self._recover > 0

    @property
    def stuck(self) -> bool:
        """True when repeated escapes from the same spot have failed.  Only the
        learner acts on this (abandons the goal); the worker keeps trying, since
        its waypoints are reachable and it must hit each one in order."""
        return self._consec >= self.MAX_CONSEC

    def update(self, pose) -> bool:
        """Call once per navigating tick.  Returns True the moment a stall is
        detected (caller should then drive the recovery via :meth:`step`)."""
        xy = (pose[0], pose[1])
        if self._ref is None or math.hypot(xy[0] - self._ref[0],
                                           xy[1] - self._ref[1]) > self.MOVE_EPS:
            self._ref = xy
            self._stall = 0
            return False
        self._stall += 1
        if self._stall > self.STALL_TICKS:
            # Reset the failure counter if we have escaped meaningfully since
            # the previous stall (a fresh, unrelated wedge); otherwise we are
            # stuck in the same spot and the count climbs toward "unreachable".
            if (self._progress_ref is None
                    or math.hypot(xy[0] - self._progress_ref[0],
                                  xy[1] - self._progress_ref[1]) > self.PROGRESS_EPS):
                self._consec = 0
            self._progress_ref = xy
            self._consec += 1
            self._recover = self.RECOVER_TICKS
            self._stall = 0
            return True
        return False

    def step(self, driver) -> bool:
        """Drive one tick of the reverse-and-turn escape.  Returns True while
        the maneuver is still running, False on the tick it completes."""
        driver.creep(-self.REVERSE_SPEED, self.TURN_SPEED)
        self._recover -= 1
        if self._recover <= 0:
            self._ref = None
            return False
        return True

    def reset(self) -> None:
        self._ref = None
        self._stall = 0
        self._recover = 0
        self._consec = 0
        self._progress_ref = None
