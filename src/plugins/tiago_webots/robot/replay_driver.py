"""Drive a Webots Pedestrian by replaying a recorded human trajectory.

The recorded path (extracted from the UF-Retail dataset, mapped into the store)
is the authority: each tick we interpolate the real ``(x, y, heading, reach)``
at the current playback time and place the actor there, animating the walking
gait from the real velocity and raising the arm exactly when the real person
reached.  Navigation/steering calls are no-ops — there is no synthetic motion to
look wrong, because the motion is real.
"""

from __future__ import annotations

import json
import math

from plugins.tiago_webots.robot.human_driver import (
    _JOINTS, _ANGLES, _HEIGHT_OFFSETS, _N_SEQ, _CYCLE_TO_DISTANCE, _ROOT_HEIGHT,
    _RIGHT_ARM, _RIGHT_LOWER, _RIGHT_HAND,
)


class ReplayDriver:
    """Position-controlled driver that plays a recorded trajectory."""

    is_human = True

    def __init__(self, robot, traj_path, loop=True, speed=1.0) -> None:
        self.robot = robot
        self.node = robot.getSelf()
        self._trans = self.node.getField("translation")
        self._rot = self.node.getField("rotation")
        self._joints = [self.node.getField(n) for n in _JOINTS]
        self._dt = robot.getBasicTimeStep() / 1000.0

        with open(traj_path) as f:
            data = json.load(f)
        self._frames = data["frames"]
        self._fdt = data["dt"]
        self._n = len(self._frames)
        self._loop = loop
        self._speed = speed

        self._t0 = None
        self._last_update_t = -1.0
        self._gait = 0.0
        self._reach = 0
        f0 = self._frames[0]
        self._last_xy = (f0[0], f0[1])
        self._place(f0[0], f0[1], f0[2], 0.0)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------
    @staticmethod
    def _lerp_angle(a, b, t):
        d = (b - a + math.pi) % (2 * math.pi) - math.pi
        return a + d * t

    def _frame_at(self, pt):
        idx = pt / self._fdt
        if self._loop:
            idx = idx % self._n
        else:
            idx = max(0.0, min(idx, self._n - 1))
        i0 = int(idx)
        i1 = (i0 + 1) % self._n if self._loop else min(i0 + 1, self._n - 1)
        a = idx - i0
        f0, f1 = self._frames[i0], self._frames[i1]
        x = f0[0] * (1 - a) + f1[0] * a
        y = f0[1] * (1 - a) + f1[1] * a
        h = self._lerp_angle(f0[2], f1[2], a)
        return x, y, h, f0[3]

    def update(self) -> None:
        """Advance playback to the current sim time (idempotent per tick)."""
        now = self.robot.getTime()
        if now == self._last_update_t:
            return
        self._last_update_t = now
        if self._t0 is None:
            self._t0 = now
        x, y, h, reach = self._frame_at((now - self._t0) * self._speed)
        self._reach = reach
        moved = math.hypot(x - self._last_xy[0], y - self._last_xy[1])
        hoff = self._animate(moved)
        self._place(x, y, h, hoff)
        self._last_xy = (x, y)

    # ------------------------------------------------------------------
    # Pose helpers (mirror HumanDriver)
    # ------------------------------------------------------------------
    def _animate(self, moved: float) -> float:
        self._gait += moved
        phase = (self._gait / _CYCLE_TO_DISTANCE) % _N_SEQ
        seq = int(phase)
        ratio = phase - seq
        nxt = (seq + 1) % _N_SEQ
        for i, field in enumerate(self._joints):
            if field is None:
                continue
            ang = _ANGLES[i][seq] * (1 - ratio) + _ANGLES[i][nxt] * ratio
            if self._reach and i in (_RIGHT_ARM, _RIGHT_LOWER, _RIGHT_HAND):
                ang = {_RIGHT_ARM: 1.35, _RIGHT_LOWER: -0.2, _RIGHT_HAND: 0.0}[i]
            field.setSFFloat(ang)
        return _HEIGHT_OFFSETS[seq] * (1 - ratio) + _HEIGHT_OFFSETS[nxt] * ratio

    def _place(self, x, y, heading, hoff) -> None:
        self._trans.setSFVec3f([x, y, _ROOT_HEIGHT + hoff])
        self._rot.setSFRotation([0, 0, 1, heading])

    # ------------------------------------------------------------------
    # Observable state for the controller
    # ------------------------------------------------------------------
    @property
    def position(self):
        return self._last_xy

    @property
    def reaching(self) -> bool:
        return bool(self._reach)

    # ------------------------------------------------------------------
    # Driver interface — every motion call just advances the replay.
    # ------------------------------------------------------------------
    def navigate_to_target(self, *a, **k) -> bool:
        self.update()
        return False

    def face(self, *a, **k) -> None:
        self.update()

    def stop(self) -> None:
        self.update()

    def creep(self, *a, **k) -> None:
        self.update()

    def yield_aside(self, *a, **k) -> None:
        self.update()

    def apply_separation(self, others) -> None:
        return  # the recorded trajectory is the authority

    def extend_arm(self, *a, **k) -> None:
        return  # arm is driven by the recorded reach flag

    def retract_arm(self, *a, **k) -> None:
        return

    def set_head_pan_tilt(self, *a, **k) -> None:
        return

    def open_gripper(self) -> None:
        return

    def close_gripper(self) -> None:
        return
