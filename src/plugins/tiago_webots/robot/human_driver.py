"""Drive a Webots Pedestrian (human) with the same interface the retail
controllers use for :class:`TiagoDriver`.

A Pedestrian has no wheels or physics: it is moved by setting its translation
each tick (a smooth glide) while a walking gait animates the legs/arms.  Because
it is position-controlled it never wedges, so the navigator's recovery/teleport
machinery simply never triggers — the human just walks its route.

Manipulation (pick/place/hand-over) is unchanged: ``TiagoObjectSensors``
teleports items to ``pose + 0.3·(cosθ, sinθ)``, i.e. into the hand, using only
the node pose — which works for a human too.
"""

from __future__ import annotations

import math


# Pedestrian PROTO joint fields and empirical walking-gait tables (from the
# stock Webots `pedestrian` controller).
_JOINTS = [
    "leftArmAngle", "leftLowerArmAngle", "leftHandAngle",
    "rightArmAngle", "rightLowerArmAngle", "rightHandAngle",
    "leftLegAngle", "leftLowerLegAngle", "leftFootAngle",
    "rightLegAngle", "rightLowerLegAngle", "rightFootAngle",
    "headAngle",
]
_HEIGHT_OFFSETS = [-0.02, 0.04, 0.08, -0.03, -0.02, 0.04, 0.08, -0.03]
_ANGLES = [
    [-0.52, -0.15, 0.58, 0.7, 0.52, 0.17, -0.36, -0.74],
    [0.0, -0.16, -0.7, -0.38, -0.47, -0.3, -0.58, -0.21],
    [0.12, 0.0, 0.12, 0.2, 0.0, -0.17, -0.25, 0.0],
    [0.52, 0.17, -0.36, -0.74, -0.52, -0.15, 0.58, 0.7],
    [-0.47, -0.3, -0.58, -0.21, 0.0, -0.16, -0.7, -0.38],
    [0.0, -0.17, -0.25, 0.0, 0.12, 0.0, 0.12, 0.2],
    [-0.55, -0.85, -1.14, -0.7, -0.56, 0.12, 0.24, 0.4],
    [1.4, 1.58, 1.71, 0.49, 0.84, 0.0, 0.14, 0.26],
    [0.07, 0.07, -0.07, -0.36, 0.0, 0.0, 0.32, -0.07],
    [-0.56, 0.12, 0.24, 0.4, -0.55, -0.85, -1.14, -0.7],
    [0.84, 0.0, 0.14, 0.26, 1.4, 1.58, 1.71, 0.49],
    [0.0, 0.0, 0.42, -0.07, 0.07, 0.07, -0.07, -0.36],
    [0.18, 0.09, 0.0, 0.09, 0.18, 0.09, 0.0, 0.09],
]
_N_SEQ = 8
_CYCLE_TO_DISTANCE = 0.22
_ROOT_HEIGHT = 1.27
# Arm indices used to override the gait with a "reaching" pose.
_RIGHT_ARM, _RIGHT_LOWER, _RIGHT_HAND = 3, 4, 5


class HumanDriver:
    """Position-controlled glide-walk driver for a Pedestrian PROTO."""

    # Tells the navigator this actor has no physics and never wedges, so it can
    # skip the furniture keep-out/detour machinery and walk straight to goals
    # (right up to a shelf).  Wheeled robots leave this False.
    is_human = True

    MAX_LINEAR_SPEED = 1.0    # m/s
    MAX_TURN_RATE = 3.2       # rad/s — cap heading change so turns look natural
    PERSONAL_STOP = 0.40      # gap (m) at which we ease to a crawl for an agent
    PERSONAL_SLOW = 0.85      # gap (m) at which we start easing off / steering

    def __init__(self, robot) -> None:
        self.robot = robot
        self.node = robot.getSelf()
        self._trans = self.node.getField("translation")
        self._rot = self.node.getField("rotation")
        self._joints = []
        for n in _JOINTS:
            f = self.node.getField(n)
            self._joints.append(f)
        self._dt = robot.getBasicTimeStep() / 1000.0
        self._gait = 0.0            # accumulated stride distance for the gait
        self._reaching = False
        self._head = 0.0
        t = self._trans.getSFVec3f()
        r = self._rot.getSFRotation()
        self._x, self._y = t[0], t[1]
        self._heading = r[3] if abs(r[2]) > 0.5 else 0.0

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------
    def _animate(self, moved: float) -> float:
        """Advance the walking gait by *moved* metres; return height offset."""
        self._gait += moved
        phase = (self._gait / _CYCLE_TO_DISTANCE) % _N_SEQ
        seq = int(phase)
        ratio = phase - seq
        nxt = (seq + 1) % _N_SEQ
        for i, field in enumerate(self._joints):
            if field is None:
                continue
            ang = _ANGLES[i][seq] * (1 - ratio) + _ANGLES[i][nxt] * ratio
            if i == 12:                       # head: add gaze pan
                ang += self._head
            if self._reaching and i in (_RIGHT_ARM, _RIGHT_LOWER, _RIGHT_HAND):
                ang = {_RIGHT_ARM: 1.3, _RIGHT_LOWER: -0.2, _RIGHT_HAND: 0.0}[i]
            field.setSFFloat(ang)
        return _HEIGHT_OFFSETS[seq] * (1 - ratio) + _HEIGHT_OFFSETS[nxt] * ratio

    def _place(self, x: float, y: float, heading: float, hoff: float) -> None:
        self._x, self._y, self._heading = x, y, heading
        self._trans.setSFVec3f([x, y, _ROOT_HEIGHT + hoff])
        self._rot.setSFRotation([0, 0, 1, heading])

    @staticmethod
    def _wrap(a: float) -> float:
        """Wrap an angle to (-pi, pi]."""
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _rotate_toward(self, cur: float, want: float) -> float:
        """Step *cur* toward *want* by at most one tick of MAX_TURN_RATE."""
        err = self._wrap(want - cur)
        limit = self.MAX_TURN_RATE * self._dt
        if err > limit:
            err = limit
        elif err < -limit:
            err = -limit
        return self._wrap(cur + err)

    def _stand(self) -> None:
        """Neutral standing pose (keeps the reaching arm if extended)."""
        for i, field in enumerate(self._joints):
            if field is None:
                continue
            if self._reaching and i in (_RIGHT_ARM, _RIGHT_LOWER, _RIGHT_HAND):
                field.setSFFloat({_RIGHT_ARM: 1.3, _RIGHT_LOWER: -0.2,
                                  _RIGHT_HAND: 0.0}[i])
            elif i == 12:
                field.setSFFloat(self._head)
            else:
                field.setSFFloat(0.0)

    # ------------------------------------------------------------------
    # Driver interface (mirrors TiagoDriver)
    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stand()

    def face(self, target_x: float, target_y: float) -> None:
        """Turn smoothly to face a point while standing still.

        Used at a shelf/counter so the reaching gesture points at the fixture
        instead of along the last walking direction.
        """
        want = math.atan2(target_y - self._y, target_x - self._x)
        self._heading = self._rotate_toward(self._heading, want)
        self._stand()
        self._rot.setSFRotation([0, 0, 1, self._heading])

    BAND = 0.6                # lateral half-width (m) of the "ahead" corridor
    CRAWL = 0.35              # never brake below this, so we can always slip past

    def _avoid(self, cx, cy, desired, obstacles):
        """Return (steered_heading, brake) to arc around agents ahead.

        Two head-on walkers must not both stop — that deadlocks forever — so we
        never brake fully: we keep a crawl and steer away from whichever side
        the blocker sits on (defaulting right when dead-ahead, like foot
        traffic), letting the pair slip past each other.
        """
        ch, sh = math.cos(desired), math.sin(desired)
        brake = 1.0
        steer = 0.0
        for ox, oy, orad in obstacles:
            odx, ody = ox - cx, oy - cy
            d = math.hypot(odx, ody)
            fwd = odx * ch + ody * sh
            lat = -odx * sh + ody * ch          # signed: +left, -right
            gap = d - orad
            if fwd <= 0 or abs(lat) > self.BAND or gap > self.PERSONAL_SLOW:
                continue
            # Ease off the closer the blocker is, but keep a crawl (CRAWL) so
            # forward motion never reaches zero.
            frac = (gap - self.PERSONAL_STOP) / (self.PERSONAL_SLOW
                                                 - self.PERSONAL_STOP)
            brake = min(brake, max(self.CRAWL, frac))
            # Steer away from the blocker's side; dead-ahead defaults right.
            side = -1.0 if lat >= 0 else 1.0
            closeness = max(0.0, 1.0 - gap / self.PERSONAL_SLOW)
            head_on = max(0.1, 1.0 - abs(lat) / self.BAND)
            steer += side * (0.5 + 1.1 * closeness) * head_on
        steer = max(-0.9, min(0.9, steer))      # cap so we never spin in place
        return self._wrap(desired + steer), brake

    def navigate_to_target(self, current_x, current_y, current_heading,
                           target_x, target_y, speed_scale=1.0,
                           obstacles=None) -> bool:
        # Use the driver's own tracked pose as the authority so a separation
        # nudge applied earlier this tick is respected (the passed pose is the
        # pre-separation reading from perception).
        current_x, current_y = self._x, self._y
        dx, dy = target_x - current_x, target_y - current_y
        dist = math.hypot(dx, dy)
        if dist < 0.15:
            self.stop()
            return True

        desired = math.atan2(dy, dx)
        brake = 1.0
        if obstacles:
            desired, brake = self._avoid(current_x, current_y, desired, obstacles)

        # Turn-rate-limited heading: walk along where we are actually facing.
        self._heading = self._rotate_toward(self._heading, desired)

        # Ease forward speed when still swinging around to face the goal, so a
        # turn reads as a turn instead of a sideways skate.
        herr = abs(self._wrap(desired - self._heading))
        turn_factor = max(0.2, 1.0 - herr / (math.pi / 2))

        step = (self.MAX_LINEAR_SPEED * max(0.0, min(1.0, speed_scale))
                * self._dt * brake * turn_factor)
        move = min(step, dist)
        nx = current_x + move * math.cos(self._heading)
        ny = current_y + move * math.sin(self._heading)
        hoff = self._animate(move)
        self._place(nx, ny, self._heading, hoff)
        return dist < 0.2

    def creep(self, linear: float, angular: float = 0.0) -> None:
        # Humans don't wedge, but keep the interface: back up slowly.
        nx = self._x + linear * self._dt * math.cos(self._heading)
        ny = self._y + linear * self._dt * math.sin(self._heading)
        hoff = self._animate(abs(linear) * self._dt)
        self._place(nx, ny, self._heading + angular * self._dt, hoff)

    PERSONAL_BUBBLE = 0.80    # m — keep at least this much clear space to others
    SEP_SPEED = 0.9           # m/s max separation push when fully overlapping

    def apply_separation(self, others) -> None:
        """Always-on personal-space repulsion so people never overlap.

        Runs every tick regardless of what the actor is doing (walking, facing a
        shelf, standing).  For each other agent inside the personal bubble, push
        directly away, scaled by how deep the overlap is.  This is what stops the
        "pushing into each other" — reactive nav avoidance alone doesn't help two
        agents that have already converged or are standing still.
        """
        px = py = 0.0
        for ox, oy in others:
            dx, dy = self._x - ox, self._y - oy
            d = math.hypot(dx, dy)
            if d < 1e-4:
                # Exactly coincident: shove in an arbitrary but stable direction.
                px += 0.3
                continue
            if d < self.PERSONAL_BUBBLE:
                f = (self.PERSONAL_BUBBLE - d) / self.PERSONAL_BUBBLE
                px += (dx / d) * f
                py += (dy / d) * f
        mag = math.hypot(px, py)
        if mag < 1e-4:
            return
        step = min(self.SEP_SPEED * self._dt, mag * self.SEP_SPEED * self._dt + 0.002)
        nx = self._x + (px / mag) * step
        ny = self._y + (py / mag) * step
        cur = self._trans.getSFVec3f()
        self._x, self._y = nx, ny
        self._trans.setSFVec3f([nx, ny, cur[2]])

    def yield_aside(self, other_x: float, other_y: float,
                    speed: float = 0.45) -> None:
        """Step sideways out of an approaching agent's path (don't reverse).

        A person gives way by stepping to the side and turning to face the
        passer, not by walking backwards.  We strafe perpendicular to the line
        to the other agent, picking the side that opens the bigger gap.
        """
        b = math.atan2(other_y - self._y, other_x - self._x)
        # Two perpendicular options; choose the one that increases separation.
        best = None
        for side in (b + math.pi / 2.0, b - math.pi / 2.0):
            tx = self._x + math.cos(side) * speed * self._dt
            ty = self._y + math.sin(side) * speed * self._dt
            gap = math.hypot(tx - other_x, ty - other_y)
            if best is None or gap > best[0]:
                best = (gap, tx, ty)
        _, nx, ny = best
        hoff = self._animate(speed * self._dt)
        # Face the passer while stepping aside (acknowledgement).
        self._place(nx, ny, b, hoff)

    # Gestures / no-ops to match the TiagoDriver surface.
    def extend_arm(self, *_a, **_k) -> None:
        self._reaching = True

    def retract_arm(self, *_a, **_k) -> None:
        self._reaching = False

    def set_head_pan_tilt(self, pan: float, tilt: float = 0.0) -> None:
        self._head = max(-0.6, min(0.6, pan))

    def open_gripper(self) -> None:
        pass

    def close_gripper(self) -> None:
        pass
