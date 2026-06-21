"""TIAGo differential-drive motor control.

Extracted from tiago_empathic.py (lines 94-321) into a reusable driver
that the social-layer navigation skill delegates to.
"""

from __future__ import annotations

import math


class TiagoDriver:
    """Low-level motor interface for the TIAGo differential-drive base."""

    # Physical constants
    MAX_SPEED = 10.0        # rad/s (motor limit)
    WHEEL_RADIUS = 0.0985   # metres
    WHEEL_BASE = 0.4044     # metres

    # Control gains
    KP_HEADING = 2.0
    MAX_ANGULAR_SPEED = 2.0
    MAX_LINEAR_SPEED = 1.0  # m/s

    def __init__(self, robot) -> None:
        self.robot = robot
        self._init_motors()
        self._tuck_arm()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_motors(self) -> None:
        self.left_motor = self.robot.getDevice("wheel_left_joint")
        self.right_motor = self.robot.getDevice("wheel_right_joint")

        if not self.left_motor or not self.right_motor:
            raise RuntimeError("Wheel motors not found on TIAGo robot")

        # Velocity-control mode
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def _tuck_arm(self) -> None:
        """Move the arm close to the body so it does not collide.

        Standard TIAGo transport pose.  It is stable, keeps the arm off the
        floor, and folds the forearm across the chest.  Navigation must keep
        furniture far enough away that the folded arm does not snag.
        """
        arm_positions = {
            "arm_1_joint": 0.07,
            "arm_2_joint": 1.02,
            "arm_3_joint": -3.16,
            "arm_4_joint": 2.02,
            "arm_5_joint": 1.32,
            "arm_6_joint": 0.0,
            "arm_7_joint": 1.41,
        }
        for name, pos in arm_positions.items():
            joint = self.robot.getDevice(name)
            if joint:
                joint.setPosition(pos)

    # ------------------------------------------------------------------
    # Motor primitives
    # ------------------------------------------------------------------
    def stop(self) -> None:
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def creep(self, linear: float, angular: float = 0.0) -> None:
        """Open-loop differential drive, used for short social maneuvers.

        Unlike :meth:`navigate_to_target`, this permits reverse motion
        (negative ``linear``) so a robot can back out of a face-to-face
        encounter to yield — the potential-field navigator deliberately
        never reverses, which is what wedges two robots together.
        """
        v_left = (linear - angular * self.WHEEL_BASE / 2.0) / self.WHEEL_RADIUS
        v_right = (linear + angular * self.WHEEL_BASE / 2.0) / self.WHEEL_RADIUS
        v_left = max(-self.MAX_SPEED, min(self.MAX_SPEED, v_left))
        v_right = max(-self.MAX_SPEED, min(self.MAX_SPEED, v_right))
        self.left_motor.setVelocity(v_left)
        self.right_motor.setVelocity(v_right)

    # ------------------------------------------------------------------
    # Arm control
    # ------------------------------------------------------------------
    def extend_arm(self, target: tuple = (0.5, 0.0, 0.8)) -> None:
        """Move arm to extended handover/pick position."""
        extend_positions = {
            "arm_1_joint": 0.20,
            "arm_2_joint": -1.34,
            "arm_3_joint": -0.20,
            "arm_4_joint": 1.94,
            "arm_5_joint": -1.57,
            "arm_6_joint": 1.37,
            "arm_7_joint": 0.0,
        }
        for name, pos in extend_positions.items():
            joint = self.robot.getDevice(name)
            if joint:
                joint.setPosition(pos)

    def retract_arm(self) -> None:
        """Retract arm to tucked position."""
        self._tuck_arm()

    def open_gripper(self) -> None:
        """Open the parallel gripper."""
        for finger in ("gripper_left_finger_joint", "gripper_right_finger_joint"):
            joint = self.robot.getDevice(finger)
            if joint:
                joint.setPosition(0.04)

    def close_gripper(self) -> None:
        """Close the parallel gripper."""
        for finger in ("gripper_left_finger_joint", "gripper_right_finger_joint"):
            joint = self.robot.getDevice(finger)
            if joint:
                joint.setPosition(0.0)

    # ------------------------------------------------------------------
    # Head control
    # ------------------------------------------------------------------
    def set_head_pan_tilt(self, pan: float, tilt: float) -> None:
        """Set head orientation for gaze control."""
        head_pan = self.robot.getDevice("head_1_joint")
        head_tilt = self.robot.getDevice("head_2_joint")
        if head_pan:
            head_pan.setPosition(pan)
        if head_tilt:
            head_tilt.setPosition(tilt)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    # Potential-field obstacle avoidance
    AVOID_DIST = 0.8          # activation distance (metres)
    K_REP = 1.2               # repulsive force gain
    MAX_REP = 0.8             # max repulsive speed (m/s)
    K_ATT = 2.0               # attractive force gain (normalised)

    def navigate_to_target(
        self,
        current_x: float,
        current_y: float,
        current_heading: float,
        target_x: float,
        target_y: float,
        speed_scale: float = 1.0,
        obstacles: list | None = None,
    ) -> bool:
        """Drive toward *target* using a potential-field controller.

        The goal exerts an attractive force and obstacles exert repulsive
        forces.  The robot follows the combined force vector, which naturally
        steers it around furniture instead of getting trapped in the
        "head-toward-goal vs steer-away-from-wall" dead-lock of the old
        two-phase HAIF controller.  Each obstacle is (x, y, radius).

        Returns ``True`` when within 0.15 m of the target.
        """
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 0.15:
            self.stop()
            return True

        # Attractive force toward the goal (unit magnitude, scaled by distance)
        if distance > 0.01:
            att_x = self.K_ATT * dx / distance
            att_y = self.K_ATT * dy / distance
        else:
            att_x, att_y = 0.0, 0.0

        # Repulsive forces from inflated obstacles
        rep_x, rep_y = 0.0, 0.0
        if obstacles:
            for ox, oy, orad in obstacles:
                odx = current_x - ox
                ody = current_y - oy
                odist = math.sqrt(odx * odx + ody * ody)
                clearance = odist - orad
                if clearance < self.AVOID_DIST and clearance > 0.02:
                    mag = self.K_REP * (1.0 / clearance - 1.0 / self.AVOID_DIST)
                    mag = min(mag, self.MAX_REP)
                    rep_x += mag * odx / odist
                    rep_y += mag * ody / odist

        # Combined force vector determines desired heading
        force_x = att_x + rep_x
        force_y = att_y + rep_y
        force_mag = math.sqrt(force_x * force_x + force_y * force_y)

        if force_mag < 0.01:
            # Stalemate: no clear direction.  Turn in place until a gradient
            # appears (usually caused by the goal attraction as the robot
            # rotates).
            self.left_motor.setVelocity(-0.5)
            self.right_motor.setVelocity(0.5)
            return False

        desired_heading = math.atan2(force_y, force_x)
        heading_error = _normalize_angle(desired_heading - current_heading)

        # Never drive backward: if the force points behind us, turn in place
        # until the goal is in front.  Backward motion is unstable in Webots
        # and easily wedges the robot against furniture.
        if abs(heading_error) > math.pi / 2:
            linear = 0.0
            angular = self.MAX_ANGULAR_SPEED if heading_error > 0 else -self.MAX_ANGULAR_SPEED
        else:
            angular = self.KP_HEADING * heading_error
            angular = max(-self.MAX_ANGULAR_SPEED, min(self.MAX_ANGULAR_SPEED, angular))

            # Drive forward proportionally to how well we are aligned with the
            # force direction.  Keep a minimum creep speed so the robot can still
            # slide along obstacle boundaries.
            alignment = max(0.25, math.cos(heading_error))
            linear = self.MAX_LINEAR_SPEED * alignment
            if distance < 0.5:
                linear *= distance / 0.5

            # Apply speed_scale from IntentPolicy / SafetyShield
            linear *= max(0.0, min(1.0, speed_scale))

        # Extra caution: slow down when an obstacle is directly ahead
        if obstacles:
            cos_h = math.cos(current_heading)
            sin_h = math.sin(current_heading)
            for ox, oy, orad in obstacles:
                odx = ox - current_x
                ody = oy - current_y
                odist = math.sqrt(odx * odx + ody * ody)
                # Projection of obstacle onto forward axis
                forward_proj = odx * cos_h + ody * sin_h
                lateral_proj = abs(-odx * sin_h + ody * cos_h)
                clearance = odist - orad
                if 0 < forward_proj < 0.8 and lateral_proj < 0.4 and clearance < 0.5:
                    linear *= max(0.1, clearance / 0.5)

        # Differential-drive kinematics
        v_left = (linear - angular * self.WHEEL_BASE / 2.0) / self.WHEEL_RADIUS
        v_right = (linear + angular * self.WHEEL_BASE / 2.0) / self.WHEEL_RADIUS

        v_left = max(-self.MAX_SPEED, min(self.MAX_SPEED, v_left))
        v_right = max(-self.MAX_SPEED, min(self.MAX_SPEED, v_right))

        # DEBUG: print commanded wheel velocities (throttled to avoid I/O lag)
        if (abs(linear) > 0.01 or abs(angular) > 0.01):
            self._debug_tick = getattr(self, "_debug_tick", 0) + 1
            if self._debug_tick % 10 == 0:
                print(f"  [DRIVER] linear={linear:.3f} angular={angular:.3f} "
                      f"force=({force_x:.2f},{force_y:.2f}) "
                      f"v_left={v_left:.2f} v_right={v_right:.2f}")

        self.left_motor.setVelocity(v_left)
        self.right_motor.setVelocity(v_right)
        return False


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle
