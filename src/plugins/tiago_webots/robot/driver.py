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
        """Move the arm close to the body so it does not collide."""
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
    # Obstacle avoidance (HAIF-style repulsive forces)
    AVOID_DIST = 0.5          # activation distance (metres)
    K_REP = 1.0               # repulsive force gain
    MAX_REP = 0.5             # max repulsive speed contribution

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
        """Drive toward *target* using two-phase (rotate-then-translate)
        differential control with reactive obstacle avoidance.

        Obstacle avoidance uses HAIF-style repulsive forces: when the
        robot is within AVOID_DIST of an obstacle, a repulsive velocity
        component steers it away.  Each obstacle is (x, y, radius).

        Returns ``True`` when within 0.15 m of the target.
        """
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 0.15:
            self.stop()
            return True

        desired_heading = math.atan2(dy, dx)
        heading_error = _normalize_angle(desired_heading - current_heading)

        # Reverse if target is behind us
        drive_backward = abs(heading_error) > math.pi / 2
        if drive_backward:
            heading_error = _normalize_angle(heading_error + math.pi)

        angular = self.KP_HEADING * heading_error
        angular = max(-self.MAX_ANGULAR_SPEED, min(self.MAX_ANGULAR_SPEED, angular))

        # Phase 1: pure rotation when misaligned
        if abs(heading_error) > math.radians(15):
            linear = 0.0
        else:
            alignment = max(0.2, math.cos(heading_error))
            linear = self.MAX_LINEAR_SPEED * alignment
            if distance < 0.5:
                linear *= distance / 0.5

        # Apply speed_scale from IntentPolicy / SafetyShield
        linear *= max(0.0, min(1.0, speed_scale))

        if drive_backward:
            linear = -linear

        # --- Obstacle avoidance: repulsive forces (HAIF-style) ---
        rep_x, rep_y = 0.0, 0.0
        if obstacles:
            for ox, oy, orad in obstacles:
                odx = current_x - ox
                ody = current_y - oy
                odist = math.sqrt(odx * odx + ody * ody)
                clearance = odist - orad
                if clearance < self.AVOID_DIST and clearance > 0.01:
                    # Repulsive magnitude: inverse-square, capped
                    mag = self.K_REP * (1.0 / clearance - 1.0 / self.AVOID_DIST)
                    mag = min(mag, self.MAX_REP)
                    # Direction: away from obstacle
                    rep_x += mag * odx / odist
                    rep_y += mag * ody / odist

        if abs(rep_x) > 0.01 or abs(rep_y) > 0.01:
            # Project repulsive force onto robot's lateral axis
            # to generate angular correction (steer away)
            cos_h = math.cos(current_heading)
            sin_h = math.sin(current_heading)
            # Lateral component (perpendicular to heading)
            rep_lateral = -rep_x * sin_h + rep_y * cos_h
            # Longitudinal component (along heading)
            rep_longitudinal = rep_x * cos_h + rep_y * sin_h

            angular += rep_lateral * 3.0
            angular = max(-self.MAX_ANGULAR_SPEED, min(self.MAX_ANGULAR_SPEED, angular))

            # Slow down when pushing against an obstacle
            if rep_longitudinal < -0.1:
                linear *= max(0.1, 1.0 + rep_longitudinal)

        # Differential-drive kinematics
        v_left = (linear - angular * self.WHEEL_BASE / 2.0) / self.WHEEL_RADIUS
        v_right = (linear + angular * self.WHEEL_BASE / 2.0) / self.WHEEL_RADIUS

        v_left = max(-self.MAX_SPEED, min(self.MAX_SPEED, v_left))
        v_right = max(-self.MAX_SPEED, min(self.MAX_SPEED, v_right))

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
