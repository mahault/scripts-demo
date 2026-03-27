
from haif_robot_control.haif.arm import Arm
from haif_robot_control.haif.agent_arm import AgentArm as Agent
import haif_robot_control.helpers.utils as utils
import yaml 
import numpy as np
from typing import List
import time

class RobotArm:
    def __init__(self, robot_config, mode):
        self.config = robot_config # Absolute path to config for haif controller
        self.goal_joints = None
        self.goal_ee = None
        self.goal_ort = None
        self.obst_pos = None
        self.ee_pos = None
        self.current_angles = None
        self.mode = mode
        self.reach_threshold = 0.005
        self.timeout_pos_control = 6.0

        if self.mode not in ['pos', 'vel']:
            raise ValueError("Invalid mode. Choose 'pos' or 'vel'.")
        
        # Initialize HAIF controller
        with open(self.config, 'r') as file:
            config = yaml.safe_load(file)
        self.arm_controller = Arm(config)
        self.arm_utils = utils.arm_utils(self.arm_controller)
        self.agent = Agent(self.arm_controller)

    def set_reach_threshold(self, threshold):
        self.reach_threshold = threshold

    def set_current_angles(self, angles):
        self.current_angles = angles

    def set_planning_params(self, nr_iter, dt):
        self.nr_iterations = nr_iter
        self.dt = dt

    def init_angles(self, init_angles):
        self.agent.reset_belief(init_angles)
        self.set_current_angles(init_angles)

    def update_angles(self, angles):
        self.arm_controller.update_with_angles(angles)
    
    def reset_beliefs(self, angles):
        self.agent.reset_belief(angles)

    def set_goal_ee(self, goal):
        self.goal_ee = goal

    def set_obst_pos(self, obst_pos):
        self.obst_pos = obst_pos

    def get_ee_pos(self):
        self.ee_pos = self.arm_utils.get_visual_obs(self.arm_controller)[-1]
        return self.ee_pos

    def get_goal_ee(self):
        return self.goal_ee

    def get_S(self):
        return self.arm_utils.get_joint_obs(self.arm_controller), self.arm_utils.get_visual_obs(self.arm_controller)

    def compute_action(self, S):
        if self.mode == 'pos':
            return self._compute_pos_action(S)
        elif self.mode == 'vel':
            return self._compute_vel_action(S)

    def _compute_vel_action(self, S):
        self.agent.a = np.zeros([self.arm_controller.n_joints]) # Avoid integrating indefinetly
        curr_angles = self.current_angles.copy() # Actual measured angles from the real robot
        for _ in range(self.nr_iterations):
            S = self.get_S()
            action_ai, _ = self.agent.inference_step(S, self.goal_joints, self.goal_ee, self.goal_ort, self.obst_pos)
            # Forward simulate the effect of the action, scaled for speed
            self.ai_angles = curr_angles + self.dt*action_ai
            self.arm_controller.update_with_angles(self.ai_angles)
        return action_ai

    def _compute_pos_action(self, S):
        curr_angles_deg = self.current_angles.copy() # Actual measured angles from the real robot
        joint_traj = []
        dist_to_goal = np.linalg.norm(self.arm_utils.get_visual_obs(self.arm_controller)[-1] - self.goal_ee)
        start_time = time.time()
        timeout = False
        while dist_to_goal > self.reach_threshold and not timeout:
        # for _ in range(100):
            S = self.get_S()
            action_ai, _ = self.agent.inference_step(S, self.goal_joints, self.goal_ee, self.goal_ort, self.obst_pos)
            # Forward simulate the effect of the action, scaled for speed
            curr_angles_deg = curr_angles_deg + 0.3*action_ai
            joint_traj.append(curr_angles_deg)
            self.arm_controller.update_with_angles(curr_angles_deg)
            dist_to_goal = np.linalg.norm(self.arm_utils.get_visual_obs(self.arm_controller)[-1] - self.goal_ee)
            if time.time() - start_time > self.timeout_pos_control: 
                timeout = True
        if timeout:
            print("Planning timed out, check goal is reachable")
            joint_traj = [self.current_angles.copy()]  # Stay still
        else:
            # Filter and execute trajectory, keep a point in the traj 
            # if its norm >= 3deg wrt the previous one
            joint_traj = self.arm_utils.filter_joint_traj(joint_traj, min_deg=3.0)
            print(f"Executing open-loop traj of length {len(joint_traj)}")
        return joint_traj

    def normalize_angle(self, angle: float) -> float:
        return (angle + 180) % 360 - 180

    def calibrate_angle(self, raw_angle: float, center: float, wrap: bool = True) -> float:
        shifted = raw_angle - center
        return self.normalize_angle(shifted) if wrap else shifted

    def calibrate_all(self, raw_angles: List[float], centers: List[float], wrap: bool = True) -> List[float]:
        return [self.calibrate_angle(raw, c, wrap) for raw, c in zip(raw_angles, centers)]

    def read_angles(self):
        raise NotImplementedError("The 'read_angles' method must be implemented.")

    def step(self, **kwargs):
        raise NotImplementedError("The 'step' method must be implemented.")

    def stop(self):
        raise NotImplementedError("The 'stop' method must be implemented.")

    def release_servos(self):
        raise NotImplementedError("The 'release_servos' method must be imppemented")