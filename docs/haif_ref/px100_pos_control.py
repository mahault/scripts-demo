
import time

from haif_robot_control.haif.arm import Arm
from haif_robot_control.haif.agent_arm import AgentArm as Agent
import haif_robot_control.helpers.utils as utils
import yaml
import os
import numpy as np

from robot_driver.arbotix import ArbotiX

port = '/dev/ttyUSB0'
baudrate = 115200 

ZERO_OFFSET = np.array([150.8, 154.57, 155.15, 145.58, 147.9])

def normalize_angle(angle: float) -> float:
    return (angle + 180) % 360 - 180

def calibrate_angle(raw_angle: float, center: float, wrap: bool = True) -> float:
    shifted = raw_angle - center
    return normalize_angle(shifted) if wrap else shifted

def calibrate_all(raw_angles, centers, wrap = True):
    return [calibrate_angle(raw, c, wrap) for raw, c in zip(raw_angles, centers)]

def read_angles():
    angles = [arbotix.getPosition(i) for i in range(1, 5)]
    return angles

def calibrate_angles(angles):
    angles_deg = [a * 0.29 for a in angles]
    calibrated_angles = calibrate_all(angles_deg, ZERO_OFFSET)
    calibrated_angles[-2:] = -np.array(calibrated_angles[-2:])  # Invert last two joints to match real px100
    return calibrated_angles

def angle2steps(angles):
    step_angles = []
    for i in range(1, 5):
        angle = angles[i-1]
        angle += ZERO_OFFSET[i-1]
        angle /= 0.29
        step_angles.append(int(angle))
    return step_angles

def send_angles(step_angles):
    for i in range(1, 5):
        arbotix.setPosition(i, int(step_angles[i-1]))

# Open real arm connection
arbotix = ArbotiX(port, baudrate, 1, open_port=True)

# Set arm speed
for i in range(1, 6):
    arbotix.setSpeed(i, 50)

time.sleep(0.5)  # Allow time for the arm to initialize

angles_steps = read_angles()

# Disable Torque for debugging
#for i in range(1, 6):
#    arbotix.disableTorque(i)

# Arm controller 
ROBOT_CONFIG = "px100.yaml"
config_file = os.path.dirname(os.path.abspath(__file__)) + "/../../configs/" + ROBOT_CONFIG
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)
arm = Arm(config)
arm_utils = utils.arm_utils(arm)
agent = Agent(arm)
# Initialize agent
curr_angles_deg = calibrate_angles(angles_steps)
agent.reset_belief(curr_angles_deg)

goal_pos = [0.2, -0.1,  0.1] 
goal_joints = None # [-45, 0, 0, 0]
obst_pos = None
goal_ort = None

print("Angles:", curr_angles_deg)
print("Step angles:", angles_steps)

# Timeouts
joint_traj = []
dist_to_goal = 1000
start_time = time.time()
timeout = False

try:
    # Compute the whole trajectory up to 5mm, then filter it and execute it open loop
    while dist_to_goal > 0.005 and not timeout:
    # for _ in range(100):
        S = arm_utils.get_joint_obs(arm), arm_utils.get_visual_obs(arm)
        action_ai, _ = agent.inference_step(S, goal_joints, goal_pos, goal_ort, obst_pos)
        # Forward simulate the effect of the action, scaled for speed
        curr_angles_deg = curr_angles_deg + 0.3*action_ai
        joint_traj.append(curr_angles_deg)
        arm.update_with_angles(curr_angles_deg)
        dist_to_goal = np.linalg.norm(arm_utils.get_visual_obs(arm)[-1] - goal_pos)
        if time.time() - start_time > 6.0:  # Timeout after 6 second
            timeout = True
    if timeout:
        print("Planning timed out, check goal is reachable")
    else:
        # Filter and execute trajectory, keep a point in the traj 
        # if its norm >= 3deg wrt the previous one
        joint_traj = arm_utils.filter_joint_traj(joint_traj, min_deg=3.0)
        print(f"Executing open-loop traj of length {len(joint_traj)}")

        for idx, element in enumerate(joint_traj):
            element[-2:] = -element[-2:]  # Invert last two joints to match real px100
            goal_steps = angle2steps(element)
            # Send goals one after the other at a certain freq
            send_angles(goal_steps)
            # time.sleep(0.05)

        angles_steps = read_angles()
        agent.reset_belief(calibrate_angles(angles_steps))
        # print("EE error:", np.linalg.norm(S[1][-1] - np.array(goal_pos)))

except KeyboardInterrupt:
    print("Exiting...")
finally:
    arbotix.closePort()
    print("Port closed.")
    print("Done.")