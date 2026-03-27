import numpy as np
import matplotlib.pyplot as plt
import yaml
from mpscenes.obstacles.dynamic_sphere_obstacle import DynamicSphereObstacle
import math
from pathlib import Path
import jax
import jax.numpy as jnp
import time
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from haif_robot_control.haif.jax_arm import JaxArm
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
import haif_robot_control.helpers.jax_fk as jax_fk

def plot_3d_trajectory(ee_traj, ee_goal_traj=None):
    ee_traj = np.array(ee_traj)
    ee_goal_traj = np.array(ee_goal_traj)

    # plot ee trajectories
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ee_traj[:,  0], ee_traj[:, 1], ee_traj[:, 2], label=f'EE Traj')
    ax.plot(ee_goal_traj[:, 0], ee_goal_traj[:, 1], ee_goal_traj[:, 2], '--', label=f'EE Goal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('End Effector Trajectories')
    ax.legend()
    plt.show()

    # plot error over time
    errors = np.linalg.norm(ee_traj - ee_goal_traj, axis=1)
    plt.figure()
    plt.plot(errors)
    plt.xlabel('Time Step')
    plt.ylabel('End Effector Position Error')
    plt.title('End Effector Position Error Over Time')
    plt.grid()
    plt.show()

def haif_to_sim_action(haif_action, env, arm):
    """Converts HAIF action to simulation action by filling non-actuated joints with zeros."""
    sim_action = np.zeros(env.n())
    sim_action[np.array(arm.static_params.controlled_joints_idxs)] = np.array(haif_action)
    return sim_action

def read_sim_angles(observation, arm):
    sim_angles_full = observation['robot_0']['joint_state']['position']
    sim_angles_controlled = jnp.array([sim_angles_full[i] for i in arm.static_params.controlled_joints_idxs])
    return sim_angles_controlled

def get_null_intentions(arm):
    n_joints = arm.static_params.n_dof + arm.static_params.n_ee
    goal_i = jnp.full(arm.static_params.n_dof, jnp.nan)
    goal_e_pos = jnp.full((n_joints, 3), jnp.nan)
    goal_e_ort = jnp.full((n_joints, 4), jnp.nan)
    obst_pos = None # No obstacles
    return goal_i, goal_e_pos, goal_e_ort, obst_pos

def load_config_and_arm(config_path: Path, urdf_path: Path) -> JaxArm:
    """Loads YAML config, adds URDF path, and initializes the JaxArm."""
    print("Loading configuration...")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['robot_params']['robot_urdf'] = str(urdf_path)

    arm = JaxArm(config)
    print("JaxArm instantiated.")
    return arm, config

def setup_environment(arm: JaxArm, config: dict, dt: float = 0.001, render: bool = True, num_sub_steps: int = 20, enforce_real_time: bool = False, camera_settings=None) -> UrdfEnv:
    """Initializes and configures the URDF simulation environment."""
    print("Setting up simulation environment...")
    robot = GenericUrdfReacher(urdf=config['robot_params']['robot_urdf'], mode="vel")
    env = UrdfEnv(dt=dt, robots=[robot], render=render, num_sub_steps=num_sub_steps, enforce_real_time=enforce_real_time)
    if camera_settings is not None:
        env.reconfigure_camera(**camera_settings)
    else:
        env.reconfigure_camera(
            camera_distance=0.5, 
            camera_pitch=-20, 
            camera_yaw=0, 
            camera_target_position=[0, 0, 0.3]
        )
    init_angles = np.array(config['robot_params']['init_angles'])

    # Check if init_angles length matches the number of controlled joints
    if len(init_angles) != env.n():
        print(f"Length of init_angles ({len(init_angles)}) does not match number of controlled joints ({env.n()}). Filling missing values.")
        init_angles = np.zeros(env.n())
        init_angles[np.array(arm.static_params.controlled_joints_idxs)] = np.array(config['robot_params']['init_angles'])
    return env, env.reset(pos=init_angles)[0]

def sample_ee_goal(key: jax.Array, arm: JaxArm) -> tuple[jax.Array, jax.Array]:
    """
    Samples a random reachable end-effector goal.
    Returns a new JAX random key and the goal position.
    """
    key, subkey = jax.random.split(key)
    # Sample joint angles within a reasonable range
    arm_limits = arm.static_params.joint_limits_pos
    random_joints = jax.random.uniform(subkey, (arm.static_params.n_dof,), minval=arm_limits[:, 0], maxval=arm_limits[:, 1])
    q_full = jnp.append(random_joints, 0.0) # Append non-actuated joint
    
    # Calculate forward kinematics to get the end-effector position
    goal_pos = jax_fk.compute_all_poses(
        arm.static_params.initial_pose_xyzquat, 
        arm.static_params.static_joint_params, 
        q_full
    )[-1, :3]
    
    return key, goal_pos

def setup_visualizations(env: UrdfEnv, n_joints: int, n_goals: int = 1, n_obstacles: int = 0):
    """Adds spheres to the environment for visualizing beliefs and goals."""
    # Blue spheres for measured joint positions (xyz)
    for _ in range(n_joints):
        env.add_visualization(shape_type="sphere", size=[0.04], rgba_color=[0.0, 0.0, 1.0, 0.7])
    # White spheres for belief joint positions (mu_e)
    for _ in range(n_joints):
        env.add_visualization(shape_type="sphere", size=[0.05], rgba_color=[1.0, 1.0, 1.0, 0.5])
    # Green sphere for the goal
    for _ in range(n_goals):
        env.add_visualization(shape_type="sphere", size=[0.05], rgba_color=[0.0, 1.0, 0.0, 0.7])
    # Red spheres for obstacles
    for _ in range(n_obstacles):
        env.add_visualization(shape_type="sphere", size=[0.05], rgba_color=[1.0, 0.0, 0.0, 1.0])

def random_linear_expr(x0, v_range=(-0.03, 0.03)):
    v = round(np.random.uniform(*v_range), 3)
    return f"{x0} + ({v}) * t"

def generate_obstacles_near_goals(goal_positions, min_dist=0.25, max_dist=0.35):
    """
    Generate one obstacle per goal position, each at a distance between min_dist and max_dist.
    
    Parameters:
        goal_positions (list of np.array): List of 3D goal coordinates
        min_dist (float): Minimum distance from the goal in meters
        max_dist (float): Maximum distance from the goal in meters
    
    Returns:
        list: Obstacles as string-format trajectories [["x", "y", "z"], ...]
    """
    obstacles = []

    for goal in goal_positions:
        while True:
            # Random direction
            direction = np.random.normal(size=3)
            direction /= np.linalg.norm(direction)
            # Random distance between min_dist and max_dist
            distance = np.random.uniform(min_dist, max_dist)
            # Obstacle position
            obstacle_pos = goal + direction * distance
            # Convert to list of strings
            obstacle_str = [f"{coord:.3f}" for coord in obstacle_pos]
            obstacles.append(obstacle_str)
            break

    return obstacles

# Normalize data
def normalize(x, limits):
    limits = np.array(limits)
    x_norm = (x - limits[0]) / (limits[1] - limits[0])
    x_norm = x_norm * 2 - 1
    return x_norm

# Denormalize data
def denormalize(x, limits):
    limits = np.array(limits)
    x_denorm = (x + 1) / 2
    x_denorm = x_denorm * (limits[1] - limits[0]) + limits[0]
    return x_denorm

def normalize_quat(q):
    # Calculate the magnitude for each quaternion (row-wise)
    magnitudes = np.linalg.norm(q, axis=1)
    
    # Avoid division by zero: normalize only non-zero magnitudes
    magnitudes[magnitudes == 0] = 1
    
    # Normalize each quaternion
    normalized_quaternions = q / magnitudes[:, np.newaxis]
    
    return normalized_quaternions

def compute_velocity(pos1, pos2, dt):
    """
    Compute the velocity vector between two positions in 3D space.
    """
    pos1 = np.array(pos1, dtype=float)
    pos2 = np.array(pos2, dtype=float)
    
    vel_xyz = (pos2 - pos1) / dt
    
    return vel_xyz

def yaw_to_quaternion(yaw):
        """Convert yaw (ψ) in radians to a quaternion (w, x, y, z)."""
        w = np.cos(yaw / 2)
        x = 0
        y = 0
        z = np.sin(yaw / 2)
        return (w, x, y, z)

def quaternion_to_euler(q):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        w,x,y,z = q
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def compute_joint_obst_cosines(velocities, positions, obstacle_position):
    """
    Compute the cosine of the angle between multiple joint velocity vectors
    and their respective vectors pointing to an obstacle.
    
    Parameters:
        velocities (array-like): A list or array of velocity vectors, each as [vx, vy, vz].
                                 Shape: (n, 3) for n joints.
        positions (array-like): A list or array of positions for each joint, each as [x, y, z].
                                Shape: (n, 3) for n joints.
        obstacle_position (array-like): The obstacle position [ox, oy, oz].
        
    Returns:
        numpy.ndarray: An array of cosines for each joint.

    Note:
        This only works for static obstacles
    """
    # Convert inputs to numpy arrays
    velocities = np.array(velocities, dtype=float)
    positions = np.array(positions, dtype=float)
    obstacle_position = np.array(obstacle_position, dtype=float)
    
    # Ensure inputs are 2D arrays (n, 3)
    if velocities.ndim == 1:
        velocities = velocities[np.newaxis, :]  # Convert single velocity vector to 2D
    if positions.ndim == 1:
        positions = positions[np.newaxis, :]    # Convert single position vector to 2D
    
    # Check that the number of joints matches for velocities and positions
    if velocities.shape[0] != positions.shape[0]:
        raise ValueError("Number of velocities and positions must match.")
    
    # Compute the vectors from the obstacle to each joint
    obstacle_vectors = obstacle_position - positions  # Shape: (n, 3)
    
    # Compute the dot products for each joint
    dot_products = np.einsum('ij,ij->i', velocities, obstacle_vectors)  # Efficient row-wise dot product
    
    # Compute the magnitudes of each velocity and obstacle vector
    velocities_magnitudes = np.linalg.norm(velocities, axis=1)
    obstacle_vectors_magnitudes = np.linalg.norm(obstacle_vectors, axis=1)
    
    # Initialize cosines with default values of -1
    cosines = np.full(velocities.shape[0], -1.0)
    
    # Avoid division by zero by masking valid entries
    valid_mask = (velocities_magnitudes > 0) & (obstacle_vectors_magnitudes > 0)
    
    # Compute cosines only for valid entries
    cosines[valid_mask] = (
        dot_products[valid_mask]
        / (velocities_magnitudes[valid_mask] * obstacle_vectors_magnitudes[valid_mask])
    )
    
    return cosines

class sampler_utils():
    def __init__(self, config_file):
        # Initialize arm parameters
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # Retrieve values from the configuration
        self.random_goal = config['benchmark']['random_goal']
        self.random_obst = config['benchmark']['random_obst']
        self.obst_type = config['benchmark']['obst_type']
        self.min_goal_coords = np.array(config['benchmark']['min_goal_coords'])
        self.max_goal_coords = np.array(config['benchmark']['max_goal_coords'])
        self.fixed_goal_coords = np.array(config['benchmark']['fixed_goal_coords'])
        self.min_obst_coords = np.array(config['benchmark']['min_obst_coords'])
        self.max_obst_coords = np.array(config['benchmark']['max_obst_coords'])
        self.min_obst_coords_dyn = np.array(config['benchmark']['min_obst_coords_dyn'])
        self.max_obst_coords_dyn = np.array(config['benchmark']['max_obst_coords_dyn'])
        self.fixed_obst_coords = np.array(config['benchmark']['fixed_obst_coords'])
        self.num_obstacles = config['benchmark']['num_obstacles']

    def sample_xyz_from_plane(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        if self.random_goal:
            return np.random.uniform(low=self.min_goal_coords, high=self.max_goal_coords)
        else:
            return self.fixed_goal_coords
    
    def sample_obstacles_from_plane(self, seed=None, xyz=(0, 0, 0), min_distance=0.3):
        def is_valid(coord):
            y = coord[1]
            distance = np.linalg.norm(np.array(coord) - np.array(xyz))
            if self.obst_type == "static":
                y_dist = min_distance
            else:   
                y_dist = min_distance*5 # Start far and approach
            return (y > y_dist or y < -y_dist) and distance >= min_distance

        if seed is not None:
            np.random.seed(seed)
        
        if not self.random_goal:
            return self.fixed_obst_coords
        else: 
            while True:
                if self.random_goal:
                    if self.obst_type == "dynamic":
                        coord = np.random.uniform(low=self.min_obst_coords_dyn, high=self.max_obst_coords_dyn, size=(3,))
                    else:
                        coord = np.random.uniform(low=self.min_obst_coords, high=self.max_obst_coords, size=(3,))
                
                if is_valid(coord):
                    return [str(coord[0]), str(coord[1]), str(coord[2])]

    def add_obstacles(self, env, obst_trajectories):
            for i, traj in enumerate(obst_trajectories):
                dynamicObstDict = {
                    "type": "sphere",
                    "geometry": {"trajectory": traj, "radius": 0.05},
                    "movable": False,
                    "rgba": [0.5, 0.0, 0.13, 1.0]
                }
                dynamicSphereObst = DynamicSphereObstacle(
                    name=f"simpleSphere_{i}", content_dict=dynamicObstDict
                )
                env.add_obstacle(dynamicSphereObst)

    def get_random_obstacle_trajectory(self, NUM_OBSTACLES, goal=(0, 0, 0), curr_seed=None):
        obst_trajectories = []
        for i in range(NUM_OBSTACLES):
            if curr_seed is not None:
                curr_seed += i
            obst_trajectories.append(self.sample_obstacles_from_plane(xyz=goal, seed=curr_seed))
        if self.obst_type == "static":
            return obst_trajectories
        else:
            obst_traj = []
            for i in range(NUM_OBSTACLES):
                x = float(obst_trajectories[i][0])
                y = float(obst_trajectories[i][1])
                z = float(obst_trajectories[i][2])
                vel = np.random.uniform(low=0.1, high=0.3)
                if y < 0:
                    vel = -vel
                obst_traj.append([f"{x}", f"{y} - {vel} * t", f"{z}"])
            return obst_traj

class arm_utils():
    def __init__(self, arm):
        # Initialize the DH parameters and the current goal
        self.dh_params = arm.dh_table
        self.n_joints = arm.n_joints
        self.joint_limits = arm.limits
        self.angles_offset = arm.dh_angle_offset
        self.current_goal = self.generate_new_xyz_goal(self.dh_params) # Initial sampled goal

    def filter_joint_traj(self, traj, min_deg=3.0):
        """Filter trajectory so a point is at least min_deg apart from the previous in norm, always including first and last."""
        if len(traj) == 0:
            return traj
        filtered = [traj[0]]
        for point in traj[1:-1]:
            if np.linalg.norm(point - filtered[-1]) >= min_deg:
                filtered.append(point)
        filtered.append(traj[-1])
        return filtered
    
    # def filter_joint_traj(self, traj, min_deg=3.0):
    #         """Filter trajectory so each point is at least min_deg apart, always including first and last."""
    #         if len(traj) == 0:
    #             return traj
    #         filtered = [traj[0]]
    #         for point in traj[1:-1]:
    #             updated_traj = filtered[-1].copy()
    #             for joint in range(len(point)):
    #                 if np.linalg.norm(point[joint] - filtered[-1][joint]) >= min_deg:
    #                     updated_traj[joint] = point[joint]
    #             filtered.append(updated_traj)
    #         filtered.append(traj[-1])
    #         return filtered
    def get_goal(self):
        return self.current_goal

    def set_goal(self, new_goal):
        self.current_goal = new_goal

    def dh_transformation(self, l, alpha, d, theta):
        theta = np.deg2rad(theta)
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), l*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), l*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self,dh_params, joint_angles):
        T = np.eye(4)
        for i in range(len(joint_angles)):
            T_i = self.dh_transformation(dh_params[i][0], dh_params[i][1], dh_params[i][2], joint_angles[i]+self.angles_offset[i])
            T = np.dot(T, T_i)
        return T[:3, 3]

    def sample_joint_angles(self):
        return np.array([np.random.uniform(low, high) for low, high in 0.5*self.joint_limits])

    def generate_new_xyz_goal(self, dh_params):
        joint_angles = self.sample_joint_angles()
        new_xyz = self.forward_kinematics(dh_params, joint_angles)
        return new_xyz

    def update_xyz_goal(self, dh_params):
        new_xyz_goal = self.generate_new_xyz_goal(dh_params)
        return new_xyz_goal

    def plot_log(self, log):
        angles, est_angles, pos, est_pos = log.angles[0], log.est_angles[0], log.pos[0], log.est_pos[0]

        # Remove dh offsets
        for i in range(len(angles)):
            angles[i,:] -= self.angles_offset
            est_angles[i,:] -= self.angles_offset

        time = np.arange(len(angles))

        fig, axs = plt.subplots(4, 1, figsize=(10, 15))

        # Plot angles and estimated angles
        for i in range(len(angles[0])):
            axs[0].plot(time, angles[:, i], label=f'True Angles joint_{i+1}')
            axs[0].plot(time, est_angles[:, i], label=f'Estimated Angles joint_{i+1}')
        axs[0].set(xlabel='Time Steps', ylabel='Angles', title='Joint Angles vs Estimated Angles')
        axs[0].legend()

        # Plot end-effector positions
        labels = ['ee_x', 'ee_y', 'ee_z']
        for i in range(3):
            axs[i+1].plot(time, pos[:, -1, i], label=f'True {labels[i]}')
            axs[i+1].plot(time, est_pos[:, -1, i], label=f'Estimated {labels[i]}')
            axs[i+1].set(xlabel='Time Steps', ylabel=f'Absolute {labels[i]} Pos', title=f'True vs Estimated {labels[i]} pos')
            axs[i+1].legend()
        # Adjust layout
        plt.tight_layout()
        # Show the plot
        plt.show()

    def get_joint_obs(self, arm):
        joint_angles_obs = self.add_gaussian_noise(arm.angles, arm.w_p)
        return normalize(joint_angles_obs, arm.norm_polar)

    def get_visual_obs(self, arm, length=None, init_pose=None, angles=None, full_pose=False):
            # TODO change for new kinematics
            if length is None:
                l = self.dh_params[:,0]
            else:
                l = length
            if angles is None:
                xyz_obs = arm.kinematics(lengths=l, init_pose=init_pose)
            else:
                xyz_obs = arm.kinematics(angles=angles, lengths=l, init_pose=init_pose)
            #return normalize(arm.kinematics()[:, :2], arm.norm_cart)
            if not full_pose:
                xyz_obs = xyz_obs[:, :3]
            return xyz_obs

    # Add Gaussian noise to array
    def add_gaussian_noise(self, array, noise):
        sigma = noise ** 0.5
        return array + np.random.normal(0, sigma, np.shape(array))


class base_utils():
    def __init__(self, base):
        # Initialize the pose
        self.pose = base.init_pose
        self.dt = base.dt

    def differential_drive_kinematics(self, x0, y0, theta0, R, L, delta_phi_L, delta_phi_R):
        """
        Computes the new pose (x, y, theta) of a differential drive robot given the
        initial pose (x0, y0, theta0) and the wheel rotations (delta_phi_L, delta_phi_R).
        
        Args:
        - x0, y0, theta0: Initial position and orientation of the robot.
        - R: Radius of the wheels.
        - L: Distance between the wheels.
        - delta_phi_L: Rotation angle of the left wheel (in radians).
        - delta_phi_R: Rotation angle of the right wheel (in radians).
        
        Returns:
        - x, y, theta: Updated position and orientation of the robot.
        """
        
        # Calculate linear displacements of each wheel
        delta_s_L = R * delta_phi_L
        delta_s_R = R * delta_phi_R
        
        # Compute the linear and angular displacements
        delta_s = (delta_s_L + delta_s_R) / 2
        delta_theta = (delta_s_R - delta_s_L) / L
        
        # If there's no change in orientation (straight-line motion)
        if delta_theta == 0:
            x = x0 + delta_s * np.cos(theta0)
            y = y0 + delta_s * np.sin(theta0)
            theta = theta0
        else:
            # For curved motion
            x = x0 + (delta_s / delta_theta) * (np.sin(theta0 + delta_theta) - np.sin(theta0))
            y = y0 + (delta_s / delta_theta) * (np.cos(theta0) - np.cos(theta0 + delta_theta))
            theta = theta0 + delta_theta
        
        return x, y, theta

    def compute_rot_wheel_velocities(self, v, omega, L, r):
        """
        Computes the left and right wheel velocities given the robot's linear and angular velocity.
        
        Args:
        - v: Linear velocity of the robot (m/s).
        - omega: Angular velocity of the robot (rad/s).
        - L: Distance between the wheels (m).
        
        Returns:
        - v_L: Velocity of the left wheel (m/s).
        - v_R: Velocity of the right wheel (m/s).
        """
        
        # Compute left and right wheel velocities
        v_L = v - (omega * L) / 2
        v_R = v + (omega * L) / 2

        return v_L / r, v_R / r

    def compute_linear_angular_velocity_from_wheel_rotations(self, w_L, w_R, R, L):
        """
        Computes the robot's linear and angular velocity given the angular velocities of the wheels.
        
        Args:
        - w_L: Rotational velocity of the left wheel (rad/s).
        - w_R: Rotational velocity of the right wheel (rad/s).
        - R: Radius of the wheels (m).
        - L: Distance between the wheels (m).
        
        Returns:
        - v: Linear velocity of the robot (m/s).
        - omega: Angular velocity of the robot (rad/s).
        """
        
        # Convert rotational velocities to linear velocities
        v_L = w_L * R
        v_R = w_R * R
        
        # Compute linear and angular velocities of the robot
        v = (v_L + v_R) / 2
        omega = (v_R - v_L) / L
        
        return v, omega

    def sample_xy_goal(self):
            x = np.random.uniform(1, -1)
            y = np.random.uniform(1., -1.)
            return np.array([x, y])

    # Add Gaussian noise to array
    def add_gaussian_noise(self, array, noise):
        sigma = noise ** 0.5
        return array + np.random.normal(0, sigma, np.shape(array))
    
    def plot_log(self, log):
        wheel_rot, est_wheel_rot, pose, est_pose = log.wheel_rot[0], log.est_wheel_rot[0], log.pose[0], log.est_pose[0]

        time = np.arange(len(wheel_rot))

        fig, axs = plt.subplots(4, 1, figsize=(10, 15))

        # Plot wheel_rot and estimated wheel_rot
        for i in range(len(wheel_rot[0])):
            axs[0].plot(time, wheel_rot[:, i], label=f'True wheel_rot joint_{i+1}')
            axs[0].plot(time, est_wheel_rot[:, i], label=f'Estimated wheel_rot joint_{i+1}')
        axs[0].set(xlabel='Time Steps', ylabel='wheel_rot', title='Joint wheel_rot vs Estimated wheel_rot')
        axs[0].legend()

        # Plot end-effector positions
        labels = ['ee_x', 'ee_y', 'theta']
        for i in range(3):
            axs[i+1].plot(time, pose[:, -1, i], label=f'True {labels[i]}')
            axs[i+1].plot(time, est_pose[:, -1, i], label=f'Estimated {labels[i]}')
            axs[i+1].set(xlabel='Time Steps', ylabel=f'Absolute {labels[i]} Pos', title=f'True vs Estimated {labels[i]} pos')
            axs[i+1].legend()
        # Adjust layout
        plt.tight_layout()
        # Show the plot
        plt.show()