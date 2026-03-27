from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv

class RobotSim:
    def __init__(self, urdf_path, control_mode="pos"):
        self.urdf_path = urdf_path
        self.control_mode = control_mode
        dt = 0.0005

        robots = [
            GenericUrdfReacher(urdf=self.urdf_path, mode=control_mode),
        ]
        self.env: UrdfEnv = UrdfEnv(
            dt=dt, robots=robots, render=True
        )

        self.env.reconfigure_camera(camera_distance=0.6, 
                                    camera_pitch=-10, 
                                    camera_yaw=35, 
                                    camera_target_position=[0, 0, 0.2]
                                    )

    def reset_sim(self, init_pos, mount_pos=None):
        ob = self.env.reset(pos=init_pos, mount_positions=mount_pos)[0]
        return ob

    def set_n_joints(self, n_joints):
        self.n_joints = n_joints

    def step_sim(self, action):
        ob, *_ = self.env.step(action)
        return ob

    def add_visualizations(self):
        # Joints
        for _ in range(self.n_joints):
            self.env.add_visualization(
                shape_type="sphere",
                size=[0.06],  # radius of the sphere
                rgba_color=[0.0, 0.0, 1.0, 0.7]  # blue color with some transparency
            )
        # Goal
        self.env.add_visualization(
            shape_type="sphere",
            size=[0.03],  # radius of the sphere
            rgba_color=[0.0, 1.0, 0.0, 0.7]  # green color with some transparency
        )

    def update_visualizations(self, joint_positions, goal_position):
        
        vis_pos = []
        for k in range (self.n_joints):
            vis_pos.append(joint_positions[k+1])
        if goal_position is not None:
            vis_pos.append(goal_position)

        self.env.update_visualizations(vis_pos)