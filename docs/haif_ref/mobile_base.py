import numpy as np

# Define arm class
class Base:
    def __init__(self, config):
        self.n_orders = config['agent']['n_orders']
        self.n_trials = config['benchmark'].get('n_trials', 1)
        self.n_steps = config['benchmark'].get('n_steps', 1000)
        self.eta = config['benchmark'].get('eta', 0.1)
        self.angles = config['base']['init_rotations']    # [phi_l, phi_r] absolute wheel rotations

        self.dt = config['agent']['dt']

        self.n_joints = 2

        self.pi_prop = config['agent']['pi_prop']
        self.pi_vis = config['agent']['pi_vis']
        self.pi_ext = config['agent']['pi_ext']

        self.w_p = config['agent']['w_p']
        self.w_a = config['agent']['w_a']

        self.a_max = config['base']['a_max']
        self.gain_a = config['agent']['gain_a']
        self.k_int = config['agent']['k_int']
        self.k_ext = config['agent']['k_ext']
        self.k_mu_ext =config['agent']['k_mu_ext']

        self.k_rep = config['agent']['k_rep']
        self.avoid_dist = config['agent']['avoid_dist']

        self.max_rep = config['agent']['max_rep']
       
        self.n_mu_ext = 7 # [x y quat] 

        self.wheel_radius = config['base']['wheel_radius']
        self.wheel_distance = config['base']['wheel_distance']
        self.k_err_cart = config['base'].get('k_err_cart', 1)
        self.k_err_ort = config['base'].get('k_err_ort', 1)

        # Initialize positions
        self.init_pose = config['agent']['init_pose']

        self.prev_phi_l = self.angles[0]
        self.prev_phi_r = self.angles[1]
        self.prev_quat = self.init_pose[-4:].copy()
        self.delta_phi = np.zeros(2)

        self.poses = self.init_pose.copy() # [x y quat] for ee and for
        self.poses = self.kinematics()

    def quaternion_multiply(self, quaternion1, quaternion0):
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

    def set_prev_phi(self, phi_l, phi_r):
        self.prev_phi_l = phi_l
        self.prev_phi_r = phi_r

    def kinematics(self, angles=None, poses=None, use_prev_quat=False):

        new_poses = np.zeros((1 + 1, self.n_mu_ext)) # [x y quat] for ee and for
        new_poses[:, 3] = 1 # unitary quaternion

        if angles is None:
            angles = self.angles
        if poses is None:
            poses = new_poses
            poses[-1] = self.init_pose
        if use_prev_quat:  
            poses[1][-4:] = self.prev_quat

        prev_pose = poses[1]
        new_poses[1] = self.generic_fw_kin(angles, prev_pose, use_prev_quat)

        return new_poses
    
    # Kinematic forward pass
    def generic_fw_kin(self, phi, prev_pose, use_prev_quat=False):
        """
        Computes forward kinematics 
        :param phi: wheel angles 
        :param prev_pose: pose of previous link [x y quat]]

        :returns position and orientation of current link
        """
        
        prev_x, prev_y, prev_quat = prev_pose[0], prev_pose[1], prev_pose[-4:] 
        phi_l = phi[0]
        phi_r = phi[1]

        q0 = prev_quat[0]
        q3 = prev_quat[-1]
        

        angle_increment = self.wheel_radius / self.wheel_distance * ((phi_r - self.prev_phi_r) - (phi_l - self.prev_phi_l))
        quat_increment = np.array([np.cos(angle_increment/2), 0, 0, np.sin(angle_increment/2)])

        quat_new = self.quaternion_multiply(quat_increment, prev_quat)
        x_new = prev_x + self.wheel_radius * 0.5 * (self.delta_phi[0] + self.delta_phi[1]) * (1-2*q3*q3)
        y_new = prev_y + self.wheel_radius * 0.5 * (self.delta_phi[0] + self.delta_phi[1]) * (2*q0*q3)

        if use_prev_quat:
            self.prev_quat = quat_new
       
        g_e = np.append([x_new, y_new, 0], quat_new)

        return g_e

    # Gradients 
    def kin_grad_phi(self, prev_quat):
        """
        Computes gradient of generative kin model, w.r.t. wheel rotations
        :param prev_quat: absolute base rotation angle

        :returns gradients 
        """
        q0 = prev_quat[0]
        q3 = prev_quat[-1]

        cos_ = 1-2*q3*q3
        sin_ = 2*q0*q3

        geom = 0.5*self.wheel_radius/self.wheel_distance
        delta_diff = geom*(self.delta_phi[0] - self.delta_phi[1])

        # Gradient of the orientation part of the kinematics
        # d_ge_phi_l = -np.array([0.5*self.wheel_radius*cos_, 0.5*self.wheel_radius*sin_, self.wheel_radius/self.wheel_distance])
        # d_ge_phi_r = -np.array([0.5*self.wheel_radius*cos_, 0.5*self.wheel_radius*sin_, -self.wheel_radius/self.wheel_distance])

        d_ge_phi_l = -np.array([0.5*self.wheel_radius*cos_, 0.5*self.wheel_radius*sin_, 0,
                                - geom*(q0*np.sin(delta_diff) + q3*np.cos(delta_diff)),
                                0, 
                                0, 
                                - geom*(-q0*np.cos(delta_diff) + q3*np.sin(delta_diff))])
        d_ge_phi_r = -np.array([0.5*self.wheel_radius*cos_, 0.5*self.wheel_radius*sin_, 0, 
                                - geom*(-q0*np.sin(delta_diff) - q3*np.cos(delta_diff)),
                                0, 
                                0, 
                                - geom*(q0*np.cos(delta_diff) - q3*np.sin(delta_diff))
                                ])

        return d_ge_phi_l, d_ge_phi_r
    
    
    # Update  with angles
    def update_with_angles(self, delta_phi, pose):
        self.angles += delta_phi
        self.delta_phi = delta_phi
        self.poses[1] = pose
