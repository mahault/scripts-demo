import numpy as np 

# Define arm class
class Arm:
    def __init__(self, config):
        self.n_orders = config['agent']['n_orders']
        self.n_trials = config['benchmark'].get('n_trials', 1)
        self.n_steps = config['benchmark'].get('n_steps', 1000)
        self.eta = config['benchmark'].get('eta', 0.1)
        self.angles = np.array(config['arm']['init_angles'])
        self.dh_angle_offset = np.array(config['arm']['dh_angle_offset'])

        self.z_offset = config['benchmark'].get('z_offset', 0)

        self.k_rep_base = config['agent'].get('k_rep_base', 0)
        self.rep_dist_base = config['agent'].get('rep_dist_base', 1000)

        self.arm_to_base_k_linear = config['agent'].get('arm_to_base_k_linear', 0)
        self.arm_to_base_k_quat = config['agent'].get('arm_to_base_k_quat', 0)

        self.limits = np.array(config['arm']['joint_limits'])
        self.norm_polar = config['arm']['norm_polar']
        self.dt = config['agent']['dt']
        self.is_dh_craig = config['arm']['is_DH_craig']
        self.dh_table = np.array(config['arm']['dh_table'])

        self.n_joints = self.dh_table.shape[0]

        self.pi_prop = config['agent']['pi_prop']
        self.pi_vis = config['agent']['pi_vis']
        self.pi_ext = config['agent']['pi_ext']

        self.w_p = config['agent']['w_p']
        self.w_a = config['agent']['w_a']

        self.lr_length = config['agent']['lr_length']

        self.a_max = config['arm']['a_max']
        self.gain_a = config['agent']['gain_a']
        self.k_int_only_arm = config['agent'].get('k_int_only_arm', 1.0)
        self.k_int_with_base = config['agent'].get('k_int_with_base', 1.0)
        self.k_int = config['agent'].get('k_int', 1.0)
        self.k_ext = config['agent']['k_ext']
        self.k_mu_ext =config['agent']['k_mu_ext']

        self.k_rep_j = config['agent']['k_rep_j']
        self.k_rep = config['agent']['k_rep']
        self.avoid_dist = config['agent']['avoid_dist']
        self.joint_lim_threshold = config['agent']['joint_lim_threshold']
        self.max_rep = config['agent']['max_rep']
       
        self.n_mu_ext = 7 # [x y z qw qx qy qz]

        # Initialize positions (real values, not normalized)

        self.init_pose = config['agent'].get('init_pose', None)
        self.poses = self.kinematics(init_pose=self.init_pose)

    # Compute pose of every link
    def kinematics(self, angles=None, lengths=None, poses=None, init_pose=None):

        new_poses = np.zeros((self.n_joints + 1, self.n_mu_ext)) # [x y z q0 q1 q2 q3] for all links + ee and one extra initial world link at 0
        new_poses[:, 3] = 1 # identity orientation
        # todotodo take the init pose from config and put in in the first value of new_poses
        if init_pose is not None:
            new_poses[0] = init_pose
        
        if angles is None:
            angles = self.angles
        if lengths is None:
            dh_table = self.dh_table.copy()
        else:
            dh_table = self.dh_table.copy()
            dh_table[:,0] = lengths     # TODO add an extra state for d and alpha in the dh table as an internal belief
        if poses is None:
            poses = new_poses
            
        poses[0] = new_poses[0]

        for k in range(self.n_joints):
            old_pose = poses[k]
            new_poses[k + 1] = self.generic_fw_kin(angles[k] + self.dh_angle_offset[k], dh_table[k], old_pose)

        return new_poses
    
    # Kinematic forward pass
    def generic_fw_kin(self, theta, DH, prev_pose):
        """
        Computes forward kinematics 
        :param theta: joint angle 
        :param DH: DH parameters for current joint
        :param prev_pose: pose of previous link [x y z quat]]

        :returns position and orientation of current link
        """
        
        pos_old, q_old = prev_pose[:3], prev_pose[3:] 

        # DH parameters for current joint
        l, alpha, d, _ = DH

        alpha = np.deg2rad(alpha)
        theta = np.deg2rad(theta)

        if not self.is_dh_craig:
            x_j = l*np.cos(theta)
            y_j = l*np.sin(theta)
            z_j = d
            s = 1 # sign
        else:
            x_j = l
            y_j = -d*np.sin(alpha)
            z_j = d*np.cos(alpha)
            s = -1

        # Compute once the necessary trigonometric functions
        c_theta2 = np.cos(0.5*theta)
        c_alpha2 = np.cos(0.5*alpha)
        s_theta2 = np.sin(0.5*theta)
        s_alpha2 = np.sin(0.5*alpha)

        # Transformed coordinates, to update joint pose
        x_tf = q_old[0]*q_old[0]*x_j + q_old[1]*q_old[1]*x_j - q_old[2]*q_old[2]*x_j - q_old[3]*q_old[3]*x_j \
            + 2*q_old[1]*q_old[2]*y_j + 2*q_old[1]*q_old[3]*z_j + 2*q_old[0]*q_old[2]*z_j - 2*q_old[0]*q_old[3]*y_j

        y_tf = q_old[0]*q_old[0]*y_j - q_old[1]*q_old[1]*y_j + q_old[2]*q_old[2]*y_j - q_old[3]*q_old[3]*y_j \
            + 2*q_old[1]*q_old[2]*x_j + 2*q_old[2]*q_old[3]*z_j - 2*q_old[0]*q_old[1]*z_j + 2*q_old[0]*q_old[3]*x_j

        z_tf = q_old[0]*q_old[0]*z_j - q_old[1]*q_old[1]*z_j - q_old[2]*q_old[2]*z_j + q_old[3]*q_old[3]*z_j \
            + 2*q_old[1]*q_old[3]*x_j + 2*q_old[2]*q_old[3]*y_j - 2*q_old[0]*q_old[2]*x_j + 2*q_old[0]*q_old[1]*y_j

        # Transformed rotation
        q_tf = [q_old[0]*c_theta2*c_alpha2 - q_old[1]*c_theta2*s_alpha2 - q_old[2]*s_theta2*s_alpha2*s - q_old[3]*s_theta2*c_alpha2, 
                q_old[0]*c_theta2*s_alpha2 + q_old[1]*c_theta2*c_alpha2 + q_old[2]*s_theta2*c_alpha2 - q_old[3]*s_theta2*s_alpha2*s, 
                q_old[0]*s_theta2*s_alpha2*s - q_old[1]*s_theta2*c_alpha2 + q_old[2]*c_theta2*c_alpha2 + q_old[3]*c_theta2*s_alpha2, 
                q_old[0]*s_theta2*c_alpha2 + q_old[1]*s_theta2*s_alpha2*s - q_old[2]*c_theta2*s_alpha2 + q_old[3]*c_theta2*c_alpha2]

        # Populate generative model for extrinsic data, these are the updated position and orientation of the joint
        g_e = np.array([pos_old[0]+x_tf, pos_old[1] + y_tf, pos_old[2] + z_tf, q_tf[0], q_tf[1], q_tf[2], q_tf[3]])

        return g_e

    # Gradients 
    def kin_grad_theta(self, theta, l, alpha, q):
        """
        Computes gradient of generative kin model, w.r.t. joint angles
        :param theta: joint angle 
        :param l, alpha: joint lenght and DH alpha angle
        :param q: quaternion with orientation of previous link

        :returns gradient
        """
        
        alpha = np.deg2rad(alpha)
        theta = np.deg2rad(theta)

        # Compute once the necessary trigonometric functions
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)

        c_theta2 = np.cos(0.5*theta)
        c_alpha2 = np.cos(0.5*alpha)
        s_theta2 = np.sin(0.5*theta)
        s_alpha2 = np.sin(0.5*alpha)

        if not self.is_dh_craig:
            s = 1 # sign
            # Gradient of the positional part of the kinematics
            d_ge_theta_pos = np.array([-q[0]*q[0]*l*s_theta - q[1]*q[1]*l*s_theta + q[2]*q[2]*l*s_theta + q[3]*q[3]*l*s_theta + 2*q[1]*q[2]*l*c_theta - 2*q[0]*q[3]*l*c_theta,
                                        q[0]*q[0]*l*c_theta - q[1]*q[1]*l*c_theta + q[2]*q[2]*l*c_theta - q[3]*q[3]*l*c_theta - 2*q[1]*q[2]*l*s_theta - 2*q[0]*q[3]*l*s_theta,
                                        -2*q[1]*q[3]*l*s_theta + 2*q[2]*q[3]*l*c_theta + 2*q[0]*q[2]*l*s_theta + 2*q[0]*q[1]*l*c_theta])
        else:
            d_ge_theta_pos = np.array([0., 0., 0.])
            s = -1 # sign

        # Gradient of the orientation part of the kinematics
        d_ge_theta_ort = 0.5*np.array([-q[0]*s_theta2*c_alpha2 + q[1]*s_theta2*s_alpha2 - q[2]*c_theta2*s_alpha2*s - q[3]*c_theta2*c_alpha2,
                                       -q[0]*s_theta2*s_alpha2 - q[1]*s_theta2*c_alpha2 + q[2]*c_theta2*c_alpha2 - q[3]*c_theta2*s_alpha2*s,
                                        q[0]*c_theta2*s_alpha2*s - q[1]*c_theta2*c_alpha2 - q[2]*s_theta2*c_alpha2 - q[3]*s_theta2*s_alpha2,
                                        q[0]*c_theta2*c_alpha2 + q[1]*c_theta2*s_alpha2*s + q[2]*s_theta2*s_alpha2 - q[3]*s_theta2*c_alpha2,
                                      ])

        d_ge_theta = np.concatenate((d_ge_theta_pos, d_ge_theta_ort))

        return d_ge_theta
    
    def kin_grad_l(self, theta, q):
        """
        Computes gradient of generative kin model, w.r.t. joint length
        :param theta: joint angle 
        :param q: quaternion with orientation of previous link

        :returns gradient
        """
        
        theta = np.deg2rad(theta)

        # Compute once the necessary trigonometric functions
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)

        if not self.is_dh_craig:
            # kinematics
            d_ge_l_pos = np.array([q[0]*q[0]*c_theta + q[1]*q[1]*c_theta - q[2]*q[2]*c_theta - q[3]*q[3]*c_theta + 2*q[1]*q[2]*s_theta - 2*q[0]*q[3]*s_theta,
                                   q[0]*q[0]*s_theta - q[1]*q[1]*s_theta + q[2]*q[2]*s_theta - q[3]*q[3]*s_theta + 2*q[1]*q[2]*c_theta + 2*q[0]*q[3]*c_theta,
                                   2*q[1]*q[3]*c_theta + 2*q[2]*q[3]*s_theta - 2*q[0]*q[2]*c_theta + 2*q[0]*q[1]*s_theta])
        else:
            d_ge_l_pos = np.array([q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3],
                                   2*q[1]*q[2] + 2*q[0]*q[3],
                                   2*q[1]*q[3] - 2*q[0]*q[2]])

        # Gradient of the orientation part of the kinematics
        d_ge_l_ort = np.array([0., 0., 0., 0.])

        d_ge_l = np.concatenate((d_ge_l_pos, d_ge_l_ort))

        return d_ge_l

    def kin_grad_ext(self, theta, l, alpha, d, q):
        """
        Computes gradient of generative kin model, w.r.t. extrinsics (position and orientation)
        :param theta: joint angle 
        :param l, alpha, d: DH parameters for current joint
        :param q: quaternion with orientation of previous link

        :returns gradient
        """

        alpha = np.deg2rad(alpha)
        theta = np.deg2rad(theta)

        # Compute once the necessary trigonometric functions
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        s_alpha = np.sin(alpha)
        c_alpha = np.cos(alpha)

        c_theta2 = np.cos(0.5*theta)
        c_alpha2 = np.cos(0.5*alpha)
        s_theta2 = np.sin(0.5*theta)
        s_alpha2 = np.sin(0.5*alpha)

        if not self.is_dh_craig:
            d_ge_q0 = np.array([2*q[0]*l*c_theta + 2*q[2]*d - 2*q[3]*l*s_theta, 
                                2*q[0]*l*s_theta - 2*q[1]*d + 2*q[3]*l*c_theta,
                                2*q[0]*d - 2*q[2]*l*c_theta + 2*q[1]*l*s_theta,
                                c_theta2*c_alpha2, c_theta2*s_alpha2, s_theta2*s_alpha2, s_theta2*c_alpha2])
            
            d_ge_q1 = np.array([2*q[1]*l*c_theta + 2*q[2]*l*s_theta + 2*q[3]*d, 
                                2*q[1]*l*s_theta + 2*q[2]*l*c_theta - 2*q[0]*d,
                                -2*q[1]*d + 2*q[3]*l*c_theta + 2*q[0]*l*s_theta,
                                -c_theta2*s_alpha2, c_theta2*c_alpha2, -s_theta2*c_alpha2, s_theta2*s_alpha2])
            
            d_ge_q2 = np.array([-2*q[2]*l*c_theta + 2*q[1]*l*s_theta + 2*q[0]*d, 
                                2*q[2]*l*s_theta + 2*q[1]*l*c_theta + 2*q[3]*d,
                                -2*q[2]*d + 2*q[3]*l*s_theta - 2*q[0]*l*c_theta,
                                -s_theta2*s_alpha2, s_theta2*c_alpha2, c_theta2*c_alpha2, -c_theta2*s_alpha2])
            
            d_ge_q3 = np.array([-2*q[3]*l*c_theta + 2*q[1]*d - 2*q[0]*l*s_theta, 
                                -2*q[3]*l*s_theta + 2*q[2]*d + 2*q[0]*l*c_theta,
                                2*q[3]*d + 2*q[1]*l*c_theta + 2*q[2]*l*s_theta,
                                -s_theta2*c_alpha2, -s_theta2*s_alpha2, c_theta2*s_alpha2, c_theta2*c_alpha2])
        else:
            d_ge_q0 = np.array([2*q[0]*l + 2*q[2]*d*c_alpha + 2*q[3]*d*s_alpha, 
                                -2*q[0]*d*s_alpha - 2*q[1]*d*c_alpha + 2*q[3]*l,
                                2*q[0]*d*c_alpha - 2*q[2]*l - 2*q[1]*d*s_alpha,
                                c_theta2*c_alpha2, c_theta2*s_alpha2, -s_theta2*s_alpha2, s_theta2*c_alpha2])
            
            d_ge_q1 = np.array([2*q[1]*l - 2*q[2]*d*s_alpha + 2*q[3]*d*c_alpha, 
                                2*q[1]*d*s_alpha + 2*q[2]*l - 2*q[0]*d*c_alpha,
                                -2*q[1]*d*c_alpha + 2*q[3]*l - 2*q[0]*d*s_alpha,
                                -c_theta2*s_alpha2, c_theta2*c_alpha2, -s_theta2*c_alpha2, -s_theta2*s_alpha2])

            d_ge_q2 = np.array([-2*q[2]*l - 2*q[2]*d*s_alpha + 2*q[0]*d*c_alpha, 
                                -2*q[2]*d*s_alpha + 2*q[1]*l + 2*q[3]*d*c_alpha,
                                -2*q[2]*d*c_alpha - 2*q[3]*d*s_alpha - 2*q[0]*l,
                                s_theta2*s_alpha2, s_theta2*c_alpha2, c_theta2*c_alpha2, -c_theta2*s_alpha2])

            d_ge_q3 = np.array([-2*q[3]*l + 2*q[1]*d*c_alpha + 2*q[0]*d*s_alpha, 
                                2*q[3]*d*s_alpha + 2*q[2]*d*c_alpha + 2*q[0]*l,
                                2*q[3]*d*c_alpha + 2*q[1]*l - 2*q[2]*d*s_alpha,
                                -s_theta2*c_alpha2, s_theta2*s_alpha2, c_theta2*s_alpha2, c_theta2*c_alpha2])
        
        d_ge_pos = np.array([[1., 0., 0., 0., 0., 0., 0.],
                             [0., 1., 0., 0., 0., 0., 0.],
                             [0., 0., 1., 0., 0., 0., 0.]])
        
        d_ge_ext = np.vstack((d_ge_pos, d_ge_q0, d_ge_q1, d_ge_q2, d_ge_q3))

        return d_ge_ext
    
    # Update arm with angles
    def update_with_angles(self, new_angles):
        self.angles[:len(new_angles)] = np.clip(new_angles, *self.limits[:len(new_angles)].T)
        self.poses = self.kinematics()
