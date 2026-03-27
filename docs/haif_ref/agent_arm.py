import numpy as np
from numpy.linalg import norm
import haif_robot_control.helpers.utils as utils


class AgentArm:
    def __init__(self, arm):

        self.arm = arm
        self.limits = utils.normalize(arm.limits, self.arm.norm_polar)
        self.a = np.zeros(self.arm.n_joints)
        self.default_k_int = self.arm.k_int
        self.cartesian_vels = None
        self.prev_cartes_pos = None
        self.dh_offset = utils.normalize(self.arm.dh_angle_offset, self.arm.norm_polar)

        # Initialize belief and action
        self.mu_int = np.zeros((self.arm.n_orders, self.arm.n_joints, 2))                       # Joint angle and joint length
        self.mu_ext = np.zeros((self.arm.n_orders, self.arm.n_joints + 1, self.arm.n_mu_ext))   # xyz quat
        
    def get_i(self, target_joint=None, target_pos=None, target_ort=None):
        """
        Get intentions
        :param target_joint: desired joint angles
        :param target_pos: desired link positions
        """
        self.target = target_pos
        i_int = self.mu_int[0].copy()
        i_ext = self.mu_ext[0].copy()

        if target_joint is not None:
            joint_norm = utils.normalize(target_joint, self.arm.norm_polar)
            i_int[:, 0] = joint_norm

        if target_pos is not None:
            i_ext[-1, :3] = target_pos  # Only final position 
            # i_ext[-2, :2] = i_ext[-1, :2].copy() # keep wrist pointing down

        if target_ort is not None:
            i_ext[-1, -4:] = np.array(target_ort) # Only final orientation

        # Experimenting with other goal types
        # Only second joint
        #i_int[-3, 0] = utils.normalize(-20, self.arm.norm_polar)

        # If both joint and position goals are present, tune down the joint goal
        if target_joint is not None and target_pos is not None:
            self.arm.k_int = self.default_k_int*0.05
        else:
            self.arm.k_int = self.default_k_int
        return i_int, i_ext

    def g_ext(self):
        """
        Get extrinsic belief
        """
        mu_int_denorm = self.mu_int[0].copy()
        mu_int_denorm[:, 0] = utils.denormalize(mu_int_denorm[:, 0],
                                                self.arm.norm_polar)

        mu_ext_normalized = self.mu_ext[0].copy()

        # Make sure quaternion is unit length 
        mu_ext_normalized[:, -4:] = utils.normalize_quat(mu_ext_normalized[:, -4:])

        new_mu_ext = self.arm.kinematics(*mu_int_denorm.T, mu_ext_normalized)

        # Make sure quaternion is unit length
        new_mu_ext[:, -4:] = utils.normalize_quat(new_mu_ext[:, -4:])

        return new_mu_ext

    def grad_ext(self, E_ext, P_ext):
        """
        Get extrinsic gradient
        :param E_ext: extrinsic prediction error
        :param P_ext: extrinsic prediction
        """
        p_ext_denorm = P_ext.copy()

        if self.arm.lr_length > 0.0:
            self.mu_int[0, -1, 1] = 0.0     # Exclude last link length from optimization. This leads to strange convergence. Might add a ghost link inn case we have to use a tool as ee.

        grad_theta = np.array([self.arm.kin_grad_theta(theta, length, alpha, q) for theta, length, alpha, q in
                               zip(self.mu_int[0, :, 0] + self.dh_offset, self.mu_int[0, :, 1], self.arm.dh_table[:,1], p_ext_denorm[1:, -4:])])
        
        grad_length = np.array([self.arm.kin_grad_l(theta, q) for theta, q in
                                zip(self.mu_int[0, :, 0] + self.dh_offset, p_ext_denorm[1:, -4:])])
        
        grad_length *= self.arm.lr_length
        
        grad_ext = np.array([self.arm.kin_grad_ext(theta, length, alpha, d, q) for theta, length, alpha, d, q in
                               zip(self.mu_int[0, :, 0] + self.dh_offset, self.mu_int[0, :, 1], self.arm.dh_table[:,1], self.arm.dh_table[:,2], p_ext_denorm[1:, -4:])])

        lkh_int = np.zeros((self.arm.n_joints, 2)) 
        for i in range(self.arm.n_joints):
            lkh_int[i, 0] = grad_theta[i].dot(E_ext[i + 1])
            lkh_int[i, 1] = grad_length[i].dot(E_ext[i + 1])

        lkh_ext = np.zeros_like(self.mu_ext[0])
        for j in range(1, self.arm.n_joints):
            lkh_ext[j] = grad_ext[j].dot(E_ext[j + 1])

        return lkh_int * self.arm.pi_ext, lkh_ext * self.arm.pi_ext


    def get_p(self):
        """
        Get predictions
        """
        p_ext = self.g_ext()                    # Forward kinematics [xyz q] predicted from joint angles 
        p_prop = self.mu_int[0, :, 0].copy()    # Joint angles [theta]
        p_vis = self.mu_ext[0, :, :3].copy()    # [xyz]

        return p_ext, p_prop, p_vis

    def get_e_g(self, S, P):
        """
        Get sensory prediction errors
        :param S: observations
        :param P: predictions
        """
        E_g = [s - p for s, p in zip(S, P)]

        return E_g

    def get_e_mu(self, I):
        """
        Get dynamics prediction errors
        :param I: intentions
        """

        E_i = [(I[0] - self.mu_int[0]) * self.arm.k_int,
               (I[1] - self.mu_ext[0]) * self.arm.k_ext]

        return self.mu_int[1] - E_i[0], self.mu_ext[1] - E_i[1]

    def get_likelihood(self, E_g, P):
        """
        Get likelihood components
        :param E_g: sensory prediction errors for each joint, [pos quat] [joint angles] [links pos]
        :param P: predictions
        """
        lkh = {}

        lkh['int'], lkh['ext'] = self.grad_ext(E_g[0], P[0]) # E_g[0] and P[0] represents respectively the sensory prediction errors for xyz quat and its prediction through forward kinematics
        lkh['prop'] = np.zeros_like(self.mu_int[0])

        lkh['prop'][:, 0] = E_g[1] * self.arm.pi_prop

        lkh['vis'] = np.c_[E_g[2] * self.arm.pi_vis, np.zeros([self.arm.n_joints + 1, 4])]

        lkh['forward_ext'] = -E_g[0] * self.arm.pi_ext

        return lkh

    def get_mu_dot(self, lkh, E_mu, obstacle_pos):
        """
        Get belief update
        :param lkh: likelihood components
        :param E_mu: dynamics prediction errors
        :param obstacle_pos: obstacle position
        """
        mu_int_dot = np.zeros_like(self.mu_int)
        mu_ext_dot = np.zeros_like(self.mu_ext)

        # Update likelihoods
        mu_int_dot[0] = self.mu_int[1] + lkh['prop'] + lkh['int'] #- E_mu[0]
        mu_ext_dot[0] = self.mu_ext[1] + lkh['ext'] + lkh['vis'] + lkh['forward_ext'] #- E_mu[1]

        # Intentions
        mu_int_dot[1] -= E_mu[0]
        mu_ext_dot[1] -= E_mu[1]    # TODO check if to add multiplier here as a gain, also Matteo had some

        # Joint limits avoidance
        avoid_j = self.get_rep_joint()
        mu_int_dot[1, :, 0] -= avoid_j

        # Avoid obstacles
        if obstacle_pos is not None:
            avoid_o = self.get_rep_force(obstacle_pos)
            mu_ext_dot[1][:,:3] -= avoid_o

        return mu_int_dot, mu_ext_dot

    def get_a_dot(self, e_prop):
        """
        Get action update
        :param e_prop: proprioceptive error
        """
        return -e_prop

    def integrate(self, mu_int_dot, mu_ext_dot, a_dot):
        """
        Integrate with gradient descent
        :param mu_int_dot: intrinsic belief update
        :param mu_ext_dot: extrinsic belief update
        :param a_dot: action update
        """
        # Update belief
        self.mu_int[0] += self.arm.dt * mu_int_dot[0]
        self.mu_int[0, :, 0] = np.clip(self.mu_int[0, :, 0],
                                           *self.limits.T)
        self.mu_int[1] += self.arm.dt * mu_int_dot[1]
        self.mu_int[0, :, 1] = np.clip(self.mu_int[0, :, 1], -1, 1)     # In Dh table, the links can have negative values

        self.mu_ext[0] += np.array(self.arm.k_mu_ext)*self.arm.dt * mu_ext_dot[0]
        self.mu_ext[1] += np.array(self.arm.k_mu_ext)*self.arm.dt * mu_ext_dot[1]
        
        # Update action
        self.a += self.arm.dt * a_dot
        self.a = np.clip(self.a, -self.arm.a_max, self.arm.a_max)
    
    def init_belief(self, angles):
        """
        Initialize belief
        :param angles: initial arm joint angles
        """

        self.arm.update_with_angles(angles)
        self.mu_int[0, :, 0] = utils.normalize(angles, self.arm.norm_polar)
        self.mu_int[0, :, 1] = self.arm.dh_table[:,0]
        
        if self.arm.lr_length > 0.0:
            self.mu_int[0, :, 1] = np.random.uniform(0, 0.2, size=(self.arm.n_joints))  # Random initialization of link lengths in case of estimation

        self.mu_ext[0] = self.arm.kinematics(angles, self.mu_int[0, :, 1])

        self.prev_cartes_pos = self.mu_ext[0, :, :3].copy()

        # Normalize quaternion
        self.mu_ext[0, :, -4:] = utils.normalize_quat(self.mu_ext[0, :, -4:])  # Normalizes only the quaternion

    def reset_belief(self, arm_angles):
        self.a = np.zeros(self.arm.n_joints)
        self.mu_int = np.zeros((self.arm.n_orders, self.arm.n_joints, 2))    
        self.mu_ext = np.zeros((self.arm.n_orders, self.arm.n_joints + 1, self.arm.n_mu_ext))
        self.init_belief(arm_angles)

    def get_rep_joint(self):
        """
        Compute repulsive force at joint level 
        """
        joint_lim_norm = utils.normalize(self.arm.limits, self.arm.norm_polar)

        gamma = utils.normalize(self.arm.joint_lim_threshold, self.arm.norm_polar)

        # Poisitive joint limits
        error_r = joint_lim_norm[:,1] - self.mu_int[0, :, 0]
       
        error_r_norm = np.maximum(abs(error_r), 0.01)

        rep_force = np.zeros((self.arm.n_joints))
        for j in range(0, self.arm.n_joints):
            if error_r_norm[j] <= gamma:
                rep_force[j] = self.arm.k_rep_j * \
                                   (1 / gamma - 1 / error_r_norm[j])

        # Negative joint limits
        error_r = joint_lim_norm[:,0] - self.mu_int[0, :, 0]
        error_r_norm = abs(error_r)
        error_r_norm = np.maximum(abs(error_r), 0.01)
        for j in range(0, self.arm.n_joints):
            if error_r_norm[j] <= gamma:
                rep_force[j] = -self.arm.k_rep_j * \
                                   (1 / gamma - 1 / error_r_norm[j])

        return  - rep_force

    def get_rep_force(self, obstacle_pos):
        """
        Compute repulsive force in cartesian space for obstacle avoidance
        :param obstacle_pos: obstacle position
        """

        q_star = self.arm.avoid_dist
        rep_force = np.zeros((self.arm.n_joints + 1, 3))
        tot_rep_force = np.zeros((self.arm.n_joints + 1, 3))

        # Loop through the obstacles
        for i in range(len(obstacle_pos)):
            curr_obst_pos_norm = obstacle_pos[i]
            error_r = curr_obst_pos_norm - self.prev_cartes_pos
            error_r_norm = norm(error_r, axis=1)

            # Compute cosine of angle between obstacle and ee velocity
            cosines = utils.compute_joint_obst_cosines(self.cartesian_vels, self.prev_cartes_pos, obstacle_pos[i])
            
            for j in range(1, self.arm.n_joints + 1):
                if error_r_norm[j] < q_star:
                    rep_force[j, :3] = self.arm.k_rep * \
                                    (1 / q_star - 1 / error_r_norm[j]) * \
                                    (1 / error_r_norm[j] ** 2) *\
                                    (error_r[j] / error_r_norm[j])
            
            # Method 0. no cosine no velocity
            rep_force = np.clip(rep_force, -self.arm.max_rep, self.arm.max_rep)
            tot_rep_force += rep_force
        
            # Method 1. only cosine
            # rep_force = rep_force * np.maximum(0., cosines)[:, np.newaxis] # F_rep += F_base * max(0, cos(θ)), so if we move tangent to an obstacle we have no rep force
            # rep_force = np.clip(rep_force, -self.arm.max_rep, self.arm.max_rep)
            # tot_rep_force += rep_force

            # Method 2. only velocity
            # vel_norm = np.clip(np.linalg.norm(self.cartesian_vels, axis=1), 0.001, 10.) 
            # rep_force = rep_force *vel_norm[:, np.newaxis]
            # tot_rep_force +=  rep_force 

            # Method 3. cosine and velocity
            # rep_force = rep_force * np.maximum(0, cosines)[:, np.newaxis] # F_rep += F_base * max(0, cos(θ)), so if we move tangent to an obstacle we have no rep force
            # vel_norm = np.clip(np.linalg.norm(self.cartesian_vels, axis=1), 0.001, 10) 
            # rep_force = rep_force *vel_norm[:, np.newaxis]
            # rep_force = np.clip(rep_force, -self.arm.max_rep, self.arm.max_rep)
            # tot_rep_force += rep_force 
        tot_rep_force = np.clip(rep_force, -self.arm.max_rep, self.arm.max_rep)
        return -tot_rep_force
    
    def compute_cart_vels(self, S):
        # Compute cartesian velocities
        self.cartesian_vels = (S[1] - self.prev_cartes_pos) / 0.01 # TODO Substitute with sim dt
        self.prev_cartes_pos = S[1].copy()

    def inference_step(self, S, target_joint, target_pos, target_ort = None, obstacle_pos = None):
        """
        Run an inference step
        :param S: observations
        :param target_joint: desired joint angles
        :param target_pos: desired link positions
        :param obstacle_pos: obstacle position
        """

        # Compute cartesian velocities for obstacle avoidance 
        self.compute_cart_vels(S)
        
        # Get predictions
        P = self.get_p()

        # Get intentions
        I = self.get_i(target_joint, target_pos, target_ort)

        # Get sensory prediction errors
        E_g = self.get_e_g((self.mu_ext[0], *S), P) # mu_ext[0] is the 0th order extrinsic belief, so position and quaternion, S is the joint angles and the link position xyz

        # Get dynamics prediction errors
        E_mu = self.get_e_mu(I)

        # Get likelihood components
        likelihood = self.get_likelihood(E_g, P)

        # Get belief update
        mu_dot = self.get_mu_dot(likelihood, E_mu, obstacle_pos)

        # Get action update
        a_dot = self.get_a_dot(E_g[1] * self.arm.pi_prop)

        # Update
        self.integrate(*mu_dot, a_dot)

        return utils.denormalize(self.a, self.arm.norm_polar) * self.arm.gain_a, P
