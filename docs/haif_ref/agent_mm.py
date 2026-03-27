import numpy as np
from numpy.linalg import norm
import haif_robot_control.helpers.utils as utils


class AgentMM:
    def __init__(self, base, arm):
        
        self.base = base
        self.arm = arm
        self.limits = arm.limits
        self.a = np.zeros(2 + self.arm.n_joints)

        # Initialize belief and action
        self.mu_int = np.zeros((self.arm.n_orders, 2 + self.arm.n_joints, 2))   # Joint angle and joint length
        self.mu_ext = np.zeros((self.arm.n_orders, 3 + self.arm.n_joints, self.arm.n_mu_ext))   # xyz quat
        
        self.base_obs = np.zeros([self.base.n_orders, self.base.n_mu_ext])
        self.base_obs_vis = np.zeros([self.base.n_orders, 3])
        self.arm_to_base_err = np.zeros(self.arm.n_mu_ext)

        self.cartesian_vels = None
        self.prev_cartes_pos = None
        self.k_int = self.arm.k_int_only_arm
        self.alpha = 0.03 # k_int smoothing factor

    def get_i(self, target_joint=None, target_pos_arm=None, target_ort_arm=None, target_pos_base=None, target_ort_base=None):
        """
        Get intentions
        :param target_joint: desired joint angles
        :param target_pos_arm: desired link positions
        """
        self.target = target_pos_arm
        i_int = self.mu_int[0].copy()
        i_ext = self.mu_ext[0].copy()

        if target_joint is not None:
            joint_norm = utils.normalize(target_joint, self.arm.norm_polar)
            i_int[2:, 0] = joint_norm
            
        if target_pos_arm is not None:
            i_ext[-1, :3] = target_pos_arm  # Only final position 

        if target_ort_arm is not None:
            i_ext[-1, -4:] = np.array(target_ort_arm) # Only final orientation

        # If a base goal is set, keep track to prevent conflicting goals with cartesian ones for the arm
        self.base_goal_active = False
        if target_pos_base is not None:
            self.base_goal_active = True
            i_ext[1, :2] = target_pos_base
        
        if target_ort_base is not None:
            self.base_goal_active = True
            i_ext[1, -4:] = target_ort_base

        new_k_int = self.arm.k_int_only_arm / 3.0 if target_pos_arm is not None else (
                    self.arm.k_int_with_base if self.base_goal_active else self.arm.k_int_only_arm
                    )

        # Change k_int smoothly
        if new_k_int > self.k_int:
            self.k_int = (1 - self.alpha) * self.k_int + self.alpha * new_k_int
        else:
            self.k_int = new_k_int
        return i_int, i_ext

    def g_ext_arm(self):
        """
        Get extrinsic belief
        """
        mu_int_denorm = self.mu_int[0][2:].copy()
        mu_int_denorm[:, 0] = utils.denormalize(mu_int_denorm[:, 0],
                                                self.arm.norm_polar)

        mu_ext_normalized = self.mu_ext[0][2:].copy()

        # Make sure quaternion is unit length 
        mu_ext_normalized[:, -4:] = utils.normalize_quat(mu_ext_normalized[:, -4:])

        new_mu_ext = self.arm.kinematics(*mu_int_denorm.T, mu_ext_normalized, init_pose=self.base.poses[1])

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
            self.mu_int[0, -1, 1] = 0.0  # Exclude last link length from optimization. This leads to strange convergence. Might add a ghost link in case we have to use a tool as ee.

        grad_theta = np.array([self.arm.kin_grad_theta(theta, length, alpha, q) for theta, length, alpha, q in
                               zip(self.mu_int[0, 2:, 0], self.mu_int[0, 2:, 1], self.arm.dh_table[:,1], p_ext_denorm[1:, -4:])])
        
        grad_length = np.array([self.arm.kin_grad_l(theta, q) for theta, q in
                                zip(self.mu_int[0, 2:, 0], p_ext_denorm[1:, -4:])])
        
        grad_length *= self.arm.lr_length
        
        grad_ext = np.array([self.arm.kin_grad_ext(theta, length, alpha, d, q) for theta, length, alpha, d, q in
                               zip(self.mu_int[0, 2:, 0], self.mu_int[0, 2:, 1], self.arm.dh_table[:,1], self.arm.dh_table[:,2], p_ext_denorm[1:, -4:])])

        lkh_int = np.zeros((self.arm.n_joints, 2)) 
        for i in range(self.arm.n_joints):
            lkh_int[i, 0] = grad_theta[i].dot(E_ext[i + 1])
            lkh_int[i, 1] = grad_length[i].dot(E_ext[i + 1])

       
        # This shoule become the goal for the base
        self.arm_to_base_err = E_ext[1]

        lkh_ext = np.zeros_like(self.mu_ext[0][2:])
        for j in range(1, self.arm.n_joints):
            lkh_ext[j] = grad_ext[j].dot(E_ext[j + 1])
        # print('err', self.arm_to_base_err)
        return lkh_int * self.arm.pi_ext, lkh_ext * self.arm.pi_ext

    def g_ext_base(self):
        """
        Get extrinsic belief
        """
        mu_int_ = self.mu_int[0][:2][:, 0].copy()
        mu_ext_ = self.mu_ext[0][:2].copy()
        new_mu_ext = self.base.kinematics(mu_int_, mu_ext_, use_prev_quat=True)
        
        return new_mu_ext

    def get_p(self):
        """
        Get predictions
        """

        p_ext = np.zeros_like(self.mu_ext[0, :, :])
        p_prop = np.zeros_like(self.mu_int[0, :, 0])
        p_vis = np.zeros_like(self.mu_ext[0, :, :3])

        p_ext[2:] = self.g_ext_arm()   # Forward kinematics [xyz q] predicted from joint angles 
        p_ext[:2] = self.g_ext_base()  # Forward kinematics [xyz q] predicted from joint angles
        p_prop = self.mu_int[0, :, 0].copy()  # Joint angles [theta]
        p_vis = p_ext[:, :3].copy()    # [xyz]

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

        E_i = [(I[0] - self.mu_int[0]) * self.k_int,
               (I[1] - self.mu_ext[0]) * self.arm.k_ext]

        return self.mu_int[1] - E_i[0], self.mu_ext[1] - E_i[1]


    def grad_ext_base(self, E_ext, P_ext):
        """
        Get extrinsic gradient
        :param E_ext: extrinsic prediction error
        :param P_ext: extrinsic prediction
        """
        p_ext_denorm = P_ext.copy()

        grad_phi_l, grad_phi_r = self.base.kin_grad_phi(p_ext_denorm[1, -4:])  # TODO this should be the previous quat

        lkh_int = np.zeros((2, 2))
        # lkh_int[0][0] = grad_phi_l.dot(E_ext[1])
        # lkh_int[1][0] = grad_phi_r.dot(E_ext[1])

        E_ext_arm = np.zeros_like(E_ext[1])
        if not self.base_goal_active:
            E_ext_arm[:2] = -self.arm_to_base_err[:2] * self.arm.arm_to_base_k_linear
            E_ext_arm[3:] = self.arm_to_base_err[3:] * self.arm.arm_to_base_k_quat
        else:
            # E_ext_arm[:2] = self.base.k_err_cart*E_ext[1][:2]  - self.arm_to_base_err[:2] * self.arm.arm_to_base_k_linear
            # E_ext_arm[3:] = self.base.k_err_ort*E_ext[1][3:]  + self.arm_to_base_err[3:] * self.arm.arm_to_base_k_quat
            E_ext_arm[:2] = self.base.k_err_cart*E_ext[1][:2]  - 0.01*self.arm_to_base_err[:2] * self.arm.arm_to_base_k_linear
            E_ext_arm[3:] = self.base.k_err_ort*E_ext[1][3:]  + 0.01*self.arm_to_base_err[3:] * self.arm.arm_to_base_k_quat
        lkh_int[0][0] = grad_phi_l.dot(E_ext_arm)
        lkh_int[1][0] = grad_phi_r.dot(E_ext_arm)

        # err = np.zeros_like(E_ext[1])
        # err[0] = 10
        # lkh_int[0] = grad_phi_l.dot(err)
        # lkh_int[1] = grad_phi_r.dot(err)

        return lkh_int * self.base.pi_ext

    def get_likelihood(self, E_g, P):
        """
        Get likelihood components
        :param E_g: sensory prediction errors for each joint, [pos quat] [joint angles] [links pos]
        :param P: predictions
        """
        lkh = {}

        lkh['int'] = np.zeros_like(self.mu_int[0])
        lkh['ext'] = np.zeros_like(self.mu_ext[0])
        lkh['prop'] = np.zeros_like(self.mu_int[0])
        lkh['forward_ext'] = np.zeros_like(self.mu_ext[0])
        lkh['vis'] = np.zeros_like(self.mu_ext[0])

        # Arm
        lkh['int'][2:], lkh['ext'][2:] = self.grad_ext(E_g[0][2:], P[0][2:]) # E_g[0] and P[0] represents respectively the sensory prediction errors for xyz quat and its prediction through forward kinematics

        lkh['prop'][2:, 0] = E_g[1][2:] * self.arm.pi_prop

        lkh['vis'][2:] = np.c_[E_g[2][2:] * self.arm.pi_vis, np.zeros([self.arm.n_joints + 1, 4])]

        lkh['forward_ext'][2:] = -E_g[0][2:] * self.arm.pi_ext

        # Base
        lkh['int'][:2] = self.grad_ext_base(E_g[0][:2], P[0][:2]) # TODO Corrado: if you do like this you cannot add goals to the base only

        lkh['prop'][:2, 0] = E_g[1][:2] * self.base.pi_prop

        lkh['vis'][:2] = np.c_[E_g[2][:2] * self.base.pi_vis, np.zeros([2, 4])] # Change to 4 instead of 1 if we have quaternions

        lkh['forward_ext'][:2] = -E_g[0][:2] * self.base.pi_ext
        # lkh['forward_ext'][:2] = E_g[0][:2] * self.base.pi_ext
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
        mu_int_dot[0] = self.mu_int[1] + lkh['prop'] + lkh['int'] 
        mu_ext_dot[0] = self.mu_ext[1] + lkh['ext'] + lkh['vis'] + lkh['forward_ext'] 

        # TODO check here for partrial derivatives of f
        # mu_int_dot[0][:2] -= E_mu[0][:2]
        # mu_ext_dot[0][:2] -= E_mu[1][:2]

        # mu_int_dot[0][2:] += E_mu[0][2:]
        # mu_ext_dot[0][2:] += E_mu[1][2:]

        # Intentions
        mu_int_dot[1] -= E_mu[0]
        mu_ext_dot[1] -= E_mu[1]   # TODO check if to add multiplier here as a gain, also Matteo had some

        # Joint limits avoidance
        avoid_j = self.get_rep_joint()
        mu_int_dot[1, 2:, 0] -= avoid_j

        # # Avoid obstacles
        if obstacle_pos is not None:
            avoid_o = self.get_rep_force(obstacle_pos)
            mu_ext_dot[1][-1,:3] -= avoid_o[-1]
            mu_ext_dot[1][-self.arm.n_joints,:1] -= avoid_o[0][:1] # Base collision avoidance # TODO Corrado, move the avoidance force to the base, changing the way we propagate back pred errors in grad_ext to allow for also base goals etc
            
            # mu_ext_dot[1][1,:2] -= avoid_o[0][:2]
            # yaw = utils.quaternion_to_euler(self.mu_ext[0][2][-4:])[2]
            # print("yaw sign", np.sign(yaw))
            # print("rep y sign", np.sign(avoid_o[0][1]))
            # print("yaw ", yaw)
            # print("rep x", avoid_o[0][0])
            
            # mu_ext_dot[1][-self.arm.n_joints:,:3] -= avoid_o[-self.arm.n_joints:]
            # mu_ext_dot[1][1,0] -= avoid_o[0][0] # Repulsive force for the base fwd component. The diff drive robot cannot move laterally, this should be translated to rotation
        
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

        # Save current beliefs for next run kinematic computation
        self.base.set_prev_phi(self.mu_int[0][0][0], self.mu_int[0][1][0])

        # Update belief
        self.mu_int[0] += self.arm.dt * mu_int_dot[0]
        # self.mu_int[0, :, 0] = np.clip(self.mu_int[0, :, 0], *self.limits.T)
        self.mu_int[1] += self.arm.dt * mu_int_dot[1]
        self.mu_int[0, 2:, 1] = np.clip(self.mu_int[0, 2:, 1], -1, 1)     # In Dh table, the links can have negative values

        self.mu_ext[0] += np.array(self.arm.k_mu_ext)*self.arm.dt * mu_ext_dot[0]
        self.mu_ext[1] += np.array(self.arm.k_mu_ext)*self.arm.dt * mu_ext_dot[1]
        
        # Update action
        self.a += self.arm.dt * a_dot
        self.a[2:] = np.clip(self.a[2:], -self.arm.a_max, self.arm.a_max)
        self.a[:2] = np.clip(self.a[:2], -self.base.a_max, self.base.a_max)
    
    def init_belief(self, base_angles, arm_angles, init_pose=None):
        """
        Initialize belief
        :param angles: initial arm joint angles
        """

        self.mu_int[0][:2][0] = base_angles
        if init_pose is not None:
            init_pose_w_world = np.vstack(([0., 0., 0., 1., 0., 0., 0.], init_pose))    # Stack world frame to the base pose
        else:
            init_pose_w_world = None
        self.mu_ext[0, :2, :] = self.base.kinematics(base_angles, init_pose_w_world)

        self.mu_int[0, 2:, 0] = utils.normalize(arm_angles, self.arm.norm_polar)
        self.mu_int[0, 2:, 1] = self.arm.dh_table[:,0]

        if self.arm.lr_length > 0.0:
            self.mu_int[0, 2:, 1] = np.random.uniform(0, 0.2, size=(self.arm.n_joints))  # Random initialization of link lengths in case of estimation

        self.mu_ext[0][2:] = self.arm.kinematics(arm_angles, self.mu_int[0, 2:, 1], init_pose=init_pose)
        
        # Normalize quaternion
        self.mu_ext[0, 2:, -4:] = utils.normalize_quat(self.mu_ext[0, 2:, -4:])  # Normalizes only the quaternion

        self.prev_cartes_pos = self.mu_ext[0, 2:, :3].copy()
    
    def reset_belief(self, base_angles, arm_angles, init_pose=None):
        self.a = np.zeros(2 + self.arm.n_joints)
        self.mu_int = np.zeros((self.arm.n_orders, 2 + self.arm.n_joints, 2))                       
        self.mu_ext = np.zeros((self.arm.n_orders, 3 + self.arm.n_joints, self.arm.n_mu_ext)) 
        self.init_belief(base_angles, arm_angles, init_pose)

    def get_rep_joint(self):
        """
        Compute repulsive force at joint level 
        """
        joint_lim_norm = utils.normalize(self.arm.limits, self.arm.norm_polar)

        gamma = utils.normalize(self.arm.joint_lim_threshold, self.arm.norm_polar)

        # Poisitive joint limits
        error_r = joint_lim_norm[:,1] - self.mu_int[0, 2:, 0]
       
        error_r_norm = np.maximum(abs(error_r), 0.01)

        rep_force = np.zeros((self.arm.n_joints))
        for j in range(0, self.arm.n_joints):
            if error_r_norm[j] <= gamma:
                rep_force[j] = self.arm.k_rep_j * \
                                   (1 / gamma - 1 / error_r_norm[j])

        # Negative joint limits
        error_r = joint_lim_norm[:,0] - self.mu_int[0, 2:, 0]
        error_r_norm = abs(error_r)
        error_r_norm = np.maximum(abs(error_r), 0.01)
        for j in range(0, self.arm.n_joints):
            if error_r_norm[j] <= gamma:
                rep_force[j] = -self.arm.k_rep_j * \
                                   (1 / gamma - 1 / error_r_norm[j])

        return self.mu_int[1, 2:, 0] - rep_force

    def get_rep_force(self, obstacle_pos):
        """
        Compute repulsive force in cartesian space for obstacle avoidance
        :param obstacle_pos: obstacle position
        """

        q_star = self.avoid_dist
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
                    rep_force[j, :3] = self.k_rep * \
                                    (1 / q_star - 1 / error_r_norm[j]) * \
                                    (1 / error_r_norm[j] ** 2) *\
                                    (error_r[j] / error_r_norm[j])
            
            # Base repulsive force
            obstacle_pos_at_base = obstacle_pos[i] + np.array([0., 0., self.arm.z_offset])
            error_r = obstacle_pos_at_base - self.prev_cartes_pos[0]
            error_r_norm = norm(error_r)
            if error_r_norm < self.rep_dist_base:
                rep_force[0, :3] =  self.k_rep_base * \
                                    (1 / self.rep_dist_base - 1 / error_r_norm) * \
                                    (1 / error_r_norm ** 2) *\
                                    (error_r / error_r_norm)
            # Method 0. no cosine no velocity
            rep_force = np.clip(rep_force, -self.max_rep, self.max_rep)
            tot_rep_force += rep_force

            # Method 1. only cosine
            # rep_force = rep_force * np.maximum(0., cosines)[:, np.newaxis] # F_rep += F_base * max(0, cos(θ)), so if we move tangent to an obstacle we have no rep force
            # tot_rep_force += rep_force

            # Method 2. only velocity
            # vel_norm = np.clip(np.linalg.norm(self.cartesian_vels, axis=1), 0.01, 1.) 
            # # print("vel norm", vel_norm)
            # rep_force = rep_force *vel_norm[:, np.newaxis]
            # tot_rep_force +=  rep_force 
            # Method 3. cosine and velocity
            # rep_force = rep_force * np.maximum(0, cosines)[:, np.newaxis] # F_rep += F_base * max(0, cos(θ)), so if we move tangent to an obstacle we have no rep force
            # vel_norm = np.clip(np.linalg.norm(self.cartesian_vels, axis=1), 0.001, 10) 
            # rep_force = rep_force *vel_norm[:, np.newaxis]
            # rep_force = np.clip(rep_force, -self.max_rep, self.max_rep)
            # tot_rep_force += rep_force 

        tot_rep_force = np.clip(rep_force, -self.max_rep, self.max_rep)
        return - tot_rep_force

    def compute_cart_vels(self, S):
        # Compute cartesian velocities
        curr_pos = np.vstack((S[1].copy(), S[3][1:].copy()))
        self.cartesian_vels = (curr_pos - self.prev_cartes_pos) / 0.01 # TODO Substitute with sim dt
        self.prev_cartes_pos = curr_pos

    def update_rep_params(self, rep_params):
        if rep_params is None:
            self.k_rep = self.arm.k_rep
            self.k_rep_base = self.arm.k_rep_base
            self.avoid_dist = self.arm.avoid_dist
            self.rep_dist_base = self.arm.rep_dist_base
            self.max_rep = self.arm.max_rep
        else:
            self.k_rep = rep_params.get('k_rep', self.arm.k_rep)
            self.k_rep_base = rep_params.get('k_rep_base', self.arm.k_rep_base)
            self.avoid_dist = rep_params.get('avoid_dist', self.arm.avoid_dist)
            self.rep_dist_base = rep_params.get('rep_dist_base', self.arm.rep_dist_base)
            self.max_rep = rep_params.get('max_rep', self.arm.max_rep)

    def inference_step(self, S, target_joint, target_pos_arm, target_ort_arm = None, obstacle_pos = None, rep_params = None, target_pos_base = None, target_ort_base = None):
        """
        Run an inference step
        :param S: observations
        :param target_joint: desired joint angles
        :param target_pos_arm: desired link positions
        :param obstacle_pos: obstacle position
        """

        # Update repulsive parameters
        self.update_rep_params(rep_params)

        # Compute cartesian velocities for obstacle avoidance 
        self.compute_cart_vels(S)

        # Re-arrange observations
        self.base_obs[0] = self.mu_ext[0][0].copy()
        self.base_obs[-1, :] = np.append(S[1], self.mu_ext[0][1][-4:]) # xyz quat
        self.base_obs_vis[1] = S[1]
        self.mu_ext[0][2] = self.base.poses[-1].copy() # Force world frame of te arm to follow the base
        base_obs = np.append(self.base_obs, [self.base.poses[-1]], axis=0)
        obs_ext = np.append(base_obs, self.mu_ext[0][3:], axis=0)
        obs_int = np.append(S[0], S[2])
        obs_vis = np.append(self.base_obs_vis, S[3], axis=0)

        # Get predictions
        P = self.get_p()

        # Get intentions
        I = self.get_i(target_joint, target_pos_arm, target_ort_arm, target_pos_base, target_ort_base)

        # Get sensory prediction errors
        E_g = self.get_e_g((obs_ext, obs_int, obs_vis), P) # mu_ext[0] is the 0th order extrinsic belief, so position and quaternion, S is the joint angles and the link position xyz
     
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

        arm_action = utils.denormalize(self.a[2:], self.arm.norm_polar)
        action = np.append(self.a[:2], arm_action)

        return  action * self.arm.gain_a, P