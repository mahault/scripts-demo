import numpy as np
from numpy.linalg import norm


class AgentBase:
    def __init__(self, base):

        self.base = base
        self.a = np.zeros(self.base.n_joints)

        # Initialize belief and action
        self.mu_int = np.zeros((self.base.n_orders, 2))                          # Wheel rotations left and right
        self.mu_ext = np.zeros((self.base.n_orders, 1 + 1, self.base.n_mu_ext))   # pose and velocity, (world joint, ee joint),  xyz quat
        
    def get_i(self, target_joint=None, target_pos=None, target_ort=None):
        """
        Get intentions
        :param target_joint: desired joint angles
        :param target_pos: desired link positions
        """
        self.target_pos = np.array(target_pos)    

        i_int = self.mu_int[0].copy()
        i_ext = self.mu_ext[0].copy()

        if target_joint is not None:
            joint_norm = target_joint
            i_int[:, 0] = joint_norm

        if target_pos is not None:
            i_ext[-1, :2] = target_pos  # Only  position 
        
        if target_ort is not None:
            i_ext[-1, -4:] = target_ort # Only  orientation

        return i_int, i_ext

    def g_ext(self):
        """
        Get extrinsic belief
        """
        mu_int_denorm = self.mu_int[0].copy()
        mu_ext_denormalized = self.mu_ext[0].copy()
        new_mu_ext = self.base.kinematics(mu_int_denorm, mu_ext_denormalized, use_prev_quat=True)
        
        return new_mu_ext

    def grad_ext(self, E_ext, P_ext):
        """
        Get extrinsic gradient
        :param E_ext: extrinsic prediction error
        :param P_ext: extrinsic prediction
        """
        p_ext_denorm = P_ext.copy()

        grad_phi_l, grad_phi_r = self.base.kin_grad_phi(p_ext_denorm[1, -4:])  # TODO this should be the previous quat
        
        lkh_int = np.zeros(2) 
        new_e_ext = np.zeros_like(E_ext[1])
        new_e_ext[:2] = self.base.k_err_cart*E_ext[1][:2]
        new_e_ext[3:] = self.base.k_err_ort*E_ext[1][3:]
        lkh_int[0] = grad_phi_l.dot(new_e_ext)
        lkh_int[1] = grad_phi_r.dot(new_e_ext)
        return lkh_int * self.base.pi_ext

    def get_p(self):
        """
        Get predictions
        """
        p_ext = self.g_ext()                    # Forward kinematics [xyz quat] predicted from extrinsic beliefs 
        p_prop = self.mu_int[0].copy()          # Wheel rotations left and right
        p_vis = self.mu_ext[0, :, :3].copy()    # [xyx] base

        return p_ext, p_prop, p_vis

    def get_e_g(self, S, P):
        """
        Get sensory prediction errors
        :param S: observations
        :param P: predictions
        """
        E_g = [s - p for s, p in zip(S, P)]
        #print('E_g', E_g[0])
        return E_g

    def get_e_mu(self, I):
        """
        Get dynamics prediction errors
        :param I: intentions
        """

        E_i = [(I[0] - self.mu_int[0]) * self.base.k_int,
               (I[1] - self.mu_ext[0]) * self.base.k_ext]

        return self.mu_int[1] - E_i[0], self.mu_ext[1] - E_i[1]

    def get_likelihood(self, E_g, P):
        """
        Get likelihood components
        :param E_g: sensory prediction errors for each joint, [pos quat] [joint angles] [links pos]
        :param P: predictions
        """
        lkh = {}

        lkh['int'] = self.grad_ext(E_g[0], P[0])   # E_g[0] and P[0] represents respectively the sensory prediction errors for xyz quat and its prediction through forward kinematics
        lkh['prop'] = np.zeros_like(self.mu_int[0])

        lkh['prop'] = E_g[1] * self.base.pi_prop

        lkh['vis'] = np.c_[E_g[2] * self.base.pi_vis, np.zeros([2, 4])] # Change to 4 instead of 1 if we have quaternions

        lkh['forward_ext'] = -E_g[0] * self.base.pi_ext

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
        mu_int_dot[0] = self.mu_int[1] + lkh['prop'] + lkh['int'] - E_mu[0]
        mu_ext_dot[0] = self.mu_ext[1] + lkh['vis'] + lkh['forward_ext'] - E_mu[1]

        # Intentions
        mu_int_dot[1] -= E_mu[0]
        mu_ext_dot[1] -= E_mu[1]    # TODO check if to add multiplier here as a gain, also Matteo had some

        # # Avoid obstacles
        # if obstacle_pos is not None:
        #     avoid_o = self.get_rep_force(obstacle_pos)
        #     mu_ext_dot[1][:,:3] -= avoid_o

        # if self.target_pos is not None:
        #     mu_ext_dot[1][-1,:2] = self.get_attr_force(self.target_pos)
        # print('mu_ext_dot', mu_ext_dot[1][-1,:2])
        return mu_int_dot, mu_ext_dot

    def get_a_dot(self, e_prop):
        """
        Get action update
        :param e_prop: proprioceptive error
        """
        return - e_prop

    def integrate(self, mu_int_dot, mu_ext_dot, a_dot):
        """
        Integrate with gradient descent
        :param mu_int_dot: intrinsic belief update
        :param mu_ext_dot: extrinsic belief update
        :param a_dot: action update
        """

        # Save current beliefs for next run kinematic computation
        self.base.set_prev_phi(self.mu_int[0][0], self.mu_int[0][1])

        # Update belief
        self.mu_int[0] += self.base.dt * mu_int_dot[0]*1
       
        self.mu_int[1] += self.base.dt * mu_int_dot[1]

        self.mu_ext[0] += np.array(self.base.k_mu_ext)*self.base.dt * mu_ext_dot[0]
        self.mu_ext[1] += np.array(self.base.k_mu_ext)*self.base.dt * mu_ext_dot[1]
        
        # Update action
        self.a += self.base.dt * a_dot
        self.a = np.clip(self.a, -self.base.a_max, self.base.a_max)
    
    def init_belief(self, angles):
        """
        Initialize belief
        :param angles: initial base joint angles
        """
        self.mu_int[0] = angles
        
        self.mu_ext[0, :, :] = self.base.kinematics(angles)

    def get_attr_force(self, goal_pos):
        """
        Compute attractive force in cartesian space for speed
        :param obstacle_pos: obstacle position
        """

        k = 0

        curr_goal_pos_norm = goal_pos
        error_r = curr_goal_pos_norm - self.prev_cartes_pos[:2]
        error_r_norm = norm(error_r)
        
        q_star = 0.05
        if error_r_norm < q_star:
            k = error_r_norm * k / q_star

        attr_force = k*error_r / error_r_norm
        # attr_force = np.clip(attr_force, -self.arm.max_rep, self.arm.max_rep)
        print('attr_force', attr_force)
        return attr_force

    def get_rep_force(self, obstacle_pos):
        """
        Compute repulsive force in cartesian space for obstacle avoidance
        :param obstacle_pos: obstacle position
        """

        pos_norm = obstacle_pos

        q_star = self.base.avoid_dist
        rep_force = np.zeros((self.base.n_joints + 1, 3))

        # Loop through the obstacles
        for i in range(len(pos_norm)):
            curr_obst_pos_norm = pos_norm[i]
            error_r = curr_obst_pos_norm - self.mu_ext[0, :, :3]
            error_r_norm = norm(error_r, axis=1)
            
            for j in range(1, self.base.n_joints + 1):
                if error_r_norm[j] < q_star:
                    rep_force[j, :3] = self.base.k_rep * \
                                    (1 / q_star - 1 / error_r_norm[j]) * \
                                    (1 / error_r_norm[j] ** 2) *\
                                    (error_r[j] / error_r_norm[j])
                    
        return self.mu_ext[1][:,:3] - np.clip(rep_force, -self.base.max_rep, self.base.max_rep)
    
    def inference_step(self, S, target_joint, target_pos, target_ort = None, obstacle_pos = None):
        """
        Run an inference step
        :param S: observations
        :param target_joint: desired joint angles
        :param target_pos: desired link positions
        :param obstacle_pos: obstacle position
        """
        # Get predictions
        P = self.get_p()

        # Get intentions
        I = self.get_i(target_joint, target_pos, target_ort)

        obs_ext = np.zeros([self.base.n_orders, self.base.n_mu_ext])
        obs_ext[0] = self.mu_ext[0][0]
        obs_ext[-1, :] = np.append([S[1][0], S[1][1], 0], self.mu_ext[0][1][-4:]) # xyz quat
        #obs_ext[-1, :] = np.append(self.mu_ext[0][1][:3], self.mu_ext[0][1][-4:]) # xyz quat
        
        obs_joint = S[0]
        obs_vis = np.zeros([self.base.n_orders, 3])
        obs_vis[1] = S[1]
        self.prev_cartes_pos = np.array(S[1])

        # Get sensory prediction errors
        E_g = self.get_e_g((obs_ext, obs_joint, obs_vis), P)

        # Get dynamics prediction errors
        E_mu = self.get_e_mu(I)

        # Get likelihood components
        likelihood = self.get_likelihood(E_g, P)

        # Get belief update
        mu_dot = self.get_mu_dot(likelihood, E_mu, obstacle_pos)

        # Get action update
        a_dot = self.get_a_dot(E_g[1] * self.base.pi_prop)

        # Update
        self.integrate(*mu_dot, a_dot)

        return self.a* self.base.gain_a, P
