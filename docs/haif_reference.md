# HAIF Robot Control - Reference Patterns

Source: `C:\Users\mahau\OneDrive\Desktop\projects\haif-robot-control-main\`

## Core Pattern: Plan → Filter → Execute Open-Loop

HAIF separates planning from execution. Active inference generates a full
trajectory during planning, then the trajectory executes without interruption.

### Position Control Mode (robot_arm.py)

```python
def _compute_pos_action(self, S):
    curr_angles_deg = self.current_angles.copy()
    joint_traj = []
    dist_to_goal = np.linalg.norm(visual_obs[-1] - self.goal_ee)
    start_time = time.time()

    # PHASE 1: PLANNING (forward simulation, no robot motion)
    while dist_to_goal > self.reach_threshold and not timeout:
        S = self.get_S()
        action_ai, _ = self.agent.inference_step(S, goal_joints, goal_ee, ...)
        curr_angles_deg = curr_angles_deg + 0.3 * action_ai
        joint_traj.append(curr_angles_deg)
        self.arm_controller.update_with_angles(curr_angles_deg)
        dist_to_goal = np.linalg.norm(visual_obs[-1] - goal_ee)
        if time.time() - start_time > 6.0:
            timeout = True

    # PHASE 2: FILTER (reduce waypoints)
    joint_traj = filter_joint_traj(joint_traj, min_deg=3.0)

    # PHASE 3: EXECUTE (open-loop, no interruption)
    for element in joint_traj:
        send_angles(angle2steps(element))

    # PHASE 4: BELIEF RESET (synchronize with reality)
    agent.reset_belief(read_angles())
```

### Inference Step (agent_arm.py)

Each step of active inference:
1. Get predictions from current beliefs (forward kinematics)
2. Get intentions from targets (goal positions/orientations)
3. Compute sensory prediction errors (observation - prediction)
4. Compute dynamics prediction errors (intention - belief velocity)
5. Get likelihood components (gradient of errors w.r.t. beliefs)
6. Compute belief updates (mu_dot)
7. Compute action update (a_dot from proprioceptive error)
8. Integrate (update beliefs and actions with dt)

### Trajectory Filtering (utils.py)

```python
def filter_joint_traj(traj, min_deg=3.0):
    """Keep waypoints at least min_deg apart in joint-space norm."""
    filtered = [traj[0]]
    for point in traj[1:-1]:
        if np.linalg.norm(point - filtered[-1]) >= min_deg:
            filtered.append(point)
    filtered.append(traj[-1])
    return filtered
```

### Belief Reset (agent_arm.py)

Called AFTER execution completes:
```python
def reset_belief(self, arm_angles):
    self.a = np.zeros(self.arm.n_joints)
    self.mu_int = np.zeros(...)
    self.mu_ext = np.zeros(...)
    self.init_belief(arm_angles)  # from actual measured state
```

## Key Design Principles

1. **Commitment**: Once a trajectory is computed, execute it fully
2. **No re-planning during execution**: Beliefs are not updated mid-motion
3. **Timeout safety**: 6-second planning limit prevents infinite loops
4. **Belief-reality sync**: Reset beliefs to measured state after execution
5. **Action clipping**: Safety bounds on joint angles and action magnitude

## How We Applied This to Social Layer

In `unified_task_controller.py`:
- `_committed_req`: when PICKUP/PLACE is selected, lock the SkillRequest
- `notify_status()`: clears commitment on SUCCESS/FAILURE/TIMEOUT
- Executive suppresses `avoid` intent during commitment (but not emergency_stop)
- This prevents the 5Hz deliberation loop from preempting arm motions
