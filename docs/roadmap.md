# Social-Layer Roadmap

## Phase 1: Core Interface Spec + Prototype (DONE)

Branch: `feat/social-layer-interface`

### Completed

- **Safety foundation**: `geometry.py` (2D geometry utilities), `constraints.py` (constraint data types)
- **Sensor interface**: `base_sensors.py` abstract `SensorInterface` with `read() -> dict` contract
- **Proxemics augmentation**: Hall's zones (intimate/personal/social/public), `ProxemicsAugmentation.augment()`
- **Perception pipeline**: Augmentation chain pattern; sensors + augmentations -> `PerceptBundle`
- **Script types + manager**: `ScriptSequence`, `ScriptStep`, status-based advancement, looping
- **Norm rules + engine**: `PersonalSpaceRule`, `KeepoutZoneRule`, `SpeedLimitRule`; merge (most restrictive wins), first veto wins
- **ToM planner adapter**: Wraps `tom_planner.ToMPlanner` from Alignment-experiments; classifies planner output into `approach/avoid/yield/wait` via geometric heuristic
- **ToM modulator**: Delegates to adapter with graceful degradation
- **Safety shield**: Emergency stop (< 0.3m), speed cap enforcement, keepout zone passthrough
- **Executive lifecycle**: `_start_skill`, `_stop_active_skill`, `_request_changed`; handles terminal statuses, notifies scripts
- **Main entry point**: Full wiring with MockSensors for headless testing
- **CI/CD**: GitHub Actions with pytest (Python 3.10-3.12) + ruff linting
- **Tests**: 30 unit + integration tests (all passing)

---

## Phase 2: Webots TIAGo Plugin + Warehouse Integration (DONE)

### Completed

- [x] Create `plugins/tiago_webots/` plugin directory structure
- [x] Extract TIAGo motor control into `robot/driver.py` (from `tiago_empathic.py`)
- [x] Extract TIAGo sensor reading into `perception/sensors.py` (from `tiago_empathic.py`)
- [x] Implement `TiagoNavSkill` (Skill interface wrapping driver)
- [x] Implement `TiagoNavIntentPolicy` (4 intents -> speed/distance params)
- [x] Create `plugin.py` registration entry point
- [x] Create `controller.py` Webots entry point (replaces monolithic `tiago_empathic.py`)
- [x] Create warehouse world file (`worlds/tiago_warehouse.wbt`)
- [x] Create stress test script (`scripts/stress_test.py`) for alpha sweep
- [ ] Validate end-to-end in Webots: both robots navigate, empathic yields, no collisions

---

## Phase 3: Augmentations (DONE)

- [x] Affect augmentation — Circumplex model (Pattisapu & Albarracin 2024), POMDP belief dynamics per agent
- [x] Engagement augmentation — gaze, interaction duration, proximity trends
- [x] Saliency augmentation — proximity/velocity/novelty/threat scoring into `pb.attention`
- [x] Wire augmentations into norm engine, safety shield, and ToM modulator (via Executive empathy + intent inference hooks)
- [x] Tests: 26 augmentation tests (all passing)

---

## Phase 4: Variational Scripts + Empathic Modulator (DONE)

### 4a. Variational script framework

- [x] Enriched `ScriptStep` with `gate` (mandatory/flexible), `deontic` mode, `expected_affect`, `expected_situation`
- [x] `SituationType` dataclass — A-matrix conceptual clusters for event-type recognition
- [x] `WeakScriptRecognizer` — softmax over feature-weighted situation types with Bayesian prior
- [x] `ScriptManager` violation detection — KL divergence between expected and observed situation
- [x] Repair script injection — activates de-escalation sequence on violation, resumes normal script after
- [x] Core types: `AffectState`, `ScriptViolation`, `DeonticMode`
- [x] Blackboard: `self_affect`, `latest_violation`, `deontic_context`

### 4b. Empathic modulator

- [x] `EmpathicModulator` with predict-observe-update loop
- [x] Robot self-affect engine — own circumplex state with decay toward neutral
- [x] Empathic contagion — `dv = alpha * (mean_other_valence - self_valence)`, capped
- [x] Violation coupling — arousal boost + valence penalty proportional to KL divergence
- [x] Affect → EFE modulation — negative valence reduces speed, high arousal adds epistemic boost
- [x] Distress interrupt — extreme valence + arousal triggers `recommend_interrupt`

### 4c. Affect-modulated norms

- [x] `AffectModulatedSpeedRule` — reduces speed cap proportionally to negative valence
- [x] `DistressVetoRule` — hard veto when both valence < -0.8 AND arousal > 0.8
- [x] Executive wiring — empathy predict/observe/modulate in tick(), affect pushed to norm rules

### 4d. Fused intent inference (engineering approximation)

- [x] `IntentInference` — log-linear product-of-experts fusing kinematic ToM, engagement, affect, script context
- [x] Script-conditioned intent priors — situation type biases intent distribution
- [x] `ToMModulator` extended with `IntentInference` + `set_situation_type()`
- [x] Executive feeds recognized situation type to ToM on each deliberation tick
- [x] Tests: 57 new tests (variational scripts, empathic modulator, affect rules, intent inference)

### Theoretical grounding

- Albarracin, Constant, Friston & Ramstead (2021) — "A Variational Approach to Scripts"
- Pattisapu, Verbelen, Pitliya, Kiefer & Albarracin (2024) — "Free Energy in a Circumplex Model of Emotion"
- Script violation = variational free energy increase = prediction error
- Empathic contagion = coupling term in generative model

---

## Phase 5: Active Inference ToM (DONE)

Replace the Phase 4 log-linear intent fusion with proper active inference
throughout.  Port patterns from empathy-prisoner-dilemma into the robotics
context.

### 5a. Intent particle filter (`cognition/tom/intent_particle_filter.py`)

Active inference opponent model adapted from `OpponentInversion`:

- [x] `IntentProfile` — per-agent behavioral parameters:
  - `approach_bias`: base tendency to approach (logit scale, like alpha)
  - `responsiveness`: sensitivity to robot's actions (reciprocity analogue)
  - `precision`: action determinism / inverse temperature (beta)
  - `empathy_j`: care for robot wellbeing (lambda_j)
- [x] `IntentParticleFilter` — particle filter maintaining beliefs over IntentProfile:
  - Particles initialized from configurable priors (Normal, Gamma, Uniform)
  - Likelihood: P(observed_signals | profile, context) from engagement, affect, kinematics
  - Systematic resampling with jitter when ESS < threshold
  - Entropy-based reliability score: `confidence = 1 - H(weights) / H_max`
  - Bayesian model averaging for intent prediction: `q(intent) = sum_k w_k * P(intent | particle_k)`
  - Epistemic value: information gain about empathy_j via hypothetical posterior entropy

### 5b. Social EFE for robotics (`cognition/tom/social_efe.py`)

- [x] `SocialEFE` class:
  - `G_social(action) = (1-lambda)*G_self(action) + lambda*G_other(action) + G_epistemic(action)`
  - `G_self`: negative expected reward for robot under opponent prediction
  - `G_other`: negative expected reward for human under robot's committed action
  - `G_epistemic`: information gain about IntentProfile (entropy reduction)
  - Action selection via softmax(-beta * G_social)
- [x] Reward matrices: `REWARD_SELF[robot_intent][human_intent]`, `REWARD_OTHER[robot_intent][human_intent]`
- [x] Affect penalty: additional EFE cost for approaching a distressed human (circumplex coupling)

### 5c. Gated ToM (`cognition/tom/gated_tom.py`)

- [x] `GatedToM` — entropy-based gating between learned and prior predictions:
  - `q_gated = reliability * q_learned + (1-reliability) * q_prior`
  - Reliability from particle filter entropy (sigmoid center=0.5, scale=0.1)
  - Graceful trust degradation when predictions fail
  - New agents start with low reliability (trust must be earned)
  - Per-agent particle filter lifecycle (create, update, prune stale)

### 5d. Wire into pipeline

- [x] `ToMModulator` three-level hierarchy: GatedToM > IntentInference > raw kinematic
- [x] ToMModulator builds `ObservationContext` from PerceptBundle (engagement, affect, kinematics)
- [x] Particle filter state persists across ticks per agent (in GatedToM._filters)
- [x] EFE breakdown (g_self, g_other, g_epistemic, g_social) in SkillUpdate.params
- [x] Agent reliability score in SkillUpdate.params

### Tests

- [x] 33 tests: particle filter mechanics, EFE computation, gating, integration (all passing)

---

## Phase 6: Script Repertoire Learning (DONE)

Active inference script acquisition: learn, compose, and consolidate
scripts from experience rather than relying on hand-authored sequences.

### 6a. Foundation types + primitive library

- [x] `ScriptPrimitive` — atomic social action units with pre/postconditions
- [x] `TrajectoryStep`, `ScriptTrajectory` — execution recording
- [x] `TransitionEntry` — B-matrix transition counts
- [x] `ScriptPattern` — learned generative model with precision, situation affinity
- [x] `RepertoireConfig` — all tunable parameters
- [x] `PrimitiveLibrary` — registry with 8 default primitives (approach-greet, yield-pass, wait-acknowledge, avoid-reroute, follow-maintain, disengage-depart, handover-extend, handover-receive)
- [x] `ScriptStep.primitive_name` — links steps back to source primitives

### 6b. Trajectory tracking + inference

- [x] `TrajectoryTracker` — records execution into ScriptTrajectory objects with bounded memory
- [x] `ScriptParticleFilter` — trajectory-level particle filter over (pattern, step_index)
  - Initialise from D-matrix (situation affinity × precision)
  - Predict: advance along B-matrix transitions
  - Update: weight by primitive match + situation KL
  - Free energy: accumulated -log(marginal likelihood)
  - Systematic resampling when ESS < threshold

### 6c. EFE-based composition

- [x] `ScriptComposer` — assembles new scripts from primitives using:
  - G_efficiency: situation distance to goal
  - G_empathy: expected affect impact
  - G_epistemic: information gain (less-used primitives preferred)
  - Softmax selection, returns weak ScriptPattern (precision=0.1)

### 6d. Repertoire orchestrator + learning manager

- [x] `ScriptRepertoire` — master component tying tracker, inference, composer
  - `on_situation_recognized()` → select or compose pattern
  - `on_step_completed()` → record + update inference
  - `on_script_completed()` → precision update + consolidation check
  - Precision dynamics: gain on low FE, loss on high FE, temporal decay
  - Weak→strong promotion when precision + count + FE thresholds met
  - `pattern_to_sequence()` bridges back to ScriptManager
- [x] `LearningScriptManager` — decorator wrapping ScriptManager with learning hooks
- [x] Executive wiring: repertoire hooks gated on `self._repertoire is not None`
- [x] Blackboard extended: `active_pattern`, `trajectory_free_energy`

### 6e. Context-conditioned semantic clusters

- [x] `ScriptPattern.primitive_cluster` — unordered semantic content (set of concepts)
- [x] `ScriptPattern.context_topology` — per-context directed pairwise connection weights (proto-B-matrix)
- [x] `ScriptPattern.is_cluster_based` — property gating cluster vs sequence mode
- [x] `ScriptComposer` produces clusters with context topology seeded from primitive pre/postconditions
- [x] `ScriptParticleFilter` dual-mode: cluster-membership likelihood (weak) vs positional match (strong)
- [x] `TrajectoryParticle.seen_primitives` — tracks observed cluster members
- [x] `ScriptRepertoire._derive_sequence_from_topology()` — context-dependent ordering via greedy walk
- [x] `ScriptRepertoire._crystallize_sequence()` — B-matrix path extraction on consolidation
- [x] Context threading through LearningScriptManager to pattern_to_sequence

### Tests

- [x] 117 tests (primitives, tracker, inference, composer, repertoire, integration, semantic clusters)

### Theoretical grounding

- Scripts as generative models of behavioural sequences
- Weak scripts = loosely-jointed semantic clusters (Albarracin et al. 2021)
- Context reshapes cluster topology: same concepts, different connection strengths
- Strong scripts = pragmatically sequenced via B-matrix crystallization
- Learning = Bayesian precision accumulation via free energy minimisation
- Composition = hierarchical generative model sampling producing clusters
- Consolidation = structural transformation from unordered cluster to ordered sequence

---

## Phase 7: Additional Skills (DONE)

### Completed

- [x] **HandoverSkill** — two-phase state machine (extend/receive) with hook methods for plugin subclassing
- [x] **HandoverIntentPolicy** — approach/avoid/yield/wait → arm_speed, retract params
- [x] **PickPlaceSkill** — pick/place state machine (approach→grasp/release→verify) with hook methods
- [x] **PickPlaceIntentPolicy** — arm_speed, grip_force, retract, caution_zone params
- [x] **GazeSkill** — four modes (look_at_agent, look_at_point, scan, avert) with stabilization tracking
- [x] **GazeIntentPolicy** — gaze_tracking, gaze_avert, gaze_hold, track_persistence params
- [x] 5 new script primitives: pick-object, place-object, gaze-at-agent, gaze-scan, gaze-avert
- [x] TIAGo plugin stubs: TiagoHandoverSkill, TiagoPickPlaceSkill, TiagoGazeSkill + driver extensions
- [x] Plugin registration updated for all skills
- [x] Tests: 58 new tests (all passing)

### Architecture pattern

Skills are robot-agnostic reference implementations in `architecture_core/skills/`:
- Express actions through `SkillUpdate.params` (gripper_action, arm_target, gaze_target)
- Read state from `PerceptBundle.world` (held_object, arm_at_target, gripper_state)
- Hook methods (`_on_extend`, `_on_grasp`, `_on_gaze_update`) are no-ops in base, overridden by plugin subclasses
- TIAGo stubs subclass reference skills and call `TiagoDriver` methods

## Phase 8: Scripts & Norms Enrichment (DONE)

### 8a. Behavior tree engine (`cognition/scripts/behavior_tree.py`)

- [x] `BTNode` abstract base with `tick(pb) -> BTStatus` and `reset()`
- [x] `ActionNode` — wraps SkillRequest, RUNNING until `set_result()` called, `on_activate` callback
- [x] `ConditionNode` — evaluates ScriptCondition predicate against PerceptBundle
- [x] `SequenceNode` — ticks children L→R, fails on first FAILURE, resumes RUNNING child
- [x] `SelectorNode` — ticks children L→R, succeeds on first SUCCESS, resumes RUNNING child
- [x] `ParallelNode` — ticks all children, succeed_on_all or succeed_on_one policy
- [x] `InverterNode` — decorator flipping SUCCESS↔FAILURE
- [x] `RepeatNode` — decorator repeating child N times (or forever if count=0)
- [x] `SucceederNode` — decorator always returning SUCCESS

### 8b. BehaviorTreeManager (`cognition/scripts/bt_manager.py`)

- [x] Implements ScriptManager interface (select, notify_status, check_violation)
- [x] Walks tree on construction to wire `on_activate` callbacks on all ActionNodes
- [x] `tree_from_sequence()` — converts existing ScriptSequence to BT SequenceNode
- [x] `is_complete` / `is_repairing` properties for Executive compatibility

### 8c. Adaptive norm discovery (`cognition/norms/adaptive_norms.py`)

- [x] `extract_norm_features(pb)` — extracts observable norm-relevant signals from PerceptBundle:
  - `mean_inter_human_distance`, `closest_human_distance`, `max_approach_speed`
  - `mean_gaze_engagement`, `mean_human_valence`, `max_interaction_duration`
- [x] `ProxemicPrior` dataclass — initial distance thresholds (replaces static cultural profiles)
- [x] `ScriptPattern.norm_features` — norms co-evolve with behavioral scripts
- [x] `TrajectoryStep.norm_snapshot` — records norm observations per step
- [x] Precision-gated EMA learning: `alpha = max(min_alpha, base_alpha / (1 + precision))`
  - High-precision patterns resist norm change (more evidence required)
  - Low-precision patterns adapt quickly
- [x] `ScriptComposer.compose()` seeds new patterns with current norm snapshot
- [x] `PersonalSpaceRule.set_norm_features()` / `SpeedLimitRule.set_norm_features()` — rules adapt from discovered norms
- [x] Executive wiring: extracts norm snapshot, pushes to repertoire, propagates to rules
- [x] `ProxemicsAugmentation.from_profile()` and `PersonalSpaceRule.from_profile()` still work via duck typing

### 8d. Dynamic keepout zones (`cognition/norms/dynamic_keepout.py`)

- [x] `DynamicKeepoutRule` — reads obstacles/hazards from PerceptBundle
- [x] Configurable margins (obstacle_margin=0.5, hazard_margin=1.0)
- [x] Veto when robot position is inside any computed zone

### 8e. Task-context-dependent norms (`cognition/norms/context_norms.py`)

- [x] `TaskContext` dataclass: named context with situation list + rule list
- [x] `ContextualNormRule` — activates rule sets based on recognized situation
- [x] Executive feeds situation to contextual rules via `set_situation()` (one-line addition, backward compatible)

### Tests

- [x] 74 tests: BT nodes, BT manager, norm feature extraction, norm discovery, adaptive rules, dynamic keepout, contextual norms, integration

## Phase 9: Safety (DONE)

### 9a. Velocity predictor (`safety/velocity_predictor.py`)

- [x] `compute_ttc()` — quadratic TTC for two circular entities via relative position/velocity
- [x] `speed_reduction_factor()` — linear speed ramp: 0.0 at ttc_stop, 1.0 at ttc_caution
- [x] `EntityState`, `CollisionRisk`, `CollisionPrediction` dataclasses
- [x] `VelocityPredictor` — enumerates robot-agent, robot-obstacle, agent-agent pairs
- [x] Robot velocity estimation from pose deltas (same pattern as SaliencyAugmentation)
- [x] Configurable max_horizon, robot_radius, default_agent_radius

### 9b. Predictive collision rule (`cognition/norms/predictive_collision.py`)

- [x] `PredictiveCollisionRule` NormRule — graduated speed constraints from TTC
- [x] Robot risks: `hard["speed_cap"] = speed_reduction_factor(min_robot_ttc)`
- [x] Veto on imminent collision (TTC < veto_ttc)
- [x] Agent-agent risks near robot: `soft["yield_agents"]` + speed reduction
- [x] Prediction data exposed in `hard["predictive_collision"]` dict
- [x] `last_prediction` property for Blackboard exposure

### 9c. Shield extension (`safety/shield.py`)

- [x] Optional `VelocityPredictor` parameter (backward compatible)
- [x] TTC-based emergency stop (TTC < emergency_ttc)
- [x] Preemptive braking between emergency and caution zones
- [x] Debug annotations: `[SHIELD:predictive-stop]`, `[SHIELD:predictive-brake]`
- [x] Distance-based emergency stop unchanged and takes precedence

### 9d. Integration

- [x] `Blackboard.collision_prediction` field for downstream access
- [x] Executive pushes collision prediction from rule to blackboard via `hasattr` guard

### Tests

- [x] 44 tests: TTC computation, speed reduction, velocity predictor, collision rule, shield extension, integration

## Phase 10: Table-Clearing Coordination via Emergent ToM (DONE)

Two TIAGo robots clear both tables (dining + coffee) in the apartment world. Coordination is fully emergent via the existing social layer — ToM, EFE rollout, empathy, norms. No explicit communication between robots.

### 10a. Task planner (`cognition/planning/task_planner.py`)

- [x] `ObjectState` dataclass: id, type, position, table, status, held_by
- [x] `TaskPlanner` — EFE-driven object selection via `SocialEFE.compute_rollout()`
- [x] For each candidate object: construct `ObservationContext`, run rollout, compute task_efe
- [x] `generate_sequence()` — creates nav→pick→nav→place 4-step ScriptSequence
- [x] Symmetry breaking via empathy asymmetry: low-empathy commits fast, high-empathy defers

### 10b. Clear-table manager (`cognition/planning/clear_table_manager.py`)

- [x] `ClearTableManager` — wraps ScriptManager + TaskPlanner + GatedToM
- [x] ToM-driven task selection: queries GatedToM for intent predictions, feeds to TaskPlanner
- [x] Observation phase: high-empathy robot waits when EFE says "wait" at step 0
- [x] ScriptManager-compatible interface (delegation for select, notify_status, check_violation)
- [x] Continuous clearing: selects next object when current sequence completes

### 10c. Object-aware sensors (`plugins/tiago_webots/perception/object_sensors.py`)

- [x] `TiagoObjectSensors` — extends TiagoWebotsSensors with Supervisor object discovery
- [x] Discovers Orange, Apple, Can nodes via world tree walk
- [x] Classifies objects by table region (dining vs. coffee)
- [x] `supervisor_grasp()` / `supervisor_release()` — teleportation-based manipulation
- [x] `update_held_position()` — keeps held object attached to robot each tick

### 10d. Controller integration

- [x] PHASE_LEVEL >= 10 activates table-clearing mode
- [x] Extended customData: `"goal_x,goal_y,alpha,agent_id,drop_x,drop_y"`
- [x] TiagoPickPlaceSkill with optional Supervisor grasping hooks
- [x] Diagnostic logging: objects remaining, current target, task completion

### 10e. Unified EFE for task selection AND navigation

The same `SocialEFE.compute_rollout()` used for moment-to-moment navigation also drives task selection. The generative model is unified across both levels:

- **Pragmatic value**: path distance to object + to drop-off
- **Social cost** (empathy × ToM): other robot near object → high obstruction → penalized
- **Collision penalty**: path crosses other robot's trajectory
- **Affect modulation**: stressed robot prefers safer/farther objects

### Tests

- [x] 23 tests: object scoring, sequence generation, symmetry breaking, ToM-driven selection, emergent coordination, ClearTableManager

## Phase 10c: Emergent Task Coordination via Inferred Commitment (MOSTLY DONE)

Fixes the multi-robot coordination deadlock where both robots select the same
object and yield to each other forever. Coordination now emerges from
ToM + empathy inference over observable behavioral cues.

### 10c-1. Globally observable object state (`object_sensors.py`)

- [ ] Z-height inference: z > 0.85 → "held", table height → "on_table", else → "placed"
- [ ] held_by = nearest agent (2D distance), no communication needed
- [ ] Replaces local `_held_object`/`_placed_ids` for status (kept for write-side manipulation)

### 10c-2. Commitment inference (`cognition/planning/commitment_inference.py`) ✅

- [x] Robot-agnostic module: infers per-object commitment from observable cues
- [x] Evidence: `w1·I[dist↓] + w2·cos(heading) + w3·I[dist < r]`
- [x] Sigmoid activation + EMA smoothing for sticky beliefs
- [x] Returns `q(C_{j,o}=1)` for each (agent, object) pair
- [x] Wired into ClearTableManager → TaskPlanner (Phase 13 Step 2)

### 10c-3. Contention cost in task EFE (`task_planner.py`) ✅

- [x] Contention flows through g_social via obstruction channel (no separate term)
- [x] Obstruction = max(physical_proximity_ratio, commitment_belief) (Phase 13 Step 3)
- [x] High obstruction → SocialEFE rollout penalizes via collision + comfort cross-entropy
- [x] Empathy modulates: high-empathy agent naturally backs off contested objects

### 10c-4. Fix yield policy (`nav_policy.py`)

- [ ] Replace escape-position target override with speed modulation
- [ ] Yield = slow down (0.3) + increase stop_distance (1.2), keep navigating to goal
- [ ] Yield timeout: after ~3s, degrade to speed 0.5 (break symmetry)
- [ ] Driver's reactive obstacle avoidance handles lateral dodging

### 10c-5. Yield-timeout task replan (`clear_table_manager.py`) ✅

- [x] Deadlock evidence from yield persistence × commitment belief (Phase 13 Step 5)
- [x] `p_deadlock = (n_yields / 10) × max(0.3, p_contend)` — EFE-grounded
- [x] Penalty = p_deadlock × 3.0, decays 0.9 per tick
- [x] Replaces fixed timer with contention-proportional penalty

### 10c-6. Fix velocity estimation (`clear_table_manager.py`) ✅

- [x] Track prev_other_dist for actual velocity from distance deltas (Phase 13 Step 1)
- [x] `approaching = velocity > 0.02` (not `distance < 5.0`)
- [x] Feed real velocity into ObservationContext for ToM

### Tests

- [x] 21 coordination tests (commitment wiring, motion tracking, deadlock yield, ToM selection)

---

## Phase 11: Strictly EFE/VFE-Grounded Refactor (COMPLETE)

Move from heuristic reward shaping to strictly EFE/VFE-grounded dynamics.
Valence = tanh(-ΔG/τ) (Joffily & Coricelli 2013), arousal = 2(H[q(π)]/log|Π|)-1,
contagion as secondary coupling through G. No reward shaping language —
everything expressed as expected surprisal under preferences.

### Step 1: Affect Emergence from Free Energy Dynamics
- [x] `AffectState` + `raw_free_energy`, `policy_entropy`, `delta_free_energy` fields in `core/types.py`
- [x] `last_social_efe`, `last_policy_entropy`, `last_num_policies` on `Blackboard`
- [x] `empathic_modulator.py` predict(): valence = tanh(-ΔG/τ), arousal = 2(H/log|Π|)-1
- [x] `empathic_modulator.py` observe(): contagion stores coupling signals only, violation KL accumulates via `consume_pending_fe()`
- [x] `executive.py`: wire effective G = base + violation + contagion → empathy.predict()
- [x] `tom_modulator.py`: compute + store policy_entropy and num_policies in params
- [x] Tests: 32 tests covering predict, observe, integration, modulate

### Step 2: Remove affect_penalty, Add Affect Risk Term
- [x] Remove `affect_penalty` param from `social_efe.py`, `gated_tom.py`, `controller.py`
- [x] Replace affect modulation with `_predict_affect_worsening()` → expected surprisal
- [x] Tests: distressed other → approach has higher g_other than yield

### Step 3: Collision Risk as Proper EFE Surprisal
- [x] Extract `_collision_probability()` as named sigmoid function
- [x] Replace collision cost with `-(p*ln(p_pref_coll) + (1-p)*ln(p_pref_safe))`
- [x] Apply to both backward induction and first-step Q in `social_efe.py`

### Step 4: Precision / Arousal Coupling
- [x] `self_arousal` parameter on `SocialEFE.compute()` and `compute_rollout()`
- [x] `effective_beta = beta * (1.5 - clamp(self_arousal))` — high arousal → lower precision
- [x] Wire through `gated_tom.py` → `tom_modulator.py` → `executive.py`

### Step 5: Simplify Affect Modulation (Remove Reward Shaping)
- [x] Remove speed_scale reduction from negative valence in `modulate()`
- [x] Remove epistemic_boost from high arousal in `modulate()`
- [x] Keep distress interrupt + diagnostic injection only

### Step 6: Yield-Timeout Replan via EFE Risk
- [x] `_yield_start_t`, `_yield_penalties`, `_yield_timeout` state in `clear_table_manager.py`
- [x] Decay logic (0.9 per tick), replan trigger after 5s continuous yield
- [x] Pass `yield_penalties` to `TaskPlanner.select_next()`

### Step 7: Intent Particle Filter A-Matrix + Arousal→Jitter
- [x] Document logit weights as `_A_MATRIX` generative model parameters
- [x] Refactor `compute_intent_probability()` to use named constants and feature dict
- [x] `set_arousal_modulation(arousal)` → modulates `_effective_jitter` in `_resample()`
- [x] Wire from `tom_modulator.py` after particle filter update

### Design decisions
- Contagion preserved as secondary coupling (G dynamics + contagion_fe_shift)
- Arousal modulates particle filter jitter_std via [0.5, 2.0] range
- Violation KL accumulated separately, consumed by Executive into G composition
- affect_penalty removed entirely (replaced by expected surprisal)
- F vs G distinction documented: currently using G as proxy, substitute VFE when available

---

## Phase 13: Emergent Coordination from EFE + ToM + Commitment ✅

Phase 12 completed the strict EFE conversion. Phase 13 wires the missing
signals so coordination **emerges** from the existing EFE/ToM machinery
rather than requiring heuristic timers or explicit communication.

### Step 1: Fix ObservationContext motion features ✅
- [x] Velocity from distance deltas: `v = max(0, (prev_dist - dist) / dt)`
- [x] Approaching from actual motion: `approaching = velocity > 0.02`
- [x] State tracking in ClearTableManager (`_prev_other_dist`, `_prev_t`)

### Step 2: Wire CommitmentInference ✅
- [x] ClearTableManager creates CommitmentInference, wires to TaskPlanner
- [x] `select()` calls `update_other_agent()` each tick with heading from perception
- [x] Commitment beliefs flow through to `_estimate_obstruction()`

### Step 3: Fix task-level obstruction geometry ✅
- [x] Other→object distance ratio replaces inter-agent distance heuristic
- [x] `d_other_obj < d_self_obj` → other is closer → contention ratio
- [x] Approaching boosts physical obstruction by 1.5×
- [x] `obstruction = max(physical_ratio, commitment_belief)`

### Step 4: Contention via existing EFE terms ✅
- [x] No separate G_contend — contention flows through g_social via obstruction
- [x] High obstruction → SocialEFE rollout penalizes via collision/comfort cross-entropy
- [x] Merged g_ambiguity + g_epistemic into single G_information term
- [x] Task EFE: `G = G_risk + G_information + G_social`

### Step 5: EFE-driven deadlock yield ✅
- [x] Replace fixed timer with `p_deadlock = (n_yields/10) × max(0.3, p_contend)`
- [x] Penalty proportional to contention strength (not fixed 2.0)
- [x] No penalty when yielding but contention is low
- [x] Decay 0.9 per tick, removed below 0.1

### Step 6: Documentation ✅
- [x] Object status sharing requirement documented in ClearTableManager.select()
- [x] Roadmap updated with Phase 13 + Phase 10c checkbox cleanup

### Design decisions
- Contention is NOT a new EFE term — it manifests through existing outcome
  variables (collision, comfort) via the obstruction signal
- Ambiguity H[p(o|s)] and epistemic -I(o;s) are coupled through the same
  hidden state (self-model precision) — combined into single G_information
- Yield-timeout from fixed timer → EFE-compatible deadlock evidence
  (persistence × commitment)

---

## Phase 12: Eliminate Reward Matrices, Strict EFE Throughout ✅

Phase 11 converted affect, collision, and affect-risk to proper EFE. Phase 12 completed
the conversion: core self/other terms now use cross-entropy under explicit preferences,
epistemic term uses full-state particle KL, and task-level EFE is documented as Boltzmann
surprisal. All 588 tests pass.

### Step 1: Replace Reward Matrices with Outcome Preferences + Expected Surprisal ✅
- [x] Extract `_immediate_cost()` helper to eliminate triplication of reward accumulation
- [x] Rename `REWARD_SELF` → `_P_PROGRESS`, `_REWARD_OTHER_*` → `_P_COMFORT_*` (outcome likelihoods)
- [x] Add preference priors: `_P_PREF_PROGRESS=0.9`, `_P_PREF_COMFORT=0.9`
- [x] Add `_cross_entropy(p_outcome, p_pref)` utility — unifies all 4 risk terms
- [x] Replace `g -= p*r` with `g += p*cross_entropy(p_outcome, p_pref)` in all 3 sites
- [x] Unify collision/affect to use same `_cross_entropy()`
- [x] Rename `reward_other()` → `_p_comfort()` (keep backward-compat alias)

### Step 2: Extend Epistemic Term to Full-State E[KL] ✅
- [x] Add `_particle_kl(posterior_weights)` — KL over full IntentProfile state
- [x] Rewrite `epistemic_value()` to use full-state KL, not empathy_j entropy bins
- [x] Remove `_empathy_j_entropy()` (replaced by `_particle_kl()`)

### Step 3: Task-Level EFE — Boltzmann Surprisal for G_risk ✅
- [x] Document `g_risk = β*cost` as proper Boltzmann surprisal: `-ln p(o|C) = β*cost + ln(Z)`
- [x] Update `task_planner.py` comments to describe G_social as proper cross-entropy EFE
- [x] No retune needed — beta_pragmatic=0.3 still works, all behavioral tests pass

### Design decisions
- Reward values R(a,h) ∈ [0,1] reinterpret as p(o=good | a,h) — observation likelihoods
- Risk = cross-entropy H(q_outcome, p_pref) = -[p*ln(p_pref) + (1-p)*ln(1-p_pref)]
- All 4 risk terms (progress, comfort, collision, affect) use same `_cross_entropy()`
- Epistemic = E_q(o|π)[KL(q(s|o,π) || q(s|π))] via particle-based KL (no binning)
- G_risk = β*cost IS Boltzmann surprisal — Z constant across candidates, cancels in argmin

---

## Phase 14: Unified Active Inference Task Controller ✅

Replaces ClearTableManager + ScriptManager with a single POMDP-based controller
where task phases emerge from posterior beliefs and actions are selected by EFE
minimisation.  Inspired by HAIF unified loop pattern — single inference per tick,
no state machine.

### Motivation

ClearTableManager uses a hardcoded state machine (nav→pick→nav→place) that breaks
in Webots: NAV SUCCESS loop (instant success → reselect same object), yield
waypoints through walls, and `active=None` as reselection trigger.

### 14a. Generative model (`cognition/planning/generative_model.py`) ✅

- [x] Factored POMDP: phase(5) × hand(2) × target_mode(3) = 30 hidden states
- [x] 7 observation modalities: d_obj, d_drop, arm, hold, obj_z, social, target
- [x] 7 policies: NAV_OBJ, PICKUP, NAV_DROP, PLACE, YIELD, WAIT, RESELECT
- [x] A matrices — P(o|s) for each modality, column-normalised
- [x] B matrices — P(s'|s,π) with distance-conditioned nav transitions
- [x] C vectors — log-preferences with empathy-scaled social cost
- [x] D vector — initial prior concentrated on APPROACH/EMPTY/FREE
- [x] `update_B_from_distances()` — sigmoid advance probability (replaces discrete SUCCESS)

### 14b. Variational inference engine (`cognition/planning/variational_engine.py`) ✅

- [x] `belief_update()` — exact Bayes with VFE decomposition (F = F_accuracy + F_complexity)
- [x] `evaluate_policies()` — EFE: G_pragmatic + G_epistemic + G_social per policy
- [x] `select_policy()` — softmax(-β·G) with arousal→precision coupling
- [x] Policy masking for infeasible actions (PICKUP only at object, YIELD only if safe)
- [x] Pure math module — no side effects, no state, independently testable

### 14c. Unified task controller (`cognition/planning/unified_task_controller.py`) ✅

- [x] Same interface as ClearTableManager (drop-in replacement at Executive level)
- [x] active=None does NOT trigger reselection — beliefs drive policy selection
- [x] Distance-conditioned B matrices (smooth sigmoid, not discrete SUCCESS)
- [x] Safe yield waypoints: world-bounds clamping, furniture keepout, hysteresis (5 ticks)
- [x] Yield masking: if no safe target → YIELD masked → WAIT fallback
- [x] VFE EMA smoothing (α=0.3) for valence computation
- [x] H[q(π)] from EFE softmax for arousal computation
- [x] Deadlock evidence → yield penalties → TaskPlanner reselection
- [x] RESELECT policy clears target and resets beliefs
- [x] DONE/LOST thresholds for automatic target advance

### 14d. Wire controller.py and executive.py ✅

- [x] Replace ClearTableManager construction with UnifiedTaskController in controller.py
- [x] Feed actual VFE to EmpathicModulator in executive.py (fulfils line 12-13 comment)
- [x] Pass world_bounds from apartment world config
- [x] Update diagnostics to show belief summary

### 14e. Tests ✅

- [x] Generative model: matrix shapes, normalization, discretisation, state roundtrip (27 tests)
- [x] Variational engine: VFE decrease, EFE ordering, belief convergence, precision coupling (18 tests)
- [x] Unified controller: policy sequence emergence, no NAV SUCCESS loop, yield safety (24 tests)

### 14f. Cleanup ✅

- [x] Remove temporary driver diagnostics (driver.py lines 223-230)
- [x] Run full test suite — 694 passed (601 existing + 93 new)

### Design decisions

- q(π) comes from EFE softmax — not hand-coded policy selection
- VFE = -E_q[ln P(o|s)] + KL[q||p] — actual free energy, not G proxy
- target_mode ∈ {FREE, CONTESTED, LOST} — hidden state for coordination
- Contention flows through g_social via obstruction (not separate EFE term)
- TaskPlanner reused for object-level selection (which object)
- UnifiedTaskController handles action-level control (what to do with object)
- ClearTableManager + ScriptManager preserved (existing tests reference them)

### Theoretical grounding

- Task phases emerge from Bayesian posterior over hidden states (not state machine)
- Policy selection via expected free energy minimisation (Parr & Friston 2019)
- Distance-conditioned transitions: p_advance = σ((threshold - d) / scale)
- Precision coupling: arousal → effective_β (Parr, Pezzulo & Friston 2022)
- VFE for affect: valence = tanh(-ΔF/τ) (Joffily & Coricelli 2013)

---

## Future: Projective Consciousness Model (PCM) integration

- [ ] Projective geometry for perspective-taking in ToM (Rudrauf et al. 2023)
- [ ] 3D projective space for affective/epistemic value computation
- [ ] Approach/avoid driven by projective magnification (1/z affective value law)

## Infrastructure

- [ ] Package `tom_planner` as installable dependency
- [ ] ROS2 plugin template
- [ ] Real robot plugin (Sphero, Tello)
- [ ] Logging and telemetry

---

## Architecture modules status

| Module | File | Status |
|---|---|---|
| Core types | `core/types.py` | Complete (+ AffectState, ScriptViolation, DeonticMode) |
| Blackboard | `core/blackboard.py` | Complete (+ active_pattern, trajectory_free_energy, norm_snapshot, active_norm_features, collision_prediction) |
| Registry | `core/registry.py` | Complete |
| Executive | `core/executive.py` | Complete (+ empathy + repertoire + collision prediction hooks) |
| Skill base | `skills/base.py` | Complete (abstract) |
| Intent policy | `cognition/tom/intent_policy.py` | Complete (abstract) |
| Script manager | `cognition/scripts/script_manager.py` | Complete (+ violation + repair + set_sequence) |
| Script types | `cognition/scripts/script_types.py` | Complete (+ primitive_name) |
| Weak recognizer | `cognition/scripts/weak_recognizer.py` | Complete |
| Repertoire types | `cognition/scripts/repertoire_types.py` | Complete (+ semantic clusters, context topology, norm_features) |
| Behavior tree | `cognition/scripts/behavior_tree.py` | Complete (8 node types) |
| BT manager | `cognition/scripts/bt_manager.py` | Complete |
| Primitive library | `cognition/scripts/primitive_library.py` | Complete (13 primitives) |
| Trajectory tracker | `cognition/scripts/trajectory_tracker.py` | Complete (+ norm_snapshot) |
| Trajectory inference | `cognition/scripts/trajectory_inference.py` | Complete (dual-mode: cluster + sequence) |
| Script composer | `cognition/scripts/script_composer.py` | Complete (EFE-based + context topology seeding + norm snapshot) |
| Script repertoire | `cognition/scripts/script_repertoire.py` | Complete (orchestrator + crystallization + norm learning) |
| Learning script mgr | `cognition/scripts/learning_script_manager.py` | Complete (decorator) |
| Norm engine | `cognition/norms/norm_engine.py` | Complete |
| Norm rules | `cognition/norms/rules.py` | Complete (3 rules + from_profile + set_norm_features) |
| Adaptive norms | `cognition/norms/adaptive_norms.py` | Complete (extract_norm_features, ProxemicPrior) |
| Dynamic keepout | `cognition/norms/dynamic_keepout.py` | Complete |
| Context norms | `cognition/norms/context_norms.py` | Complete |
| Affect norm rules | `cognition/norms/affect_rules.py` | Complete (2 rules) |
| ToM modulator | `cognition/tom/tom_modulator.py` | Complete (3-level hierarchy) |
| ToM planner adapter | `cognition/tom/models/tom_planner_adapter.py` | Complete |
| Intent inference | `cognition/tom/intent_inference.py` | Complete (log-linear fallback) |
| Intent particle filter | `cognition/tom/intent_particle_filter.py` | Complete |
| Social EFE | `cognition/tom/social_efe.py` | Complete |
| Gated ToM | `cognition/tom/gated_tom.py` | Complete |
| Empathic modulator | `cognition/empathy/empathic_modulator.py` | Complete |
| Safety shield | `safety/shield.py` | Complete (+ velocity-aware emergency stop) |
| Velocity predictor | `safety/velocity_predictor.py` | Complete (TTC, multi-pair prediction) |
| Predictive collision | `cognition/norms/predictive_collision.py` | Complete (graduated speed constraints) |
| Safety geometry | `safety/geometry.py` | Complete |
| Safety constraints | `safety/constraints.py` | Complete |
| Perception pipeline | `perception/perception_pipeline.py` | Complete |
| Sensor interface | `perception/sensors/base_sensors.py` | Complete (abstract) |
| Proxemics augmentation | `perception/augmentations/proxemics.py` | Complete (+ from_profile) |
| Affect augmentation | `perception/augmentations/affect.py` | Complete |
| Engagement augmentation | `perception/augmentations/engagement.py` | Complete |
| Saliency augmentation | `perception/augmentations/saliency.py` | Complete |
| Task planner | `cognition/planning/task_planner.py` | Complete (EFE-driven object selection) |
| Clear-table manager | `cognition/planning/clear_table_manager.py` | Complete (ToM + ScriptManager orchestration) |
| Generative model | `cognition/planning/generative_model.py` | Complete (30-state factored POMDP) |
| Variational engine | `cognition/planning/variational_engine.py` | Complete (VFE + EFE computation) |
| Unified task ctrl | `cognition/planning/unified_task_controller.py` | Complete (replaces ClearTableManager) |

**Test count: 601 (all passing)**
