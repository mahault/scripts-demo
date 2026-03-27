"""Webots controller entry point for TIAGo with the social-layer stack.

Each TIAGo robot runs its own independent social-layer stack;
coordination happens indirectly via Supervisor API observations.

The social cognition pipeline:
  Perception → WeakScriptRecognizer → ToMModulator (GatedToM) → IntentPolicy
    → SafetyShield → Skill.tick()

Phase-gated via PHASE_LEVEL env var (default=99):
  1  = Phases 1+2: basic nav + proxemics + personal space + shield
  3  = + Phase 3: affect/engagement/saliency augmentations
  4  = + Phase 4: empathic modulator + affect norm rules
  5  = + Phase 5: GatedToM active inference
  9  = + Phase 9: VelocityPredictor + PredictiveCollisionRule
  10 = + Phase 10: Table-clearing coordination (TaskPlanner + ClearTableManager)
  99 = Full stack

Usage in a .wbt file:
    Tiago {
      controller "tiago_social_layer"
      supervisor TRUE
      customData "goal_x,goal_y,alpha,agent_id[,drop_x,drop_y]"
    }
"""

from __future__ import annotations

import sys
import os
import logging
import math

# Webots puts the controller module on the path automatically
from controller import Supervisor  # noqa: F401 (Webots built-in)

# File-based logging so diagnostics are captured even when Webots
# swallows stdout.  Each robot gets its own log file.
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.pardir, os.pardir, os.pardir, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Bootstrap: ensure social-layer src/ is importable
# ------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.normpath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# --- Always-on imports (Phases 1+2) ---
from architecture_core.core.blackboard import Blackboard
from architecture_core.core.registry import SkillRegistry
from architecture_core.core.executive import Executive
from architecture_core.core.types import PerceptBundle, SkillRequest, SkillUpdate
from architecture_core.cognition.norms.norm_engine import NormEngine
from architecture_core.cognition.norms.rules import PersonalSpaceRule, SpeedLimitRule
from architecture_core.cognition.scripts.script_manager import ScriptManager
from architecture_core.cognition.scripts.script_types import ScriptSequence, ScriptStep, SituationType
from architecture_core.cognition.scripts.weak_recognizer import WeakScriptRecognizer
from architecture_core.cognition.tom.tom_modulator import ToMModulator
from architecture_core.safety.shield import SafetyShield
from architecture_core.perception.perception_pipeline import PerceptionPipeline
from architecture_core.perception.augmentations.proxemics import ProxemicsAugmentation

from plugins.tiago_webots.robot.driver import TiagoDriver
from plugins.tiago_webots.perception.sensors import TiagoWebotsSensors
from plugins.tiago_webots import plugin as tiago_plugin

# --- Phase 3: Perception augmentations ---
from architecture_core.perception.augmentations.affect import AffectAugmentation
from architecture_core.perception.augmentations.engagement import EngagementAugmentation
from architecture_core.perception.augmentations.saliency import SaliencyAugmentation

# --- Phase 4: Empathic modulator + affect norm rules ---
from architecture_core.cognition.empathy.empathic_modulator import EmpathicModulator
from architecture_core.cognition.norms.affect_rules import (
    AffectModulatedSpeedRule,
    DistressVetoRule,
)

# --- Phase 5: GatedToM ---
from architecture_core.cognition.tom.gated_tom import GatedToM

# --- Phase 9: Predictive safety ---
from architecture_core.safety.velocity_predictor import VelocityPredictor
from architecture_core.cognition.norms.predictive_collision import PredictiveCollisionRule

# --- Phase 10: Table-clearing coordination ---
from architecture_core.cognition.planning.task_planner import TaskPlanner
from architecture_core.cognition.planning.clear_table_manager import ClearTableManager
from architecture_core.cognition.planning.unified_task_controller import UnifiedTaskController
from architecture_core.cognition.planning.affordance import EmbodimentModel
from plugins.tiago_webots.perception.object_sensors import TiagoObjectSensors

# --- Kinematic ToM adapter (provides yield waypoints) ---
from architecture_core.cognition.tom.models.tom_planner_adapter import ToMPlannerAdapter

# --- Script learning pipeline ---
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.scripts.script_repertoire import ScriptRepertoire
from architecture_core.cognition.scripts.repertoire_types import RepertoireConfig
from architecture_core.cognition.scripts.learning_script_manager import LearningScriptManager

# --- Nav skill (for deterministic teacher loop) ---
from plugins.tiago_webots.skills.nav_skill import TiagoNavSkill


class DemoLearningScriptManager(LearningScriptManager):
    """LearningScriptManager that only installs strong (crystallized) patterns.

    The base LearningScriptManager immediately composes and installs patterns,
    overwriting the learner's naive script.  For the demo, the learner should
    follow its naive script until a pattern crystallizes (is_strong=True),
    then switch to the learned script.
    """

    def select(self, pb, active=None):
        if self._repertoire is None:
            return self._base.select(pb, active)

        # Feed situation belief to repertoire (precision accumulates)
        situation_belief = self._extract_situation_belief(pb)
        if situation_belief:
            pattern_name = self._repertoire.on_situation_recognized(
                situation_belief, t=pb.t,
            )
            # Only install STRONG patterns (post-crystallization)
            if pattern_name is not None and pattern_name != self._current_pattern_name:
                pattern = self._repertoire.patterns.get(pattern_name)
                if pattern is not None and pattern.is_strong:
                    self._current_pattern_name = pattern_name
                    most_likely_sit = (
                        max(situation_belief, key=situation_belief.get)
                        if situation_belief else None
                    )
                    seq = self._repertoire.pattern_to_sequence(
                        pattern, context=most_likely_sit,
                    )
                    self._base.set_sequence(seq)

        return self._base.select(pb, active)


class QueueWaitRule(PersonalSpaceRule):
    """Queue norm: if a queue cue is active, don't overtake the agent ahead.

    Implementation strategy (simple + demo-friendly):
    - Detect a deontic cue named CUE_QUEUE_HERE (-> cue type 'queue_here').
    - If an agent is within 'queue_follow_distance' *and* roughly ahead of the
      robot's heading, cap speed to 0.0 (=> intent policy compiles to WAIT).

    This is intentionally minimal: it demonstrates artifact-shaped normativity
    without requiring OCR/vision pipelines inside Webots.
    """

    name = "queue_wait"

    def __init__(self, queue_follow_distance: float = 1.6, ahead_cos: float = 0.2) -> None:
        super().__init__(min_distance=0.8, veto_distance=0.4)
        self.queue_follow_distance = queue_follow_distance
        self.ahead_cos = ahead_cos

    def evaluate(self, pb, candidate):
        constraints = super().evaluate(pb, candidate)

        # Only apply to navigation-like behaviors
        if getattr(candidate, "skill", "") != "navigate":
            return constraints

        cues = pb.world.get("deontic_cues", [])
        queue_active = any(
            (c.get("active", False) and "queue" in str(c.get("type", "")))
            for c in cues
        )
        if not queue_active:
            return constraints

        agents = pb.world.get("agents", [])
        if not agents:
            return constraints

        robot_pose = pb.world.get("robot_pose", (0, 0, 0, 0))
        rx, ry, heading = robot_pose[0], robot_pose[1], robot_pose[3]
        # Heading unit vector
        hx, hy = (math.cos(heading), math.sin(heading))

        # Find closest agent
        best = None
        best_d = 1e9
        for a in agents:
            ax, ay = a.get("pose", (0.0, 0.0))
            dx, dy = ax - rx, ay - ry
            d = (dx * dx + dy * dy) ** 0.5
            if d < best_d:
                best_d = d
                best = (dx, dy, d)

        if best is None:
            return constraints

        dx, dy, d = best
        # Is the agent roughly ahead?
        if d > 1e-6:
            cosang = (dx * hx + dy * hy) / d
        else:
            cosang = 1.0

        if d <= self.queue_follow_distance and cosang >= self.ahead_cos:
            # Hard stop (unless Executive chose YIELD, which shield exempts)
            constraints.hard["speed_cap"] = 0.0
            constraints.hard["min_agent_distance"] = max(constraints.hard.get("min_agent_distance", 0.0), 0.8)
        return constraints


def main() -> None:
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    name = robot.getName()
    self_node = robot.getSelf()

    # ---- File-based diagnostic logging ----
    _log_path = os.path.join(_LOG_DIR, f"{name}.log")
    _log_fh = open(_log_path, "w")
    import builtins
    _orig_print = builtins.print
    def _print(*args, **kwargs):
        _orig_print(*args, **kwargs)          # still goes to Webots console
        _orig_print(*args, **kwargs, file=_log_fh, flush=True)  # also to file
    builtins.print = _print

    # ---- PHASE_LEVEL configuration ----
    phase_level = int(os.environ.get("PHASE_LEVEL", "99"))

    # ---- parse customData: goal_x, goal_y, alpha, agent_id[, drop_x, drop_y] ----
    goal_x, goal_y, alpha, agent_id = 1.25, 0.0, 0.5, 0
    drop_x, drop_y = -2.0, -0.5  # default drop-off (kitchen counter)
    custom_data = robot.getCustomData()
    if custom_data:
        parts = custom_data.strip().split(",")
        if len(parts) >= 2:
            goal_x, goal_y = float(parts[0]), float(parts[1])
        if len(parts) >= 3:
            alpha = float(parts[2])
        if len(parts) >= 4:
            agent_id = int(parts[3])
        if len(parts) >= 6:
            drop_x, drop_y = float(parts[4]), float(parts[5])

    # ---- Queue demo mode ----
    is_queue_demo = os.environ.get("QUEUE_DEMO") == "1"
    is_teacher = is_queue_demo and agent_id <= 2

    print(f"{'='*60}")
    print(f"{name}: PHASE_LEVEL={phase_level}  goal=({goal_x},{goal_y})  alpha={alpha}  id={agent_id}")
    if is_teacher:
        print(f"{name}: MODE=teacher (deterministic)")
    elif is_queue_demo:
        print(f"{name}: MODE=learner (full social layer + repertoire learning)")
    if phase_level >= 10:
        print(f"{name}: drop_off=({drop_x},{drop_y})")
    print(f"{'='*60}")

    # ==================================================================
    # TEACHER: train formation (no social layer)
    #   All teachers navigate the SAME waypoint circuit on the SAME track.
    #   T1 (agent_id=0) = engine — also dwells at CUE markers
    #   T2 (agent_id=1) = car 2 — stops if within FOLLOW_DIST of T1
    #   T3 (agent_id=2) = car 3 — stops if within FOLLOW_DIST of T2
    #   Robots start in a column (same x, staggered y) so they're
    #   already on the same rail from the start.
    # ==================================================================
    if is_teacher:
        driver = TiagoDriver(robot)
        sensors = TiagoWebotsSensors(robot, self_node, name)
        nav = TiagoNavSkill(driver)

        FOLLOW_DIST = 1.5       # stop when this close to car ahead
        DWELL_TICKS = 180       # ~3s dwell at CUE markers (16ms timestep)

        waypoints = [
            (0.0, -2.0),    # CUE_QUEUE_HERE
            (3.0, -2.0),    # CUE_QUEUE_SERVICE
            (5.0, -5.0),    # exit south-east
            (-4.0, -5.0),   # return south-west
        ]
        dwell_at = {0, 1}   # only the leader dwells at CUE markers

        # Robot ahead in the chain (None for the leader)
        car_ahead = f"TIAGo_{agent_id}" if agent_id > 0 else None

        dummy_update = SkillUpdate(intent="approach",
                                   params={"speed_scale": 1.0})

        # Staggered start: each car waits so the one ahead pulls away
        start_delay = agent_id * 180  # ~3s per position
        for _ in range(start_delay):
            robot.step(timestep)

        wp_idx = 0
        dwell_remaining = 0
        nav.start(SkillRequest(skill="navigate",
                               goal={"x": waypoints[0][0], "y": waypoints[0][1]},
                               params={"goal_tolerance": 0.6}))

        role = "LEADER" if agent_id == 0 else f"FOLLOWER→{car_ahead}"
        print(f"{name}: {role} — {len(waypoints)} waypoints (train mode)")
        print(f"{name}: start_delay was {start_delay} ticks ({start_delay*0.016:.1f}s)")

        # List all agents visible to this robot
        raw0 = sensors.read()
        agents0 = raw0.get("agents", [])
        print(f"{name}: sees {len(agents0)} agents: "
              f"{[a.get('id') for a in agents0]}")

        tick_count = 0
        while robot.step(timestep) != -1:
            raw = sensors.read()
            pb = PerceptBundle(t=robot.getTime(), world=raw)
            pose = raw.get("robot_pose", (0, 0, 0, 0))

            # Check distance to car ahead (followers only)
            too_close = False
            ahead_dist = -1.0
            car_ahead_found = False
            if car_ahead is not None:
                for agent in raw.get("agents", []):
                    if agent.get("id") == car_ahead:
                        car_ahead_found = True
                        ap = agent.get("pose", (0, 0))
                        ahead_dist = math.sqrt(
                            (ap[0] - pose[0])**2 + (ap[1] - pose[1])**2)
                        if ahead_dist <= FOLLOW_DIST:
                            too_close = True
                        break

            # Determine state
            if too_close:
                state = "HOLD(too_close)"
                driver.stop()
            elif dwell_remaining > 0:
                state = f"DWELL({dwell_remaining})"
                driver.stop()
                dwell_remaining -= 1
                if dwell_remaining == 0:
                    wp_idx = (wp_idx + 1) % len(waypoints)
                    wx, wy = waypoints[wp_idx]
                    nav.start(SkillRequest(skill="navigate",
                                           goal={"x": wx, "y": wy},
                                           params={"goal_tolerance": 0.6}))
            else:
                status = nav.tick(pb, dummy_update)
                state = f"NAV({status})"
                if status == "SUCCESS":
                    if agent_id == 0 and wp_idx in dwell_at:
                        dwell_remaining = DWELL_TICKS
                        print(f"{name}: DWELL START at wp={wp_idx} "
                              f"pos=({pose[0]:.2f},{pose[1]:.2f})")
                    else:
                        wp_idx = (wp_idx + 1) % len(waypoints)
                        wx, wy = waypoints[wp_idx]
                        nav.start(SkillRequest(skill="navigate",
                                               goal={"x": wx, "y": wy},
                                               params={"goal_tolerance": 0.6}))

            tick_count += 1
            # Log every 30 ticks (~0.5s) for detailed visibility
            if tick_count % 30 == 0:
                agents_str = ""
                for a in raw.get("agents", []):
                    aid = a.get("id", "?")
                    ap = a.get("pose", (0, 0))
                    ad = math.sqrt((ap[0]-pose[0])**2 + (ap[1]-pose[1])**2)
                    agents_str += f" {aid}={ad:.1f}m"
                print(f"{name}: t={robot.getTime():.1f}  "
                      f"pos=({pose[0]:.2f},{pose[1]:.2f})  "
                      f"wp={wp_idx}/{len(waypoints)} "
                      f"tgt=({waypoints[wp_idx][0]:.1f},{waypoints[wp_idx][1]:.1f})  "
                      f"state={state}  ahead_d={ahead_dist:.1f} "
                      f"found={car_ahead_found}{agents_str}")

        return  # Teachers exit here

    # ==================================================================
    # LEARNER / DEFAULT: full social layer
    # ==================================================================

    # ---- instantiate components ----
    driver = TiagoDriver(robot)

    # Phase 10: use object-aware sensors
    object_sensors = None
    if phase_level >= 10:
        sensors = TiagoObjectSensors(robot, self_node, name)
        object_sensors = sensors
        print(f"{name}: Phase 10 active — TiagoObjectSensors ({len(sensors._object_nodes)} objects)")
    else:
        sensors = TiagoWebotsSensors(robot, self_node, name)

    registry = SkillRegistry()
    tiago_plugin.register(registry, driver, object_sensors=object_sensors,
                          drop_off_pos=(drop_x, drop_y, 0.74))

    # ---- Perception pipeline ----
    augmentations = [ProxemicsAugmentation()]

    if phase_level >= 3:
        augmentations.extend([
            AffectAugmentation(),
            EngagementAugmentation(),
            SaliencyAugmentation(),
        ])
        print(f"{name}: Phase 3 active — affect/engagement/saliency augmentations")

    perception = PerceptionPipeline(
        sensors=sensors,
        augmentations=augmentations,
    )

    # ---- Norm rules ----
    norm_rules = [
        PersonalSpaceRule(min_distance=0.8, veto_distance=0.4),
        SpeedLimitRule(max_speed=1.0),
        QueueWaitRule(queue_follow_distance=1.6),
    ]

    if phase_level >= 4:
        norm_rules.extend([
            AffectModulatedSpeedRule(base_max_speed=1.0),
            DistressVetoRule(),
        ])
        print(f"{name}: Phase 4 affect norm rules active")

    # ---- Phase 9: Predictive safety ----
    predictor = None
    if phase_level >= 9:
        predictor = VelocityPredictor(robot_radius=0.25, default_agent_radius=0.25)
        norm_rules.append(PredictiveCollisionRule(predictor=predictor))
        print(f"{name}: Phase 9 active — VelocityPredictor + PredictiveCollisionRule")

    # ---- Phase 5: GatedToM (active inference) ----
    gated_tom = None
    empathy = 0.0
    if phase_level >= 5:
        empathy = min(1.0, alpha / 6.0) if alpha > 0 else 0.0
        gated_tom = GatedToM(
            empathy_factor=empathy,
            n_particles=200,
            beta=2.0,
            epistemic_weight=0.5,
        )
        print(f"{name}: Phase 5 active — GatedToM  empathy_factor={empathy:.2f}")

    # ---- Phase 4: Empathic modulator ----
    empathy_mod = None
    if phase_level >= 4:
        empathy_mod = EmpathicModulator()
        print(f"{name}: Phase 4 active — EmpathicModulator")

    # ---- Script sequence + weak-script recognizer ----
    recognizer = None
    if phase_level >= 4:
        # Demo-friendly recognizer: adds artifact-script situations.
        # - 'queue' fires when a queue cue is active and an agent is near.
        # - 'approach' fires when no queue cue is present and velocity is nonzero.
        recognizer = WeakScriptRecognizer(
            situation_types=[
                SituationType(
                    name="queue",
                    feature_weights={
                        "cue_queue_here": 4.0,
                        "inverse_min_distance": 1.2,
                        "mean_velocity": -1.0,
                    },
                ),
                SituationType(
                    name="approach",
                    feature_weights={
                        "cue_queue_here": -3.0,
                        "mean_velocity": 2.0,
                        "inverse_min_distance": -0.2,
                    },
                ),
                # Keep the original navigation contexts for non-demo worlds.
                SituationType(
                    name="corridor",
                    feature_weights={
                        "inverse_min_distance": -0.5,
                        "mean_velocity": 3.0,
                    },
                ),
                SituationType(
                    name="blocked",
                    feature_weights={
                        "inverse_min_distance": 2.0,
                        "mean_velocity": -5.0,
                    },
                ),
            ],
            prior={"queue": 0.2, "approach": 0.5, "corridor": 0.25, "blocked": 0.05},
        )
        print(f"{name}: Weak-script recognizer active (queue/approach + corridor/blocked)")

    # ---- Script sequences (QUEUE_DEMO or default) ----
    repertoire = None

    if is_queue_demo and agent_id == 3:
        # Learner: naive "go to counter" script + full POMDP learning pipeline
        # The learner follows the SAME path as the teacher chain so it
        # encounters them queued up.  QueueWaitRule forces reactive waiting;
        # ScriptRepertoire observes and crystallizes the queue script.
        library = PrimitiveLibrary(register_defaults=True)
        seq = ScriptSequence(
            name="learner_naive",
            steps=[
                ScriptStep(SkillRequest(skill="navigate",
                           goal={"x": 0.0, "y": -2.0}),
                           expected_situation="queue"),                  # queue zone (CUE_QUEUE_HERE)
                ScriptStep(SkillRequest(skill="navigate",
                           goal={"x": 3.0, "y": -2.0}),
                           expected_situation="queue"),                  # service counter (CUE_SERVICE)
                ScriptStep(SkillRequest(skill="navigate",
                           goal={"x": 5.0, "y": -5.0})),                # exit
                ScriptStep(SkillRequest(skill="navigate",
                           goal={"x": -4.0, "y": -5.0})),               # return south-west
                ScriptStep(SkillRequest(skill="navigate",
                           goal={"x": -4.0, "y": -10.5})),              # back to start
            ],
            loop=True,
        )
        repertoire = ScriptRepertoire(library, config=RepertoireConfig(
            strong_precision_threshold=2.0,
            min_trajectories_for_promotion=5,
            precision_gain_on_success=0.3,
        ))
        print(f"{name}: Learner script — naive 4-step loop + ScriptRepertoire learning")

    else:
        # Default: single-step navigate to goal
        seq = ScriptSequence(
            name="navigate_to_goal",
            steps=[ScriptStep(
                SkillRequest(skill="navigate", goal={"x": goal_x, "y": goal_y}),
                expected_situation="approach" if phase_level >= 4 else None,
            )],
        )

    # ---- Script manager (with optional learning wrapper) ----
    scripts_manager = ScriptManager(sequence=seq, recognizer=recognizer)
    if repertoire is not None:
        scripts_manager = DemoLearningScriptManager(scripts_manager, repertoire)

    # ---- Phase 10: Table-clearing coordination ----
    clear_table_mgr = None

    if phase_level >= 10 and gated_tom is not None:
        from architecture_core.cognition.tom.social_efe import SocialEFE
        task_efe = SocialEFE(
            empathy_factor=empathy,
            beta=2.0,
            epistemic_weight=0.5,
        )
        planner = TaskPlanner(
            drop_off=(drop_x, drop_y),
            efe=task_efe,
            empathy_factor=empathy,
            embodiment=EmbodimentModel(arm_reach=0.85, body_radius=0.27),
        )
        clear_table_mgr = UnifiedTaskController(
            planner=planner,
            gated_tom=gated_tom,
            empathy_factor=empathy,
            world_bounds=(-8.5, 0.5, -6.0, 0.5),
        )
        if recognizer is not None:
            clear_table_mgr._recognizer_ref = recognizer
        print(f"{name}: Phase 10 active — TaskPlanner + UnifiedTaskController")

    # ---- Kinematic ToM adapter (yield waypoints) ----
    tom_adapter = None
    if phase_level >= 5:
        _planner_path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.pardir, os.pardir, os.pardir, os.pardir,
            "Alignment-experiments", "webots_sim", "controllers", "tiago_empathic",
        ))
        try:
            tom_adapter = ToMPlannerAdapter(
                agent_id=agent_id,
                goal_x=goal_x,
                goal_y=goal_y,
                alpha=alpha,
                planner_path=_planner_path,
            )
            # Configure arena for apartment world (larger than corridor default)
            # Coarse grid (~108 states) to keep JAX JIT fast — corridor used 77.
            tom_adapter.configure_arena(
                x_min=-8.5, x_max=0.5,
                y_min=-6.0, y_max=0.5,
                hazards=[],  # furniture handled by nav skill obstacle avoidance
                n_x=12, n_y=9,
            )
            print(f"{name}: ToMPlannerAdapter active (planner_path={_planner_path})")
        except Exception as exc:
            print(f"{name}: ToMPlannerAdapter failed: {exc} — falling back to no adapter")
            tom_adapter = None

    # ---- Safety shield ----
    shield = SafetyShield(
        emergency_stop_distance=0.4,
        predictor=predictor,
    )

    # ---- Executive ----
    bb = Blackboard()
    executive = Executive(
        bb=bb,
        registry=registry,
        scripts=clear_table_mgr if clear_table_mgr is not None else scripts_manager,
        norms=NormEngine(rules=norm_rules),
        tom=ToMModulator(adapter=tom_adapter, gated_tom=gated_tom),
        shield=shield,
        deliberation_hz=5.0,
        empathy=empathy_mod,
    )

    # Let simulation stabilise (learner waits longer so teacher chain forms first)
    stabilise_ticks = 600 if (is_queue_demo and agent_id == 3) else 10
    for _ in range(stabilise_ticks):
        robot.step(timestep)

    phases_str = []
    if phase_level >= 1:
        phases_str.append("1+2:core")
    if phase_level >= 3:
        phases_str.append("3:augment")
    if phase_level >= 4:
        phases_str.append("4:empathy")
    if phase_level >= 5:
        phases_str.append("5:ToM")
    if phase_level >= 9:
        phases_str.append("9:safety")
    if phase_level >= 10:
        phases_str.append("10:coordination")
    print(f"{name}: entering main loop  phases=[{', '.join(phases_str)}]")

    # ---- main loop with diagnostics ----
    tick_count = 0
    log_interval = 30

    while robot.step(timestep) != -1:
        t = robot.getTime()
        bb.percept = perception.tick(t)

        # Update held object position each tick (Phase 10)
        if object_sensors is not None:
            object_sensors.update_held_position()

        executive.tick(t)

        tick_count += 1
        if tick_count % log_interval == 0:
            _log_diagnostics(name, bb, goal_x, goal_y, t, phase_level,
                             gated_tom, clear_table_mgr, repertoire)


def _log_diagnostics(
    name: str,
    bb,
    goal_x: float,
    goal_y: float,
    t: float,
    phase_level: int,
    gated_tom,
    clear_table_mgr=None,
    repertoire=None,
) -> None:
    """Print one-line diagnostic summary."""
    if not bb.percept:
        return

    pose = bb.percept.world.get("robot_pose")
    if not pose:
        return

    agents = bb.percept.world.get("agents", [])
    dist_to_goal = ((goal_x - pose[0])**2 + (goal_y - pose[1])**2)**0.5

    # Agent distance
    agent_dist = "none"
    if agents:
        ap = agents[0].get("pose", (0, 0))
        agent_dist = f"{((pose[0]-ap[0])**2 + (pose[1]-ap[1])**2)**0.5:.2f}"

    # TTC (Phase 9)
    ttc_str = "-"
    if phase_level >= 9 and bb.collision_prediction is not None:
        ttc_str = f"{bb.collision_prediction.min_robot_ttc:.1f}"

    # Self-affect (Phase 4)
    affect_str = "-"
    if phase_level >= 4 and bb.self_affect is not None:
        affect_str = f"({bb.self_affect.arousal:.2f},{bb.self_affect.valence:.2f})"

    # Intent / reliability (Phase 5)
    intent_str = "-"
    rel_str = "-"
    if phase_level >= 5 and gated_tom is not None and agents:
        agent_id = agents[0].get("id", 0)
        rel = gated_tom.agent_reliability(agent_id)
        rel_str = f"{rel:.2f}"

    # Compiled intent + speed from Executive (set by shield)
    intent_str = getattr(bb, '_compiled_intent', intent_str)
    speed_str = "-"
    compiled_spd = getattr(bb, '_compiled_speed_scale', None)
    if compiled_spd is not None:
        speed_str = f"{compiled_spd:.2f}"

    # Rollout policy (Phase 5+)
    policy_str = "-"
    rollout_policy = getattr(bb, '_rollout_policy', None)
    if rollout_policy:
        abbrev = [a[0] for a in rollout_policy[:5]]
        policy_str = "".join(abbrev)

    # Yield target and obstruction (debug)
    yield_str = "-"
    obs_str = "-"
    compiled_params = getattr(bb, '_compiled_params', None)
    if compiled_params:
        tx = compiled_params.get("target_x")
        ty = compiled_params.get("target_y")
        if tx is not None:
            yield_str = f"({tx:.2f},{ty:.2f})"
        io = compiled_params.get("initial_obstruction")
        if io is not None:
            obs_str = f"{io:.2f}"

    # Phase 10: task status
    task_str = ""
    if phase_level >= 10 and clear_table_mgr is not None:
        planner = clear_table_mgr._planner
        available = len(planner.available_objects)
        target = planner.current_target
        target_id = target.id if target else "none"
        committed = clear_table_mgr._committed_phase or "-"
        resel_ev = clear_table_mgr._reselect_evidence
        task_str = f"  objs={available}  tgt={target_id}  commit={committed}  resel_ev={resel_ev:.2f}"
        if hasattr(clear_table_mgr, 'belief_summary'):
            task_str += f"  q={clear_table_mgr.belief_summary()}"
        if planner.is_task_complete:
            task_str += "  DONE"

    # Learner learning status
    learn_str = ""
    if repertoire is not None:
        patterns = repertoire.patterns
        if patterns:
            for pname, p in patterns.items():
                learn_str += (
                    f"\n  [LEARN] pattern={pname} prec={p.precision:.2f} "
                    f"strong={p.is_strong} trajs={p.trajectory_count}"
                )
        else:
            learn_str = "\n  [LEARN] no patterns yet"

    print(
        f"{name}: t={t:.1f}  pos=({pose[0]:.2f},{pose[1]:.2f})  "
        f"gd={dist_to_goal:.2f}  ad={agent_dist}  "
        f"ttc={ttc_str}  affect={affect_str}  "
        f"intent={intent_str}  rel={rel_str}  "
        f"spd={speed_str}  policy={policy_str}  "
        f"ytgt={yield_str}  obs={obs_str}  active={'Y' if bb.active_req else 'N'}"
        f"{task_str}{learn_str}"
    )


if __name__ == "__main__":
    main()
