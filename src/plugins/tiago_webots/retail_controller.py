"""Retail demo controller for Webots TIAGo.

Modes (set via agent_id in customData):
  0 = Teacher (Worker_T) — deterministic restock loop, no social layer
  1 = Learner (Learner_L) — full social layer + script repertoire learning
  2+ = Customer — simple patrol navigation

Env vars:
  RETAIL_DEMO=1 — enables retail mode in tiago_social_layer shim
  PHASE_LEVEL — learner phase gating (default 9 for demo)
"""

from __future__ import annotations

import sys
import os
import math
import json

from controller import Supervisor

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.pardir, os.pardir, os.pardir, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.normpath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Core imports
from architecture_core.core.blackboard import Blackboard
from architecture_core.core.registry import SkillRegistry
from architecture_core.core.executive import Executive
from architecture_core.core.types import PerceptBundle, SkillRequest, SkillUpdate
from architecture_core.cognition.norms.norm_engine import NormEngine
from architecture_core.cognition.norms.rules import PersonalSpaceRule, SpeedLimitRule
from architecture_core.cognition.scripts.script_manager import ScriptManager
from architecture_core.cognition.scripts.script_types import ScriptSequence, ScriptStep, SituationType
from architecture_core.cognition.scripts.weak_recognizer import WeakScriptRecognizer
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.scripts.script_repertoire import ScriptRepertoire
from architecture_core.cognition.scripts.learning_script_manager import LearningScriptManager
from architecture_core.cognition.tom.tom_modulator import ToMModulator
from architecture_core.safety.shield import SafetyShield
from architecture_core.perception.perception_pipeline import PerceptionPipeline
from architecture_core.perception.augmentations.proxemics import ProxemicsAugmentation

# Phase-gated imports
from architecture_core.perception.augmentations.affect import AffectAugmentation
from architecture_core.perception.augmentations.engagement import EngagementAugmentation
from architecture_core.perception.augmentations.saliency import SaliencyAugmentation
from architecture_core.cognition.empathy.empathic_modulator import EmpathicModulator
from architecture_core.cognition.norms.affect_rules import AffectModulatedSpeedRule, DistressVetoRule
from architecture_core.cognition.tom.gated_tom import GatedToM
from architecture_core.safety.velocity_predictor import VelocityPredictor
from architecture_core.cognition.norms.predictive_collision import PredictiveCollisionRule

from plugins.tiago_webots.robot.driver import TiagoDriver
from plugins.tiago_webots.perception.object_sensors import TiagoObjectSensors
from plugins.tiago_webots import plugin as tiago_plugin
from plugins.tiago_webots.skills.nav_skill import TiagoNavSkill

from plugins.tiago_webots.retail_primitives import (
    register_retail_primitives,
    make_retail_fragments,
    make_retail_repertoire_config,
    make_restock_seed_pattern,
)
from plugins.tiago_webots.observation import (
    TeacherObserver,
    build_learned_sequence,
)
from plugins.tiago_webots.social_interactions import WorkerEncounterManager


class DemoLearningScriptManager(LearningScriptManager):
    """Only installs strong (crystallized) patterns.

    Also fixes loop-wrap detection: looping scripts never set
    is_complete=True, so trajectories would stay open forever.
    We detect index wrap and manually finalize the trajectory.
    """

    def select(self, pb, active=None):
        if self._repertoire is None:
            return self._base.select(pb, active)
        situation_belief = self._extract_situation_belief(pb)
        if situation_belief:
            pattern_name = self._repertoire.on_situation_recognized(
                situation_belief, t=pb.t,
            )
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

    def notify_status(self, status):
        # Detect loop wrap BEFORE base.notify_status advances the index
        seq = self._base._sequence
        wrapped = False
        if (
            seq is not None
            and seq.loop
            and status in ("SUCCESS", "FAILURE", "TIMEOUT")
        ):
            idx_before = self._base._current_idx
            step = seq.steps[idx_before] if idx_before < len(seq.steps) else None
            if step is not None and status == step.transition_on:
                if idx_before >= len(seq.steps) - 1:
                    wrapped = True

        super().notify_status(status)

        # If we wrapped around, finalize trajectory manually
        if wrapped and self._repertoire is not None:
            self._repertoire.on_script_completed(status)
            self._current_pattern_name = None


class RetailWaitRule(PersonalSpaceRule):
    """If a customer is at the counter, wait before approaching."""
    name = "retail_wait"

    def __init__(self, counter_x=4.5, counter_y=-7.5, wait_radius=2.0):
        super().__init__(min_distance=0.8, veto_distance=0.4)
        self.counter = (counter_x, counter_y)
        self.wait_radius = wait_radius

    def evaluate(self, pb, candidate):
        constraints = super().evaluate(pb, candidate)
        if getattr(candidate, "skill", "") != "navigate":
            return constraints
        agents = pb.world.get("agents", [])
        rx, ry = pb.world.get("robot_pose", (0, 0))[:2]
        # Check if any agent is near counter
        customer_near = False
        for a in agents:
            ax, ay = a.get("pose", (0, 0))
            cd = ((ax - self.counter[0])**2 + (ay - self.counter[1])**2)**0.5
            if cd < self.wait_radius:
                customer_near = True
                break
        if customer_near:
            rd = ((rx - self.counter[0])**2 + (ry - self.counter[1])**2)**0.5
            if rd > self.wait_radius:
                # Approaching counter while customer there — slow down
                constraints.soft["speed_cap"] = 0.2
        return constraints


# ============================================================================
# TEACHER MODE
# ============================================================================
def _run_teacher(robot, timestep, name, agent_id):
    driver = TiagoDriver(robot)
    self_node = robot.getSelf()
    sensors = TiagoObjectSensors(robot, self_node, name)
    nav = TiagoNavSkill(driver)

    # Restock waypoints — stay in the wide open corridors so the base (and
    # the folded transport arm) never scrapes shelves/counters.  The arm
    # extends at the waypoint for the simulated pick/place/handover.
    # Perimeter-style restock loop.  Stay well clear of the divider/shelves
    # by moving in orthogonal segments along the south aisle and corridor.
    waypoints = [
        (-5.5, -1.0),   # 0 stock room
        (-4.9, -1.5),   # 1 corridor entry, clear of stock_shelf & divider
        (-4.9, -5.0),   # 2 south of stock divider
        (-4.9, -7.0),   # 3 south aisle (clear of shelf A/B inflated keepout)
        (-3.5, -7.0),   # 4 shelf A aisle
        (-0.25, -7.0),  # 5 central aisle between shelf A and shelf B
        (2.0, -7.0),    # 6 approach counter from west
        (3.0, -7.0),    # 7 worker queue (south-west of service counter)
        (1.5, -7.0),    # 8 shelf B restock point — aisle, south of shelf_B
                        #   (teleport place; never enters the shelf keepout)
        (-1.0, -7.0),   # 9 continue west along south aisle
        (-4.9, -7.0),   # 10 west along south aisle
        (-4.9, -5.0),   # 11 north toward corridor
        (-4.9, -1.5),   # 12 corridor
        (-5.5, -1.0),   # 13 stock room
    ]

    # Container destinations for restocking.  Items are dropped into the
    # matching tray; if a tray is full, the worker falls back to the raw
    # surface position and freezes the prop so it cannot roll.
    PLACE_POSITIONS = {
        "shelf_a": (-2.0, -5.75, 0.43),
        "shelf_b": (1.5, -5.75, 0.43),
        "counter": (4.5, -6.85, 0.93),
    }
    DESTINATION_CONTAINER = {
        "shelf_a": "CONTAINER_shelf_a",
        "shelf_b": "CONTAINER_shelf_b",
        "counter": "CONTAINER_counter",
    }
    DESTINATION_WP = {
        "shelf_a": 4,
        "shelf_b": 8,
        "counter": 7,
    }
    WP_NAMES = {
        0: "stock", 4: "shelf_a", 7: "counter", 8: "shelf_b",
    }

    dwell_ticks = 80    # ~1.3s dwell at each waypoint
    goal_tolerance = 0.6

    dummy_update = SkillUpdate(intent="approach", params={"speed_scale": 0.8})

    wp_idx = 0
    dwell_remaining = 0
    nav.start(SkillRequest(skill="navigate",
                           goal={"x": waypoints[0][0], "y": waypoints[0][1]},
                           params={"goal_tolerance": goal_tolerance}))

    # Manipulation state
    holding = False
    held_id = None
    destination = "shelf_a"   # cycles through shelf_a / shelf_b / counter
    action_done = False
    restocked_count = 0
    at_stock = False
    at_dest = False
    waiting_for_customer = False

    # Give-way + greet manager: turns shopper encounters (anywhere on the
    # floor) into a structured social act instead of a collision.
    encounter = WorkerEncounterManager(name, customer_id="Customer_1")

    # Counter zone for yielding to the customer
    COUNTER_CENTER = (4.5, -7.5)
    CUSTOMER_PROXIMITY = 2.5
    WORKER_COUNTER_PROXIMITY = 1.8

    def customer_at_counter(pb):
        for a in pb.world.get("agents", []):
            if a.get("id") == "Customer_1":
                ax, ay = a.get("pose", (0, 0))
                return ((ax - COUNTER_CENTER[0])**2 +
                        (ay - COUNTER_CENTER[1])**2)**0.5 < CUSTOMER_PROXIMITY
        return False

    def worker_near_counter(pose):
        return ((pose[0] - COUNTER_CENTER[0])**2 +
                (pose[1] - COUNTER_CENTER[1])**2)**0.5 < WORKER_COUNTER_PROXIMITY

    print(f"{name}: TEACHER mode — restock loop with real item transport")
    tick_count = 0
    state = "NAV"
    loop_count = 0

    def pick_from_stock():
        """Grasp the next available stock item, respawning if stock is empty."""
        nonlocal holding, held_id
        obj_id = sensors.find_object_in_region("stock")
        if obj_id is None:
            # Respawn only the stock items so the customer's basket/items are
            # not teleported away mid-shopping trip.
            sensors.reset_stock()
            obj_id = sensors.find_object_in_region("stock")
        if obj_id is None:
            return False
        sensors.set_manipulation_target(obj_id)
        if sensors.supervisor_grasp(obj_id):
            holding = True
            held_id = obj_id
            print(f"{name}: GRASPED {obj_id}")
            return True
        return False

    def place_at_destination():
        """Release the currently held item at the active destination."""
        nonlocal holding, held_id, restocked_count
        if not holding or held_id is None:
            return False
        sensors.set_manipulation_target(held_id)
        cid = DESTINATION_CONTAINER[destination]
        if sensors.supervisor_release_into_container(cid):
            print(f"{name}: PLACED {held_id} into {cid}")
            holding = False
            held_id = None
            restocked_count += 1
            return True
        # Fallback: place directly on the surface as a frozen static prop
        pos = PLACE_POSITIONS[destination]
        if sensors.supervisor_release(pos, freeze=True):
            print(f"{name}: PLACED {held_id} at {destination} {pos} (frozen)")
            holding = False
            held_id = None
            restocked_count += 1
            return True
        return False

    while robot.step(timestep) != -1:
        raw = sensors.read()
        pb = PerceptBundle(t=robot.getTime(), world=raw)
        pose = raw.get("robot_pose", (0, 0, 0, 0))

        # Keep any held object attached to the gripper while moving, and keep
        # items dropped into trays visually locked to those trays.
        if holding:
            sensors.update_held_position()
        sensors.update_containers()

        # Social interaction: give way to an approaching shopper anywhere on
        # the floor.  The manager owns the base/arm/head while the give-way +
        # greet plays out and reports the social act as a discrete action
        # label, which the learner's observer segments into a primitive.
        enc = encounter.update(pb, pose, driver)
        if enc.active:
            state = f"SOCIAL:{enc.action}"
            if tick_count % 10 == 0:
                try:
                    self_node.getField("customData").setSFString(json.dumps({
                        "waypoint": wp_idx,
                        "loop": loop_count,
                        "state": state,
                        "action": enc.action,
                        "holding": holding,
                        "held_id": held_id,
                        "destination": destination,
                        "restocked": restocked_count,
                        # Keep the nav target published during give-ways too, so
                        # navigate segments interrupted by a yield still learn
                        # their destination (no goal-less primitives).
                        "tx": waypoints[wp_idx][0],
                        "ty": waypoints[wp_idx][1],
                    }))
                except Exception:
                    pass
            tick_count += 1
            if tick_count % 60 == 0:
                print(f"{name}: t={robot.getTime():.1f} pos=({pose[0]:.2f},{pose[1]:.2f}) "
                      f"wp={wp_idx} {state} (shopper give-way)")
            continue
        if enc.just_finished and dwell_remaining == 0:
            # Re-issue the current nav goal so navigation resumes cleanly after
            # the give-way maneuver moved the base.
            wx, wy = waypoints[wp_idx]
            nav.start(SkillRequest(skill="navigate",
                                   goal={"x": wx, "y": wy},
                                   params={"goal_tolerance": goal_tolerance}))

        if dwell_remaining > 0:
            driver.stop()

            # Perform one pick/place action per dwell, halfway through
            if dwell_remaining == dwell_ticks // 2 and not action_done:
                action_done = True
                at_stock = wp_idx == 0
                at_dest = wp_idx == DESTINATION_WP[destination]

                if at_stock and not holding:
                    driver.open_gripper()
                    driver.extend_arm()
                    if pick_from_stock():
                        driver.close_gripper()
                elif at_dest and holding:
                    driver.open_gripper()
                    driver.extend_arm()
                    if place_at_destination():
                        driver.close_gripper()
                else:
                    # No real manipulation at this waypoint; just gesture
                    driver.extend_arm()

            if dwell_remaining == dwell_ticks // 4:
                driver.retract_arm()
                driver.close_gripper()

            dwell_remaining -= 1
            state = f"WORK({dwell_remaining})"

            if dwell_remaining == 0:
                wp_idx = (wp_idx + 1) % len(waypoints)
                wx, wy = waypoints[wp_idx]
                nav.start(SkillRequest(skill="navigate",
                                       goal={"x": wx, "y": wy},
                                       params={"goal_tolerance": goal_tolerance}))
                action_done = False

                if wp_idx == 0:
                    loop_count += 1
                    # Cycle destination for the next loop
                    destination = ["shelf_a", "shelf_b", "counter"][loop_count % 3]
        else:
            status = nav.tick(pb, dummy_update)
            state = f"NAV({status})"
            if status == "SUCCESS":
                dwell_remaining = dwell_ticks
                wp_name = WP_NAMES.get(wp_idx, "transit")
                print(f"{name}: arrived at wp={wp_idx} ({wp_name}) "
                      f"pos=({pose[0]:.2f},{pose[1]:.2f}) holding={holding}")

        # Publish HUD state
        if tick_count % 10 == 0:
            try:
                self_node.getField("customData").setSFString(json.dumps({
                    "waypoint": wp_idx,
                    "loop": loop_count,
                    "state": state,
                    "action": "wait" if waiting_for_customer else (
                        "pick" if at_stock else ("place" if at_dest else "transit")),
                    "holding": holding,
                    "held_id": held_id,
                    "destination": destination,
                    "restocked": restocked_count,
                    # Current target waypoint position, so the observer can learn
                    # per-destination navigate primitives and retrace the route.
                    "tx": waypoints[wp_idx][0],
                    "ty": waypoints[wp_idx][1],
                }))
            except Exception:
                pass

        tick_count += 1
        if tick_count % 60 == 0:
            print(f"{name}: t={robot.getTime():.1f} pos=({pose[0]:.2f},{pose[1]:.2f}) "
                  f"wp={wp_idx} state={state} loops={loop_count} "
                  f"dest={destination} holding={holding}")


# ============================================================================
# CUSTOMER MODE
# ============================================================================
def _run_customer(robot, timestep, name, agent_id):
    driver = TiagoDriver(robot)
    self_node = robot.getSelf()
    sensors = TiagoObjectSensors(robot, self_node, name)
    nav = TiagoNavSkill(driver)

    # Shopping sources: where the customer stands to pick items, and the
    # container id of the tray to pick from.
    # Service points follow the same open orthogonal corridor route that the
    # original customer patrol used, so the robot never scrapes furniture.
    # Grasping is done via Supervisor teleport, so the robot does not need to
    # stand right next to the tray.
    # Visit all three service points so the customer traverses the whole store.
    # Goals are aisle *approach* points (south of each fixture), not the
    # fixture centres — the shelves/counter occupy their centres, so a goal
    # there is unreachable and the navigator stalls at the keepout boundary.
    # Grasping teleports via the Supervisor, so standing in the aisle is fine.
    SOURCES = [
        {"state": "COLLECT_COUNTER", "goal": (4.0, -6.0),
         "container": "CONTAINER_counter", "label": "counter"},
        {"state": "COLLECT_SHELF_B", "goal": (1.5, -6.6),
         "container": "CONTAINER_shelf_b", "label": "shelf_b"},
        {"state": "COLLECT_SHELF_A", "goal": (-2.0, -6.6),
         "container": "CONTAINER_shelf_a", "label": "shelf_a"},
    ]
    BASKET_DROP_GOAL = (4.0, -6.0)
    BASKET_PICK_GOAL = (4.0, -6.0)
    EXIT_GOAL = (0.0, -8.5)
    BASKET_EXIT_POS = (1.5, -8.5, 0.075)
    TARGET_COUNT = len(SOURCES)

    dwell_ticks = 80
    goal_tolerance = 0.4
    dummy_update = SkillUpdate(intent="approach", params={"speed_scale": 0.5})

    # Discover the basket and remember which items belong in the trays so the
    # shopping trip can be reset cleanly after checkout.
    basket_id = sensors.find_object_by_type("Basket")
    # The basket's container id is its Solid node name ("BASKET_1"); the
    # object id returned by find_object_by_type is prefixed with "basket_".
    basket_container_id = basket_id.split("_", 1)[1] if basket_id else None
    home_item_ids = []
    for src in SOURCES:
        c = sensors.find_container_by_id(src["container"])
        if c:
            home_item_ids.extend([it["id"] for it in c["items"]])

    print(f"{name}: CUSTOMER mode — fill basket ({len(home_item_ids)} items in stock)")

    # State machine
    source_index = 0
    collected = 0
    holding_item = False
    held_item_id = None
    holding_basket = False
    action_done = False
    dwell_remaining = 0
    state = SOURCES[0]["state"]

    def start_nav(x, y):
        nav.start(SkillRequest(
            skill="navigate",
            goal={"x": x, "y": y},
            params={"goal_tolerance": goal_tolerance},
        ))

    start_nav(SOURCES[0]["goal"][0], SOURCES[0]["goal"][1])

    def perform_action():
        """Execute the one-shot manipulation for the current state."""
        nonlocal state, source_index, holding_item, held_item_id, holding_basket, collected
        if state == "DROP_BASKET":
            if holding_item and basket_container_id:
                driver.open_gripper()
                driver.extend_arm()
                if sensors.supervisor_release_into_container(basket_container_id):
                    holding_item = False
                    held_item_id = None
                    collected += 1
                    driver.close_gripper()
                    print(f"{name}: dropped item in basket (collected {collected})")
            return

        if state == "PICKUP_BASKET":
            if basket_id and not holding_basket and not holding_item:
                driver.open_gripper()
                driver.extend_arm()
                sensors.set_manipulation_target(basket_id)
                if sensors.supervisor_grasp(basket_id):
                    holding_basket = True
                    driver.close_gripper()
                    print(f"{name}: picked up basket")
            return

        if state == "TO_EXIT":
            if holding_basket and basket_id:
                driver.open_gripper()
                driver.extend_arm()
                sensors.set_manipulation_target(basket_id)
                if sensors.supervisor_release(BASKET_EXIT_POS):
                    holding_basket = False
                    driver.close_gripper()
                    print(f"{name}: dropped basket at exit")
            return

        if state == "RESET":
            reset_ids = list(home_item_ids)
            if basket_id:
                reset_ids.append(basket_id)
            sensors.reset_objects_to_initial(reset_ids)
            collected = 0
            source_index = 0
            print(f"{name}: reset shopping props")
            return

        # Collect actions (COUNTER, SHELF_A, SHELF_B)
        src = SOURCES[source_index]
        item_id = sensors.find_object_in_container(src["container"])
        if item_id is not None and not holding_item:
            driver.open_gripper()
            driver.extend_arm()
            sensors.set_manipulation_target(item_id)
            if sensors.supervisor_grasp(item_id):
                holding_item = True
                held_item_id = item_id
                driver.close_gripper()
                print(f"{name}: grabbed {item_id} from {src['label']}")
        elif item_id is None:
            print(f"{name}: no items available in {src['label']}")

    def advance_state():
        """Decide where to go after the current dwell finishes."""
        nonlocal state, source_index
        if state == "DROP_BASKET":
            source_index += 1
            if source_index < len(SOURCES):
                state = SOURCES[source_index]["state"]
                start_nav(SOURCES[source_index]["goal"][0],
                          SOURCES[source_index]["goal"][1])
            elif collected >= TARGET_COUNT:
                state = "PICKUP_BASKET"
                start_nav(BASKET_PICK_GOAL[0], BASKET_PICK_GOAL[1])
            else:
                # Loop sources again until the basket has enough items
                source_index = 0
                state = SOURCES[0]["state"]
                start_nav(SOURCES[0]["goal"][0], SOURCES[0]["goal"][1])
            return

        if state.startswith("COLLECT_"):
            state = "DROP_BASKET"
            start_nav(BASKET_DROP_GOAL[0], BASKET_DROP_GOAL[1])
            return

        if state == "PICKUP_BASKET":
            state = "TO_EXIT"
            start_nav(EXIT_GOAL[0], EXIT_GOAL[1])
            return

        if state == "TO_EXIT":
            state = "RESET"
            start_nav(EXIT_GOAL[0], EXIT_GOAL[1])
            return

        if state == "RESET":
            state = SOURCES[0]["state"]
            source_index = 0
            start_nav(SOURCES[0]["goal"][0], SOURCES[0]["goal"][1])
            return

    tick_count = 0
    while robot.step(timestep) != -1:
        try:
            raw = sensors.read()
        except Exception as e:
            print(f"{name}: ERROR in sensors.read(): {e}")
            import traceback
            traceback.print_exc()
            continue
        pb = PerceptBundle(t=robot.getTime(), world=raw)
        pose = raw.get("robot_pose", (0, 0, 0, 0))

        # Keep the held object (item or basket) attached to the gripper, then
        # update any contained items so they follow the basket.
        if sensors.held_object_id:
            sensors.update_held_position()
        sensors.update_containers()

        if dwell_remaining > 0:
            driver.stop()

            if dwell_remaining == dwell_ticks // 2 and not action_done:
                action_done = True
                perform_action()

            if dwell_remaining == dwell_ticks // 4:
                driver.retract_arm()
                driver.close_gripper()

            dwell_remaining -= 1
            display_state = f"{state}({dwell_remaining})"
            if dwell_remaining == 0:
                advance_state()
                action_done = False
        else:
            status = nav.tick(pb, dummy_update)
            display_state = f"NAV({status})"
            if status == "SUCCESS":
                dwell_remaining = dwell_ticks
                print(f"{name}: arrived at {state} pos=({pose[0]:.2f},{pose[1]:.2f}) "
                      f"collected={collected}")

        # Publish HUD state
        if tick_count % 10 == 0:
            try:
                basket_count = sensors.get_container_count(basket_container_id) \
                    if basket_container_id else 0
                self_node.getField("customData").setSFString(json.dumps({
                    "state": display_state,
                    "action": state,
                    "holding_item": held_item_id,
                    "holding_basket": holding_basket,
                    "collected": collected,
                    "basket_count": basket_count,
                }))
            except Exception:
                pass

        tick_count += 1
        if tick_count % 60 == 0:
            print(f"{name}: t={robot.getTime():.1f} pos=({pose[0]:.2f},{pose[1]:.2f}) "
                  f"state={state} collected={collected} basket={basket_container_id}")


# ============================================================================
# LEARNER MODE — observational learning + execution
# ============================================================================
def _run_learner(robot, timestep, name, agent_id, goal_x, goal_y, alpha):
    phase_level = int(os.environ.get("PHASE_LEVEL", "9"))
    driver = TiagoDriver(robot)
    sensors = TiagoObjectSensors(robot, robot.getSelf(), name)
    registry = SkillRegistry()
    tiago_plugin.register(registry, driver)

    # Perception
    augmentations = [ProxemicsAugmentation()]
    if phase_level >= 3:
        augmentations.extend([
            AffectAugmentation(),
            EngagementAugmentation(),
            SaliencyAugmentation(),
        ])
    perception = PerceptionPipeline(sensors=sensors, augmentations=augmentations)

    # Norms
    norm_rules = [
        PersonalSpaceRule(min_distance=0.8, veto_distance=0.4),
        SpeedLimitRule(max_speed=1.0),
        RetailWaitRule(),
    ]
    if phase_level >= 4:
        norm_rules.extend([
            AffectModulatedSpeedRule(base_max_speed=1.0),
            DistressVetoRule(),
        ])

    predictor = None
    if phase_level >= 9:
        predictor = VelocityPredictor(robot_radius=0.25, default_agent_radius=0.25)
        norm_rules.append(PredictiveCollisionRule(predictor=predictor))

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

    empathy_mod = None
    if phase_level >= 4:
        empathy_mod = EmpathicModulator()

    # Weak-script recognizer with retail situations
    recognizer = WeakScriptRecognizer(
        situation_types=[
            SituationType(
                name="stock_zone",
                feature_weights={"cue_stock": 4.0, "mean_velocity": -1.0},
            ),
            SituationType(
                name="shelf_zone",
                feature_weights={"cue_shelf": 4.0, "mean_velocity": -0.5},
            ),
            SituationType(
                name="counter_zone",
                feature_weights={"cue_counter": 4.0, "mean_velocity": -0.5},
            ),
            SituationType(
                name="customer_present",
                feature_weights={"inverse_min_distance": 2.0, "mean_velocity": 0.5},
            ),
            SituationType(
                name="approach",
                feature_weights={"mean_velocity": 2.0, "inverse_min_distance": -0.2},
            ),
        ],
        prior={
            "stock_zone": 0.2,
            "shelf_zone": 0.2,
            "counter_zone": 0.2,
            "customer_present": 0.15,
            "approach": 0.25,
        },
    )

    # Library with defaults only — retail primitives discovered via observation
    library = PrimitiveLibrary(register_defaults=True)

    # Naive script: same waypoints as teacher (fallback if observation fails)
    seq = ScriptSequence(
        name="learner_naive",
        steps=[
            ScriptStep(SkillRequest(skill="navigate",
                       goal={"x": -5.5, "y": -1.0}),
                       expected_situation="stock_zone"),
            ScriptStep(SkillRequest(skill="navigate",
                       goal={"x": -2.0, "y": -5.0}),
                       expected_situation="shelf_zone"),
            ScriptStep(SkillRequest(skill="navigate",
                       goal={"x": 1.5, "y": -5.0}),
                       expected_situation="shelf_zone"),
            ScriptStep(SkillRequest(skill="navigate",
                       goal={"x": 4.5, "y": -7.5}),
                       expected_situation="counter_zone"),
        ],
        loop=True,
    )

    # Repertoire starts EMPTY — patterns discovered from scratch via observation
    config = make_retail_repertoire_config()
    repertoire = ScriptRepertoire(
        library,
        config=config,
        initial_patterns=[],
        observational_mode=True,
    )

    scripts_manager = ScriptManager(sequence=seq, recognizer=recognizer)
    scripts_manager = DemoLearningScriptManager(scripts_manager, repertoire)

    shield = SafetyShield(emergency_stop_distance=0.4, predictor=predictor)

    bb = Blackboard()
    executive = Executive(
        bb=bb,
        registry=registry,
        scripts=scripts_manager,
        norms=NormEngine(rules=norm_rules),
        tom=ToMModulator(adapter=None, gated_tom=gated_tom),
        shield=shield,
        deliberation_hz=5.0,
        empathy=empathy_mod,
    )

    # ---- Observation phase ----
    observer = TeacherObserver(robot, teacher_name="Worker_T", library=library)
    observe_mode = True
    observed_loops = 0
    crystallized = False

    # Stabilise (teacher starts moving while learner watches)
    for _ in range(300):
        robot.step(timestep)

    print(f"{name}: LEARNER mode — starting OBSERVATION phase")

    tick_count = 0
    log_interval = 30
    self_node = robot.getSelf()

    while robot.step(timestep) != -1:
        t = robot.getTime()

        if observe_mode:
            # --- OBSERVATION: stay still, watch teacher, feed synthetic trajectories ---
            driver.stop()
            obs_status = observer.tick(t, repertoire, recognizer)

            if obs_status["loop_count"] > observed_loops:
                observed_loops = obs_status["loop_count"]
                bp = (max(repertoire.patterns.values(), key=lambda p: p.precision)
                      if repertoire.patterns else None)
                print(f"{name}: OBSERVED loop {observed_loops} complete "
                      f"(teacher at {obs_status['last_wp']}) | "
                      f"patterns={len(repertoire.patterns)} "
                      f"best={bp.name if bp else '-'} "
                      f"prec={bp.precision:.2f} traj={bp.trajectory_count if bp else 0} "
                      f"prims={len(obs_status['discovered_primitives'])}")

            # Check for crystallization
            strong = [p for p in repertoire.patterns.values() if p.is_strong]
            if strong and not crystallized:
                crystallized = True
                best = max(strong, key=lambda p: p.precision)
                print(f"{name}: >>> CRYSTALLIZED from observation! "
                      f"pattern={best.name} prec={best.precision:.2f}")

            # Transition to execution after crystallization + 1 extra observed loop
            if crystallized and observed_loops >= 2:
                learned_seq = build_learned_sequence(repertoire)
                if learned_seq is not None:
                    scripts_manager.base.set_sequence(learned_seq)
                    print(f"{name}: >>> SWITCHING TO EXECUTION — learned script installed "
                          f"({len(learned_seq.steps)} steps)")
                observe_mode = False
                observer.reset()
                # Re-initialise executive with fresh blackboard for execution
                bb = Blackboard()
                executive = Executive(
                    bb=bb,
                    registry=registry,
                    scripts=scripts_manager,
                    norms=NormEngine(rules=norm_rules),
                    tom=ToMModulator(adapter=None, gated_tom=gated_tom),
                    shield=shield,
                    deliberation_hz=5.0,
                    empathy=empathy_mod,
                )

            # Publish HUD state during observation
            if tick_count % 10 == 0:
                best_pattern = None
                if repertoire.patterns:
                    best_pattern = max(repertoire.patterns.values(), key=lambda p: p.precision)
                hud_state = {
                    "intent": "OBSERVE",
                    "affect": "0.00,0.00",
                    "speed": "0.00",
                    "pattern": best_pattern.name if best_pattern else f"observing {obs_status.get('last_wp', 'none')}",
                    "precision": round(best_pattern.precision, 2) if best_pattern else 0.0,
                    "is_strong": best_pattern.is_strong if best_pattern else False,
                    "patterns": len(repertoire.patterns),
                    "primitives": len(library.names),
                    "observed_loops": observed_loops,
                    "teacher_pos": obs_status.get("teacher_pos"),
                }
                try:
                    self_node.getField("customData").setSFString(json.dumps(hud_state))
                except Exception:
                    pass

        else:
            # --- EXECUTION: run the learned script ---
            bb.percept = perception.tick(t)
            executive.tick(t)

            if tick_count % 10 == 0:
                _publish_learner_state(self_node, bb, phase_level, gated_tom, repertoire)

        tick_count += 1
        if tick_count % log_interval == 0 and not observe_mode:
            _log_learner(name, bb, t, phase_level, gated_tom, repertoire)


def _extract_learner_state(bb, phase_level, gated_tom, repertoire):
    """Build a dict of learner cognitive state for HUD."""
    state = {
        "intent": "-",
        "affect": "-,-",
        "speed": "-",
        "pattern": "none",
        "precision": 0.0,
        "is_strong": False,
        "patterns": 0,
    }
    if not bb.percept:
        return state

    intent_str = getattr(bb, '_compiled_intent', "-")
    state["intent"] = intent_str

    compiled_spd = getattr(bb, '_compiled_speed_scale', None)
    if compiled_spd is not None:
        state["speed"] = f"{compiled_spd:.2f}"

    if phase_level >= 4 and bb.self_affect is not None:
        state["affect"] = f"{bb.self_affect.arousal:.2f},{bb.self_affect.valence:.2f}"

    if repertoire is not None:
        patterns = repertoire.patterns
        state["patterns"] = len(patterns)
        if patterns:
            # Report the highest-precision pattern
            best = max(patterns.values(), key=lambda p: p.precision)
            state["pattern"] = best.name
            state["precision"] = round(best.precision, 2)
            state["is_strong"] = best.is_strong

    return state


def _publish_learner_state(self_node, bb, phase_level, gated_tom, repertoire):
    state = _extract_learner_state(bb, phase_level, gated_tom, repertoire)
    try:
        self_node.getField("customData").setSFString(json.dumps(state))
    except Exception:
        pass


def _log_learner(name, bb, t, phase_level, gated_tom, repertoire):
    if not bb.percept:
        return
    pose = bb.percept.world.get("robot_pose")
    if not pose:
        return

    agents = bb.percept.world.get("agents", [])
    agent_dist = "none"
    if agents:
        ap = agents[0].get("pose", (0, 0))
        agent_dist = f"{((pose[0]-ap[0])**2 + (pose[1]-ap[1])**2)**0.5:.2f}"

    affect_str = "-"
    if phase_level >= 4 and bb.self_affect is not None:
        affect_str = f"({bb.self_affect.arousal:.2f},{bb.self_affect.valence:.2f})"

    intent_str = getattr(bb, '_compiled_intent', "-")
    speed_str = "-"
    compiled_spd = getattr(bb, '_compiled_speed_scale', None)
    if compiled_spd is not None:
        speed_str = f"{compiled_spd:.2f}"

    rel_str = "-"
    if phase_level >= 5 and gated_tom is not None and agents:
        rel = gated_tom.agent_reliability(agents[0].get("id", 0))
        rel_str = f"{rel:.2f}"

    learn_str = ""
    if repertoire is not None:
        patterns = repertoire.patterns
        if patterns:
            for pname, p in patterns.items():
                learn_str += (
                    f" | {pname}:prec={p.precision:.2f},strong={p.is_strong},n={p.trajectory_count}"
                )
        else:
            learn_str = " | no patterns"

    print(
        f"{name}: t={t:.1f} pos=({pose[0]:.2f},{pose[1]:.2f}) "
        f"ad={agent_dist} affect={affect_str} intent={intent_str} "
        f"spd={speed_str} rel={rel_str}{learn_str}"
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main() -> None:
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    name = robot.getName()
    self_node = robot.getSelf()

    # Logging
    _log_path = os.path.join(_LOG_DIR, f"{name}.log")
    _log_fh = open(_log_path, "w")
    import builtins
    _orig_print = builtins.print
    def _print(*args, **kwargs):
        _orig_print(*args, **kwargs)
        _orig_print(*args, **kwargs, file=_log_fh, flush=True)
    builtins.print = _print

    # Parse customData: goal_x,goal_y,alpha,agent_id
    goal_x, goal_y, alpha, agent_id = -3.0, -1.0, 3.0, 1
    custom_data = robot.getCustomData()
    if custom_data:
        parts = custom_data.strip().split(",")
        if len(parts) >= 2:
            goal_x, goal_y = float(parts[0]), float(parts[1])
        if len(parts) >= 3:
            alpha = float(parts[2])
        if len(parts) >= 4:
            agent_id = int(parts[3])

    print(f"{'='*60}")
    print(f"{name}: RETAIL DEMO  goal=({goal_x},{goal_y}) alpha={alpha} id={agent_id}")
    print(f"{'='*60}")

    try:
        if agent_id == 0:
            _run_teacher(robot, timestep, name, agent_id)
        elif agent_id == 1:
            _run_learner(robot, timestep, name, agent_id, goal_x, goal_y, alpha)
        else:
            _run_customer(robot, timestep, name, agent_id)
    except Exception as e:
        import traceback
        print(f"{name}: FATAL ERROR: {e}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
