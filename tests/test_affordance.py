"""Tests for embodied affordance reasoning."""

from __future__ import annotations

import math
import pytest

from architecture_core.safety.geometry import Rect
from architecture_core.cognition.planning.affordance import (
    ApproachCandidate,
    ClearingAction,
    ClearingPlan,
    EmbodimentModel,
    FurnitureItem,
    SceneGraph,
    compute_approach_candidates,
    compute_pull_position,
    compute_push_position,
    select_clearing_action,
    _epistemic_for_action,
)


# =====================================================================
# Helpers
# =====================================================================
def _make_furniture(
    fid: str = "chair_1",
    ftype: str = "Chair",
    pos: tuple = (1.0, 0.0),
    width: float = 0.5,
    depth: float = 0.5,
    mass: float = 3.0,
    movable: bool = True,
    padding: float = 0.3,
) -> FurnitureItem:
    """Create a FurnitureItem with auto-computed keepout."""
    half_w = width / 2.0 + padding
    half_d = depth / 2.0 + padding
    keepout = Rect(
        pos[0] - half_w, pos[1] - half_d,
        pos[0] + half_w, pos[1] + half_d,
    )
    return FurnitureItem(
        id=fid, type=ftype, position=pos, rotation=0.0,
        width=width, depth=depth, mass=mass, movable=movable,
        keepout=keepout,
    )


# =====================================================================
# TestEmbodimentModel
# =====================================================================
class TestEmbodimentModel:
    def test_default_values(self):
        em = EmbodimentModel()
        assert em.arm_reach == 0.6
        assert em.body_radius == 0.27
        assert em.max_push_mass == 5.0
        assert em.grip_strength == 5.0
        assert em.push_precision == 0.1
        assert em.grip_precision == 0.1

    def test_custom_embodiment(self):
        em = EmbodimentModel(arm_reach=0.8, body_radius=0.3, max_push_mass=10.0)
        assert em.arm_reach == 0.8
        assert em.max_push_mass == 10.0

    def test_precision_increases_on_success(self):
        em = EmbodimentModel()
        old = em.push_precision
        em.update_precision(ClearingAction.PUSH_FORWARD, success=True)
        assert em.push_precision > old

    def test_precision_decreases_on_failure(self):
        em = EmbodimentModel(push_precision=1.0)
        old = em.push_precision
        em.update_precision(ClearingAction.PUSH_FORWARD, success=False)
        assert em.push_precision < old

    def test_precision_does_not_go_below_minimum(self):
        em = EmbodimentModel(push_precision=0.05)
        em.update_precision(ClearingAction.PUSH_FORWARD, success=False)
        assert em.push_precision >= 0.01

    def test_grip_precision_updates_independently(self):
        em = EmbodimentModel()
        em.update_precision(ClearingAction.GRAB_AND_PULL, success=True)
        assert em.grip_precision > 0.1
        assert em.push_precision == 0.1  # unchanged

    def test_navigate_around_no_precision_change(self):
        em = EmbodimentModel()
        em.update_precision(ClearingAction.NAVIGATE_AROUND, success=True)
        assert em.push_precision == 0.1
        assert em.grip_precision == 0.1


# =====================================================================
# TestApproachCandidates
# =====================================================================
class TestApproachCandidates:
    def test_generates_n_candidates_no_obstacles(self):
        cands = compute_approach_candidates(
            object_pos=(0, 0), obstacles=[], embodiment=EmbodimentModel(),
            robot_pos=(2, 0), n_candidates=8,
        )
        assert len(cands) == 8

    def test_candidates_at_arm_reach_distance(self):
        em = EmbodimentModel(arm_reach=0.6)
        cands = compute_approach_candidates(
            object_pos=(0, 0), obstacles=[], embodiment=em,
            robot_pos=(2, 0), n_candidates=4,
        )
        for c in cands:
            dist = math.sqrt(c.x ** 2 + c.y ** 2)
            assert abs(dist - 0.6) < 0.01

    def test_filters_immovable_obstacles(self):
        # Place a large immovable obstacle covering all approach positions
        table = _make_furniture(
            fid="table_1", ftype="Table", pos=(0, 0),
            width=3.0, depth=3.0, mass=15.0, movable=False, padding=0.5,
        )
        cands = compute_approach_candidates(
            object_pos=(0, 0), obstacles=[table], embodiment=EmbodimentModel(),
            robot_pos=(5, 0), n_candidates=8,
        )
        # All candidates should be inside the keepout -> filtered out
        assert len(cands) == 0

    def test_movable_obstacle_adds_clearing_cost(self):
        chair = _make_furniture(
            pos=(0.6, 0.0), movable=True, mass=3.0,
        )
        cands = compute_approach_candidates(
            object_pos=(0, 0), obstacles=[chair], embodiment=EmbodimentModel(),
            robot_pos=(3, 0), n_candidates=8,
        )
        blocked = [c for c in cands if c.blocked_by]
        clear = [c for c in cands if not c.blocked_by]
        # Some candidates should be blocked by the chair
        if blocked:
            assert blocked[0].clearing_cost != 0 or len(blocked[0].clearing_plans) > 0
        # Clear candidates should have zero clearing cost
        for c in clear:
            assert c.clearing_cost == 0.0

    def test_clear_candidate_has_zero_clearing_cost(self):
        cands = compute_approach_candidates(
            object_pos=(0, 0), obstacles=[], embodiment=EmbodimentModel(),
            robot_pos=(2, 0), n_candidates=4,
        )
        for c in cands:
            assert c.clearing_cost == 0.0
            assert len(c.clearing_plans) == 0

    def test_all_blocked_by_immovable_returns_empty(self):
        walls = [
            _make_furniture(
                fid=f"wall_{i}", pos=(0, 0), width=5, depth=5,
                mass=100, movable=False, padding=1.0,
            )
            for i in range(1)
        ]
        cands = compute_approach_candidates(
            object_pos=(0, 0), obstacles=walls, embodiment=EmbodimentModel(),
            robot_pos=(10, 0), n_candidates=8,
        )
        assert len(cands) == 0

    def test_nearest_candidate_has_lowest_nav_cost(self):
        cands = compute_approach_candidates(
            object_pos=(0, 0), obstacles=[], embodiment=EmbodimentModel(),
            robot_pos=(2, 0), n_candidates=8,
        )
        # First candidate (sorted) should have lowest nav_cost
        nav_costs = [c.nav_cost for c in cands]
        assert nav_costs[0] == min(nav_costs)

    def test_candidates_sorted_by_total_cost(self):
        cands = compute_approach_candidates(
            object_pos=(0, 0), obstacles=[], embodiment=EmbodimentModel(),
            robot_pos=(2, 0), n_candidates=8,
        )
        total_costs = [c.nav_cost + c.clearing_cost + c.epistemic_value for c in cands]
        assert total_costs == sorted(total_costs)


# =====================================================================
# TestClearingActionSelection
# =====================================================================
class TestClearingActionSelection:
    def test_push_feasible_when_mass_under_limit(self):
        chair = _make_furniture(mass=3.0, movable=True)
        em = EmbodimentModel(max_push_mass=5.0)
        plans = select_clearing_action(
            chair, (2, 0), (3, 0), em, [],
        )
        actions = [p.action for p in plans]
        assert ClearingAction.PUSH_FORWARD in actions

    def test_push_infeasible_when_mass_over_limit(self):
        heavy = _make_furniture(mass=10.0, movable=True)
        em = EmbodimentModel(max_push_mass=5.0)
        plans = select_clearing_action(
            heavy, (2, 0), (3, 0), em, [],
        )
        actions = [p.action for p in plans]
        assert ClearingAction.PUSH_FORWARD not in actions

    def test_grab_and_pull_feasible_when_mass_under_grip(self):
        chair = _make_furniture(mass=3.0, movable=True)
        em = EmbodimentModel(grip_strength=5.0)
        plans = select_clearing_action(
            chair, (2, 0), (3, 0), em, [],
        )
        actions = [p.action for p in plans]
        assert ClearingAction.GRAB_AND_PULL in actions

    def test_grab_and_pull_infeasible_when_mass_over_grip(self):
        heavy = _make_furniture(mass=10.0, movable=True)
        em = EmbodimentModel(grip_strength=5.0)
        plans = select_clearing_action(
            heavy, (2, 0), (3, 0), em, [],
        )
        actions = [p.action for p in plans]
        assert ClearingAction.GRAB_AND_PULL not in actions

    def test_no_plans_when_too_heavy_for_all(self):
        """When obstacle exceeds both push and grip, no clearing plans."""
        heavy = _make_furniture(mass=100.0, movable=True)
        em = EmbodimentModel(max_push_mass=5.0, grip_strength=5.0)
        plans = select_clearing_action(
            heavy, (2, 0), (3, 0), em, [],
        )
        assert len(plans) == 0

    def test_push_needs_clear_space_behind(self):
        chair = _make_furniture(fid="chair", pos=(1.0, 0.0), mass=3.0)
        # Wall behind the chair (blocking push start position)
        wall = _make_furniture(
            fid="wall", pos=(0.0, 0.0), width=2.0, depth=2.0,
            mass=100.0, movable=False, padding=0.5,
        )
        em = EmbodimentModel(max_push_mass=5.0)
        plans = select_clearing_action(
            chair, (2, 0), (3, 0), em, [wall],
        )
        actions = [p.action for p in plans]
        assert ClearingAction.PUSH_FORWARD not in actions

    def test_pull_needs_clear_space_in_front(self):
        chair = _make_furniture(fid="chair", pos=(1.0, 0.0), mass=3.0)
        # Wall on the approach side (blocking pull nav-to and pull-to)
        wall = _make_furniture(
            fid="wall", pos=(2.5, 0.0), width=3.0, depth=3.0,
            mass=100.0, movable=False, padding=0.5,
        )
        em = EmbodimentModel(grip_strength=5.0)
        plans = select_clearing_action(
            chair, (2, 0), (3, 0), em, [wall],
        )
        actions = [p.action for p in plans]
        assert ClearingAction.GRAB_AND_PULL not in actions

    def test_returns_sorted_by_efe(self):
        chair = _make_furniture(mass=3.0, movable=True)
        em = EmbodimentModel()
        plans = select_clearing_action(
            chair, (2, 0), (3, 0), em, [],
        )
        efes = [p.efe for p in plans]
        assert efes == sorted(efes)

    def test_push_lower_efe_than_pull_when_both_feasible(self):
        chair = _make_furniture(mass=3.0, movable=True)
        # Equal precision -> push has lower effort cost (0.3 vs 0.5)
        em = EmbodimentModel(push_precision=1.0, grip_precision=1.0)
        plans = select_clearing_action(
            chair, (2, 0), (3, 0), em, [],
        )
        push_plans = [p for p in plans if p.action == ClearingAction.PUSH_FORWARD]
        pull_plans = [p for p in plans if p.action == ClearingAction.GRAB_AND_PULL]
        if push_plans and pull_plans:
            assert push_plans[0].efe < pull_plans[0].efe


# =====================================================================
# TestEpistemicDrive
# =====================================================================
class TestEpistemicDrive:
    def test_low_push_precision_gives_high_epistemic_value(self):
        em = EmbodimentModel(push_precision=0.1)
        val = _epistemic_for_action(ClearingAction.PUSH_FORWARD, em)
        assert val < 0  # negative = good (drives exploration)

    def test_high_push_precision_gives_low_epistemic_value(self):
        em_low = EmbodimentModel(push_precision=0.1)
        em_high = EmbodimentModel(push_precision=5.0)
        val_low = _epistemic_for_action(ClearingAction.PUSH_FORWARD, em_low)
        val_high = _epistemic_for_action(ClearingAction.PUSH_FORWARD, em_high)
        # High precision -> less negative (less exploration drive)
        assert val_high > val_low

    def test_epistemic_can_favor_novel_action_over_cheaper(self):
        # Push has lower effort (0.3) but high precision (well-known)
        # Pull has higher effort (0.5) but low precision (novel)
        em = EmbodimentModel(
            push_precision=10.0,  # very well-known
            grip_precision=0.01,  # very novel
            epistemic_weight=1.0,  # strong epistemic drive
        )
        push_efe = 0.3 + _epistemic_for_action(ClearingAction.PUSH_FORWARD, em)
        pull_efe = 0.5 + _epistemic_for_action(ClearingAction.GRAB_AND_PULL, em)
        # Epistemic value should make pull cheaper despite higher effort
        assert pull_efe < push_efe

    def test_navigate_around_has_zero_epistemic(self):
        em = EmbodimentModel()
        val = _epistemic_for_action(ClearingAction.NAVIGATE_AROUND, em)
        assert val == 0.0


# =====================================================================
# TestClearingPlans
# =====================================================================
class TestClearingPlans:
    def test_push_plan_has_nav_to_and_push_through(self):
        chair = _make_furniture(mass=3.0, movable=True)
        em = EmbodimentModel()
        plans = select_clearing_action(
            chair, (2, 0), (3, 0), em, [],
        )
        push_plans = [p for p in plans if p.action == ClearingAction.PUSH_FORWARD]
        assert len(push_plans) > 0
        p = push_plans[0]
        assert p.nav_to is not None
        assert p.push_through is not None

    def test_pull_plan_has_nav_to_grasp_target_pull_to(self):
        chair = _make_furniture(mass=3.0, movable=True)
        em = EmbodimentModel()
        plans = select_clearing_action(
            chair, (2, 0), (3, 0), em, [],
        )
        pull_plans = [p for p in plans if p.action == ClearingAction.GRAB_AND_PULL]
        assert len(pull_plans) > 0
        p = pull_plans[0]
        assert p.nav_to is not None
        assert p.grasp_target is not None
        assert p.pull_to is not None

    def test_no_navigate_around_in_clearing_plans(self):
        """NAVIGATE_AROUND is implicit (choose different candidate), not in plans."""
        chair = _make_furniture(mass=3.0, movable=True)
        em = EmbodimentModel()
        plans = select_clearing_action(
            chair, (2, 0), (3, 0), em, [],
        )
        nav_plans = [p for p in plans if p.action == ClearingAction.NAVIGATE_AROUND]
        assert len(nav_plans) == 0


# =====================================================================
# TestPushPosition
# =====================================================================
class TestPushPosition:
    def test_push_position_opposite_approach_direction(self):
        chair = _make_furniture(pos=(1.0, 0.0))
        em = EmbodimentModel()
        start, through = compute_push_position(
            chair, approach_pos=(2.0, 0.0), embodiment=em,
        )
        # Start should be on the side opposite the approach (x < 1.0)
        assert start[0] < chair.position[0]

    def test_push_through_past_obstacle_center(self):
        chair = _make_furniture(pos=(1.0, 0.0))
        em = EmbodimentModel()
        start, through = compute_push_position(
            chair, approach_pos=(2.0, 0.0), embodiment=em,
        )
        # Through should be past obstacle center toward approach
        assert through[0] > chair.position[0]


# =====================================================================
# TestPullPosition
# =====================================================================
class TestPullPosition:
    def test_pull_nav_to_at_arm_reach_from_edge(self):
        chair = _make_furniture(pos=(1.0, 0.0))
        em = EmbodimentModel(arm_reach=0.6)
        nav_to, grasp, pull_to = compute_pull_position(
            chair, approach_pos=(2.0, 0.0), embodiment=em,
        )
        # Nav-to should be arm_reach away from grasp point
        dist = math.sqrt(
            (nav_to[0] - grasp[0]) ** 2 + (nav_to[1] - grasp[1]) ** 2,
        )
        assert abs(dist - 0.6) < 0.01

    def test_grasp_point_on_nearest_edge(self):
        chair = _make_furniture(pos=(1.0, 0.0), width=0.5, depth=0.5)
        em = EmbodimentModel()
        _, grasp, _ = compute_pull_position(
            chair, approach_pos=(2.0, 0.0), embodiment=em,
        )
        # Grasp should be on the edge toward approach (x > 1.0)
        assert grasp[0] > chair.position[0]

    def test_pull_to_behind_nav_position(self):
        chair = _make_furniture(pos=(1.0, 0.0))
        em = EmbodimentModel()
        nav_to, _, pull_to = compute_pull_position(
            chair, approach_pos=(2.0, 0.0), embodiment=em,
        )
        # Pull-to should be further from obstacle than nav-to
        dist_nav = math.sqrt(
            (nav_to[0] - chair.position[0]) ** 2
            + (nav_to[1] - chair.position[1]) ** 2,
        )
        dist_pull = math.sqrt(
            (pull_to[0] - chair.position[0]) ** 2
            + (pull_to[1] - chair.position[1]) ** 2,
        )
        assert dist_pull > dist_nav


# =====================================================================
# TestSceneGraph
# =====================================================================
class _FakeObject:
    """Minimal object stand-in for SceneGraph tests."""
    def __init__(self, oid, pos):
        self.id = oid
        self.position = pos


class TestSceneGraph:
    def test_infers_support_from_containment(self):
        """Object inside table's physical rect → supported by table."""
        table = _make_furniture(
            fid="dining", pos=(0, 0), width=1.2, depth=0.8,
            movable=False, mass=15,
        )
        obj = _FakeObject("orange_1", (0.1, 0.1))
        sg = SceneGraph(furniture=[table], objects=[obj])
        assert sg.support_surface("orange_1") is table

    def test_no_support_when_outside(self):
        table = _make_furniture(
            fid="dining", pos=(0, 0), width=1.2, depth=0.8,
            movable=False, mass=15,
        )
        obj = _FakeObject("orange_1", (5, 5))  # far from table
        sg = SceneGraph(furniture=[table], objects=[obj])
        assert sg.support_surface("orange_1") is None

    def test_reachable_near_edge(self):
        table = _make_furniture(
            fid="t", pos=(0, 0), width=1.0, depth=1.0, movable=False, mass=15,
        )
        em = EmbodimentModel(arm_reach=0.6, body_radius=0.27)
        sg = SceneGraph(furniture=[table])
        # Object 0.1m from edge → 0.4m from center on a 1.0m table (half=0.5)
        assert sg.reachable((0.4, 0), em) is True

    def test_unreachable_at_center_of_large_table(self):
        table = _make_furniture(
            fid="t", pos=(0, 0), width=3.0, depth=3.0, movable=False, mass=15,
        )
        em = EmbodimentModel(arm_reach=0.6, body_radius=0.27)
        sg = SceneGraph(furniture=[table])
        # Object at dead center of 3m table — 1.5m from edge, way beyond reach
        assert sg.reachable((0, 0), em) is False

    def test_reachable_when_not_on_surface(self):
        """Object not on any surface is always reachable."""
        sg = SceneGraph(furniture=[])
        em = EmbodimentModel()
        assert sg.reachable((5, 5), em) is True

    def test_support_relations_dict(self):
        table = _make_furniture(fid="t1", pos=(0, 0), width=2, depth=2, movable=False, mass=15)
        objs = [_FakeObject("a", (0, 0)), _FakeObject("b", (5, 5))]
        sg = SceneGraph(furniture=[table], objects=objs)
        rels = sg.support_relations
        assert rels["a"] == "t1"
        assert "b" not in rels


# =====================================================================
# TestIntegrationWithTaskPlanner
# =====================================================================
class TestIntegrationWithTaskPlanner:
    """Integration-level tests verifying affordance module output
    is compatible with TaskPlanner expectations."""

    def test_approach_position_not_on_object(self):
        """Approach candidates should be at arm_reach, not at the object."""
        cands = compute_approach_candidates(
            object_pos=(0, 0), obstacles=[], embodiment=EmbodimentModel(),
            robot_pos=(3, 0), n_candidates=8,
        )
        for c in cands:
            dist = math.sqrt(c.x ** 2 + c.y ** 2)
            assert dist > 0.3  # not at the object

    def test_push_steps_possible_when_needed(self):
        """When a movable obstacle blocks, clearing_plans should be non-empty."""
        chair = _make_furniture(pos=(0.5, 0.0), mass=3.0, movable=True)
        cands = compute_approach_candidates(
            object_pos=(0, 0), obstacles=[chair], embodiment=EmbodimentModel(),
            robot_pos=(3, 0), n_candidates=8,
        )
        blocked = [c for c in cands if c.blocked_by]
        for c in blocked:
            assert len(c.clearing_plans) > 0

    def test_no_clearing_steps_when_path_clear(self):
        cands = compute_approach_candidates(
            object_pos=(0, 0), obstacles=[], embodiment=EmbodimentModel(),
            robot_pos=(3, 0), n_candidates=8,
        )
        for c in cands:
            assert len(c.clearing_plans) == 0

    def test_object_on_table_has_valid_candidates(self):
        """Object near table edge — arm reaches across, body at table edge.

        The table is an obstacle for the body, but arm_reach extends
        beyond the body.  Candidates at arm_reach from the object that
        have the body outside the physical table footprint are valid.
        Objects near edges are reachable; objects at exact center of
        large tables may not be (correct physics).
        """
        # Dining table: 1.2 x 0.8, object near the edge (not dead center)
        table = _make_furniture(
            fid="dining_table", ftype="Table", pos=(0, 0),
            width=1.2, depth=0.8, mass=15.0, movable=False, padding=0.62,
        )
        em = EmbodimentModel(arm_reach=0.6, body_radius=0.27)
        # Object near the short-axis edge: (0, 0.25) — 0.15m from edge
        cands = compute_approach_candidates(
            object_pos=(0, 0.25), obstacles=[table], embodiment=em,
            robot_pos=(3, 0), n_candidates=16,
        )
        # Should have feasible candidates from the nearby edge
        assert len(cands) > 0

    def test_body_outside_physical_footprint(self):
        """Candidates use physical footprint, not padded keepout."""
        # Small table with huge keepout padding
        table = _make_furniture(
            fid="t", pos=(0, 0), width=0.5, depth=0.5,
            mass=15.0, movable=False, padding=5.0,  # enormous padding
        )
        em = EmbodimentModel(arm_reach=0.6, body_radius=0.27)
        cands = compute_approach_candidates(
            object_pos=(0, 0), obstacles=[table], embodiment=em,
            robot_pos=(3, 0), n_candidates=8,
        )
        # Despite huge keepout padding, candidates should still work
        # because collision uses physical footprint (0.5x0.5), not keepout
        assert len(cands) > 0

    def test_efe_includes_clearing_cost(self):
        chair = _make_furniture(pos=(0.5, 0.0), mass=3.0, movable=True)
        cands_blocked = compute_approach_candidates(
            object_pos=(0, 0), obstacles=[chair], embodiment=EmbodimentModel(),
            robot_pos=(3, 0), n_candidates=8,
        )
        cands_clear = compute_approach_candidates(
            object_pos=(0, 0), obstacles=[], embodiment=EmbodimentModel(),
            robot_pos=(3, 0), n_candidates=8,
        )
        # Find a blocked candidate and compare total cost
        blocked = [c for c in cands_blocked if c.blocked_by]
        if blocked:
            # Clearing cost should be non-zero for blocked candidates
            assert any(c.clearing_cost != 0 for c in blocked)
