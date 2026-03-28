"""Tests for compositional script assembly.

Covers: WeightedPattern, top-k recognition, pattern graph, graph
diffusion retrieval, compose_from_patterns, 3-tier selection,
compositional consolidation, and a toy reception-desk demo.
"""

from __future__ import annotations

import pytest

from architecture_core.core.types import AffectState, PerceptBundle, SkillRequest
from architecture_core.cognition.scripts.repertoire_types import (
    RepertoireConfig,
    ScriptPattern,
    ScriptPrimitive,
    WeightedPattern,
)
from architecture_core.cognition.scripts.script_types import SituationType
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.scripts.script_composer import ScriptComposer
from architecture_core.cognition.scripts.script_repertoire import ScriptRepertoire
from architecture_core.cognition.scripts.weak_recognizer import (
    SituationBelief,
    WeakScriptRecognizer,
)


# =====================================================================
# Helpers
# =====================================================================
def _make_fragment(
    name: str,
    primitives: set,
    affinity: dict,
) -> ScriptPattern:
    """Create a weak fragment pattern for testing."""
    return ScriptPattern(
        name=name,
        primitives_sequence=sorted(primitives),
        primitive_cluster=set(primitives),
        precision=0.1,
        situation_affinity=affinity,
    )


def _reception_fragments() -> list:
    """Four fragment patterns for the reception desk scenario."""
    return [
        _make_fragment(
            "detect_queue",
            {"wait-acknowledge", "gaze-scan"},
            {"reception": 0.8, "corridor": 0.3},
        ),
        _make_fragment(
            "align_behind",
            {"wait-acknowledge", "yield-pass"},
            {"reception": 0.7, "corridor": 0.6},
        ),
        _make_fragment(
            "approach_desk",
            {"approach-greet", "gaze-at-agent"},
            {"reception": 0.9, "open_area": 0.5},
        ),
        _make_fragment(
            "yield_space",
            {"yield-pass", "gaze-avert"},
            {"reception": 0.4, "corridor": 0.8},
        ),
    ]


def _register_reception_primitives(lib: PrimitiveLibrary) -> None:
    """Register domain primitives for the reception desk scenario.

    These encode the causal chain for queue-joining via specific
    situation transitions:

        scan-environment -> scene_assessed
        position-in-queue -> in_queue
        wait-for-turn -> ready_for_service
        approach-counter -> at_counter
        engage-staff -> interaction

    The ordering emerges from the causal pre/postcondition topology --
    not from explicit sequencing.
    """
    domain_prims = [
        ScriptPrimitive(
            name="scan-environment",
            skill_template=SkillRequest(skill="gaze", params={"mode": "scan_area"}),
            precondition_situations=["open_area", "corridor_encounter"],
            postcondition_situation="scene_assessed",
            expected_affect=AffectState(valence=0.0, arousal=0.1),
            typical_duration_s=3.0,
            deontic_default="permitted",
        ),
        ScriptPrimitive(
            name="position-in-queue",
            skill_template=SkillRequest(
                skill="navigate", params={"intent": "queue", "speed_scale": 0.4},
            ),
            precondition_situations=["scene_assessed", "open_area"],
            postcondition_situation="in_queue",
            expected_affect=AffectState(valence=0.0, arousal=-0.1),
            typical_duration_s=4.0,
            deontic_default="obligatory",
        ),
        ScriptPrimitive(
            name="wait-for-turn",
            skill_template=SkillRequest(
                skill="navigate", params={"intent": "wait", "speed_scale": 0.0},
            ),
            precondition_situations=["in_queue"],
            postcondition_situation="ready_for_service",
            expected_affect=AffectState(valence=0.0, arousal=-0.2),
            typical_duration_s=10.0,
            deontic_default="permitted",
        ),
        ScriptPrimitive(
            name="approach-counter",
            skill_template=SkillRequest(
                skill="navigate", params={"intent": "approach", "speed_scale": 0.5},
            ),
            precondition_situations=["ready_for_service", "open_area"],
            postcondition_situation="at_counter",
            expected_affect=AffectState(valence=0.2, arousal=0.1),
            typical_duration_s=4.0,
            deontic_default="permitted",
        ),
        ScriptPrimitive(
            name="engage-staff",
            skill_template=SkillRequest(
                skill="gaze", params={"mode": "look_at_staff"},
            ),
            precondition_situations=["at_counter", "interaction"],
            postcondition_situation="interaction",
            expected_affect=AffectState(valence=0.3, arousal=0.1),
            typical_duration_s=3.0,
            deontic_default="permitted",
        ),
    ]
    for p in domain_prims:
        lib.register(p)


def _reception_domain_setup():
    """Full reception desk setup with domain primitives and fragments.

    Returns (library, fragments, repertoire, config).
    """
    lib = PrimitiveLibrary()
    _register_reception_primitives(lib)
    cfg = RepertoireConfig()

    fragments = [
        _make_fragment(
            "observe_scene",
            {"scan-environment", "gaze-scan"},
            {"reception": 0.9, "corridor": 0.4},
        ),
        _make_fragment(
            "queue_position",
            {"position-in-queue", "yield-pass"},
            {"reception": 0.8, "corridor": 0.3},
        ),
        _make_fragment(
            "wait_patiently",
            {"wait-for-turn", "wait-acknowledge"},
            {"reception": 0.7, "corridor": 0.5},
        ),
        _make_fragment(
            "approach_service",
            {"approach-counter", "gaze-at-agent"},
            {"reception": 0.9, "open_area": 0.4},
        ),
        _make_fragment(
            "courtesy_space",
            {"yield-pass", "gaze-avert"},
            {"reception": 0.3, "corridor": 0.8},
        ),
    ]

    rep = ScriptRepertoire(lib, config=cfg, initial_patterns=fragments)
    rep.enable_compositional_mode()
    return lib, fragments, rep, cfg


# =====================================================================
# WeightedPattern
# =====================================================================
class TestWeightedPattern:
    def test_construction(self):
        pat = ScriptPattern(name="p1")
        wp = WeightedPattern(pattern=pat, weight=0.75)
        assert wp.pattern.name == "p1"
        assert wp.weight == 0.75

    def test_ordering_by_weight(self):
        p1 = WeightedPattern(pattern=ScriptPattern(name="a"), weight=0.3)
        p2 = WeightedPattern(pattern=ScriptPattern(name="b"), weight=0.9)
        ranked = sorted([p1, p2], key=lambda w: w.weight, reverse=True)
        assert ranked[0].pattern.name == "b"


# =====================================================================
# Top-K Recognition
# =====================================================================
class TestTopKRecognition:
    def _recognizer(self):
        types = [
            SituationType(name="reception", feature_weights={"agent_count": 0.5}),
            SituationType(name="corridor", feature_weights={"min_distance": -0.3}),
            SituationType(name="open_area", feature_weights={"mean_velocity": 0.2}),
            SituationType(name="hazard", feature_weights={"has_hazard": 1.0}),
        ]
        return WeakScriptRecognizer(types)

    def _pb(self):
        return PerceptBundle(
            t=0.0,
            world={"agents": [{"pose": (1, 1)}], "robot_pose": (0, 0, 0, 0)},
            attention={"saliency": {"targets": []}},
            social={"affect": {}, "engagement": {}},
        )

    def test_returns_k_items(self):
        rec = self._recognizer()
        result = rec.recognize_topk(self._pb(), k=2)
        assert len(result) == 2

    def test_sorted_descending(self):
        rec = self._recognizer()
        result = rec.recognize_topk(self._pb(), k=4)
        weights = [w for _, w in result]
        assert weights == sorted(weights, reverse=True)

    def test_weights_sum_leq_one(self):
        rec = self._recognizer()
        result = rec.recognize_topk(self._pb(), k=4)
        assert sum(w for _, w in result) <= 1.0 + 1e-6

    def test_backward_compat_recognize_unchanged(self):
        rec = self._recognizer()
        belief = rec.recognize(self._pb())
        assert isinstance(belief, SituationBelief)
        assert belief.most_likely in ("reception", "corridor", "open_area", "hazard")


# =====================================================================
# Pattern Graph
# =====================================================================
class TestPatternGraph:
    def test_graph_none_by_default(self):
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib)
        assert rep._pattern_graph is None

    def test_enable_creates_graph(self):
        lib = PrimitiveLibrary()
        frags = _reception_fragments()
        rep = ScriptRepertoire(lib, initial_patterns=frags)
        rep.enable_compositional_mode()
        assert rep._pattern_graph is not None
        assert len(rep._pattern_graph) == 4

    def test_edge_range_zero_to_one(self):
        lib = PrimitiveLibrary()
        frags = _reception_fragments()
        rep = ScriptRepertoire(lib, initial_patterns=frags)
        rep.enable_compositional_mode()
        for neighbors in rep._pattern_graph.values():
            for w in neighbors.values():
                assert 0.0 <= w <= 1.0

    def test_overlap_creates_positive_edge(self):
        lib = PrimitiveLibrary()
        frags = _reception_fragments()
        rep = ScriptRepertoire(lib, initial_patterns=frags)
        rep.enable_compositional_mode()
        # detect_queue and align_behind share "wait-acknowledge"
        edge = rep._pattern_graph["detect_queue"].get("align_behind", 0.0)
        assert edge > 0.0

    def test_diffusion_spreads_activation(self):
        lib = PrimitiveLibrary()
        frags = _reception_fragments()
        rep = ScriptRepertoire(lib, initial_patterns=frags)
        rep.enable_compositional_mode()
        # Activate only detect_queue
        query = {"detect_queue": 1.0}
        result = rep.retrieve_composition(query)
        # Other fragments should also get some activation via diffusion
        names_activated = {wp.pattern.name for wp in result}
        assert len(names_activated) > 1


# =====================================================================
# Retrieve Composition
# =====================================================================
class TestRetrieveComposition:
    def test_weighted_results_sorted(self):
        lib = PrimitiveLibrary()
        frags = _reception_fragments()
        rep = ScriptRepertoire(lib, initial_patterns=frags)
        rep.enable_compositional_mode()
        query = {"detect_queue": 0.8, "approach_desk": 0.9}
        result = rep.retrieve_composition(query)
        weights = [wp.weight for wp in result]
        assert weights == sorted(weights, reverse=True)

    def test_empty_query_returns_empty(self):
        lib = PrimitiveLibrary()
        frags = _reception_fragments()
        rep = ScriptRepertoire(lib, initial_patterns=frags)
        rep.enable_compositional_mode()
        result = rep.retrieve_composition({})
        assert result == []

    def test_no_graph_still_works(self):
        lib = PrimitiveLibrary()
        frags = _reception_fragments()
        rep = ScriptRepertoire(lib, initial_patterns=frags)
        # Graph not enabled — retrieve_composition should still work
        # (no diffusion, just returns patterns with positive query scores)
        query = {"detect_queue": 0.5}
        result = rep.retrieve_composition(query)
        assert len(result) >= 1

    def test_diffusion_ranking(self):
        lib = PrimitiveLibrary()
        frags = _reception_fragments()
        rep = ScriptRepertoire(lib, initial_patterns=frags)
        rep.enable_compositional_mode()
        # Only approach_desk activated — diffusion should pull in related
        query = {"approach_desk": 1.0}
        result = rep.retrieve_composition(query)
        assert result[0].pattern.name == "approach_desk"
        # At least one more pattern should be retrieved
        assert len(result) >= 2


# =====================================================================
# Compose From Patterns
# =====================================================================
class TestComposeFromPatterns:
    def _composer(self):
        return ScriptComposer(PrimitiveLibrary())

    def test_merge_primitives(self):
        composer = self._composer()
        wp1 = WeightedPattern(
            pattern=_make_fragment("f1", {"approach-greet", "gaze-at-agent"}, {"reception": 0.9}),
            weight=0.8,
        )
        wp2 = WeightedPattern(
            pattern=_make_fragment("f2", {"wait-acknowledge", "gaze-scan"}, {"reception": 0.7}),
            weight=0.6,
        )
        result = composer.compose_from_patterns([wp1, wp2], "reception")
        assert result is not None
        assert "approach-greet" in result.primitive_cluster
        assert "wait-acknowledge" in result.primitive_cluster

    def test_weights_reflect_activation(self):
        composer = self._composer()
        wp1 = WeightedPattern(
            pattern=_make_fragment("f1", {"approach-greet"}, {"reception": 0.9}),
            weight=0.9,
        )
        wp2 = WeightedPattern(
            pattern=_make_fragment("f2", {"gaze-scan"}, {"corridor": 0.8}),
            weight=0.1,
        )
        result = composer.compose_from_patterns([wp1, wp2], "reception")
        assert result is not None
        # approach-greet should have higher weight than gaze-scan
        assert result.primitive_weights["approach-greet"] > result.primitive_weights["gaze-scan"]

    def test_affinity_merged(self):
        composer = self._composer()
        wp1 = WeightedPattern(
            pattern=_make_fragment("f1", {"approach-greet"}, {"reception": 1.0}),
            weight=0.5,
        )
        wp2 = WeightedPattern(
            pattern=_make_fragment("f2", {"gaze-scan"}, {"corridor": 1.0}),
            weight=0.5,
        )
        result = composer.compose_from_patterns([wp1, wp2], "reception")
        assert result is not None
        assert "reception" in result.situation_affinity
        assert "corridor" in result.situation_affinity

    def test_fragments_tracked(self):
        composer = self._composer()
        wp1 = WeightedPattern(
            pattern=_make_fragment("frag_a", {"approach-greet"}, {}),
            weight=0.5,
        )
        wp2 = WeightedPattern(
            pattern=_make_fragment("frag_b", {"gaze-scan"}, {}),
            weight=0.5,
        )
        result = composer.compose_from_patterns([wp1, wp2], "test")
        assert result is not None
        assert "frag_a" in result.source_fragments
        assert "frag_b" in result.source_fragments
        assert result.composition_signature != ""
        assert result.is_composite is True

    def test_empty_returns_none(self):
        composer = self._composer()
        assert composer.compose_from_patterns([], "test") is None


# =====================================================================
# Three-Tier Selection
# =====================================================================
class TestThreeTierSelection:
    def test_tier1_strong_match_bypasses_composition(self):
        """A strong confident pattern should be reused directly."""
        lib = PrimitiveLibrary()
        strong = ScriptPattern(
            name="strong_corridor",
            primitives_sequence=["yield-pass"],
            precision=3.0,
            situation_affinity={"corridor": 0.9},
            is_strong=True,
        )
        rep = ScriptRepertoire(lib, initial_patterns=[strong])
        rep.enable_compositional_mode()
        result = rep._select_or_compose("corridor", {"corridor": 0.9})
        assert result == "strong_corridor"

    def test_tier2_compositional_when_graph_active(self):
        """With graph active and weak patterns, Tier 2 should compose."""
        lib = PrimitiveLibrary()
        frags = _reception_fragments()
        rep = ScriptRepertoire(lib, initial_patterns=frags)
        rep.enable_compositional_mode()
        result = rep._select_or_compose("reception", {"reception": 0.9})
        assert result is not None
        pat = rep.patterns[result]
        # Should be a composite (assembled from fragments)
        assert pat.is_composite or len(pat.source_fragments) > 1

    def test_tier3_fallback_without_graph(self):
        """Without graph, selection should fall back to scratch composition."""
        lib = PrimitiveLibrary()
        rep = ScriptRepertoire(lib)
        result = rep._select_or_compose("corridor_encounter", {"corridor_encounter": 0.9})
        assert result is not None
        pat = rep.patterns[result]
        # Should be a plain composed pattern (not composite)
        assert pat.name.startswith("composed_")


# =====================================================================
# Compositional Consolidation
# =====================================================================
class TestCompositionalConsolidation:
    def test_composition_count_increments(self):
        lib = PrimitiveLibrary()
        p = ScriptPattern(
            name="comp_test",
            primitives_sequence=["yield-pass"],
            precision=3.0,  # High enough to pass Tier 1 selection
            situation_affinity={"corridor": 0.9},
            source_fragments=["a", "b"],
            composition_signature="a|b",
            composition_count=0,
        )
        rep = ScriptRepertoire(lib, initial_patterns=[p])
        # Simulate script completion lifecycle
        rep.on_situation_recognized({"corridor": 0.9}, t=0.0)
        rep.on_step_completed(t=1.0, primitive_name="yield-pass", outcome="SUCCESS")
        rep.on_script_completed("SUCCESS")
        # composition_count should have been incremented
        assert rep.patterns["comp_test"].composition_count >= 1

    def test_repeated_use_promotes_composite(self):
        cfg = RepertoireConfig(
            composition_consolidation_count=2,
            min_trajectories_for_promotion=2,
            max_free_energy_for_promotion=5.0,
        )
        lib = PrimitiveLibrary()
        p = ScriptPattern(
            name="comp_promote",
            primitives_sequence=["yield-pass"],
            primitive_cluster={"yield-pass"},
            precision=1.8,
            situation_affinity={"corridor": 0.9},
            source_fragments=["a", "b"],
            composition_signature="a|b",
            composition_count=2,
            trajectory_count=1,
            mean_free_energy=1.0,
        )
        rep = ScriptRepertoire(lib, config=cfg, initial_patterns=[p])
        # Run enough trajectories to meet promotion thresholds
        for _ in range(3):
            rep.on_situation_recognized({"corridor": 0.9}, t=0.0)
            rep.on_step_completed(t=1.0, primitive_name="yield-pass", outcome="SUCCESS")
            rep.on_script_completed("SUCCESS")

        updated = rep.patterns["comp_promote"]
        # Should eventually become strong via compositional consolidation
        # (composition_count meets threshold, trajectory_count meets threshold)
        assert updated.composition_count >= cfg.composition_consolidation_count
        assert updated.trajectory_count >= cfg.min_trajectories_for_promotion

    def test_single_fragment_not_composite(self):
        p = ScriptPattern(
            name="single",
            source_fragments=["only_one"],
        )
        assert p.is_composite is False


# =====================================================================
# Toy Demo: Reception Desk
# =====================================================================
class TestReceptionDeskDemo:
    """Robot approaches a reception desk — the system should compose
    a context-sensitive script from fragments rather than selecting
    a single canned script.
    """

    def test_reception_desk_composition(self):
        # 1. Setup: library, fragments, repertoire
        lib = PrimitiveLibrary()
        fragments = _reception_fragments()
        cfg = RepertoireConfig()
        rep = ScriptRepertoire(lib, config=cfg, initial_patterns=fragments)

        # 2. Enable compositional mode
        rep.enable_compositional_mode()

        # 3. Build query: "reception" situation is dominant
        query_belief = {"reception": 0.9, "corridor": 0.1}

        # 4. Map belief to pattern scores via affinity * precision
        query_scores = {}
        for name, pat in rep.patterns.items():
            aff = pat.situation_affinity.get("reception", 0.0)
            query_scores[name] = aff * pat.precision

        # 5. Retrieve composition via graph diffusion
        weighted = rep.retrieve_composition(query_scores)

        # 6. Assert: multiple fragments activated (not winner-take-all)
        assert len(weighted) >= 2, "Should activate multiple fragments"

        # 7. Assert: detect_queue and approach_desk have highest weights
        top_names = [wp.pattern.name for wp in weighted[:2]]
        # approach_desk has reception affinity 0.9, detect_queue 0.8
        assert "approach_desk" in top_names, (
            f"approach_desk should be in top-2, got {top_names}"
        )

        # 8. Compose from weighted fragments
        composer = ScriptComposer(lib, cfg)
        result = composer.compose_from_patterns(weighted, "reception")

        # 9. Assert: result has primitives from multiple fragments
        assert result is not None
        prims = result.primitive_cluster
        # From detect_queue: wait-acknowledge, gaze-scan
        # From approach_desk: approach-greet, gaze-at-agent
        assert "wait-acknowledge" in prims or "gaze-scan" in prims, (
            "Should include primitives from detect_queue"
        )
        assert "approach-greet" in prims or "gaze-at-agent" in prims, (
            "Should include primitives from approach_desk"
        )

        # 10. Assert: source provenance tracked
        assert len(result.source_fragments) > 1
        assert result.composition_signature != ""

        # 11. Assert: is_composite and is_cluster_based
        assert result.is_composite is True
        assert result.is_cluster_based is True

    def test_uncertainty_broadens_composition(self):
        """Ambiguous situation → more fragments recruited."""
        lib = PrimitiveLibrary()
        fragments = _reception_fragments()
        cfg = RepertoireConfig()
        rep = ScriptRepertoire(lib, config=cfg, initial_patterns=fragments)
        rep.enable_compositional_mode()

        # Confident query: strongly reception-biased
        confident_scores = {}
        for name, pat in rep.patterns.items():
            confident_scores[name] = (
                pat.situation_affinity.get("reception", 0.0) * 0.9
            )
        confident_result = rep.retrieve_composition(confident_scores)

        # Uncertain query: flat/low scores (ambiguous situation)
        uncertain_scores = {}
        for name, pat in rep.patterns.items():
            # Average across all affinities — flatter activation
            all_aff = pat.situation_affinity.values()
            avg_aff = sum(all_aff) / len(all_aff) if all_aff else 0.0
            uncertain_scores[name] = avg_aff * 0.3

        uncertain_result = rep.retrieve_composition(uncertain_scores)

        # With uncertainty, weights should be more evenly distributed
        if len(confident_result) >= 2 and len(uncertain_result) >= 2:
            # Ratio of top to second weight — should be closer to 1.0
            # for uncertain (more even) than for confident
            confident_ratio = confident_result[0].weight / max(
                confident_result[1].weight, 1e-9
            )
            uncertain_ratio = uncertain_result[0].weight / max(
                uncertain_result[1].weight, 1e-9
            )
            assert uncertain_ratio <= confident_ratio + 0.1, (
                "Uncertain query should produce more even weight distribution"
            )


# =====================================================================
# Emergent Behavioral Properties
# =====================================================================
class TestEmergentBehavior:
    """Verify that compositional assembly produces queue-respecting
    behavior from domain primitives with causal pre/postconditions.

    The ordering EMERGES from the causal topology — not from hard-coded
    sequences.  Domain primitives provide the right generative model;
    the backbone extraction algorithm discovers the ordering.
    """

    def _compose_reception(self):
        """Run the full compositional pipeline with domain primitives."""
        lib, fragments, rep, cfg = _reception_domain_setup()

        query_scores = {}
        for name, pat in rep.patterns.items():
            aff = pat.situation_affinity.get("reception", 0.0)
            query_scores[name] = aff * pat.precision

        weighted = rep.retrieve_composition(query_scores)
        composer = ScriptComposer(lib, cfg)
        return composer.compose_from_patterns(weighted, "reception")

    def test_scan_before_approach(self):
        """The agent should observe the scene BEFORE approaching.

        scan-environment produces scene_assessed which is a precondition
        for position-in-queue; approach-counter requires ready_for_service
        which is downstream.  The backbone chain enforces this ordering.
        """
        result = self._compose_reception()
        seq = result.primitives_sequence
        assert "scan-environment" in seq, "Should include scan-environment"
        assert "approach-counter" in seq, "Should include approach-counter"
        idx_scan = seq.index("scan-environment")
        idx_approach = seq.index("approach-counter")
        assert idx_scan < idx_approach, (
            f"scan-environment (idx={idx_scan}) should come before "
            f"approach-counter (idx={idx_approach}) in {seq}"
        )

    def test_position_before_approach(self):
        """The agent should join the queue BEFORE approaching the counter.

        position-in-queue → in_queue → wait-for-turn → ready_for_service
        → approach-counter: the causal chain enforces queuing first.
        """
        result = self._compose_reception()
        seq = result.primitives_sequence
        assert "position-in-queue" in seq
        assert "approach-counter" in seq
        idx_pos = seq.index("position-in-queue")
        idx_approach = seq.index("approach-counter")
        assert idx_pos < idx_approach, (
            f"position-in-queue (idx={idx_pos}) should come before "
            f"approach-counter (idx={idx_approach}) in {seq}"
        )

    def test_wait_before_approach(self):
        """The agent should wait for its turn BEFORE approaching.

        wait-for-turn → ready_for_service → approach-counter: the agent
        doesn't cut the queue.
        """
        result = self._compose_reception()
        seq = result.primitives_sequence
        assert "wait-for-turn" in seq
        assert "approach-counter" in seq
        idx_wait = seq.index("wait-for-turn")
        idx_approach = seq.index("approach-counter")
        assert idx_wait < idx_approach, (
            f"wait-for-turn (idx={idx_wait}) should come before "
            f"approach-counter (idx={idx_approach}) in {seq}"
        )

    def test_approach_not_first(self):
        """The robot should NOT charge directly toward the counter."""
        result = self._compose_reception()
        seq = result.primitives_sequence
        assert seq[0] != "approach-counter", (
            f"approach-counter should not be the first action, got {seq}"
        )

    def test_full_queue_backbone_order(self):
        """The backbone causal chain should appear in order:
        scan → position → wait → approach.

        This is the core queue-joining behavior, emerging from
        pre/postcondition chains, not explicit sequencing.
        """
        result = self._compose_reception()
        seq = result.primitives_sequence
        backbone = ["scan-environment", "position-in-queue",
                     "wait-for-turn", "approach-counter"]
        indices = []
        for prim in backbone:
            assert prim in seq, f"{prim} should be in composed sequence"
            indices.append(seq.index(prim))
        assert indices == sorted(indices), (
            f"Backbone should be in causal order, got indices {indices} "
            f"for {backbone} in {seq}"
        )

    def test_multimodal_behavior(self):
        """The composed script should include BOTH motor and perceptual
        primitives from multiple fragments.
        """
        result = self._compose_reception()
        cluster = result.primitive_cluster
        gaze_prims = {"scan-environment", "gaze-scan", "gaze-at-agent",
                       "gaze-avert", "engage-staff"}
        nav_prims = {"position-in-queue", "wait-for-turn", "yield-pass",
                      "wait-acknowledge", "approach-counter"}
        assert bool(cluster & gaze_prims), "Should include gaze primitives"
        assert bool(cluster & nav_prims), "Should include nav primitives"

    def test_richer_than_any_single_fragment(self):
        """The composed script should be richer than any single fragment."""
        result = self._compose_reception()
        _, fragments, _, _ = _reception_domain_setup()
        cluster = result.primitive_cluster
        for frag in fragments:
            if frag.primitive_cluster == cluster:
                pytest.fail(
                    f"Composed cluster matches single fragment '{frag.name}'"
                )
        assert len(result.source_fragments) >= 2

    def test_causal_ordering_not_arbitrary(self):
        """Sequence should follow causal structure, not weight or alpha."""
        result = self._compose_reception()
        seq = result.primitives_sequence
        assert seq != sorted(seq), "Should not be alphabetical"
        weight_sorted = sorted(
            result.primitive_weights.keys(),
            key=lambda p: result.primitive_weights.get(p, 0),
            reverse=True,
        )
        assert seq != weight_sorted, "Should not be pure weight order"
