"""Computational Experiments — Environment Shapes Robot Cognition.

Runs three experiments testing how material environment structure shapes
robot social cognition via compositional script assembly under active inference.

Experiment 1: Material Cue Factorial (48 conditions)
    2^4 cue combinations × 3 contexts — tests how material cues modify
    the generative model's B-matrix and C-matrix.

Experiment 2: Fragment Scaling (189 conditions)
    C(6,k) for k=1..6 × 3 contexts — tests graceful degradation as the
    generative model becomes sparser.

Experiment 3: Context Transfer (3 conditions)
    All 6 fragments × 3 contexts — tests how different C-matrices yield
    different policies from identical knowledge.

Outputs CSV and JSON to experiment_results/.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from architecture_core.core.types import AffectState, PerceptBundle, SkillRequest
from architecture_core.cognition.scripts.script_types import SituationType
from architecture_core.cognition.scripts.weak_recognizer import WeakScriptRecognizer, SituationBelief
from architecture_core.cognition.scripts.repertoire_types import (
    RepertoireConfig,
    ScriptPattern,
    ScriptPrimitive,
    WeightedPattern,
)
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary
from architecture_core.cognition.scripts.script_composer import ScriptComposer
from architecture_core.cognition.scripts.script_repertoire import ScriptRepertoire
from architecture_core.cognition.scripts.affinity_learner import AffinityLearner
from architecture_core.cognition.scripts.threshold_learner import PrecisionGating
from architecture_core.cognition.scripts.recognizer_learner import (
    RecognizerLearner,
    LabeledSample,
)


# ================================================================
# Constants
# ================================================================
EPSILON = 1e-6

# Context-dependent fragment gating threshold.
# In active inference terms, this is the minimum D-matrix prior for a
# generative sub-model (fragment) to be activated by the context.  Below
# this threshold the environment suppresses the fragment — modelling how
# material structure gates script activation (Guénin-Carlut & Albarracin 2023).
CONTEXT_GATE_THRESHOLD = 0.4

CONTEXTS = ["reception", "corridor", "hospital"]

ALL_FRAGMENT_NAMES = [
    "observe_scene",
    "queue_position",
    "wait_patiently",
    "approach_service",
    "courtesy_space",
    "direct_approach",
]

# Situation phase ordering — used for metrics and phase classification
_SITUATION_PHASE = {
    "open_area": 0,
    "corridor_encounter": 0,
    "scene_assessed": 1,
    "in_queue": 2,
    "ready_for_service": 3,
    "at_counter": 4,
    "interaction": 5,
    "narrow_passage": 0,
    "doorway": 0,
    "hazard": 0,
    "meeting_point": 5,
    "handover": 5,
}

DOMAIN_SITUATIONS = {
    "scene_assessed", "in_queue", "ready_for_service",
    "at_counter", "interaction",
}

# Material cue → fragment mapping for Experiment 1
CUE_FRAGMENT_MAP = {
    "stanchions": "queue_position",
    "waiting_area": "wait_patiently",
    "service_sign": "direct_approach",
    "social_density": "courtesy_space",
}

# Base fragments always present in Exp1
EXP1_BASE_FRAGMENTS = ["observe_scene", "approach_service"]

# Situation types for WeakScriptRecognizer (A-matrix)
# Feature weights tuned to the 10 features extracted by _extract_features():
#   agent_count, min_distance, inverse_min_distance, mean_velocity,
#   max_arousal, max_engagement, has_hazard,
#   cue_queue_here, cue_staff_only, cue_quiet_zone
SITUATION_TYPES = [
    SituationType(name="reception", feature_weights={
        "agent_count": 0.5,
        "inverse_min_distance": 0.8,
        "max_engagement": 1.5,
        "cue_queue_here": 3.0,
    }),
    SituationType(name="corridor", feature_weights={
        "mean_velocity": 2.0,
        "min_distance": 0.5,
        "agent_count": -0.3,
    }),
    SituationType(name="hospital", feature_weights={
        "cue_quiet_zone": 3.0,
        "max_arousal": -1.0,
        "agent_count": 0.8,
        "inverse_min_distance": 0.5,
        "max_engagement": -0.5,
    }),
]


def _make_base_percept(context: str) -> PerceptBundle:
    """Build a base PerceptBundle encoding the environmental character of a context."""
    if context == "reception":
        return PerceptBundle(
            t=0.0,
            world={
                "agents": [{"pose": (2, 1)}, {"pose": (3, 2)}, {"pose": (4, 1)}],
                "robot_pose": (0, 0, 0, 0),
                "hazards": [],
                "deontic_cues": [{"type": "queue_here", "active": True}],
            },
            social={
                "affect": {"readings": [{"arousal": 0.3}]},
                "engagement": {"readings": [{"score": 0.7}]},
            },
            attention={
                "saliency": {"targets": [{"velocity": 0.1}]},
            },
        )
    elif context == "corridor":
        return PerceptBundle(
            t=0.0,
            world={
                "agents": [{"pose": (5, 0)}],
                "robot_pose": (0, 0, 0, 0),
                "hazards": [],
                "deontic_cues": [],
            },
            social={
                "affect": {"readings": [{"arousal": 0.1}]},
                "engagement": {"readings": [{"score": 0.2}]},
            },
            attention={
                "saliency": {"targets": [{"velocity": 0.8}]},
            },
        )
    else:  # hospital
        return PerceptBundle(
            t=0.0,
            world={
                "agents": [{"pose": (2, 1)}, {"pose": (3, 0)}, {"pose": (1, 2)}, {"pose": (4, 1)}],
                "robot_pose": (0, 0, 0, 0),
                "hazards": [],
                "deontic_cues": [{"type": "quiet_zone", "active": True}],
            },
            social={
                "affect": {"readings": [{"arousal": 0.1}]},
                "engagement": {"readings": [{"score": 0.3}]},
            },
            attention={
                "saliency": {"targets": [{"velocity": 0.05}]},
            },
        )


def _apply_material_cues(pb: PerceptBundle, cue_flags: dict) -> PerceptBundle:
    """Modify PerceptBundle based on material cues present."""
    if cue_flags.get("stanchions"):
        cues = pb.world.get("deontic_cues", [])
        if not any(c.get("type") == "queue_here" for c in cues):
            cues.append({"type": "queue_here", "active": True})
        pb.world["deontic_cues"] = cues

    if cue_flags.get("waiting_area"):
        agents = pb.world.get("agents", [])
        agents.extend([{"pose": (1.5, 0.5)}, {"pose": (2.5, 0.5)}])
        pb.world["agents"] = agents

    if cue_flags.get("service_sign"):
        eng = pb.social.get("engagement", {})
        readings = eng.get("readings", [])
        readings.append({"score": 0.8})
        pb.social["engagement"] = {"readings": readings}

    if cue_flags.get("social_density"):
        agents = pb.world.get("agents", [])
        agents.extend([{"pose": (1, 1)}, {"pose": (2, 2)}, {"pose": (3, 1)}])
        pb.world["agents"] = agents

    return pb


# ================================================================
# Domain setup
# ================================================================
def _register_reception_primitives(lib: PrimitiveLibrary) -> None:
    """Register the 5 domain primitives (identical to demo_variations_anim.py)."""
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


def _make_fragment(name: str, primitives: set, affinity: dict) -> ScriptPattern:
    """Create a fragment pattern (partial generative model)."""
    return ScriptPattern(
        name=name,
        primitives_sequence=sorted(primitives),
        primitive_cluster=set(primitives),
        precision=0.1,
        situation_affinity=affinity,
    )


def _make_all_fragments() -> Dict[str, ScriptPattern]:
    """Create all 6 fragments with reception, corridor, AND hospital affinities.

    Each fragment's situation_affinity is the D-matrix prior parameterized
    per context. Hospital affinities encode the preference structure of a
    medical waiting environment.
    """
    return {
        "observe_scene": _make_fragment(
            "observe_scene",
            {"scan-environment", "gaze-scan"},
            {"reception": 0.9, "corridor": 0.4, "hospital": 0.85},
        ),
        "queue_position": _make_fragment(
            "queue_position",
            {"position-in-queue", "yield-pass"},
            {"reception": 0.8, "corridor": 0.3, "hospital": 0.90},
        ),
        "wait_patiently": _make_fragment(
            "wait_patiently",
            {"wait-for-turn", "wait-acknowledge"},
            {"reception": 0.7, "corridor": 0.5, "hospital": 0.95},
        ),
        "approach_service": _make_fragment(
            "approach_service",
            {"approach-counter", "gaze-at-agent"},
            {"reception": 0.9, "corridor": 0.4, "hospital": 0.80},
        ),
        "courtesy_space": _make_fragment(
            "courtesy_space",
            {"yield-pass", "gaze-avert"},
            {"reception": 0.3, "corridor": 0.8, "hospital": 0.60},
        ),
        "direct_approach": _make_fragment(
            "direct_approach",
            {"approach-counter", "engage-staff"},
            {"reception": 0.95, "corridor": 0.2, "hospital": 0.15},
        ),
    }


# ================================================================
# Principled free energy computations
# ================================================================
def compute_transition_VFE(topology: dict, from_prim: str, to_prim: str) -> float:
    """Variational free energy of a single transition under B-matrix.

    F(s_tau -> s_tau+1) = -ln P(s_tau+1 | s_tau, pi) = -ln B[s_tau+1, s_tau]

    This is the surprisal of the transition under the generative model's
    transition matrix.
    """
    w = topology.get((from_prim, to_prim), 0.0)
    return -math.log(max(w, EPSILON))


def compute_sequence_VFE(topology: dict, sequence: List[str]) -> float:
    """Total variational free energy of composed policy.

    F(pi) = sum_tau -ln B[s_tau+1, s_tau, pi_tau]

    Total surprisal of the full sequence under the B-matrix.
    """
    if len(sequence) < 2:
        return 0.0
    return sum(
        compute_transition_VFE(topology, sequence[i], sequence[i + 1])
        for i in range(len(sequence) - 1)
    )


def compute_policy_EFE(
    topology: dict,
    sequence: List[str],
    lib: PrimitiveLibrary,
    context_affinities: Dict[str, float],
) -> float:
    """Expected free energy of composed policy.

    G(pi) = sum_tau [Risk(tau) + Ambiguity(tau)] + Risk_terminal

    Risk: does this primitive lead to preferred observations (high affinity)?
    Ambiguity: how uncertain is the transition (low B-matrix weight)?
    Risk_terminal: C-matrix penalty for not reaching preferred terminal state.

    The terminal risk encodes the C-matrix preference: policies that fail
    to reach high-phase situations (interaction, at_counter) are penalized.
    This prevents degenerate EFE=0 for empty/short policies.

    We use a principled approximation:
    - Risk term: cross-entropy between predicted outcome and context preference
    - Ambiguity term: entropy of outgoing transition weights from this primitive
    - Terminal risk: -ln P(preferred | final_phase) from C-matrix
    """
    if len(sequence) < 1:
        # Empty policy: maximum terminal risk (failed to reach ANY preferred state)
        return -math.log(0.01)  # ~4.6 nats

    G = 0.0
    max_phase_reached = 0
    for i, prim_name in enumerate(sequence):
        # Risk: does the postcondition situation align with context preferences?
        prim = lib.get(prim_name)
        if prim:
            post_sit = prim.postcondition_situation
            # P(preferred | post_state) approximated by context affinity
            p_pref = context_affinities.get(post_sit, 0.1)
            p_pref = max(min(p_pref, 0.999), 0.001)
            risk = -math.log(p_pref)
            # Track highest phase reached (C-matrix terminal preference)
            phase = _SITUATION_PHASE.get(post_sit, 0)
            max_phase_reached = max(max_phase_reached, phase)
        else:
            risk = -math.log(0.1)  # unknown primitive → high risk

        # Ambiguity: entropy of outgoing transitions from this primitive
        outgoing = {}
        for (a, b), w in topology.items():
            if a == prim_name:
                outgoing[b] = w
        if outgoing:
            total = sum(outgoing.values())
            if total > 0:
                probs = [w / total for w in outgoing.values()]
                ambiguity = -sum(p * math.log(max(p, EPSILON)) for p in probs)
            else:
                ambiguity = math.log(max(len(sequence), 1))
        else:
            # No outgoing edges → maximum ambiguity for this context
            ambiguity = math.log(max(len(sequence), 1))

        G += risk + ambiguity

    # Terminal risk: C-matrix penalty for not reaching preferred end state
    # P(preferred | phase) increases with phase advancement
    # Phase 5 (interaction) = fully preferred → P=0.95 → risk ≈ 0.05
    # Phase 0 (no progress) = not preferred → P=0.05 → risk ≈ 3.0
    max_possible_phase = 5
    p_preferred_terminal = 0.05 + 0.90 * (max_phase_reached / max_possible_phase)
    terminal_risk = -math.log(p_preferred_terminal)
    G += terminal_risk

    return G


# ================================================================
# Experiment condition dataclass
# ================================================================
@dataclass
class ExperimentCondition:
    experiment: str
    condition_id: str
    context: str
    fragment_names: List[str]
    # Exp1-specific cue flags
    cue_stanchions: int = 0
    cue_waiting_area: int = 0
    cue_service_sign: int = 0
    cue_social_density: int = 0


@dataclass
class ExperimentResult:
    """All metrics for one experimental condition."""
    # Condition info
    experiment: str
    condition_id: str
    context: str
    fragment_names: str  # comma-separated (requested fragments)
    n_fragments: int  # requested fragment count
    active_fragments: str  # comma-separated (context-gated fragments)
    n_active_fragments: int  # fragments that passed D-matrix gating
    # Cue flags (Exp1)
    cue_stanchions: int
    cue_waiting_area: int
    cue_service_sign: int
    cue_social_density: int
    # Composed policy
    n_primitives: int
    primitive_set: str  # comma-separated
    sequence: str  # arrow-separated
    sequence_length: int
    backbone_length: int
    # Free energy metrics
    total_VFE: float
    mean_VFE: float
    max_VFE: float
    total_EFE: float
    efe_rate: float  # G/T: EFE per step for fair variable-horizon comparison
    per_transition_VFE: float  # observation-level VFE (total_VFE / n_transitions)
    # Primitive classification
    n_obligatory: int
    n_advancing: int
    n_returning: int
    # Topology metrics
    topology_density: float
    # Perception pipeline (A-matrix)
    inferred_context: str  # belief.most_likely
    belief_confidence: float  # belief.confidence
    belief_entropy: float  # belief.entropy
    belief_reception: float  # belief.distribution["reception"]
    belief_corridor: float  # belief.distribution["corridor"]
    belief_hospital: float  # belief.distribution["hospital"]
    effective_affinities: str  # JSON of marginalized affinities
    # Weights
    weighted_scores: str  # JSON
    primitive_weights: str  # JSON
    # Transfer metric (Exp3)
    D_KL_from_baseline: float


# ================================================================
# Pipeline runner
# ================================================================
def run_single_condition(cond: ExperimentCondition) -> ExperimentResult:
    """Run the composition pipeline for one experimental condition.

    Follows the pattern from demo_variations_anim.py:run_pipeline().
    Applies context-dependent fragment gating: only fragments whose
    D-matrix affinity for the current context meets the threshold are
    activated.  This models how material environments gate script
    activation — a corridor suppresses queue_position (affinity 0.3)
    even when the agent "knows" how to queue.

    Returns all metrics needed for analysis.
    """
    all_frags = _make_all_fragments()
    lib = PrimitiveLibrary()
    _register_reception_primitives(lib)

    # Requested fragments (from condition)
    requested = [all_frags[name] for name in cond.fragment_names]

    # --- Perception-driven situation recognition (A-matrix) ---
    recognizer = WeakScriptRecognizer(SITUATION_TYPES, temperature=1.0)
    pb = _make_base_percept(cond.context)
    pb = _apply_material_cues(pb, {
        "stanchions": cond.cue_stanchions,
        "waiting_area": cond.cue_waiting_area,
        "service_sign": cond.cue_service_sign,
        "social_density": cond.cue_social_density,
    })
    belief = recognizer.recognize(pb)
    inferred_context = belief.most_likely

    # Marginalize affinity over posterior: effective_aff(f) = sum_s P(s|o) * aff(f,s)
    fragments = []
    effective_affinities = {}
    for f in requested:
        eff_aff = sum(
            belief.distribution.get(s, 0.0) * f.situation_affinity.get(s, 0.0)
            for s in belief.distribution
        )
        effective_affinities[f.name] = round(eff_aff, 6)
        if eff_aff >= CONTEXT_GATE_THRESHOLD:
            fragments.append(f)
    active_frag_names = [f.name for f in fragments]

    # Handle empty fragment set — no viable behaviour for this context
    if not fragments:
        # Empty policy gets maximum terminal risk from C-matrix
        empty_efe = -math.log(0.01)  # ~4.6 nats
        return ExperimentResult(
            experiment=cond.experiment,
            condition_id=cond.condition_id,
            context=cond.context,
            fragment_names=",".join(cond.fragment_names),
            n_fragments=len(cond.fragment_names),
            active_fragments="",
            n_active_fragments=0,
            cue_stanchions=cond.cue_stanchions,
            cue_waiting_area=cond.cue_waiting_area,
            cue_service_sign=cond.cue_service_sign,
            cue_social_density=cond.cue_social_density,
            n_primitives=0,
            primitive_set="",
            sequence="",
            sequence_length=0,
            backbone_length=0,
            total_VFE=0.0,
            mean_VFE=0.0,
            max_VFE=0.0,
            total_EFE=round(empty_efe, 6),
            efe_rate=round(empty_efe, 6),
            per_transition_VFE=0.0,
            n_obligatory=0,
            n_advancing=0,
            n_returning=0,
            topology_density=0.0,
            inferred_context=inferred_context,
            belief_confidence=round(belief.confidence, 6),
            belief_entropy=round(belief.entropy, 6),
            belief_reception=round(belief.distribution.get("reception", 0.0), 6),
            belief_corridor=round(belief.distribution.get("corridor", 0.0), 6),
            belief_hospital=round(belief.distribution.get("hospital", 0.0), 6),
            effective_affinities=json.dumps(effective_affinities),
            weighted_scores="{}",
            primitive_weights="{}",
            D_KL_from_baseline=0.0,
        )

    cfg = RepertoireConfig()
    rep = ScriptRepertoire(lib, cfg, initial_patterns=fragments)
    rep.enable_compositional_mode()

    # D-matrix scoring: w_f(c) = affinity(f, inferred_context) * precision(f)
    # Uses inferred_context from perceptual recognition instead of hardcoded context
    query_scores = {}
    for name, pat in rep.patterns.items():
        aff = pat.situation_affinity.get(inferred_context, 0.0)
        query_scores[name] = aff * pat.precision

    # Belief propagation on factor graph (graph diffusion)
    weighted = rep.retrieve_composition(query_scores)

    # B-matrix construction + backbone extraction
    composer = ScriptComposer(lib, cfg)
    composite = composer.compose_from_patterns(weighted, inferred_context)
    if composite is None:
        composite = ScriptPattern(name="empty_composite")

    # Extract topology and sequence
    topo = composite.context_topology.get(inferred_context, {})
    if not topo:
        topo = next(iter(composite.context_topology.values()), {})

    sequence = list(composite.primitives_sequence)
    primitive_set = set(sequence)

    # --- Compute metrics ---

    # Free energy metrics
    total_vfe = compute_sequence_VFE(topo, sequence)
    mean_vfe = total_vfe / max(len(sequence) - 1, 1)
    max_vfe = 0.0
    if len(sequence) >= 2:
        max_vfe = max(
            compute_transition_VFE(topo, sequence[i], sequence[i + 1])
            for i in range(len(sequence) - 1)
        )

    # Context affinities for EFE computation (using inferred context)
    context_affinities = {}
    for sit in _SITUATION_PHASE:
        # Approximate: situations used by high-affinity fragments get high preference
        best_aff = 0.1
        for frag in fragments:
            aff = frag.situation_affinity.get(inferred_context, 0.0)
            for p_name in frag.primitive_cluster:
                prim = lib.get(p_name)
                if prim and prim.postcondition_situation == sit:
                    best_aff = max(best_aff, aff)
        context_affinities[sit] = best_aff

    total_efe = compute_policy_EFE(topo, sequence, lib, context_affinities)

    # Backbone length: count primitives whose postcondition advances the phase
    backbone_length = 0
    for prim_name in sequence:
        prim = lib.get(prim_name)
        if prim:
            phase = _SITUATION_PHASE.get(prim.postcondition_situation, 0)
            if phase > 0:
                backbone_length += 1

    # Primitive classification
    n_obligatory = 0
    n_advancing = 0
    n_returning = 0
    for prim_name in sequence:
        prim = lib.get(prim_name)
        if prim:
            if prim.deontic_default == "obligatory":
                n_obligatory += 1
            phase = _SITUATION_PHASE.get(prim.postcondition_situation, 0)
            if phase > 0:
                n_advancing += 1
            else:
                n_returning += 1
        else:
            n_returning += 1

    # Topology density: |edges| / |possible edges|
    n_prims_in_topo = len(primitive_set)
    possible_edges = n_prims_in_topo * (n_prims_in_topo - 1) if n_prims_in_topo > 1 else 1
    actual_edges = sum(1 for (a, b), w in topo.items() if a in primitive_set and b in primitive_set and w > 0.0)
    topology_density = actual_edges / possible_edges

    # Weighted scores
    weighted_scores_dict = {wp.pattern.name: round(wp.weight, 6) for wp in weighted}
    primitive_weights_dict = dict(composite.primitive_weights) if composite.primitive_weights else {}

    return ExperimentResult(
        experiment=cond.experiment,
        condition_id=cond.condition_id,
        context=cond.context,
        fragment_names=",".join(cond.fragment_names),
        n_fragments=len(cond.fragment_names),
        active_fragments=",".join(active_frag_names),
        n_active_fragments=len(active_frag_names),
        cue_stanchions=cond.cue_stanchions,
        cue_waiting_area=cond.cue_waiting_area,
        cue_service_sign=cond.cue_service_sign,
        cue_social_density=cond.cue_social_density,
        n_primitives=len(sequence),
        primitive_set=",".join(sorted(primitive_set)),
        sequence=" -> ".join(sequence),
        sequence_length=len(sequence),
        backbone_length=backbone_length,
        total_VFE=round(total_vfe, 6),
        mean_VFE=round(mean_vfe, 6),
        max_VFE=round(max_vfe, 6),
        total_EFE=round(total_efe, 6),
        efe_rate=round(total_efe / max(len(sequence), 1), 6),
        per_transition_VFE=round(total_vfe / max(len(sequence) - 1, 1), 6),
        n_obligatory=n_obligatory,
        n_advancing=n_advancing,
        n_returning=n_returning,
        topology_density=round(topology_density, 6),
        inferred_context=inferred_context,
        belief_confidence=round(belief.confidence, 6),
        belief_entropy=round(belief.entropy, 6),
        belief_reception=round(belief.distribution.get("reception", 0.0), 6),
        belief_corridor=round(belief.distribution.get("corridor", 0.0), 6),
        belief_hospital=round(belief.distribution.get("hospital", 0.0), 6),
        effective_affinities=json.dumps(effective_affinities),
        weighted_scores=json.dumps(weighted_scores_dict),
        primitive_weights=json.dumps({k: round(v, 6) for k, v in primitive_weights_dict.items()}),
        D_KL_from_baseline=0.0,  # Computed post-hoc for Exp3
    )


# ================================================================
# Stochastic pipeline runner (for multi-trial experiments)
# ================================================================
def run_single_condition_stochastic(
    cond: ExperimentCondition,
    noise_std: float = 0.1,
    seed: Optional[int] = None,
    threshold: Optional[float] = None,
    learned_affinities: Optional[Dict[str, Dict[str, float]]] = None,
    learned_weights: Optional[List[SituationType]] = None,
    precision_gating: Optional[PrecisionGating] = None,
) -> ExperimentResult:
    """Run the composition pipeline with sensory precision modulation.

    Extended version of run_single_condition that supports:
    - Sensory precision via noise_std (pi = 1/sigma^2) — scales likelihood
    - Configurable random seed for variability
    - Precision-based gating (from PrecisionGating) or fixed threshold fallback
    - Learned affinities override (from AffinityLearner Dirichlet posterior)
    - Learned A-matrix weights (from RecognizerLearner generative model)

    The sensory_precision parameter controls how much the observation
    likelihood contributes to the posterior — this is the active inference
    formulation of sensory noise (not additive corruption).

    Backward compatible: with default params, uses hard threshold.
    """
    gate_threshold = threshold if threshold is not None else CONTEXT_GATE_THRESHOLD
    all_frags = _make_all_fragments()

    # Apply learned affinities if provided
    if learned_affinities:
        for frag_name, aff_dict in learned_affinities.items():
            if frag_name in all_frags:
                all_frags[frag_name].situation_affinity.update(aff_dict)

    lib = PrimitiveLibrary()
    _register_reception_primitives(lib)

    # Requested fragments (from condition)
    requested = [all_frags[name] for name in cond.fragment_names]

    # --- Perception-driven situation recognition (A-matrix) ---
    # Sensory precision: pi = 1/sigma^2. Low precision = uncertain observations.
    sensory_precision = 1.0 / (noise_std ** 2) if noise_std > 0 else None
    sit_types = learned_weights if learned_weights else SITUATION_TYPES
    recognizer = WeakScriptRecognizer(
        sit_types,
        temperature=1.0,
        sensory_precision=sensory_precision,
        seed=seed,
    )
    pb = _make_base_percept(cond.context)
    pb = _apply_material_cues(pb, {
        "stanchions": cond.cue_stanchions,
        "waiting_area": cond.cue_waiting_area,
        "service_sign": cond.cue_service_sign,
        "social_density": cond.cue_social_density,
    })
    belief = recognizer.recognize(pb)
    inferred_context = belief.most_likely

    # Marginalize affinity over posterior: effective_aff(f) = sum_s P(s|o) * aff(f,s)
    fragments = []
    effective_affinities = {}
    for f in requested:
        eff_aff = sum(
            belief.distribution.get(s, 0.0) * f.situation_affinity.get(s, 0.0)
            for s in belief.distribution
        )
        effective_affinities[f.name] = round(eff_aff, 6)

    # Fragment gating: precision-based (principled) or threshold (fallback)
    if precision_gating is not None:
        # Precision-based gating: P(activate) = sigma(gamma * (aff - baseline))
        gating_result = precision_gating.gate_fragments(
            effective_affinities, n_contexts=len(CONTEXTS)
        )
        for f in requested:
            if f.name in gating_result.active_fragments:
                fragments.append(f)
    else:
        # Fallback: hard threshold (backward compatible)
        for f in requested:
            if effective_affinities.get(f.name, 0.0) >= gate_threshold:
                fragments.append(f)

    active_frag_names = [f.name for f in fragments]

    # Handle empty fragment set
    if not fragments:
        # Empty policy gets maximum terminal risk from C-matrix
        empty_efe = -math.log(0.01)  # ~4.6 nats
        return ExperimentResult(
            experiment=cond.experiment,
            condition_id=cond.condition_id,
            context=cond.context,
            fragment_names=",".join(cond.fragment_names),
            n_fragments=len(cond.fragment_names),
            active_fragments="",
            n_active_fragments=0,
            cue_stanchions=cond.cue_stanchions,
            cue_waiting_area=cond.cue_waiting_area,
            cue_service_sign=cond.cue_service_sign,
            cue_social_density=cond.cue_social_density,
            n_primitives=0,
            primitive_set="",
            sequence="",
            sequence_length=0,
            backbone_length=0,
            total_VFE=0.0,
            mean_VFE=0.0,
            max_VFE=0.0,
            total_EFE=round(empty_efe, 6),
            efe_rate=round(empty_efe, 6),
            per_transition_VFE=0.0,
            n_obligatory=0,
            n_advancing=0,
            n_returning=0,
            topology_density=0.0,
            inferred_context=inferred_context,
            belief_confidence=round(belief.confidence, 6),
            belief_entropy=round(belief.entropy, 6),
            belief_reception=round(belief.distribution.get("reception", 0.0), 6),
            belief_corridor=round(belief.distribution.get("corridor", 0.0), 6),
            belief_hospital=round(belief.distribution.get("hospital", 0.0), 6),
            effective_affinities=json.dumps(effective_affinities),
            weighted_scores="{}",
            primitive_weights="{}",
            D_KL_from_baseline=0.0,
        )

    cfg = RepertoireConfig()
    rep = ScriptRepertoire(lib, cfg, initial_patterns=fragments)
    rep.enable_compositional_mode()

    # D-matrix scoring
    query_scores = {}
    for name, pat in rep.patterns.items():
        aff = pat.situation_affinity.get(inferred_context, 0.0)
        query_scores[name] = aff * pat.precision

    weighted = rep.retrieve_composition(query_scores)
    composer = ScriptComposer(lib, cfg)
    composite = composer.compose_from_patterns(weighted, inferred_context)
    if composite is None:
        composite = ScriptPattern(name="empty_composite")

    topo = composite.context_topology.get(inferred_context, {})
    if not topo:
        topo = next(iter(composite.context_topology.values()), {})

    sequence = list(composite.primitives_sequence)
    primitive_set = set(sequence)

    # --- Compute metrics ---
    total_vfe = compute_sequence_VFE(topo, sequence)
    mean_vfe = total_vfe / max(len(sequence) - 1, 1)
    max_vfe = 0.0
    if len(sequence) >= 2:
        max_vfe = max(
            compute_transition_VFE(topo, sequence[i], sequence[i + 1])
            for i in range(len(sequence) - 1)
        )

    context_affinities = {}
    for sit in _SITUATION_PHASE:
        best_aff = 0.1
        for frag in fragments:
            aff = frag.situation_affinity.get(inferred_context, 0.0)
            for p_name in frag.primitive_cluster:
                prim = lib.get(p_name)
                if prim and prim.postcondition_situation == sit:
                    best_aff = max(best_aff, aff)
        context_affinities[sit] = best_aff

    total_efe = compute_policy_EFE(topo, sequence, lib, context_affinities)

    backbone_length = 0
    for prim_name in sequence:
        prim = lib.get(prim_name)
        if prim:
            phase = _SITUATION_PHASE.get(prim.postcondition_situation, 0)
            if phase > 0:
                backbone_length += 1

    n_obligatory = 0
    n_advancing = 0
    n_returning = 0
    for prim_name in sequence:
        prim = lib.get(prim_name)
        if prim:
            if prim.deontic_default == "obligatory":
                n_obligatory += 1
            phase = _SITUATION_PHASE.get(prim.postcondition_situation, 0)
            if phase > 0:
                n_advancing += 1
            else:
                n_returning += 1
        else:
            n_returning += 1

    n_prims_in_topo = len(primitive_set)
    possible_edges = n_prims_in_topo * (n_prims_in_topo - 1) if n_prims_in_topo > 1 else 1
    actual_edges = sum(1 for (a, b), w in topo.items() if a in primitive_set and b in primitive_set and w > 0.0)
    topology_density = actual_edges / possible_edges

    weighted_scores_dict = {wp.pattern.name: round(wp.weight, 6) for wp in weighted}
    primitive_weights_dict = dict(composite.primitive_weights) if composite.primitive_weights else {}

    return ExperimentResult(
        experiment=cond.experiment,
        condition_id=cond.condition_id,
        context=cond.context,
        fragment_names=",".join(cond.fragment_names),
        n_fragments=len(cond.fragment_names),
        active_fragments=",".join(active_frag_names),
        n_active_fragments=len(active_frag_names),
        cue_stanchions=cond.cue_stanchions,
        cue_waiting_area=cond.cue_waiting_area,
        cue_service_sign=cond.cue_service_sign,
        cue_social_density=cond.cue_social_density,
        n_primitives=len(sequence),
        primitive_set=",".join(sorted(primitive_set)),
        sequence=" -> ".join(sequence),
        sequence_length=len(sequence),
        backbone_length=backbone_length,
        total_VFE=round(total_vfe, 6),
        mean_VFE=round(mean_vfe, 6),
        max_VFE=round(max_vfe, 6),
        total_EFE=round(total_efe, 6),
        efe_rate=round(total_efe / max(len(sequence), 1), 6),
        per_transition_VFE=round(total_vfe / max(len(sequence) - 1, 1), 6),
        n_obligatory=n_obligatory,
        n_advancing=n_advancing,
        n_returning=n_returning,
        topology_density=round(topology_density, 6),
        inferred_context=inferred_context,
        belief_confidence=round(belief.confidence, 6),
        belief_entropy=round(belief.entropy, 6),
        belief_reception=round(belief.distribution.get("reception", 0.0), 6),
        belief_corridor=round(belief.distribution.get("corridor", 0.0), 6),
        belief_hospital=round(belief.distribution.get("hospital", 0.0), 6),
        effective_affinities=json.dumps(effective_affinities),
        weighted_scores=json.dumps(weighted_scores_dict),
        primitive_weights=json.dumps({k: round(v, 6) for k, v in primitive_weights_dict.items()}),
        D_KL_from_baseline=0.0,
    )


# ================================================================
# Learning phase: train all learners from synthetic episodes
# ================================================================
def run_learning_phase(
    n_episodes_per_context: int = 50,
    noise_std: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[AffinityLearner, PrecisionGating, RecognizerLearner]:
    """Run the learning phase: train D-matrix, precision, and A-matrix.

    All learning follows active inference principles:
    1. A-matrix: Variational free energy minimization on generative model
       (Gaussian likelihood, sufficient statistics accumulation)
    2. D-matrix: Dirichlet concentration accumulation from trajectory evidence
       (no reward signal, just Bayesian updating)
    3. Precision: Variational Laplace on the gating precision gamma
       (gradient descent on free energy w.r.t. precision)

    Parameters
    ----------
    n_episodes_per_context : int
        Training episodes per context.
    noise_std : float
        Controls sensory precision (pi = 1/sigma^2).
    seed : int | None
        Random seed.

    Returns
    -------
    (affinity_learner, precision_gating, recognizer_learner) tuple.
    """
    import numpy as np
    from experiments.episode_generator import EpisodeGenerator

    rng = np.random.default_rng(seed)

    # --- 1. Generate synthetic episodes ---
    gen = EpisodeGenerator(seed=seed, noise_std=noise_std)
    train_episodes, test_episodes = gen.generate_training_set(
        n_per_context=n_episodes_per_context,
        train_ratio=0.8,
    )

    # --- 2. Train A-matrix via VFE minimization (generative model) ---
    recognizer_learner = RecognizerLearner(
        context_names=CONTEXTS,
        prior_count=1.0,
        prior_variance=1.0,
    )
    # Warm-start from hand-coded weights (interpreted as precision-weighted means)
    recognizer_learner.warm_start(SITUATION_TYPES)

    # Convert episodes to labeled samples
    recognizer = WeakScriptRecognizer(SITUATION_TYPES, temperature=1.0)
    train_samples = []
    for ep in train_episodes:
        for pb in ep.percepts:
            features = recognizer._extract_features(pb)
            train_samples.append(LabeledSample(features=features, label=ep.label))

    test_samples = []
    for ep in test_episodes:
        for pb in ep.percepts:
            features = recognizer._extract_features(pb)
            test_samples.append(LabeledSample(features=features, label=ep.label))

    # VFE minimization: multiple passes accumulating sufficient statistics
    recognizer_learner.train(
        train_samples, val_samples=test_samples, epochs=10
    )

    # --- 3. Initialize D-matrix learner (Dirichlet concentrations) ---
    # F_0 (free_energy_scale) controls evidence sensitivity.
    # Typical EFE differences from ablation are 1-4, so F_0=2.0
    # gives evidence ratios of exp(±2) ≈ 7:1 between good/bad fragments.
    affinity_learner = AffinityLearner(
        fragment_names=ALL_FRAGMENT_NAMES,
        contexts=CONTEXTS,
        prior_concentration=1.0,  # symmetric uninformative prior
        free_energy_scale=2.0,
    )

    # --- 4. Initialize precision gating (variational Laplace on gamma) ---
    # Start with moderate gamma; prior_precision regularizes toward mean.
    precision_gating = PrecisionGating(
        initial_gamma=2.0,
        gamma_prior_mean=2.0,
        gamma_prior_precision=0.1,
        learning_rate=0.05,
        min_gamma=0.5,
    )

    # --- 5. Warm-up: run episodes, accumulate Dirichlet evidence + update precision ---
    #
    # Learning proceeds in two sub-phases:
    # Phase 5a: Exploratory — run ALL fragments (no gating) to accumulate
    #   differential evidence. Fragments that achieve low VFE in a context
    #   get more concentration; those that don't get less.
    # Phase 5b: Exploitation — apply precision gating with learned affinities
    #   to refine the precision parameter gamma.
    #
    # This mirrors the explore-exploit structure inherent in active inference:
    # initially, epistemic value (information gain) dominates; as concentrations
    # grow, pragmatic value (goal-directed behavior) takes over.
    lib = PrimitiveLibrary()
    _register_reception_primitives(lib)

    # Phase 5a: Exploratory (no gating, accumulate differential evidence)
    #
    # Uses ABLATION-based evidence: for each context, run the full
    # multi-fragment composition and measure each fragment's marginal
    # contribution to reducing Expected Free Energy (EFE).
    #
    # This is principled active inference: the D-matrix concentration for
    # (fragment, context) grows proportionally to how much that fragment
    # reduces G (expected free energy) in that context.
    #
    # prediction_error(f, c) = EFE_full(c) - EFE_without_f(c)
    #   - Negative → f reduces EFE in c → high evidence (good fit)
    #   - Positive → f increases EFE in c → low evidence (bad fit)
    #
    explore_episodes = max(n_episodes_per_context * 4 // 5, 10)
    for context in CONTEXTS:
        for ep_idx in range(explore_episodes):
            ep_seed = int(rng.integers(0, 2**31))

            # 1. Run full composition with ALL fragments (no gating)
            full_cond = ExperimentCondition(
                experiment="learning",
                condition_id=f"learn_{context}_{ep_idx}_full",
                context=context,
                fragment_names=list(ALL_FRAGMENT_NAMES),
            )
            full_result = run_single_condition_stochastic(
                full_cond,
                noise_std=noise_std,
                seed=ep_seed,
                threshold=0.0,  # no gating — all fragments active
            )
            efe_full = full_result.total_EFE

            # 2. For each fragment, run WITHOUT it (ablation)
            for frag_name in ALL_FRAGMENT_NAMES:
                ablated_frags = [f for f in ALL_FRAGMENT_NAMES if f != frag_name]
                ablated_cond = ExperimentCondition(
                    experiment="learning",
                    condition_id=f"learn_{context}_{ep_idx}_no_{frag_name}",
                    context=context,
                    fragment_names=ablated_frags,
                )
                ablated_result = run_single_condition_stochastic(
                    ablated_cond,
                    noise_std=noise_std,
                    seed=ep_seed,
                    threshold=0.0,
                )
                efe_without = ablated_result.total_EFE

                # Marginal contribution: prediction_error = EFE_full - EFE_without
                # If f REDUCES EFE (good fit): efe_full < efe_without → PE < 0 → high evidence
                # If f INCREASES EFE (bad fit): efe_full > efe_without → PE > 0 → low evidence
                prediction_error = efe_full - efe_without

                # Accuracy scales evidence by recognizer certainty
                accuracy = full_result.belief_confidence

                affinity_learner.update(
                    fragment=frag_name,
                    context=context,
                    prediction_error=prediction_error,
                    accuracy=accuracy,
                )

    # Phase 5b: Exploitation (precision gating calibration only)
    # After exploration establishes differential affinities, use them with
    # precision gating to calibrate gamma via variational Laplace.
    # Affinity learning is NOT continued here — the ablation-based evidence
    # from Phase 5a provides the principled signal; the exploitation phase
    # would only apply uniform evidence to all active fragments, diluting
    # the differential structure learned during exploration.
    exploit_episodes = n_episodes_per_context - explore_episodes
    for context in CONTEXTS:
        for ep_idx in range(exploit_episodes):
            ep_seed = int(rng.integers(0, 2**31))
            cond = ExperimentCondition(
                experiment="learning",
                condition_id=f"exploit_{context}_{ep_idx}",
                context=context,
                fragment_names=list(ALL_FRAGMENT_NAMES),
            )

            # Use precision gating with learned affinities
            result = run_single_condition_stochastic(
                cond,
                noise_std=noise_std,
                seed=ep_seed,
                precision_gating=precision_gating,
                learned_affinities=affinity_learner.affinities,
            )

            # Update precision via variational Laplace
            # Per-transition VFE: observation-level prediction error
            # (calibrates gamma against per-observation surprisal, not total trajectory)
            n_active = result.n_active_fragments
            n_total = len(ALL_FRAGMENT_NAMES)
            precision_gating.update(
                prediction_error=result.per_transition_VFE,
                n_active=n_active,
                n_total=n_total,
            )

    return affinity_learner, precision_gating, recognizer_learner


# ================================================================
# Condition generators
# ================================================================
def generate_exp1_conditions() -> List[ExperimentCondition]:
    """Experiment 1: Material Cue Factorial — 48 conditions.

    2 base fragments always present + 4 binary cues × 3 contexts.
    Material cues modify which fragments are available, reshaping
    the generative model's B-matrix and C-matrix.
    """
    cue_names = ["stanchions", "waiting_area", "service_sign", "social_density"]
    conditions = []

    for context in CONTEXTS:
        for bits in range(16):  # 0..15 → 2^4 combinations
            cue_flags = {
                "stanchions": (bits >> 0) & 1,
                "waiting_area": (bits >> 1) & 1,
                "service_sign": (bits >> 2) & 1,
                "social_density": (bits >> 3) & 1,
            }

            frag_names = list(EXP1_BASE_FRAGMENTS)
            for cue, flag in cue_flags.items():
                if flag:
                    frag_names.append(CUE_FRAGMENT_MAP[cue])

            # Remove duplicates while preserving order
            seen = set()
            unique_frags = []
            for f in frag_names:
                if f not in seen:
                    seen.add(f)
                    unique_frags.append(f)

            cond_id = f"exp1_{context}_{''.join(str(v) for v in cue_flags.values())}"
            conditions.append(ExperimentCondition(
                experiment="exp1",
                condition_id=cond_id,
                context=context,
                fragment_names=unique_frags,
                cue_stanchions=cue_flags["stanchions"],
                cue_waiting_area=cue_flags["waiting_area"],
                cue_service_sign=cue_flags["service_sign"],
                cue_social_density=cue_flags["social_density"],
            ))

    assert len(conditions) == 48, f"Expected 48 Exp1 conditions, got {len(conditions)}"
    return conditions


def generate_exp2_conditions() -> List[ExperimentCondition]:
    """Experiment 2: Fragment Scaling — 189 conditions.

    All C(6,k) for k=1..6 × 3 contexts. Tests how the pipeline
    degrades gracefully as the generative model becomes sparser.
    """
    conditions = []

    for context in CONTEXTS:
        for k in range(1, 7):
            for combo in itertools.combinations(ALL_FRAGMENT_NAMES, k):
                frag_list = list(combo)
                cond_id = f"exp2_{context}_k{k}_{'_'.join(frag_list)}"
                conditions.append(ExperimentCondition(
                    experiment="exp2",
                    condition_id=cond_id,
                    context=context,
                    fragment_names=frag_list,
                ))

    assert len(conditions) == 189, f"Expected 189 Exp2 conditions, got {len(conditions)}"
    return conditions


def generate_exp3_conditions() -> List[ExperimentCondition]:
    """Experiment 3: Context Transfer — 3 conditions.

    All 6 fragments × 3 contexts. Same knowledge (A/B matrices),
    different C-matrix → different composed behavior.
    """
    conditions = []
    for context in CONTEXTS:
        cond_id = f"exp3_{context}_all6"
        conditions.append(ExperimentCondition(
            experiment="exp3",
            condition_id=cond_id,
            context=context,
            fragment_names=list(ALL_FRAGMENT_NAMES),
        ))

    assert len(conditions) == 3, f"Expected 3 Exp3 conditions, got {len(conditions)}"
    return conditions


# ================================================================
# KL divergence for Exp3 policy comparison
# ================================================================
def _primitive_distribution(result: ExperimentResult) -> Dict[str, float]:
    """Convert primitive weights to a probability distribution."""
    weights = json.loads(result.primitive_weights)
    if not weights:
        # Uniform over primitives in sequence
        prims = result.primitive_set.split(",")
        if prims and prims[0]:
            return {p: 1.0 / len(prims) for p in prims}
        return {}
    total = sum(weights.values())
    if total <= 0:
        return {k: 1.0 / len(weights) for k in weights}
    return {k: v / total for k, v in weights.items()}


def compute_kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """D_KL(P || Q) between two distributions over primitives."""
    all_keys = set(p.keys()) | set(q.keys())
    kl = 0.0
    for k in all_keys:
        pk = p.get(k, EPSILON)
        qk = q.get(k, EPSILON)
        pk = max(pk, EPSILON)
        qk = max(qk, EPSILON)
        kl += pk * math.log(pk / qk)
    return kl


def compute_exp3_kl_divergences(results: List[ExperimentResult]) -> None:
    """Compute KL divergence from reception baseline for Exp3 results."""
    exp3 = [r for r in results if r.experiment == "exp3"]
    baseline = None
    for r in exp3:
        if r.context == "reception":
            baseline = r
            break
    if baseline is None:
        return

    p_base = _primitive_distribution(baseline)
    for r in exp3:
        p_r = _primitive_distribution(r)
        r.D_KL_from_baseline = round(compute_kl_divergence(p_r, p_base), 6)


# ================================================================
# CSV output
# ================================================================
CSV_COLUMNS = [
    "experiment", "condition_id", "context", "fragment_names", "n_fragments",
    "active_fragments", "n_active_fragments",
    "cue_stanchions", "cue_waiting_area", "cue_service_sign", "cue_social_density",
    "n_primitives", "primitive_set", "sequence", "sequence_length", "backbone_length",
    "total_VFE", "mean_VFE", "max_VFE", "total_EFE", "efe_rate", "per_transition_VFE",
    "n_obligatory", "n_advancing", "n_returning",
    "topology_density",
    "inferred_context", "belief_confidence", "belief_entropy",
    "belief_reception", "belief_corridor", "belief_hospital",
    "effective_affinities",
    "weighted_scores", "primitive_weights",
    "D_KL_from_baseline",
]


def write_csv(results: List[ExperimentResult], filepath: str) -> None:
    """Write results to CSV."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            row = {col: getattr(r, col) for col in CSV_COLUMNS}
            writer.writerow(row)


def write_json(results: List[ExperimentResult], filepath: str) -> None:
    """Write results to JSON for detailed analysis."""
    data = []
    for r in results:
        d = {col: getattr(r, col) for col in CSV_COLUMNS}
        data.append(d)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ================================================================
# Main
# ================================================================
def main():
    out_dir = os.path.join(os.path.dirname(__file__), "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    all_results = []

    # --- Experiment 1: Material Cue Factorial ---
    print("=" * 60)
    print("Experiment 1: Material Cue Factorial (48 conditions)")
    print("=" * 60)
    exp1_conditions = generate_exp1_conditions()
    exp1_results = []
    for i, cond in enumerate(exp1_conditions):
        result = run_single_condition(cond)
        exp1_results.append(result)
        print(f"  [{i+1:2d}/48] {cond.condition_id}: "
              f"{result.n_active_fragments}/{result.n_fragments} active, "
              f"{result.n_primitives} primitives, "
              f"EFE={result.total_EFE:.3f} (rate={result.efe_rate:.3f}), "
              f"inferred={result.inferred_context} ({result.belief_confidence:.2f})")
    write_csv(exp1_results, os.path.join(out_dir, "exp1_material_cues.csv"))
    all_results.extend(exp1_results)
    print(f"  -> Wrote {len(exp1_results)} rows to exp1_material_cues.csv")

    # --- Experiment 2: Fragment Scaling ---
    print("\n" + "=" * 60)
    print("Experiment 2: Fragment Scaling (189 conditions)")
    print("=" * 60)
    exp2_conditions = generate_exp2_conditions()
    exp2_results = []
    for i, cond in enumerate(exp2_conditions):
        result = run_single_condition(cond)
        exp2_results.append(result)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1:3d}/189] {cond.condition_id}: "
                  f"{result.n_active_fragments}/{result.n_fragments} active, "
                  f"{result.n_primitives} prims, "
                  f"EFE={result.total_EFE:.3f} (rate={result.efe_rate:.3f})")
    write_csv(exp2_results, os.path.join(out_dir, "exp2_fragment_scaling.csv"))
    all_results.extend(exp2_results)
    print(f"  -> Wrote {len(exp2_results)} rows to exp2_fragment_scaling.csv")

    # --- Experiment 3: Context Transfer ---
    print("\n" + "=" * 60)
    print("Experiment 3: Context Transfer (3 conditions)")
    print("=" * 60)
    exp3_conditions = generate_exp3_conditions()
    exp3_results = []
    for cond in exp3_conditions:
        result = run_single_condition(cond)
        exp3_results.append(result)
        print(f"  {cond.condition_id}: "
              f"{result.n_active_fragments}/{result.n_fragments} active, "
              f"{result.n_primitives} prims, "
              f"seq=[{result.sequence}]")

    # Compute KL divergences for Exp3
    compute_exp3_kl_divergences(exp3_results)
    write_csv(exp3_results, os.path.join(out_dir, "exp3_context_transfer.csv"))
    all_results.extend(exp3_results)
    print(f"  -> Wrote {len(exp3_results)} rows to exp3_context_transfer.csv")

    # --- Combined output ---
    write_json(all_results, os.path.join(out_dir, "all_experiments.json"))
    print(f"\n{'=' * 60}")
    print(f"Total: {len(all_results)} conditions "
          f"(48 + 189 + 3 = {48 + 189 + 3})")
    print(f"Results in: {out_dir}")

    # --- Verification ---
    print(f"\n{'=' * 60}")
    print("Verification checks:")
    print(f"  Exp1 count: {len(exp1_results)} (expected 48) {'OK' if len(exp1_results) == 48 else 'FAIL'}")
    print(f"  Exp2 count: {len(exp2_results)} (expected 189) {'OK' if len(exp2_results) == 189 else 'FAIL'}")
    print(f"  Exp3 count: {len(exp3_results)} (expected 3) {'OK' if len(exp3_results) == 3 else 'FAIL'}")

    # VFE non-negativity
    vfe_violations = [r for r in all_results if r.total_VFE < -0.001]
    print(f"  VFE >= 0: {len(vfe_violations)} violations {'OK' if not vfe_violations else 'FAIL'}")

    # Determinism check: run one condition twice
    test_cond = exp1_conditions[0]
    r1 = run_single_condition(test_cond)
    r2 = run_single_condition(test_cond)
    deterministic = (r1.sequence == r2.sequence and r1.total_VFE == r2.total_VFE)
    print(f"  Determinism: {'OK' if deterministic else 'FAIL'}")

    # Recognition accuracy: inferred_context == ground_truth context
    correct = sum(1 for r in all_results if r.inferred_context == r.context)
    total = len(all_results)
    print(f"  Recognition accuracy: {correct}/{total} = {correct/total:.1%}")

    # Hospital sanity: in hospital context with all fragments, direct_approach should be weak
    hospital_full = [r for r in exp3_results if r.context == "hospital"]
    if hospital_full:
        h = hospital_full[0]
        ws = json.loads(h.weighted_scores)
        da_w = ws.get("direct_approach", 0.0)
        wp_w = ws.get("wait_patiently", 0.0)
        print(f"  Hospital sanity: direct_approach={da_w:.4f}, wait_patiently={wp_w:.4f} "
              f"{'OK' if wp_w > da_w else 'CHECK'}")

    print(f"\nDone.")


def main_learned():
    """Run the full experimental protocol with learned parameters.

    New protocol:
    Phase 1: Learning (replaces hand-tuning)
    Phase 2: Evaluation (multi-trial with noise)
    Phase 3: Statistical analysis

    This demonstrates that starting from uninformative priors, the system
    learns context-specific parameters that outperform baselines.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from experiments.stochastic_runner import (
        StochasticRunner,
        StochasticExperimentConfig,
    )
    from experiments.baselines import run_all_baselines
    from experiments.statistical_analysis import (
        bootstrap_ci,
        cohens_d,
        permutation_test,
    )
    import numpy as np

    out_dir = os.path.join(os.path.dirname(__file__), "experiment_results", "learned")
    os.makedirs(out_dir, exist_ok=True)

    # ================================================================
    # Phase 1: Learning
    # ================================================================
    print("=" * 60)
    print("PHASE 1: Learning from interaction (replacing hand-tuning)")
    print("=" * 60)

    affinity_learner, precision_gating, recognizer_learner = run_learning_phase(
        n_episodes_per_context=100,
        noise_std=0.1,
        seed=12345,
    )

    # Report learning results
    print(f"\n  Precision gamma: {precision_gating.gamma:.4f} "
          f"(VFE contribution: {precision_gating.variational_free_energy():.4f})")
    print(f"  Precision converged: {precision_gating.has_converged()}")

    # Expert affinities for correlation
    expert_affinities = {}
    all_frags = _make_all_fragments()
    for name, frag in all_frags.items():
        expert_affinities[name] = dict(frag.situation_affinity)

    correlation = affinity_learner.correlation_with_expert(expert_affinities)
    print(f"  Learned-expert affinity correlation: r={correlation:.4f}")

    # Recognizer accuracy
    if recognizer_learner.history:
        last_metrics = recognizer_learner.history[-1]
        print(f"  Recognizer accuracy: {last_metrics.accuracy:.3f} (VFE: {last_metrics.vfe:.3f})")

    # Save learning curves
    learning_data = {
        "affinity_history": [
            {"episode": p.episode, "fragment": p.fragment,
             "context": p.context, "affinity": p.affinity,
             "concentration": p.concentration, "precision": p.precision}
            for p in affinity_learner.learning_curve
        ],
        "precision_history": [
            {"episode": h.episode, "gamma": h.new_gamma,
             "prediction_error": h.prediction_error,
             "gated_fraction": h.gated_fraction}
            for h in precision_gating.history
        ],
        "recognizer_history": [
            {"epoch": m.episode, "vfe": m.vfe,
             "accuracy": m.accuracy,
             "kl_divergence": m.kl_divergence}
            for m in recognizer_learner.history
        ],
        "learned_affinities": affinity_learner.affinities,
        "learned_concentrations": affinity_learner.concentrations,
        "learned_gamma": precision_gating.gamma,
        "expert_correlation": correlation,
    }
    with open(os.path.join(out_dir, "learning_curves.json"), "w") as f:
        json.dump(learning_data, f, indent=2)
    print(f"  -> Saved learning curves to {out_dir}/learning_curves.json")

    # ================================================================
    # Phase 2: Evaluation (multi-trial)
    # ================================================================
    print(f"\n{'=' * 60}")
    print("PHASE 2: Multi-trial evaluation with learned parameters")
    print("=" * 60)

    learned_affinities = affinity_learner.affinities
    learned_sit_types = recognizer_learner.get_situation_types()

    # Define run function for stochastic runner
    def run_fn(cond, noise_std=0.1, seed=None):
        result = run_single_condition_stochastic(
            cond,
            noise_std=noise_std,
            seed=seed,
            precision_gating=precision_gating,
            learned_affinities=learned_affinities,
            learned_weights=learned_sit_types,
        )
        return {
            "n_active_fragments": result.n_active_fragments,
            "n_primitives": result.n_primitives,
            "total_VFE": result.total_VFE,
            "mean_VFE": result.mean_VFE,
            "total_EFE": result.total_EFE,
            "efe_rate": result.efe_rate,
            "per_transition_VFE": result.per_transition_VFE,
            "inferred_context": result.inferred_context,
            "belief_confidence": result.belief_confidence,
            "belief_entropy": result.belief_entropy,
            "topology_density": result.topology_density,
            "sequence": result.sequence,
            "primitive_set": result.primitive_set,
        }

    config = StochasticExperimentConfig(
        n_trials=50,
        noise_levels=[0.05, 0.1, 0.2],
        output_dir=out_dir,
    )
    runner = StochasticRunner(run_fn, config)

    # Run Exp3 (context transfer) with multi-trial — most important
    exp3_conditions = generate_exp3_conditions()
    conditions_tuples = [
        (cond, cond.condition_id, cond.context) for cond in exp3_conditions
    ]

    def progress(completed, total):
        if completed % 3 == 0 or completed == total:
            print(f"  Progress: {completed}/{total}")

    multi_results = runner.run_all_conditions(
        conditions_tuples,
        progress_callback=progress,
    )

    # Save multi-trial results
    results_path = runner.save_results(os.path.join(out_dir, "multi_trial_results.json"))
    print(f"  -> Saved {len(multi_results)} multi-trial results to {results_path}")

    # ================================================================
    # Phase 3: Statistical Analysis
    # ================================================================
    print(f"\n{'=' * 60}")
    print("PHASE 3: Statistical analysis")
    print("=" * 60)

    # Compare three systems at noise=0.1 using EFE rate (G/T) for fair
    # variable-horizon comparison:
    # 1. Learned (precision-gated, Dirichlet affinities, generative A-matrix)
    # 2. Learned affinities + hard threshold (isolates affinity learning)
    # 3. Hand-coded (original hard threshold, expert affinities)
    # 4. No-gating baseline (all fragments active, no differentiation)
    learned_efe_rate_values = []
    learned_hard_efe_rate_values = []
    handcoded_efe_rate_values = []
    nogating_efe_rate_values = []

    for cond in exp3_conditions:
        for trial in range(50):
            seed = int(np.random.default_rng(trial + 100).integers(0, 2**31))

            # Learned with precision gating
            r_learned = run_single_condition_stochastic(
                cond, noise_std=0.1, seed=seed,
                precision_gating=precision_gating,
                learned_affinities=learned_affinities,
                learned_weights=learned_sit_types,
            )
            learned_efe_rate_values.append(r_learned.efe_rate)

            # Learned affinities with hard threshold (same threshold as expert)
            r_learned_hard = run_single_condition_stochastic(
                cond, noise_std=0.1, seed=seed,
                threshold=CONTEXT_GATE_THRESHOLD,
                learned_affinities=learned_affinities,
                learned_weights=learned_sit_types,
            )
            learned_hard_efe_rate_values.append(r_learned_hard.efe_rate)

            # Hand-coded (expert affinities + hard threshold)
            r_handcoded = run_single_condition_stochastic(
                cond, noise_std=0.1, seed=seed,
            )
            handcoded_efe_rate_values.append(r_handcoded.efe_rate)

            # No-gating baseline (threshold=0)
            r_nogating = run_single_condition_stochastic(
                cond, noise_std=0.1, seed=seed,
                threshold=0.0,
            )
            nogating_efe_rate_values.append(r_nogating.efe_rate)

    learned_arr = np.array(learned_efe_rate_values)
    learned_hard_arr = np.array(learned_hard_efe_rate_values)
    handcoded_arr = np.array(handcoded_efe_rate_values)
    nogating_arr = np.array(nogating_efe_rate_values)

    # --- Effect sizes and CIs (using EFE rate for length-independent comparison) ---
    print("\n  --- Comparison: Learned (precision gating) vs Handcoded [EFE rate G/T] ---")
    effect = cohens_d(handcoded_arr, learned_arr, seed=42)
    perm = permutation_test(handcoded_arr, learned_arr, seed=42)
    ci_learned = bootstrap_ci(learned_arr, seed=42)
    ci_handcoded = bootstrap_ci(handcoded_arr, seed=42)
    print(f"    Cohen's d: {effect.d:.4f} [{effect.ci_lower:.4f}, {effect.ci_upper:.4f}] ({effect.interpretation})")
    print(f"    Permutation p: {perm.p_value:.6f}")
    print(f"    Learned EFE rate: {ci_learned.statistic:.4f} [{ci_learned.ci_lower:.4f}, {ci_learned.ci_upper:.4f}]")
    print(f"    Handcoded EFE rate: {ci_handcoded.statistic:.4f} [{ci_handcoded.ci_lower:.4f}, {ci_handcoded.ci_upper:.4f}]")

    print("\n  --- Comparison: Learned (hard threshold) vs Handcoded [EFE rate G/T] ---")
    effect_hard = cohens_d(handcoded_arr, learned_hard_arr, seed=42)
    ci_learned_hard = bootstrap_ci(learned_hard_arr, seed=42)
    print(f"    Cohen's d: {effect_hard.d:.4f} [{effect_hard.ci_lower:.4f}, {effect_hard.ci_upper:.4f}] ({effect_hard.interpretation})")
    print(f"    Learned (hard) EFE rate: {ci_learned_hard.statistic:.4f} [{ci_learned_hard.ci_lower:.4f}, {ci_learned_hard.ci_upper:.4f}]")

    print("\n  --- Comparison: No-gating baseline vs Handcoded [EFE rate G/T] ---")
    effect_nogating = cohens_d(handcoded_arr, nogating_arr, seed=42)
    ci_nogating = bootstrap_ci(nogating_arr, seed=42)
    print(f"    Cohen's d: {effect_nogating.d:.4f} [{effect_nogating.ci_lower:.4f}, {effect_nogating.ci_upper:.4f}] ({effect_nogating.interpretation})")
    print(f"    No-gating EFE rate: {ci_nogating.statistic:.4f} [{ci_nogating.ci_lower:.4f}, {ci_nogating.ci_upper:.4f}]")

    print("\n  --- Comparison: Learned (precision gating) vs No-gating [EFE rate G/T] ---")
    effect_vs_baseline = cohens_d(nogating_arr, learned_arr, seed=42)
    perm_vs_baseline = permutation_test(nogating_arr, learned_arr, seed=42)
    print(f"    Cohen's d: {effect_vs_baseline.d:.4f} [{effect_vs_baseline.ci_lower:.4f}, {effect_vs_baseline.ci_upper:.4f}] ({effect_vs_baseline.interpretation})")
    print(f"    Permutation p: {perm_vs_baseline.p_value:.6f}")

    # Save statistical results
    stats_data = {
        "learned_vs_handcoded": {
            "cohens_d": {"d": effect.d, "ci": [effect.ci_lower, effect.ci_upper],
                         "interpretation": effect.interpretation},
            "permutation_test": {"p_value": perm.p_value,
                                 "observed_stat": perm.observed_statistic},
        },
        "learned_hard_vs_handcoded": {
            "cohens_d": {"d": effect_hard.d, "ci": [effect_hard.ci_lower, effect_hard.ci_upper],
                         "interpretation": effect_hard.interpretation},
        },
        "nogating_vs_handcoded": {
            "cohens_d": {"d": effect_nogating.d, "ci": [effect_nogating.ci_lower, effect_nogating.ci_upper],
                         "interpretation": effect_nogating.interpretation},
        },
        "learned_vs_nogating": {
            "cohens_d": {"d": effect_vs_baseline.d, "ci": [effect_vs_baseline.ci_lower, effect_vs_baseline.ci_upper],
                         "interpretation": effect_vs_baseline.interpretation},
            "permutation_test": {"p_value": perm_vs_baseline.p_value,
                                 "observed_stat": perm_vs_baseline.observed_statistic},
        },
        "efe_rate_summary": {
            "metric": "EFE rate (G/T) — length-independent comparison",
            "learned_precision_gating": {"mean": ci_learned.statistic, "ci": [ci_learned.ci_lower, ci_learned.ci_upper]},
            "learned_hard_threshold": {"mean": ci_learned_hard.statistic, "ci": [ci_learned_hard.ci_lower, ci_learned_hard.ci_upper]},
            "handcoded": {"mean": ci_handcoded.statistic, "ci": [ci_handcoded.ci_lower, ci_handcoded.ci_upper]},
            "no_gating": {"mean": ci_nogating.statistic, "ci": [ci_nogating.ci_lower, ci_nogating.ci_upper]},
        },
        "learning_metrics": {
            "expert_correlation": correlation,
            "precision_gamma": precision_gating.gamma,
            "precision_converged": precision_gating.has_converged(),
        },
    }
    with open(os.path.join(out_dir, "statistical_results.json"), "w") as f:
        json.dump(stats_data, f, indent=2)
    print(f"\n  -> Saved statistical results to {out_dir}/statistical_results.json")

    print(f"\n{'=' * 60}")
    print("DONE — Full learned experimental protocol complete.")
    print(f"Results in: {out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import sys
    if "--learned" in sys.argv:
        main_learned()
    else:
        main()
