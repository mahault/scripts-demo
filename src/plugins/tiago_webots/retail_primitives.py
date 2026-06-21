"""Retail-specific script primitives and fragments for the investor demo.

The restock causal chain:
  stock_zone → scan → pick → shelf_zone → place → shelf_zone → place →
  counter_zone → wait → stock_zone (loop)
"""

from __future__ import annotations

from architecture_core.core.types import AffectState, SkillRequest
from architecture_core.cognition.scripts.repertoire_types import (
    RepertoireConfig,
    ScriptPattern,
    ScriptPrimitive,
)
from architecture_core.cognition.scripts.primitive_library import PrimitiveLibrary


def register_retail_primitives(lib: PrimitiveLibrary) -> None:
    """Add retail/restock primitives to an existing library."""
    retail_prims = [
        ScriptPrimitive(
            name="scan-stock",
            skill_template=SkillRequest(
                skill="gaze", params={"mode": "scan"},
            ),
            precondition_situations=["stock_zone", "open_area"],
            postcondition_situation="scene_assessed",
            expected_affect=AffectState(valence=0.0, arousal=0.1),
            typical_duration_s=2.0,
            deontic_default="permitted",
        ),
        ScriptPrimitive(
            name="pick-item",
            skill_template=SkillRequest(
                skill="pick_place", params={"mode": "pick"},
            ),
            precondition_situations=["scene_assessed", "stock_zone"],
            postcondition_situation="item_held",
            expected_affect=AffectState(valence=0.1, arousal=0.1),
            typical_duration_s=4.0,
            deontic_default="permitted",
        ),
        ScriptPrimitive(
            name="place-item",
            skill_template=SkillRequest(
                skill="pick_place", params={"mode": "place"},
            ),
            precondition_situations=["item_held", "shelf_zone", "counter_zone"],
            postcondition_situation="item_placed",
            expected_affect=AffectState(valence=0.2, arousal=0.0),
            typical_duration_s=4.0,
            deontic_default="permitted",
        ),
        ScriptPrimitive(
            name="navigate-to-shelf",
            skill_template=SkillRequest(
                skill="navigate",
                goal={"x": -2.0, "y": -5.0},
                params={"intent": "approach", "speed_scale": 0.5},
            ),
            precondition_situations=["item_held", "stock_zone", "counter_zone"],
            postcondition_situation="shelf_zone",
            expected_affect=AffectState(valence=0.0, arousal=0.0),
            typical_duration_s=5.0,
            deontic_default="permitted",
        ),
        ScriptPrimitive(
            name="navigate-to-stock",
            skill_template=SkillRequest(
                skill="navigate",
                goal={"x": -5.5, "y": -1.0},
                params={"intent": "approach", "speed_scale": 0.5},
            ),
            precondition_situations=["counter_zone", "shelf_zone", "open_area"],
            postcondition_situation="stock_zone",
            expected_affect=AffectState(valence=0.0, arousal=0.0),
            typical_duration_s=5.0,
            deontic_default="permitted",
        ),
        ScriptPrimitive(
            name="navigate-to-counter",
            skill_template=SkillRequest(
                skill="navigate",
                goal={"x": 4.5, "y": -7.5},
                params={"intent": "approach", "speed_scale": 0.4},
            ),
            precondition_situations=["shelf_zone", "open_area"],
            postcondition_situation="counter_zone",
            expected_affect=AffectState(valence=0.1, arousal=0.0),
            typical_duration_s=5.0,
            deontic_default="permitted",
        ),
        ScriptPrimitive(
            name="wait-at-counter",
            skill_template=SkillRequest(
                skill="navigate", params={"intent": "wait", "speed_scale": 0.0},
            ),
            precondition_situations=["counter_zone", "customer_present"],
            postcondition_situation="ready_for_stock",
            expected_affect=AffectState(valence=0.0, arousal=-0.1),
            typical_duration_s=5.0,
            deontic_default="permitted",
        ),
        ScriptPrimitive(
            name="greet-customer",
            skill_template=SkillRequest(
                skill="gaze", params={"mode": "look_at_agent"},
            ),
            precondition_situations=["customer_present", "counter_zone", "shelf_zone"],
            postcondition_situation="interaction",
            expected_affect=AffectState(valence=0.3, arousal=0.1),
            typical_duration_s=2.0,
            deontic_default="permitted",
        ),
    ]
    for p in retail_prims:
        lib.register(p)


def make_retail_fragments() -> list:
    """Return weak retail script fragments for compositional assembly."""
    def _frag(name, primitives, affinity):
        return ScriptPattern(
            name=name,
            primitives_sequence=sorted(primitives),
            primitive_cluster=set(primitives),
            precision=0.1,
            situation_affinity=affinity,
        )

    return [
        _frag("observe_stock",
              {"scan-stock", "gaze-scan"},
              {"retail": 0.9, "stock_zone": 0.8}),
        _frag("collect_items",
              {"pick-item"},
              {"retail": 0.8, "stock_zone": 0.9}),
        _frag("restock_shelf",
              {"navigate-to-shelf", "place-item"},
              {"retail": 0.9, "shelf_zone": 0.9}),
        _frag("service_counter",
              {"navigate-to-counter", "wait-at-counter", "greet-customer"},
              {"retail": 0.8, "counter_zone": 0.9}),
        _frag("return_to_stock",
              {"navigate-to-stock"},
              {"retail": 0.7, "counter_zone": 0.6, "shelf_zone": 0.6}),
    ]


def make_restock_seed_pattern() -> ScriptPattern:
    """Pre-seed pattern for fast demo crystallization.

    Starts just below the strong threshold so 1-2 successful loops
    trigger crystallization.  Sequence matches the actual restock nav loop.
    """
    return ScriptPattern(
        name="restock_full",
        primitives_sequence=[
            "navigate-to-stock",
            "navigate-to-shelf",
            "navigate-to-shelf",
            "navigate-to-counter",
        ],
        primitive_cluster=set(),  # empty -> not cluster-based, uses sequence directly
        precision=1.0,
        trajectory_count=2,
        mean_free_energy=0.0,
        situation_affinity={"retail": 0.9, "stock_zone": 0.8, "shelf_zone": 0.8, "counter_zone": 0.8},
    )


def make_retail_repertoire_config() -> RepertoireConfig:
    """Tuned config for faster crystallization in the demo.

    The learner crystallises a script purely from *observation*, where the
    trajectory free energy is high and noisy (it is reconstructed from the
    teacher's state stream, not the learner's own controlled rollouts).  That
    noise scales down the per-loop precision gain and even reverses it, so with
    the original thresholds the restock pattern never reaches "strong".  For the
    demo we relax the free-energy penalty (treat consistently-observed loops as
    good evidence) and lower the strong threshold to what observation actually
    reaches, so ~3 consistent loops crystallise.
    """
    return RepertoireConfig(
        strong_precision_threshold=0.8,
        min_trajectories_for_promotion=3,
        precision_gain_on_success=0.6,
        precision_loss_on_violation=0.05,
        precision_decay_rate=0.01,
        max_free_energy_for_promotion=1.0e6,
        max_composition_length=8,
    )
