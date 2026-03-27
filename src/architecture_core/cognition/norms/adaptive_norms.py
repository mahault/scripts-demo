"""Adaptive norm discovery from observable behavior.

Replaces static cultural profiles with runtime inference. Norms are
not assigned by cultural label — they are discovered from observable
cues: inter-human distances, approach speeds, gaze engagement, affect
reactions, and interaction durations.

Norm features are embedded into ScriptPatterns so they co-evolve with
behavioral scripts in the same learning loop (precision accumulation,
context topology, composition, consolidation).

Key function:
    extract_norm_features(pb) -> Dict[str, float]
        Extracts all observable norm-relevant signals from a PerceptBundle.
        Called by Executive each tick; stored on Blackboard and passed to
        ScriptRepertoire for trajectory recording.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

from architecture_core.core.types import PerceptBundle
from architecture_core.safety.geometry import point_distance


# =====================================================================
# Proxemic prior -- generic initial values (NOT a cultural identity)
# =====================================================================
@dataclass
class ProxemicPrior:
    """Initial prior for proxemic distance thresholds.

    These are starting estimates used by ``from_profile()`` factories
    on ProxemicsAugmentation and PersonalSpaceRule.  They are NOT
    cultural identities — the adaptive norm system will refine them
    from observations.

    Distances in metres.
    """
    intimate_distance: float = 0.45
    personal_distance: float = 1.2
    social_distance: float = 3.6
    public_distance: float = 7.6


DEFAULT_PRIOR = ProxemicPrior()


# =====================================================================
# Norm feature extraction
# =====================================================================
def extract_norm_features(pb: PerceptBundle) -> Dict[str, float]:
    """Extract observable norm-relevant features from perception.

    Returns a dict of named features. Keys are standardised strings
    that ScriptPattern.norm_features uses to store learned norms and
    PersonalSpaceRule/SpeedLimitRule use to adapt thresholds.

    Feature set:
        mean_inter_human_distance  — pairwise agent distances
        closest_human_distance     — robot to nearest human
        max_approach_speed         — fastest agent velocity
        mean_gaze_engagement       — average gaze/attention score
        mean_human_valence         — average emotional valence
        max_interaction_duration   — longest ongoing interaction
    """
    features: Dict[str, float] = {}
    agents: List[Dict[str, Any]] = pb.world.get("agents", [])

    # ------------------------------------------------------------------
    # 1. Inter-human pairwise distances
    # ------------------------------------------------------------------
    if len(agents) >= 2:
        distances: List[float] = []
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                pi = agents[i].get("pose", (0, 0))
                pj = agents[j].get("pose", (0, 0))
                d = point_distance(pi[0], pi[1], pj[0], pj[1])
                distances.append(d)
        if distances:
            features["mean_inter_human_distance"] = sum(distances) / len(distances)

    # ------------------------------------------------------------------
    # 2. Closest human distance (robot → nearest agent)
    # ------------------------------------------------------------------
    proxemics = pb.social.get("proxemics", {})
    closest = proxemics.get("closest_distance")
    if closest is not None and closest < float("inf"):
        features["closest_human_distance"] = closest

    # ------------------------------------------------------------------
    # 3. Approach speed (from agent velocities)
    # ------------------------------------------------------------------
    max_speed = 0.0
    found_speed = False
    for agent in agents:
        vel = agent.get("velocity")
        if vel is not None:
            if isinstance(vel, (list, tuple)):
                speed = math.sqrt(sum(v * v for v in vel))
            else:
                speed = abs(vel)
            if speed > max_speed:
                max_speed = speed
                found_speed = True
    if found_speed:
        features["max_approach_speed"] = max_speed

    # ------------------------------------------------------------------
    # 4. Gaze / engagement
    # ------------------------------------------------------------------
    engagement = pb.social.get("engagement", {})
    eng_readings = engagement.get("readings", [])
    if eng_readings:
        gaze_scores: List[float] = []
        max_duration = 0.0
        for r in eng_readings:
            if isinstance(r, dict):
                gs = r.get("gaze_score")
                if gs is not None:
                    gaze_scores.append(gs)
                dur = r.get("duration")
                if dur is not None and dur > max_duration:
                    max_duration = dur
        if gaze_scores:
            features["mean_gaze_engagement"] = sum(gaze_scores) / len(gaze_scores)
        if max_duration > 0:
            features["max_interaction_duration"] = max_duration

    # ------------------------------------------------------------------
    # 5. Affect (mean human valence)
    # ------------------------------------------------------------------
    affect = pb.social.get("affect", {})
    affect_readings = affect.get("readings", [])
    if affect_readings:
        valences: List[float] = []
        for r in affect_readings:
            if isinstance(r, dict):
                v = r.get("valence")
                if v is not None:
                    valences.append(v)
        if valences:
            features["mean_human_valence"] = sum(valences) / len(valences)

    return features
