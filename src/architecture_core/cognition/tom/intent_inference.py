"""Fused intent inference — multi-signal human intent prediction.

Combines four observation channels into a unified per-agent intent
posterior:

1. **Kinematic ToM** — approach/avoid/yield/wait from position/velocity
2. **Engagement** — gaze direction, body orientation, interaction duration
3. **Weak script** — situation type belief (corridor, handover, open area)
4. **Affect** — emotional state of the observed agent

Each channel produces a likelihood over intent hypotheses.  The module
fuses them via a weighted log-linear combination (product of experts)
and returns a posterior distribution with confidence.

Theoretical grounding:
- The kinematic channel is a geometric heuristic (existing ToMPlannerAdapter)
- Engagement + affect are A-matrix observations (Albarracin et al. 2021)
- The weak script provides a contextual prior: "in a handover situation,
  approach is much more likely than avoid"
- Fusion follows the multi-modal evidence integration pattern from
  the active inference literature (Friston et al. 2017)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from architecture_core.core.types import Intent, PerceptBundle


# All possible intent hypotheses
INTENT_LABELS: List[Intent] = ["approach", "avoid", "yield", "wait", "neutral"]


@dataclass
class IntentHypothesis:
    """Single intent hypothesis with probability and evidence sources."""
    intent: Intent
    probability: float
    evidence: Dict[str, float] = field(default_factory=dict)


@dataclass
class AgentIntentBelief:
    """Posterior distribution over intents for one observed agent."""
    entity_id: str
    hypotheses: List[IntentHypothesis]
    most_likely: Intent
    confidence: float          # probability of most_likely
    entropy: float             # Shannon entropy of distribution
    evidence_sources: List[str]  # which channels contributed


# =====================================================================
# Script-conditioned intent priors
# =====================================================================
# For each situation type, prior probabilities over intent
# These encode: "in a corridor encounter, yield is likely; in a
# handover, approach is likely"
DEFAULT_SCRIPT_PRIORS: Dict[str, Dict[Intent, float]] = {
    "corridor_encounter": {
        "approach": 0.15, "avoid": 0.25, "yield": 0.35, "wait": 0.20, "neutral": 0.05,
    },
    "open_area": {
        "approach": 0.30, "avoid": 0.15, "yield": 0.10, "wait": 0.10, "neutral": 0.35,
    },
    "handover": {
        "approach": 0.55, "avoid": 0.05, "yield": 0.10, "wait": 0.25, "neutral": 0.05,
    },
}

# Uniform fallback
_UNIFORM_PRIOR: Dict[Intent, float] = {
    i: 1.0 / len(INTENT_LABELS) for i in INTENT_LABELS
}


# =====================================================================
# Engagement → intent likelihood
# =====================================================================
def _engagement_likelihood(
    gaze: float, body_orient: float, proximity_trend: float,
) -> Dict[Intent, float]:
    """Map engagement signals to intent likelihoods.

    High gaze + facing robot + approaching → likely approach/handover
    Low gaze + turning away + retreating → likely avoid/neutral
    """
    # Approaching factor: negative trend = approaching
    approach_signal = max(0.0, min(1.0, -proximity_trend / 0.5 + 0.5))

    likelihoods: Dict[Intent, float] = {}
    engagement_score = 0.4 * gaze + 0.3 * body_orient + 0.3 * approach_signal

    likelihoods["approach"] = 0.1 + 0.8 * engagement_score
    likelihoods["wait"] = 0.1 + 0.5 * gaze * (1 - approach_signal)
    likelihoods["yield"] = 0.1 + 0.4 * (1 - engagement_score) * approach_signal
    likelihoods["avoid"] = 0.1 + 0.7 * (1 - gaze) * (1 - body_orient)
    likelihoods["neutral"] = 0.1 + 0.5 * (1 - engagement_score) * (1 - approach_signal)

    return likelihoods


# =====================================================================
# Affect → intent likelihood
# =====================================================================
def _affect_likelihood(
    valence: float, arousal: float,
) -> Dict[Intent, float]:
    """Map emotional state to intent likelihoods.

    Positive valence + low arousal → approach/wait (cooperative)
    Negative valence + high arousal → avoid/yield (threat/frustration)
    """
    # Map valence [-1,1] and arousal [-1,1] to factors
    pos_v = max(0.0, valence)
    neg_v = max(0.0, -valence)
    hi_a = max(0.0, arousal)
    lo_a = max(0.0, -arousal)

    likelihoods: Dict[Intent, float] = {}
    likelihoods["approach"] = 0.15 + 0.5 * pos_v + 0.1 * lo_a
    likelihoods["avoid"] = 0.1 + 0.5 * neg_v + 0.3 * hi_a
    likelihoods["yield"] = 0.1 + 0.3 * neg_v + 0.2 * lo_a
    likelihoods["wait"] = 0.15 + 0.2 * pos_v + 0.3 * lo_a
    likelihoods["neutral"] = 0.2 + 0.2 * (1 - abs(valence)) * (1 - abs(arousal))

    return likelihoods


# =====================================================================
# Kinematic → intent likelihood
# =====================================================================
def _kinematic_likelihood(
    kinematic_intent: Intent,
) -> Dict[Intent, float]:
    """Convert a discrete kinematic intent into a soft likelihood.

    The kinematic classifier is noisy, so we spread probability:
    80% on the classified intent, 5% on each alternative.
    """
    likelihoods: Dict[Intent, float] = {}
    for i in INTENT_LABELS:
        likelihoods[i] = 0.8 if i == kinematic_intent else 0.05
    return likelihoods


# =====================================================================
# IntentInference
# =====================================================================
class IntentInference:
    """Fuse multi-modal signals into per-agent intent posteriors.

    Parameters
    ----------
    script_priors : dict | None
        Mapping from situation type name to intent prior distribution.
        Defaults to ``DEFAULT_SCRIPT_PRIORS``.
    channel_weights : dict | None
        Relative log-linear weights for each channel.
        Keys: "kinematic", "engagement", "affect", "script_prior".
    """

    def __init__(
        self,
        script_priors: Optional[Dict[str, Dict[Intent, float]]] = None,
        channel_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._script_priors = script_priors or DEFAULT_SCRIPT_PRIORS
        self._weights = channel_weights or {
            "kinematic": 1.0,
            "engagement": 0.8,
            "affect": 0.5,
            "script_prior": 0.7,
        }

    def infer(
        self,
        pb: PerceptBundle,
        kinematic_intent: Intent = "neutral",
        situation_type: Optional[str] = None,
    ) -> List[AgentIntentBelief]:
        """Compute intent belief for each observed agent.

        Parameters
        ----------
        pb : PerceptBundle
            Current perception with social/attention sections populated.
        kinematic_intent : Intent
            The intent from the kinematic ToM (ToMPlannerAdapter).
        situation_type : str | None
            The most likely situation from WeakScriptRecognizer.

        Returns
        -------
        list[AgentIntentBelief]
            One belief per observed agent, sorted by confidence (highest first).
        """
        agents = pb.world.get("agents", [])
        if not agents:
            return []

        # Gather augmentation data
        eng_data = pb.social.get("engagement", {})
        eng_readings = {
            r.get("entity_id", ""): r
            for r in eng_data.get("readings", [])
        }

        aff_data = pb.social.get("affect", {})
        aff_readings = {
            r.get("entity_id", ""): r
            for r in aff_data.get("readings", [])
        }

        # Script prior
        if situation_type and situation_type in self._script_priors:
            script_prior = self._script_priors[situation_type]
        else:
            script_prior = _UNIFORM_PRIOR

        beliefs: List[AgentIntentBelief] = []

        for agent in agents:
            eid = agent.get("id", "unknown")
            sources: List[str] = []

            # --- Collect likelihoods per channel ---
            log_posterior: Dict[Intent, float] = {i: 0.0 for i in INTENT_LABELS}

            # 1. Kinematic
            kin_lik = _kinematic_likelihood(kinematic_intent)
            w_k = self._weights.get("kinematic", 1.0)
            for i in INTENT_LABELS:
                log_posterior[i] += w_k * math.log(max(kin_lik[i], 1e-8))
            sources.append("kinematic")

            # 2. Engagement
            eng = eng_readings.get(eid)
            if eng:
                gaze = eng.get("gaze_on_robot", 0.5)
                body = eng.get("body_orientation", 0.5)
                trend = eng.get("proximity_trend", 0.0)
                eng_lik = _engagement_likelihood(gaze, body, trend)
                w_e = self._weights.get("engagement", 0.8)
                for i in INTENT_LABELS:
                    log_posterior[i] += w_e * math.log(max(eng_lik[i], 1e-8))
                sources.append("engagement")

            # 3. Affect
            aff = aff_readings.get(eid)
            if aff:
                val = aff.get("valence", 0.0)
                aro = aff.get("arousal", 0.0)
                aff_lik = _affect_likelihood(val, aro)
                w_a = self._weights.get("affect", 0.5)
                for i in INTENT_LABELS:
                    log_posterior[i] += w_a * math.log(max(aff_lik[i], 1e-8))
                sources.append("affect")

            # 4. Script prior
            w_s = self._weights.get("script_prior", 0.7)
            for i in INTENT_LABELS:
                log_posterior[i] += w_s * math.log(max(script_prior.get(i, 0.05), 1e-8))
            sources.append("script_prior")

            # --- Softmax to get posterior ---
            posterior = _softmax_dict(log_posterior)

            most_likely = max(posterior, key=posterior.get)  # type: ignore[arg-type]
            confidence = posterior[most_likely]
            entropy = _entropy(posterior)

            hypotheses = [
                IntentHypothesis(
                    intent=i,
                    probability=round(posterior[i], 4),
                    evidence={
                        s: round(
                            _get_channel_evidence(s, i, kin_lik,
                                                  eng_lik if eng else None,
                                                  aff_lik if aff else None,
                                                  script_prior),
                            3,
                        )
                        for s in sources
                    },
                )
                for i in INTENT_LABELS
            ]

            beliefs.append(AgentIntentBelief(
                entity_id=eid,
                hypotheses=sorted(hypotheses, key=lambda h: h.probability, reverse=True),
                most_likely=most_likely,
                confidence=round(confidence, 4),
                entropy=round(entropy, 4),
                evidence_sources=sources,
            ))

        beliefs.sort(key=lambda b: b.confidence, reverse=True)
        return beliefs


# =====================================================================
# Helpers
# =====================================================================
def _softmax_dict(log_probs: Dict[Intent, float]) -> Dict[Intent, float]:
    max_val = max(log_probs.values())
    exps = {k: math.exp(v - max_val) for k, v in log_probs.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def _entropy(dist: Dict[Intent, float]) -> float:
    h = 0.0
    for p in dist.values():
        if p > 0:
            h -= p * math.log(p)
    return h


def _get_channel_evidence(
    channel: str,
    intent: Intent,
    kin_lik: Dict[Intent, float],
    eng_lik: Optional[Dict[Intent, float]],
    aff_lik: Optional[Dict[Intent, float]],
    script_prior: Dict[Intent, float],
) -> float:
    """Get the raw likelihood value from a specific channel for an intent."""
    if channel == "kinematic":
        return kin_lik.get(intent, 0.05)
    if channel == "engagement" and eng_lik:
        return eng_lik.get(intent, 0.1)
    if channel == "affect" and aff_lik:
        return aff_lik.get(intent, 0.1)
    if channel == "script_prior":
        return script_prior.get(intent, 0.2)
    return 0.0
