# Social Layer — A Robot-Agnostic Meta-Protocol for Social Robots

A **robot-agnostic social cognition layer** built on active inference. The architecture completely separates social reasoning from physical reasoning: the core reasons in abstract social primitives (`approach`, `avoid`, `yield`, `wait`, `neutral`) while robot-specific implementations live in plugins.

## Current Status

**734 tests passing** | Branch: `feat/social-layer-interface`

| Phase | Status |
|-------|--------|
| 1. Core interface spec + prototype | Done |
| 2. Webots TIAGo plugin + warehouse | Done |
| 3. Perception augmentations (affect, engagement, saliency) | Done |
| 4. Variational scripts + empathic modulator | Done |
| 5. Active inference ToM (particle filter, Social EFE, GatedToM) | Done |
| 6. Script repertoire learning (primitives, composition, consolidation) | Done |
| 7. Additional skills (handover, pick/place, gaze) | Done |
| 8. Scripts & norms enrichment (BT engine, adaptive norms, keepout, context norms) | Done |
| 9. Safety (velocity prediction, multi-agent collision) | Done |
| 10. Table-clearing coordination via emergent ToM | Done |
| 10b. Embodied affordance reasoning via active inference | Done |
| Webots phase-by-phase validation | In Progress |

See [docs/roadmap.md](docs/roadmap.md) for detailed phase descriptions.
See [docs/phase_validation.md](docs/phase_validation.md) for Webots simulation validation results.

---

## Compositional Assembly Demos

The `demo_variations_anim.py` script runs 10 experiment variants through the real composition pipeline, producing animated GIFs that visualize how **environment structure shapes robot cognition**.

### Set A: Environment Changes (same robot, different space)

| Variant | Environment | Key finding |
|---------|-------------|-------------|
| **A1** | Reception desk (baseline) | Full queue norm emerges from stanchions/desk/signs |
| **A2** | Corridor (one-sided) | Same fragments, different ordering — courtesy_space dominates, queue recedes |
| **A3** | Degraded reception (stanchions removed) | Robot drops from 9 to 6 primitives — queuing and waiting norms vanish with the stanchions |
| **A4** | Enriched reception (sign + queue) | Sign adds engage-staff but cannot override queue norms (additive composition) |
| **A5** | Corridor (both sides) | Robot zigzags between pedestrians via proximity-weighted repulsion |
| **A6** | Sign + no queue cues | Sign displaces queue norms entirely — robot goes directly to counter (7 vs 9 primitives) |

### Set B: System Changes (same environment, different fragment repertoire)

| Variant | Fragments | Key finding |
|---------|-----------|-------------|
| **B5** | All 5 (baseline) | Full queue behavior in reception |
| **B6** | Missing wait_patiently | System lacks wait knowledge — skips waiting phase |
| **B7** | Extra direct_approach | Conflicting knowledge — adds engage-staff on top of queue |
| **B8** | Minimal (2 fragments) | Sparse knowledge still composes a viable 4-primitive sequence |

### Key theoretical claims validated

- **Lefebvre**: Change the space, change the cognition (A1 vs A2)
- **Akrich**: The norm was in the environment, not the robot (A3)
- **Latour**: Material cues reshape the affordance landscape (A4, A6)
- **Compositional robustness**: The pipeline degrades gracefully with sparse fragments (B8)
- **Additive vs substitutive**: Adding cues is additive; displacing cues is substitutive (A4 vs A6)

### Running the demos

```bash
cd social-layer-feat-social-layer-interface
python demo_variations_anim.py    # 10 animated GIFs in demo_figures/
python demo_variations.py         # Static 2x2 comparison PNGs
```

Each animation shows four panels: top-down scene with gaze, situation state machine with G labels, fragment composition weights, and execution sequence with progress tracking.

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Social Layer                        │
│  (robot-agnostic — zero hardware imports)             │
│                                                       │
│  Perception ──→ Cognition ──→ Safety                  │
│  (augments)     (ToM, norms,    (shield)              │
│                  scripts,                             │
│                  empathy)                             │
│         ↑            ↓                                │
│    PerceptBundle  SkillUpdate                         │
│                                                       │
├──────── Three Plugin Interfaces ─────────────────────┤
│   SensorInterface    Skill    IntentPolicy            │
├──────────────────────────────────────────────────────┤
│                Plugin (robot-specific)                │
│  e.g. plugins/tiago_webots/                           │
└──────────────────────────────────────────────────────┘
```

See [docs/architecture.md](docs/architecture.md) for the full architecture documentation.

### Key Design Principles

1. **Intent as abstraction.** Social reasoning produces one of five intents. This is the universal vocabulary between the social layer and any robot.

2. **One-way dependency.** Plugins depend on the social layer. The social layer depends on nothing robot-specific.

3. **Graceful degradation.** ToM has three levels (GatedToM → IntentInference → kinematic). Empathy is optional. Script learning is optional. The system works at varying levels of sophistication.

---

## Social Cognition Stack

### Perception Augmentations
- **Proxemics** — Hall's zones (intimate/personal/social/public)
- **Affect** — Circumplex emotion estimation per agent (Pattisapu & Albarracin 2024)
- **Engagement** — Gaze, body orientation, interaction duration
- **Saliency** — Attention priority scoring

### Theory of Mind (Active Inference)
- **IntentParticleFilter** — Particle filter over behavioural profiles (approach_bias, responsiveness, precision, empathy_j)
- **SocialEFE** — Empathy-weighted Expected Free Energy: `G = (1-λ)*G_self + λ*G_other + G_epistemic`
- **GatedToM** — Entropy-based trust gating: `q_gated = reliability * q_learned + (1-reliability) * q_prior`

### Empathic Modulator
- Robot self-affect engine with predict-observe-update loop
- Empathic contagion from observed human affect
- Script violation → arousal coupling
- Affect-modulated norm rules (speed reduction, distress veto)

### Variational Scripts
- **WeakScriptRecognizer** — A-matrix event type classification (Albarracin et al. 2021)
- Script violation detection via KL divergence + repair injection
- Enriched steps with deontic mode, expected affect, gate type

### Script Repertoire Learning
- **Semantic clusters** — Weak scripts as loosely-jointed concept sets with context-conditioned topology (Albarracin et al. 2021)
- **Context topology** — Same cluster, different context → different connection strengths (proto-B-matrix)
- **Primitive Library** — 13 atomic social action units (navigation, handover, pick/place, gaze)
- **ScriptParticleFilter** — Dual-mode: cluster-membership (weak) vs positional (strong) inference
- **ScriptComposer** — EFE-based composition producing clusters with context topology
- **Compositional Assembly** — Graph-based fragment retrieval with diffusion; 3-tier selection (strong match → compositional → scratch); causal ordering via backbone extraction
- **Crystallization** — B-matrix path extraction transforms unordered cluster → ordered sequence
- **Consolidation** — Reliable patterns promoted to strong scripts via precision accumulation

### Reference Skills
- **HandoverSkill** — Two-phase handover (extend/receive) with state machine and hook methods for plugin subclassing
- **PickPlaceSkill** — Pick/place with approach→grasp/release→verify state machine
- **GazeSkill** — Four gaze modes (look_at_agent, look_at_point, scan, avert) with stabilization tracking
- Each skill has an **IntentPolicy** mapping approach/avoid/yield/wait to skill-specific parameters

### Behavior Trees
- **Full BT engine** — ActionNode, ConditionNode, SequenceNode, SelectorNode, ParallelNode, InverterNode, RepeatNode, SucceederNode
- **BehaviorTreeManager** — implements ScriptManager interface; `tree_from_sequence()` bridges existing linear scripts to BT
- Conditional branching via ConditionNode + ScriptCondition predicates

### Norms & Safety
- Personal space, keepout zones, speed limits
- Affect-modulated speed reduction, distress veto
- **Adaptive norm discovery** — norms co-evolve with scripts via precision-gated EMA; observable features (inter-human distance, approach speed, gaze engagement, affect, interaction duration) are extracted from perception and accumulated into ScriptPattern norm_features
- **ProxemicPrior** — initial distance thresholds (replaces static cultural profiles); rules adapt from observed behavior via `set_norm_features()`
- **Dynamic keepout zones** — from perceived obstacles/hazards with configurable margins
- **Task-context-dependent norms** — situation-activated rule sets via ContextualNormRule
- **Predictive collision** — TTC-based velocity prediction for all entity pairs (robot-agent, robot-obstacle, agent-agent); graduated speed reduction via `speed_reduction_factor(ttc)`
- **Velocity-aware emergency stop** — SafetyShield with optional VelocityPredictor for TTC-based preemptive braking
- Emergency stop, speed cap enforcement

### Embodied Affordance Reasoning (Phase 10b)
- **EmbodimentModel** — Robot self-model (A-matrix): arm_reach, body_radius, grip_strength, max_push_mass with precision tracking
- **FurnitureItem** — Environment model: furniture with mass, movable flag, keepout bounding box
- **Affordance EFE** — `G_total = G_pragmatic + G_epistemic + G_social` for joint (object, approach_pos, clearing_action) selection
- **Approach candidates** — 8 positions at arm_reach distance around each object, filtered by keepout collision
- **Clearing actions** — Push forward (drive into obstacle slowly), grab-and-pull (grasp edge, reverse), navigate around (choose different candidate)
- **Epistemic drive** — Low self-model precision drives exploration of novel actions; precision accumulates on success, decays on failure

### Multi-Robot Coordination (Phase 10)
- **Emergent coordination** — No explicit communication; robots coordinate through perception + Theory of Mind
- **Unified EFE for task AND navigation** — Same `SocialEFE.compute_rollout()` drives both object selection and moment-to-moment navigation
- **Symmetry breaking** — Low-empathy robot commits fast (small social cost), high-empathy robot defers and observes (large social cost from empathy-weighted EFE)
- **TaskPlanner** — EFE-driven object selection; scores each candidate by rollout value + path cost
- **ClearTableManager** — Orchestrates TaskPlanner + GatedToM + ScriptManager for continuous clearing

### Webots Simulation Validation
- **Apartment world** — Two TIAGo robots clearing dining + coffee tables (7 objects)
- **Retail world** — `tiago_retail_demo.wbt`: Worker_T performs a real pick→transport→place restock loop with Can/Orange/Apple items, Customer_1 picks up and carries a shopping basket, Learner_L observes and learns the script; live navigation and manipulation verified in R2025a
- **Phase-gated controller** — `PHASE_LEVEL` env var enables phases incrementally (1→3→4→5→9→10→99)
- **Validated phases**: Phases 1+2 (baseline deadlock), Phase 3 (augmentations, no side effects), Phase 4 (empathic modulator, violation-driven affect oscillation)
- See [docs/phase_validation.md](docs/phase_validation.md) for predictions vs observations

---

## Three Plugin Interfaces

### SensorInterface
```python
class SensorInterface(ABC):
    @abstractmethod
    def read(self) -> Dict[str, Any]: ...
```
Returns robot-agnostic data: `robot_pose`, `agents`, `obstacles`, `timestamp`.

### Skill
```python
class Skill(ABC):
    @abstractmethod
    def start(self, req: SkillRequest) -> None: ...
    @abstractmethod
    def tick(self, pb: PerceptBundle, update: SkillUpdate) -> Status: ...
    @abstractmethod
    def stop(self, reason: str = "") -> None: ...
```
Receives abstract intents and generic params. Translates to hardware commands.

### IntentPolicy
```python
class IntentPolicy(ABC):
    @abstractmethod
    def approach(self, pb, req, base) -> SkillUpdate: ...
    @abstractmethod
    def avoid(self, pb, req, base) -> SkillUpdate: ...
    @abstractmethod
    def yield_(self, pb, req, base) -> SkillUpdate: ...
    @abstractmethod
    def wait(self, pb, req, base) -> SkillUpdate: ...
```
Compiles social intents into skill-specific parameters.

---

## Adding a New Robot

1. Create a plugin directory (e.g. `src/plugins/your_robot/`)
2. Implement `SensorInterface` — translate sensors to common dict format
3. Implement `Skill` — translate SkillUpdates to motor commands
4. Implement `IntentPolicy` — define what approach/avoid/yield/wait mean
5. Register with `SkillRegistry`

No changes to the social layer needed.

---

## Motivational Biases, Not Motor Commands

The core design insight: `approach`, `avoid`, `yield`, `wait` are **motivational biases**, not navigation primitives.

- `approach` during navigation = "reduce hesitation, close distance"
- `approach` during grasping = "commit to reach, increase execution gain"
- `yield` during navigation = "give right-of-way, slow down"
- `yield` during manipulation = "retract, de-escalate"

The same ToM intent is interpreted differently by each skill's IntentPolicy.

---

## Theoretical Foundations

- **Active inference**: Friston et al. (2017) — Expected Free Energy for action selection
- **Variational scripts**: Albarracin, Constant, Friston & Ramstead (2021) — A-matrix event recognition, script violation as prediction error
- **Circumplex affect**: Pattisapu, Verbelen, Pitliya, Kiefer & Albarracin (2024) — Free energy in a circumplex model of emotion
- **Social EFE**: Empathy-weighted expected free energy adapted from empathy-prisoner-dilemma

---

## Project Structure

```
src/
├── architecture_core/          # Robot-agnostic social layer
│   ├── core/                   # Executive, blackboard, types, registry
│   ├── cognition/
│   │   ├── tom/                # ToM modulator, particle filter, Social EFE, GatedToM
│   │   ├── scripts/            # Script manager, recognizer, repertoire, behavior trees
│   │   ├── norms/              # Norm engine, rules, affect rules, adaptive norms, keepout
│   │   ├── empathy/            # Empathic modulator
│   │   └── planning/           # TaskPlanner, ClearTableManager (multi-robot coordination)
│   ├── perception/             # Pipeline, sensors, augmentations
│   ├── safety/                 # Shield, geometry, constraints
│   └── skills/                 # Skill base + reference impls (handover, pick/place, gaze)
└── plugins/
    └── tiago_webots/           # TIAGo Webots plugin (nav, handover, pick/place, gaze)
tests/                          # 734 tests
demo_variations.py              # Static 2x2 comparison figures
demo_variations_anim.py         # 10 animated variant GIFs
demo_figures/                   # Generated PNGs and GIFs
docs/
├── architecture.md             # Detailed architecture documentation
└── roadmap.md                  # Phase-by-phase implementation roadmap
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

TBD

## Citation

TBD
