# Social-Layer Architecture

## Design Principle: Robot-Agnostic Social Reasoning

The social layer **completely separates social reasoning from physical reasoning**. The core architecture knows nothing about motors, joints, wheels, or any specific robot platform. It reasons exclusively in terms of abstract social primitives — `approach`, `avoid`, `yield`, `wait`, `neutral` — and generic parameter dictionaries.

This separation means the same social cognition stack (Theory of Mind, empathy, norms, scripts) works identically whether the robot underneath is a TIAGo in Webots, a Sphero on a desk, a TurtleBot in ROS2, or a drone.

> "Abstracting the social intents and actions from physical actions so that we can define these abstract ideas like approach, avoid, yield — these social primitives — effectively just separates completely the social reasoning from the physical reasoning."


---

## Abstraction Boundary

```
┌─────────────────────────────────────────────────────┐
│                   Social Layer                       │
│  (robot-agnostic — no hardware imports)              │
│                                                      │
│  ┌─────────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Perception  │  │ Cognition│  │    Safety      │  │
│  │  Pipeline +  │→ │ (ToM,    │→ │    Shield      │  │
│  │  Augments    │  │  Norms,  │  │               │  │
│  │             │  │  Scripts, │  └───────────────┘  │
│  │             │  │  Empathy) │                      │
│  └─────────────┘  └──────────┘                      │
│         ↑               ↓                            │
│    PerceptBundle    SkillUpdate                      │
│    (generic dict)   (intent + params dict)           │
│                                                      │
├──────────── Three Plugin Interfaces ─────────────────┤
│                                                      │
│   SensorInterface    Skill    IntentPolicy           │
│   read() → dict     start()  approach/avoid/         │
│                     tick()    yield/wait              │
│                     stop()                           │
│                                                      │
├──────────────────────────────────────────────────────┤
│                  Plugin (robot-specific)              │
│                                                      │
│  e.g. plugins/tiago_webots/                          │
│    ├── robot/driver.py      (motor control)          │
│    ├── perception/sensors.py (Webots sensor read)    │
│    ├── skills/nav_skill.py  (Skill implementation)   │
│    ├── skills/nav_policy.py (IntentPolicy impl)      │
│    └── plugin.py            (registration)           │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**The boundary is strict:** `src/architecture_core/` has zero imports from any plugin or robot-specific module. All robot knowledge lives in `src/plugins/`.

---

## Three Plugin Interfaces

### 1. `SensorInterface` (`perception/sensors/base_sensors.py`)

```python
class SensorInterface(ABC):
    @abstractmethod
    def read(self) -> Dict[str, Any]: ...
```

Returns robot-agnostic data: `robot_pose`, `agents`, `obstacles`, `timestamp`. The plugin translates hardware-specific sensor data (lidar, camera, Webots API) into this common format.

### 2. `Skill` (`skills/base.py`)

```python
class Skill(ABC):
    @abstractmethod
    def start(self, req: SkillRequest) -> None: ...
    @abstractmethod
    def tick(self, pb: PerceptBundle, update: SkillUpdate) -> Status: ...
    @abstractmethod
    def stop(self, reason: str = "") -> None: ...
```

Receives a `SkillUpdate` containing an abstract intent (e.g. `"yield"`) and generic params (e.g. `{"speed_scale": 0.5}`). The skill translates these into hardware commands. The social layer never specifies motor voltages or joint angles.

### 3. `IntentPolicy` (`cognition/tom/intent_policy.py`)

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

Compiles social intents into skill-specific parameters. For a navigation skill, `yield_()` might set `speed_scale=0.3` and `lateral_offset=1.0`. For a handover skill, `yield_()` might set `retract=True`. The social layer decides *which* intent; the IntentPolicy decides *how* to execute it on this robot.

---

## Data Flow

```
Sensors ──read()──→ PerceptBundle ──augmentations──→ Enriched PerceptBundle
                                                          │
                    ┌─────────────────────────────────────┘
                    ↓
              ┌─ Scripts ──→ SkillRequest (what to do)
              │
              ├─ Norms ──→ NormativeConstraints (what's allowed)
              │
              ├─ ToM ──→ SkillUpdate.intent (approach/avoid/yield/wait)
              │    └─ Active inference: ParticleFilter → GatedToM → SocialEFE
              │
              ├─ Empathy ──→ affect modulation (speed reduction, epistemic boost)
              │
              └─ IntentPolicy ──→ SkillUpdate.params (robot-specific translation)
                    │
                    ↓
              Safety Shield ──→ final SkillUpdate
                    │
                    ↓
              Skill.tick(pb, update) ──→ hardware commands
```

Everything above the IntentPolicy is robot-agnostic. Everything below is robot-specific.

---

## Social Cognition Stack (all robot-agnostic)

| Layer | Module | Purpose |
|-------|--------|---------|
| Perception | `perception/augmentations/proxemics.py` | Hall's zones (intimate/personal/social/public) |
| Perception | `perception/augmentations/affect.py` | Circumplex emotion estimation per agent |
| Perception | `perception/augmentations/engagement.py` | Gaze, body orientation, interaction duration |
| Perception | `perception/augmentations/saliency.py` | Attention priority scoring |
| Scripts | `cognition/scripts/script_manager.py` | Variational script execution + violation detection |
| Scripts | `cognition/scripts/weak_recognizer.py` | A-matrix situation type recognition |
| Norms | `cognition/norms/norm_engine.py` | Merge constraints (most restrictive wins) |
| Norms | `cognition/norms/rules.py` | Personal space, keepout zones, speed limits |
| Norms | `cognition/norms/affect_rules.py` | Affect-modulated speed, distress veto |
| ToM | `cognition/tom/intent_particle_filter.py` | Active inference particle filter over IntentProfile |
| ToM | `cognition/tom/social_efe.py` | Social Expected Free Energy (empathy-weighted) |
| ToM | `cognition/tom/gated_tom.py` | Entropy-based trust gating |
| ToM | `cognition/tom/tom_modulator.py` | Three-level hierarchy (GatedToM → IntentInference → kinematic) |
| ToM | `cognition/tom/intent_inference.py` | Log-linear fusion fallback |
| Empathy | `cognition/empathy/empathic_modulator.py` | Self-affect, contagion, EFE modulation |
| Planning | `cognition/planning/task_planner.py` | EFE-driven object selection for multi-robot tasks |
| Planning | `cognition/planning/clear_table_manager.py` | ToM-driven task orchestration (wraps ScriptManager) |
| Safety | `safety/shield.py` | Emergency stop, speed cap, keepout enforcement |
| Core | `core/executive.py` | Tick loop orchestration |

None of these modules import anything from `plugins/`.

---

## Script Repertoire Learning

Scripts are not hardcoded — they are **learned generative models** of behavioural sequences that strengthen through experience.

### Weak Scripts as Semantic Clusters

Following Albarracin, Constant, Friston & Ramstead (2021), weak scripts are **loosely-jointed semantic clusters** — unordered sets of concepts whose internal topology reshapes depending on context. This is fundamentally different from strong scripts:

- **Weak script**: `{approach-greet, yield-pass, wait-acknowledge}` — a bag of relevant social actions with no fixed order. In a corridor, yield-pass and wait-acknowledge tighten their connections. In a reception area, approach-greet dominates.
- **Strong script**: `[wait-acknowledge → yield-pass → approach-greet]` — a specific temporal ordering crystallized from repeated experience.

The transition from weak to strong is a **structural transformation**, not just precision increasing. An unordered semantic cluster becomes an ordered behavioral sequence as the B-matrix crystallizes through experience.

### Context-Conditioned Topology

The same cluster of concepts has different connection strengths in different contexts:

```
primitive_cluster: {approach-greet, yield-pass, wait-acknowledge}

corridor_encounter:                     open_area:
  wait → yield: 0.9 (strong)            approach → wait: 0.8 (strong)
  yield → approach: 0.3                 approach → yield: 0.6
  wait → approach: 0.2 (weak)           wait → yield: 0.2 (weak)
```

This is implemented via `ScriptPattern.context_topology`: a per-context dictionary of directed pairwise connection weights (the proto-B-matrix). When a pattern is converted to a `ScriptSequence` for execution, the topology for the current context determines the step ordering via greedy walk.

### Active Inference Framing

| Concept | Implementation |
|---------|---------------|
| **Weak script** | `ScriptPattern` with `primitive_cluster` (semantic set) + `context_topology` (proto-B-matrix) |
| **Strong script** | `ScriptPattern` with crystallized `primitives_sequence` from B-matrix |
| **A-matrix** | `WeakScriptRecognizer` — maps observations to situation type posterior |
| **B-matrix** | `TransitionEntry` counts (empirical) + `context_topology` (initial, context-conditioned) |
| **D-matrix** | `ScriptPattern.situation_affinity` — prior over which situations a pattern applies to |
| **Precision** | Accumulates on successful prediction (low free energy), decays on violations |
| **Free energy** | `ScriptParticleFilter.trajectory_free_energy` — accumulated -log(marginal likelihood) |

### Learning Cycle

```
1. OBSERVE situation → WeakScriptRecognizer → SituationBelief
       │
2. SELECT pattern (highest precision × affinity)
   or COMPOSE new cluster (EFE-scored primitives + context topology)
       │
3. DERIVE ordering from context_topology for current situation
       │
4. EXECUTE steps → TrajectoryTracker records each step
       │
5. INFER via ScriptParticleFilter
   - Cluster mode (weak): membership + topology-weighted likelihood
   - Sequence mode (strong): positional match likelihood
       │
6. UPDATE precision: low free energy → increase, high → decrease
   UPDATE B-matrix: observed transitions strengthen empirical counts
       │
7. CONSOLIDATE: precision > threshold → crystallize B-matrix path → promote to strong
```

### Composition (when no pattern matches)

The `ScriptComposer` assembles a semantic cluster from the `PrimitiveLibrary` using EFE scoring:

```
G_compose = w_eff * G_efficiency + w_emp * G_empathy + w_epi * G_epistemic
```

- **G_efficiency**: does this primitive advance toward the goal situation?
- **G_empathy**: does it improve or worsen human affect?
- **G_epistemic**: have we tried this primitive before? (prefer novel ones)

Composed patterns include both the `primitive_cluster` and a `context_topology` seeded from primitive pre/postcondition overlap. They start weak (precision=0.1) and can strengthen through repeated successful use.

### Crystallization (weak → strong)

When a pattern meets consolidation thresholds (precision >= 2.0, trajectory_count >= 5, mean_free_energy <= 3.0):

1. The most-probable path through the accumulated B-matrix is extracted
2. This path becomes the definitive `primitives_sequence`
3. The pattern is promoted to `is_strong = True`
4. The particle filter switches to sequence-mode evaluation
5. Strong scripts decay 10x slower than weak ones

---

## Adding a New Robot

To integrate a new robot platform:

1. **Create a plugin directory** (e.g. `src/plugins/turtlebot_ros/`)

2. **Implement `SensorInterface`** — translate your robot's sensors into the common dict format:
   ```python
   class TurtleBotSensors(SensorInterface):
       def read(self) -> dict:
           return {"robot_pose": ..., "agents": [...], "timestamp": ...}
   ```

3. **Implement `Skill`** — translate abstract SkillUpdates into motor commands:
   ```python
   class TurtleBotNavSkill(Skill):
       def tick(self, pb, update):
           speed = update.params.get("speed_scale", 1.0) * self.max_speed
           # ... send to motors
   ```

4. **Implement `IntentPolicy`** — define what approach/avoid/yield/wait mean for your robot:
   ```python
   class TurtleBotNavPolicy(IntentPolicy):
       def yield_(self, pb, req, base):
           return SkillUpdate(params={"speed_scale": 0.3, "lateral_offset": 0.5})
   ```

5. **Register** — wire into the SkillRegistry:
   ```python
   registry.register(SkillEntry(skill=nav_skill, policy=nav_policy))
   ```

No changes to the social layer are needed. The same ToM, empathy, norms, and scripts work unchanged.

---

## Existing Plugin: TIAGo Webots

```
plugins/tiago_webots/
├── robot/driver.py          # TiagoDriver: wheel velocities, motor control
├── perception/sensors.py    # TiagoSensors(SensorInterface): Webots API → dict
├── skills/nav_skill.py      # TiagoNavSkill(Skill): SkillUpdate → wheel commands
├── skills/nav_policy.py     # TiagoNavIntentPolicy(IntentPolicy): intent → speed/distance
├── plugin.py                # register(registry, driver) entry point
└── controller.py            # Webots controller main loop
```

The plugin imports *from* the social layer (`architecture_core.core.types`, `architecture_core.skills.base`, etc.) but the social layer never imports from the plugin.

---

## Multi-Robot Coordination

Coordination between robots is **emergent** — there is no explicit communication, task negotiation, or shared planning. Each robot runs its own independent social-layer stack and coordinates through perception + Theory of Mind.

### Unified EFE for Task and Navigation

The same `SocialEFE.compute_rollout()` drives both levels:

1. **Task selection**: For each candidate object, construct an `ObservationContext` as if approaching it. The EFE rollout evaluates social cost (is the other robot near this object?), collision risk, and affect. The object with lowest total EFE wins.

2. **Navigation**: During movement, the rollout continuously evaluates approach/yield/wait actions moment-to-moment. If paths cross, the empathic robot yields.

### Symmetry Breaking

Two identical robots with identical scoring would deadlock. Symmetry is broken by **empathy asymmetry**:

- **Low-empathy robot** (alpha ≈ 0): social cost term is small → selects nearest object, commits immediately
- **High-empathy robot** (alpha ≈ 6): social cost term is large → avoids objects near the other robot, enters observation phase when uncertain

This emerges naturally from the EFE: when the high-empathy robot's rollout produces "wait" at step 0 (due to high social cost + epistemic uncertainty), the `ClearTableManager` enters an observation phase. During this brief pause, the low-empathy robot commits, the high-empathy robot observes this via ToM, and selects the complement.

### Architecture

```
TaskPlanner (architecture_core)     — EFE-driven object selection
    ↓ selects next object
ClearTableManager                   — wraps ScriptManager, generates sequences
    ↓ nav→pick→nav→place
ScriptManager (existing)            — executes steps via set_sequence()
    ↓ SkillRequest
Executive → ToM → EFE → Shield     — existing social layer
    ↓ SkillUpdate
Navigate / PickPlace skills         — existing skills
```

---

## Key Design Decisions

1. **Intent as the abstraction unit.** Social reasoning produces one of five intents (`approach`, `avoid`, `yield`, `wait`, `neutral`). This is the universal vocabulary shared between the social layer and any robot.

2. **Params as generic dicts.** The `SkillUpdate.params` and `SkillRequest.goal` are `Dict[str, Any]`. This allows the IntentPolicy to inject arbitrary robot-specific parameters without the social layer needing to know what they are.

3. **One-way dependency.** Plugins depend on the social layer. The social layer depends on nothing robot-specific. This is enforced by directory structure and import discipline.

4. **Graceful degradation.** The ToM modulator has three levels (GatedToM → IntentInference → kinematic). The empathic modulator is optional (`None` = no behavior change). Augmentations are chainable. This means the social layer works at varying levels of sophistication without requiring all components.
