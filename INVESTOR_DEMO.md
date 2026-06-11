# Social Layer — Retail Script Learning Demo

A 2-minute Webots simulation demonstrating how a robot **observes a retail worker**, **discovers behavioral primitives by segmenting its motion**, **learns a script from scratch**, and **adaptively performs the same tasks** while respecting social norms.

---

## Quick Start

```bash
python launch_retail_demo.py
```

Requires **Webots R2025a**. The launcher now uses `--mode=realtime` so the simulation starts running immediately.

---

## What You Will See

### Scene Layout
- **Stock room** (back-left, blue marker) — where items are collected
- **Shelf A & B** (center, orange markers) — where items are placed
- **Service counter** (front-right, yellow marker) — where the worker waits
- **Entrance** (front-center, green marker) — where customers enter

### Robots
| Robot | Role | Color in HUD |
|-------|------|-------------|
| **Worker_T** | Teacher — performs a deterministic restock loop | Blue |
| **Learner_L** | Learner — observes, learns, then replicates | Red |
| **Customer_1** | Customer — patrols entrance→counter→shelf | Green |

---

## Narrative Arc (~2 minutes)

### Phase 1: Observation & Segmentation (0:00 – 0:45)
> *"This is the Worker. It follows a simple restock script: stock room → shelf A → shelf B → counter → repeat."*
>
> *"The Learner starts with zero knowledge of retail — no pre-programmed primitives, no seed script."*

- Watch **Worker_T** (blue) cycle through its waypoints
- **Learner_L** (red) stays still, watching via the Supervisor API
- The Learner segments the Worker's continuous motion into discrete behaviors:
  - **Navigation** — moving between waypoints
  - **Arm extension** — reaching for items
  - **Arm retraction** — returning to neutral pose
  - **Waiting** — pausing at the counter
- Each discovered behavior becomes a new primitive in the library
- HUD shows **PHASE 1: OBSERVATION**

### Phase 2: Pattern Discovery (0:45 – 1:15)
> *"After one complete loop, the Learner assembles the discovered primitives into a pattern."*
>
> *"After each subsequent loop, precision accumulates — the pattern gets stronger."*

- **Learner_L** continues observing
- HUD shows the discovered pattern name and `prec=` value increasing
- Primitive count grows as new behavior segments are observed

### Phase 3: Crystallization & Execution (1:15 – 2:00)
> *"Crystallization complete — the pattern is now strong enough to execute."*
>
> *"The Learner switches from observation to autonomous execution of the learned script."*

- HUD turns green: **PHASE 3: EXECUTING LEARNED SCRIPT**
- **Learner_L** now independently runs stock→shelf→counter
- The script was learned entirely from observation — the Learner never moved during training

### Social Adaptation (throughout)
> *"When the Customer approaches, the Learner's social layer detects it, infers intent via Theory of Mind, and yields."*
>
> *"No re-programming — the same learned script resumes after the customer passes."*

- Watch **Learner_L** slow down or wait when **Customer_1** is near
- Intent in HUD changes from `approach` to `yield` or `wait`

---

## Key Metrics to Watch

| HUD Field | Meaning |
|-----------|---------|
| **Intent** | Current social intent: approach / yield / wait / neutral |
| **Affect** | Robot's emotional state (arousal, valence) — changes when customers appear |
| **Speed** | Current speed scale — drops when yielding |
| **Pattern** | Name of the best-learned script pattern |
| **Precision** | Confidence in the learned pattern — must cross threshold to crystallize |
| **Rel** | Theory of Mind reliability — how well the robot understands the other agent |

---

## Technical Highlights

- **734 unit tests** passing
- **Zero robot-specific imports** in the social cognition core
- **Active inference** drives all decisions (Expected Free Energy minimization)
- **Automatic behavior segmentation** — primitives discovered from raw observation, not hand-coded
- **From-scratch learning** — no seed patterns or pre-defined retail scripts required
- **Compositional assembly** — behavior emerges from fragment composition, not hard-coded programs
- **Theory of Mind** — robot maintains beliefs about other agents' intents and empathy levels

---

## Implementation Notes

The live simulation now runs end-to-end in Webots R2025a. Key geometry fixes that made this possible:

- **Accurate furniture footprints** — `TiagoObjectSensors` reads Cabinet `depth`, `outerThickness`, and `columnsWidths` so shelf/counter keepout zones match the real collision geometry.
- **Safe orthogonal routes** — Worker_T stays in the south aisle, Customer_1 stays in the north aisle, and their counter queues are on opposite sides to avoid head-to-head deadlock.
- **No backward driving** — the low-level driver turns in place when the goal is behind, eliminating the instability that wedged robots against furniture.
- **Agent-only reactive avoidance** — reactive repulsion is reserved for other robots; static obstacles are handled by the detour planner using the corrected furniture sizes.

## Troubleshooting

**Webots not found?**
- Install Webots R2025a from https://cyberbotics.com
- Or edit `launch_retail_demo.py` to set the correct path

**Controllers not loading?**
- Ensure `src/` is in PYTHONPATH
- Check `logs/*.log` for error messages

**Robots not moving?**
- Check that `PHASE_LEVEL=9` is set (done automatically by launcher)
- Verify all 4 robots appear in the Scene Tree
