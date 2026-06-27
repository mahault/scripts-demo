# Movement-Based Observation & Inference — Plan

## Problem

The retail demo's claim is *"observe movement → infer relationships → infer scripts/roles."*
The current implementation does none of that honestly:

- **The learner reads flags, not movement.** `TeacherObserver` reads the worker's
  published `customData` *state string* (`NAV`/`WORK`/`SOCIAL`, waypoint index, nav
  target `tx/ty`) and the `SegmentationEngine` classifies actions from that string. It
  is reading the teacher's internal state machine, then calling it "observation."
- **Relations are scripted handshakes.** "Shopper asks → worker escorts" is one agent
  writing `asking:"milk"` and the other reading it. No relation is inferred.
- **Manipulation is a Supervisor teleport** (`supervisor_grasp`) — the item pops into
  the hand. Looks bad.
- **Agents move on fixed waypoint loops** with collision-recovery hacks → bumping,
  backing up, nonsensical paths.

## Principle

The learner may only consume **observable kinematics** — signals a watcher could
extract from seeing the scene — never another agent's internal state. How the teacher
*decides* to move is its own business; the learner observes only its **motion**.

Observable (all Supervisor-readable today, currently discarded):
- Each agent's **position** over time → velocity, speed, heading.
- The worker's **arm-joint angle** (`rightArmAngle`, set by `HumanDriver` when
  reaching) → "reaching/manipulating" read from the *body*, not a label.
- **Inter-agent geometry** (distance, closing speed, facing) → raw material for
  inferring interaction / following / attention-seeking.
- Static **fixture landmarks** (shelves, counter, stock) — visible, so "nearest
  fixture" is fair game.

## Workstreams

### A. Movement-based inference  *(the actual claim — highest priority)*
A1. **Kinematic segmenter.** Replace state-string classification with a segmenter that
    derives behavior from `(speed, arm_angle, nearest_fixture)`:
    - `travel` — speed above threshold; destination = nearest fixture where it stops.
    - `reach` — stopped near a fixture **and** arm extended (sustained high arm angle).
    - mid-aisle stops (not near a fixture) are *not* script steps (this is where a
      give-way lives) — naturally excludes reactive social acts from the routine.
A2. **Honest observer.** `TeacherObserver` reads only position + heading + arm angle;
    **zero** reads of the worker's state string / `tx,ty` / `asking` / `escorting`.
A3. **Script from motion.** Feed movement-derived primitives (`obs_move_<zone>`,
    `obs_reach`) to the existing repertoire; it crystallises the routine. Zone-keying
    (stock / shelf / counter, by nearest fixture) preserves generalisation across
    shelves. Loop = observed return to the stock landmark.
A4. **Relation inferrer.** From relative motion over a window: *interaction/talking*
    (close + facing + low rel-velocity, sustained); *following* (lagged trajectory
    tracking); *approach/seeking-attention* (closing on a stationary agent). Emit
    relational events (used by the HUD bubbles and, later, the learner).

### B. Believable agent behavior
B1. **Steering** — replace fixed waypoints + teleport-recovery with seek-goal +
    separation so motion looks sensible and agents don't collide/back up.
    (Goals can stay; only the *locomotion* changes.)

### C. Manipulation
C1. **Smooth attach** — interpolate the item to the hand over ~0.5 s (and to the
    shelf on release) instead of an instant teleport. Cheap, kills the worst visual.

### D. Presentation
D1. **Sims-style speech bubbles** over heads at interaction moments, driven by the
    inferred relations from A4 (honest: they show *inferred* interaction, not a flag).

## Execution order

1. **A1–A3** — movement-based observer + segmenter; verify the learner crystallises
   the restock routine reading **zero** state-string fields. *(This session.)*
2. **C1** — smooth manipulation. *(This session if A lands.)*
3. **A4 + D1** — relation inference + speech bubbles.
4. **B1** — steering for believable motion.

## Acceptance for stage 1

- `TeacherObserver` makes no call that reads the worker's `state`/`tx`/`ty`/`asking`/
  `escorting`. Grep-clean.
- Headless run: learner observes loops, crystallises a routine, switches to execution,
  and performs a **shuttle** (stock ↔ shelf, reaching at each) — not parked.
- No (0,0) wedge; no dependence on the teacher's published intent.
