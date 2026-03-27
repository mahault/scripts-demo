# Phase Validation Results

Systematic validation of each social-layer phase in Webots apartment simulation.
Two TIAGo robots, 3m apart, head-on in apartment corridor.
- TIAGo_1: selfish (alpha=0.0, empathy=0.0)
- TIAGo_2: empathic (alpha=6.0, empathy=1.0)

## Test Matrix

| Test | PHASE_LEVEL | Phases Active | Prediction | Result | Pass? |
|------|-------------|---------------|------------|--------|-------|
| 1 | 1 | Core + Plugin | Both full speed, abrupt stop ~0.4m, deadlock | Deadlock at 0.54m, spd 0.50→0.30 | PASS |
| 2 | 3 | + Augmentations | Same as Test 1, augmentation data visible in logs | Identical to Test 1, no side effects | PASS |
| 3 | 4 | + Empathy | Slight slowdown if arousal builds, self_affect > 0 | Affect oscillates (0.02–0.24), violation detection works | PASS |
| 4 | 5 | + GatedToM | Empathic robot yields, selfish pushes through | | |
| 5 | 9 | + Safety | Smooth deceleration from ~3m, TTC in logs | | |
| 6 | 99 | Full stack | Early yield + smooth brake + empathy combined | | |

---

## Test 1: PHASE_LEVEL=1 (Baseline: Phases 1+2)

### Prediction
- Both robots drive at full speed (~0.5 m/s), neutral intent (no ToM)
- PersonalSpaceRule triggers speed_cap at 0.8m distance
- SafetyShield emergency stop at 0.4m
- **Expected behavior:** Both drive straight, abrupt stop at ~0.4m. Deadlock — neither yields.
- **Key metrics:**
  - min_agent_distance ≈ 0.4-0.5m
  - Both robots' intent = "neutral" throughout
  - speed_scale = 1.0 until very close, then 0.0

### Observations
- **PASS** — Baseline behavior confirmed
- Both robots drove at spd=0.50 (SpeedLimitRule cap) with intent=neutral throughout
- Agent distance decreased steadily: 2.89 → 2.22 → 1.28 → 0.80 → 0.61 → 0.54
- PersonalSpaceRule triggered at ad≈0.80: spd dropped from 0.50 → 0.30
- Deadlock at ad=0.54 from t=6.4s onward — neither yields
- Emergency stop (0.4m) never reached: robots physically blocked at 0.54m (body radii ~0.25m each)
- All phase-specific fields correctly show `-` (no augmentations, no ToM, no TTC)
- **Correction to prediction:** speed_scale goes 0.50 → 0.30 (not 1.0 → 0.0). SpeedLimitRule caps at 0.5, PersonalSpaceRule caps at 0.3

---

## Test 2: PHASE_LEVEL=3 (+ Perception Augmentations)

### Prediction
- Affect, engagement, saliency augmentations now active in perception pipeline
- These enrich PerceptBundle.social but NO rules consume them yet (no empathy, no affect rules)
- **Expected behavior:** Identical to Test 1 — same deadlock
- **Purpose:** Verify augmentations compute and log without errors or side effects
- **Key metrics:**
  - Logs show affect=(arousal, valence) values
  - Logs show engagement readings
  - Robot behavior unchanged from Test 1

### Observations
- **PASS** — Behavior identical to Test 1
- Same trajectory, same deadlock at ad=0.54 from t=6.4s, same spd 0.50→0.30 transition
- Augmentations (affect/engagement/saliency) running without errors or behavioral side effects
- Diagnostic shows affect=`-` because `bb.self_affect` is only set by EmpathicModulator (Phase 4)
- Augmentation data lives in `pb.social` (not surfaced in current diagnostics but confirmed no errors)

---

## Test 3: PHASE_LEVEL=4 (+ Empathic Modulator)

### Prediction
- EmpathicModulator predicts/observes/modulates robot self-affect
- AffectModulatedSpeedRule reduces speed when self-affect has negative valence
- DistressVetoRule vetoes if self-affect is extreme (arousal > 0.8, valence < -0.8)
- First encounter: both start with neutral affect (0, 0) → no affect-based change
- Near-miss at ~0.4m may cause arousal spike → modest speed reduction IF the robots oscillate
- **Expected behavior:** Very similar to Test 1 on first run. Arousal may build slightly.
- **Key metrics:**
  - self_affect arousal > 0 after close encounter
  - If arousal > 0.5: visible speed reduction
  - speed_scale slightly lower than Test 1 near encounter

### Observations
- **PASS** — Empathic modulator working, violations detected, affect oscillates
- Required wiring fix: added WeakScriptRecognizer with "corridor"/"blocked" situation types
- Required bug fix: violation coupling was applied every tick (not once per detection) → instant saturation
- After fix: violation fires at 5Hz, affect oscillates stably:
  - Arousal: 0.02 → 0.24 (spike) → 0.02 (decay), repeating ~1.5s cycle
  - Valence: -0.01 → -0.20 (trough) → -0.02 (decay), repeating
- AffectModulatedSpeedRule threshold (-0.3) not crossed (valence only reaches -0.20) — parameter tuning issue
- DistressVetoRule did NOT fire (arousal 0.24 well below 0.8 threshold) — correct
- Behavioral outcome: same deadlock at 0.54m as baseline, but affect state now active
- Added `inverse_min_distance` feature to WeakScriptRecognizer for blocked detection

---

## Test 4: PHASE_LEVEL=5 (+ GatedToM — KEY PHASE)

### Prediction
- GatedToM with IntentParticleFilter + SocialEFE now active
- TIAGo_1 (empathy=0.0): G_social ≈ G_self → selfish action selection → approach
- TIAGo_2 (empathy=1.0): G_social ≈ G_other → considers other's wellbeing → yield/wait
- First 3-5 seconds: both use cautious prior (reliability ~0.2) → yield/wait biased → slow
- After ~5s: particle filter observes kinematic intent → reliability rises
  - TIAGo_1 learns TIAGo_2 is approaching → selects approach (best for self)
  - TIAGo_2 learns TIAGo_1 is approaching → selects yield (best for other)
- **Expected behavior:** ASYMMETRIC — empathic robot yields, selfish robot passes through
- **Key metrics:**
  - TIAGo_1 intent transitions: neutral → approach (after reliability > 0.5)
  - TIAGo_2 intent transitions: neutral → yield or wait (after reliability > 0.5)
  - TIAGo_2 speed_scale < 0.3 when agent_dist < 2m
  - TIAGo_1 speed_scale ≈ 0.8-1.0 throughout
  - Reliability increases in logs for both robots

### Observations
_(to be filled after test run)_

---

## Test 5: PHASE_LEVEL=9 (+ Predictive Safety)

### Prediction
- VelocityPredictor estimates velocities from pose deltas, computes TTC for all pairs
- PredictiveCollisionRule applies graduated speed_cap = speed_reduction_factor(TTC)
- SafetyShield applies preemptive braking when TTC < 2.0s
- At 3m apart, closing at ~1.0 m/s (0.5 each) → TTC ≈ 3.0s → speed_factor ≈ 0.6
- At 2m apart → TTC ≈ 2.0s → speed_factor ≈ 0.4
- At 1m → TTC ≈ 1.0s → speed_factor ≈ 0.1
- **Expected behavior:** Both robots visibly decelerate starting at ~3m. Smooth curve, no jerky stop.
- **Key metrics:**
  - TTC transitions visible: inf → 6s → 3s → 1.5s → ...
  - speed_scale decreases gradually (not binary 1.0 → 0.0)
  - Smoother deceleration than Test 4

### Observations
_(to be filled after test run)_

---

## Test 6: PHASE_LEVEL=99 (Full Stack)

### Prediction
- All systems combine: GatedToM + predictive braking + empathic modulation
- TIAGo_2 yields early (GatedToM) with smooth deceleration (predictor)
- TIAGo_1 pushes through but also decelerates smoothly near TIAGo_2
- **Expected behavior:** Most natural-looking interaction — empathic yields, selfish passes, no collision, smooth motion
- **Key metrics:**
  - All metrics from Tests 4+5 combined
  - Minimum distance > 0.5m (safety maintained)
  - No abrupt stops

### Observations
_(to be filled after test run)_
