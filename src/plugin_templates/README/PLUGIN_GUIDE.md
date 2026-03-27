# Plugin guide (TODO)

This folder is an illustrative example of what the clients will implement

## What plugin authors implement
- A `Skill` (start/tick/stop)
- An `IntentPolicy` (approach/avoid/yield/wait) for that skill

## What plugins should NOT do
- Import cognition modules (scripts/norms/ToM)
- Implement their own outer control loop

## Registration
Expose a `register(registry)` function that registers SkillEntry objects.
