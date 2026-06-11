"""HUD overlay controller for the retail investor demo.

Reads robot cognitive states from their customData fields and renders
on-screen labels using Supervisor.setLabel().

Screen layout (normalized 0-1 coordinates):
  Top banner    : demo phase + title
  Bottom panel  : robot status cards
"""

import json
import math
from controller import Supervisor


ROBOTS = ["Worker_T", "Learner_L", "Customer_1"]
COLORS = {
    "Worker_T": "0x3498db",    # blue
    "Learner_L": "0xe74c3c",   # red
    "Customer_1": "0x2ecc71",  # green
}
LABELS = {
    "Worker_T": "WORKER",
    "Learner_L": "LEARNER",
    "Customer_1": "CUSTOMER",
}


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    # Cache node references
    nodes = {}
    for name in ROBOTS:
        node = robot.getFromDef(name)
        if node:
            nodes[name] = {
                "node": node,
                "field": node.getField("customData"),
            }

    # Label IDs — keep stable so we overwrite rather than create new
    label_id = 0

    def draw_label(text, x, y, size=0.04, color="0xffffff", bold=False):
        nonlocal label_id
        font = "Arial" + (" Black" if bold else "")
        robot.setLabel(label_id, text, x, y, size, color, 0.8, font)
        label_id += 1

    tick = 0
    while robot.step(timestep) != -1:
        label_id = 0  # reset and reuse IDs each frame

        # --- Top banner ---
        draw_label("SOCIAL LAYER  —  Retail Script Learning Demo", 0.02, 0.02,
                   size=0.055, color="0xffffff", bold=True)

        # --- Read robot states ---
        states = {}
        for name, info in nodes.items():
            try:
                raw = info["field"].getSFString()
                if raw:
                    states[name] = json.loads(raw)
                else:
                    states[name] = {}
            except Exception:
                states[name] = {}

        # --- Infer demo phase from Learner state ---
        learner_state = states.get("Learner_L", {})
        phase = "PHASE 1: OBSERVATION"
        phase_color = "0xf1c40f"
        prec = learner_state.get("precision", 0.0)
        is_strong = learner_state.get("is_strong", False)
        patterns = learner_state.get("patterns", 0)

        if patterns == 0:
            phase = "PHASE 1: OBSERVATION"
            phase_color = "0xf1c40f"  # yellow
        elif not is_strong:
            phase = f"PHASE 2: LEARNING  (precision={prec:.2f})"
            phase_color = "0xe67e22"  # orange
        else:
            phase = "PHASE 3: EXECUTING LEARNED SCRIPT"
            phase_color = "0x2ecc71"  # green

        draw_label(phase, 0.02, 0.09, size=0.05, color=phase_color, bold=True)

        # --- Bottom robot cards ---
        card_y = 0.88
        card_w = 0.32
        for i, name in enumerate(ROBOTS):
            x = 0.02 + i * card_w
            color = COLORS[name]
            st = states.get(name, {})

            # Card header
            draw_label(f"[{LABELS[name]}]", x, card_y, size=0.045, color=color, bold=True)

            # Card body lines
            lines = []
            if name == "Worker_T":
                wp = st.get("waypoint", "-")
                loop = st.get("loop", 0)
                state = st.get("state", "-")
                lines = [
                    f"Waypoint: {wp}",
                    f"Loops: {loop}",
                    f"State: {state}",
                ]
            elif name == "Learner_L":
                intent = st.get("intent", "-")
                aff = st.get("affect", "-,-")
                spd = st.get("speed", "-")
                pat = st.get("pattern", "none")
                lines = [
                    f"Intent: {intent}",
                    f"Affect: ({aff})",
                    f"Speed: {spd}",
                    f"Pattern: {pat}",
                ]
            elif name == "Customer_1":
                wp = st.get("waypoint", "-")
                state = st.get("state", "-")
                lines = [
                    f"Waypoint: {wp}",
                    f"State: {state}",
                ]

            for j, line in enumerate(lines):
                draw_label(line, x, card_y + 0.055 + j * 0.04, size=0.035, color="0xeeeeee")

        tick += 1


if __name__ == "__main__":
    main()
