"""HUD overlay controller for the retail investor demo.

Reads robot cognitive states from their customData fields and renders
on-screen labels using Supervisor.setLabel().  Also drives a simple cinematic
camera that follows the most interesting actor or cycles through overview shots.

Screen layout (normalized 0-1 coordinates):
  Top banner    : demo phase + title
  Top right     : live camera mode
  Bottom panel  : robot status cards
"""

import json
import math
import os
from controller import Supervisor


_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.pardir, os.pardir, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)


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

# Static overview camera.  Kept simple while the demo mechanics are being
# tuned: a fixed top-down shot that shows the whole store.
OVERVIEW_CAMERA = {
    "name": "OVERVIEW",
    "position": (-4.5, -9.0, 8.5),
    "orientation": (-0.55, 0.45, 0.70, 2.1),
}


def _trunc(text, length=14):
    """Shorten long object ids so they fit on the HUD cards."""
    if text is None:
        return "-"
    text = str(text)
    if len(text) <= length:
        return text
    return text[:length - 1] + "…"


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    _log_path = os.path.join(_LOG_DIR, "HUD_Robot.log")
    _log_fh = open(_log_path, "w")
    import builtins
    _orig_print = builtins.print
    def _print(*args, **kwargs):
        _orig_print(*args, **kwargs)
        _orig_print(*args, **kwargs, file=_log_fh, flush=True)
    builtins.print = _print

    # Cache robot node references
    nodes = {}
    for name in ROBOTS:
        node = robot.getFromDef(name)
        if node:
            nodes[name] = {
                "node": node,
                "field": node.getField("customData"),
            }

    # Cache Viewpoint fields for cinematic switching
    vp = robot.getFromDef("VIEWPOINT")
    vp_follow = vp.getField("follow") if vp else None
    vp_position = vp.getField("position") if vp else None
    vp_orientation = vp.getField("orientation") if vp else None
    def set_overview():
        if vp is None:
            print("CAMERA: no VIEWPOINT node")
            return
        try:
            if vp_follow is not None:
                vp_follow.setSFString("")
            pos = OVERVIEW_CAMERA["position"]
            if vp_position is not None:
                vp_position.setSFVec3f(list(pos))
            ori = OVERVIEW_CAMERA["orientation"]
            if vp_orientation is not None:
                vp_orientation.setSFRotation(list(ori))
            print("CAMERA: locked to static overview")
        except Exception as e:
            print(f"CAMERA: error setting overview: {e}")

    set_overview()

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

        # --- Top banner ---
        draw_label("SOCIAL LAYER  —  Retail Script Learning Demo", 0.02, 0.02,
                   size=0.055, color="0xffffff", bold=True)

        # --- Camera indicator ---
        draw_label("CAM: OVERVIEW", 0.78, 0.02, size=0.04,
                   color="0xf1c40f", bold=True)

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
        card_y = 0.84
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
                action = st.get("action", "-")
                dest = st.get("destination", "-")
                held = _trunc(st.get("held_id"))
                restocked = st.get("restocked", 0)
                loop = st.get("loop", 0)
                lines = [
                    f"Action: {action} -> {dest}",
                    f"Holding: {held}",
                    f"Restocked: {restocked}  (loop {loop})",
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
                action = st.get("action", "-")
                held = _trunc(st.get("holding_item"))
                basket = "yes" if st.get("holding_basket") else "no"
                collected = st.get("collected", 0)
                basket_count = st.get("basket_count", 0)
                lines = [
                    f"Action: {action}",
                    f"Item: {held}  Basket: {basket}",
                    f"Collected: {collected}  In basket: {basket_count}",
                ]

            for j, line in enumerate(lines):
                draw_label(line, x, card_y + 0.055 + j * 0.04, size=0.035, color="0xeeeeee")

        tick += 1


if __name__ == "__main__":
    main()
