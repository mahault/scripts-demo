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
    # Known-good framing of the store (oblique overhead from the SW).
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

    # ------------------------------------------------------------------
    # World -> screen projection for floating speech bubbles (Sims-style).
    # The overview camera is fixed, so we build its view basis once and project
    # each actor's head position to normalized screen coords for setLabel.
    # ------------------------------------------------------------------
    def _rot_from_axis_angle(ax, ay, az, th):
        n = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
        x, y, z = ax / n, ay / n, az / n
        c, s = math.cos(th), math.sin(th)
        C = 1.0 - c
        return [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]

    _cam = OVERVIEW_CAMERA["position"]
    _o = OVERVIEW_CAMERA["orientation"]
    _R = _rot_from_axis_angle(_o[0], _o[1], _o[2], _o[3])
    _right = (_R[0][0], _R[1][0], _R[2][0])
    _up = (_R[0][1], _R[1][1], _R[2][1])
    _fwd = (-_R[0][2], -_R[1][2], -_R[2][2])   # camera looks along local -z
    _FOVX = 0.785398                            # default Webots viewpoint FOV
    _ASPECT = 16.0 / 9.0
    _FOVY = 2.0 * math.atan(math.tan(_FOVX / 2.0) / _ASPECT)
    _TX, _TY = math.tan(_FOVX / 2.0), math.tan(_FOVY / 2.0)

    def project(wx, wy, wz):
        """World point -> (screen_x, screen_y) in 0..1, or None if behind/off."""
        vx, vy, vz = wx - _cam[0], wy - _cam[1], wz - _cam[2]
        depth = vx * _fwd[0] + vy * _fwd[1] + vz * _fwd[2]
        if depth <= 0.05:
            return None
        px = vx * _right[0] + vy * _right[1] + vz * _right[2]
        py = vx * _up[0] + vy * _up[1] + vz * _up[2]
        sx = 0.5 + 0.5 * (px / depth) / _TX
        sy = 0.5 - 0.5 * (py / depth) / _TY
        if -0.1 <= sx <= 1.1 and -0.1 <= sy <= 1.1:
            return (sx, sy)
        return None

    def head_screen(name):
        """Screen coords just above an actor's head, or None if off-camera."""
        info = nodes.get(name)
        if not info:
            return None
        try:
            p = info["node"].getPosition()
            return project(p[0], p[1], p[2] + 0.75)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Optional movie + interactive-animation capture (RECORD_DEMO=1).
    # Records the actual 3D scene — walking humans + TIAGo learner — for one
    # full learn->perform arc, then stops and quits Webots.  Outputs land in
    # dashboard/recorded_run/ for embedding in the dashboard.
    # ------------------------------------------------------------------
    recording = os.environ.get("RECORD_DEMO") == "1"
    rec_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir, os.pardir, "dashboard", "recorded_run"))
    movie_path = os.path.join(rec_dir, "demo.mp4")
    anim_path = os.path.join(rec_dir, "demo.html")
    REC_ACCEL = int(os.environ.get("RECORD_ACCEL", "20"))   # movie speed-up
    REC_PERFORM_S = float(os.environ.get("RECORD_PERFORM_S", "45"))  # film this
    REC_MAX_S = float(os.environ.get("RECORD_MAX_S", "600"))         # hard cap
    rec_stage = "off"
    perform_s = 0.0
    if recording:
        os.makedirs(rec_dir, exist_ok=True)
        try:
            # Render as fast as possible while still drawing frames for capture.
            robot.simulationSetMode(robot.SIMULATION_MODE_RUN)
        except Exception as e:
            print(f"REC: could not set run mode: {e}")
        try:
            robot.animationStartRecording(anim_path)
            print(f"REC: animation recording -> {anim_path}")
        except Exception as e:
            print(f"REC: animation start failed: {e}")
        try:
            robot.movieStartRecording(
                movie_path, width=1280, height=720, codec=0,
                quality=80, acceleration=REC_ACCEL, caption=False)
            print(f"REC: movie recording -> {movie_path} (accel={REC_ACCEL})")
            rec_stage = "recording"
        except Exception as e:
            print(f"REC: movie start failed: {e}")

    # Telemetry for the synced web dashboard: sample once per sim-second while
    # recording so the charts can animate in lockstep with the movie.  The movie
    # maps linearly (sim_t = sim_start + video_t * acceleration), so the web side
    # just needs these samples plus the acceleration and the sim span.
    telemetry = []
    tele_last = -999.0
    rec_sim_start = None
    tele_path = os.path.join(rec_dir, "telemetry.json")
    _gv_prev = _ho_prev = _wf_prev = False
    gv_count = ho_count = wf_count = 0

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

        # --- Two-phase banner, inferred from the learner ---
        learner_state = states.get("Learner_L", {})
        worker_state = states.get("Worker_T", {})
        cust_state = states.get("Customer_1", {})
        prec = learner_state.get("precision", 0.0)
        is_strong = learner_state.get("is_strong", False)

        if is_strong:
            phase = "PHASE 2:  LEARNER NOW PERFORMING THE ROUTINE IT LEARNED"
            phase_color = "0x2ecc71"  # green
        else:
            phase = f"PHASE 1:  LEARNER WATCHING  —  learned {int(prec * 100)}%"
            phase_color = "0xf1c40f"  # yellow
        draw_label(phase, 0.02, 0.09, size=0.05, color=phase_color, bold=True)

        # --- Telemetry sampling for the synced dashboard ---
        if recording and rec_stage == "recording":
            now_t = robot.getTime()
            if rec_sim_start is None:
                rec_sim_start = now_t
            # Rising-edge counters for the social-act tallies.
            w_state = str(worker_state.get("state", ""))
            gv = w_state.startswith("SOCIAL")
            if gv and not _gv_prev:
                gv_count += 1
            _gv_prev = gv
            ho = worker_state.get("action", "") == "handover"
            if ho and not _ho_prev:
                ho_count += 1
            _ho_prev = ho
            wf = bool(worker_state.get("escorting"))
            if wf and not _wf_prev:
                wf_count += 1
            _wf_prev = wf
            if now_t - tele_last >= 1.0:
                tele_last = now_t
                performing = (is_strong
                              and learner_state.get("intent", "OBSERVE") != "OBSERVE")
                telemetry.append({
                    "t": round(now_t, 2),
                    "precision": round(float(prec or 0.0), 3),
                    "strong": bool(is_strong),
                    "phase": "perform" if performing else "watch",
                    "intent": learner_state.get("intent", "OBSERVE"),
                    "restocked": int(worker_state.get("restocked", 0) or 0),
                    "loops": int(worker_state.get("loop", 0) or 0),
                    "collected": int(cust_state.get("collected", 0) or 0),
                    "basket": int(cust_state.get("basket_count", 0) or 0),
                    "giveway": gv_count,
                    "handover": ho_count,
                    "wayfinding": wf_count,
                })

        # --- Plain-language narration of the current beat ---
        w_act = worker_state.get("action", "")
        c_act = cust_state.get("action", "")
        c_state = cust_state.get("state", "")
        asking = cust_state.get("asking", "") or worker_state.get("escort_product", "")
        narration, ncol = "", "0xffffff"
        if w_act == "escorting":
            narration = f"WORKER: \"Follow me!\"  leading the shopper to the {asking or 'shelf'}"
            ncol = "0x9b59b6"
        elif w_act == "handover":
            narration = "WORKER hands the item to the shopper at the counter"
            ncol = "0xe74c3c"
        elif c_state.startswith("FOLLOW"):
            narration = "CUSTOMER follows the worker to the shelf"
            ncol = "0x9b59b6"
        elif c_state.startswith("ASK"):
            narration = f"CUSTOMER: \"Excuse me, where is the {asking or 'item'}?\""
            ncol = "0x2ecc71"
        elif str(worker_state.get("state", "")).startswith("SOCIAL:yield"):
            narration = "WORKER steps aside to let the shopper pass"
            ncol = "0x3498db"
        elif w_act in ("pick", "place", "transit"):
            narration = f"WORKER restocking  ({worker_state.get('destination', '')})"
            ncol = "0x3498db"
        if narration:
            draw_label(narration, 0.02, 0.15, size=0.045, color=ncol, bold=True)

        # --- Floating speech bubbles over actors' heads (Sims-style) ---
        def bubble(name, text, color):
            scr = head_screen(name)
            if scr is None:
                return
            bx = min(0.80, max(0.01, scr[0] - 0.05))
            by = min(0.90, max(0.03, scr[1] - 0.05))
            draw_label(text, bx, by, size=0.05, color=color, bold=True)

        asking = cust_state.get("asking", "")
        if asking:
            bubble("Customer_1", '"Where\'s the %s?"' % asking, "0x2ecc71")
        elif worker_state.get("escorting"):
            bubble("Worker_T", '"Follow me!"', "0x9b59b6")
        if worker_state.get("action") == "handover":
            bubble("Worker_T", '"Here you go!"', "0xe74c3c")

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

        # --- Recording lifecycle: film through the learn->perform arc ---
        if rec_stage == "recording":
            now = robot.getTime()
            # "Performing" = crystallized AND actually executing the learned
            # script (intent leaves OBSERVE once the learner switches phases).
            performing = is_strong and learner_state.get("intent", "OBSERVE") != "OBSERVE"
            if performing:
                perform_s += timestep / 1000.0
            done = perform_s >= REC_PERFORM_S or now >= REC_MAX_S
            if int(now) % 30 == 0:
                print(f"REC: t={now:.0f}s strong={is_strong} "
                      f"performing={performing} "
                      f"performed={perform_s:.0f}/{REC_PERFORM_S:.0f}s")
            if done:
                print(f"REC: stopping at t={now:.0f}s (performed {perform_s:.0f}s)")
                try:
                    robot.movieStopRecording()
                except Exception as e:
                    print(f"REC: movie stop error: {e}")
                try:
                    robot.animationStopRecording()
                    print("REC: animation saved")
                except Exception as e:
                    print(f"REC: animation stop error: {e}")
                try:
                    with open(tele_path, "w") as tf:
                        json.dump({
                            "accel": REC_ACCEL,
                            "sim_start": rec_sim_start or 0.0,
                            "sim_end": now,
                            "samples": telemetry,
                        }, tf)
                    print(f"REC: telemetry saved ({len(telemetry)} samples) "
                          f"-> {tele_path}")
                except Exception as e:
                    print(f"REC: telemetry write failed: {e}")
                rec_stage = "finishing"
        elif rec_stage == "finishing":
            # Let Webots flush/encode the movie before quitting.
            try:
                if robot.movieIsReady():
                    if robot.movieFailed():
                        print("REC: movie FAILED to encode")
                    else:
                        print("REC: movie encoded OK")
                    robot.simulationQuit(0)
            except Exception as e:
                print(f"REC: finishing error: {e}")
                robot.simulationQuit(0)

        tick += 1


if __name__ == "__main__":
    main()
