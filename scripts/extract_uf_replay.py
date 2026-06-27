#!/usr/bin/env python3
"""Extract real UF-Retail trajectories (human shopper + robot worker) and map
them into the Webots store, writing replay files the ReplayDriver consumes.

Output: data/replay/shopper.json and data/replay/worker.json, each
  {"dt": 0.1, "n": N, "frames": [[x, y, heading, reach], ...]}
in store coordinates, time-synced (same index = same instant), starting at t=0.

The SAME affine map is applied to both so their relative geometry — the
interaction — is preserved.  The shopper carries a `reach` flag from the real
hand height; the worker (robot path) has reach=0 (no recorded arm).
"""

from __future__ import annotations

import io
import csv
import json
import math
import os
import zipfile

import openpyxl

# UF-Retail archive lives in the main project's dataset dir.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
ZIP = os.path.normpath(os.path.join(
    PROJ, os.pardir, os.pardir, "social-layer", "data", "datasets",
    "uf_retail", "UF-Retail-HRI-Dataset-main.zip"))
MOCAP = "UF-Retail-HRI-Dataset-main/Example code/P001_006.xlsx"
AMCL = "UF-Retail-HRI-Dataset-main/Example code/amcldata.txt"

OUT_DIR = os.path.join(PROJ, "data", "replay")
DT = 0.1                     # replay timestep (s)
REACH_HAND_Z = 1.20          # m — right hand above this = reaching up
REACH_HAND_R = 0.40          # m — or hand this far horizontally from pelvis

# Target store region the dataset bounding box is mapped into (uniform scale,
# centred).  Chosen to put the action across the aisles in front of the shelves.
STORE_CENTER = (-0.5, -5.2)
STORE_W = 9.5
STORE_H = 6.0


def load_human(zf):
    with zf.open(MOCAP) as f:
        wb = openpyxl.load_workbook(io.BytesIO(f.read()), read_only=True,
                                    data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    col = {h: i for i, h in enumerate(rows[0])}

    def g(r, name):
        try:
            return float(r[col[name]])
        except (TypeError, ValueError, KeyError):
            return None

    out = []   # (t, x, y, reach)
    for r in rows[1:]:
        t = g(r, "Timestamp")
        px, py = g(r, "Pelvis_x"), g(r, "Pelvis_y")
        if t is None or px is None or py is None:
            continue
        rhz = g(r, "RightHand_z") or 0.0
        rhx, rhy = g(r, "RightHand_x") or px, g(r, "RightHand_y") or py
        hand_r = math.hypot(rhx - px, rhy - py)
        reach = 1 if (rhz > REACH_HAND_Z or hand_r > REACH_HAND_R) else 0
        out.append((t / 1000.0, px, py, reach))
    return out


def load_robot(zf):
    out = []   # (t, x, y)
    with zf.open(AMCL) as f:
        rdr = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8",
                                              errors="replace"))
        for row in rdr:
            try:
                t = float(row["%time"]) / 1e9
                x = float(row["field.pose.pose.position.x"])
                y = float(row["field.pose.pose.position.y"])
                out.append((t, x, y))
            except (KeyError, ValueError, TypeError):
                continue
    return out


def sample(series, t):
    """Linear interp of an (t, x, y, [reach]) series at time t."""
    if t <= series[0][0]:
        return series[0][1:]
    if t >= series[-1][0]:
        return series[-1][1:]
    lo, hi = 0, len(series) - 1
    while lo + 1 < hi:
        m = (lo + hi) // 2
        if series[m][0] <= t:
            lo = m
        else:
            hi = m
    t0, t1 = series[lo][0], series[hi][0]
    a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    res = []
    for i in range(1, len(series[lo])):
        v0, v1 = series[lo][i], series[hi][i]
        res.append(v0 + a * (v1 - v0) if i < 3 else (v1 if a > 0.5 else v0))
    return res


def main():
    zf = zipfile.ZipFile(ZIP)
    human = load_human(zf)
    robot = load_robot(zf)
    print(f"human frames {len(human)}, robot frames {len(robot)}")

    # Shared time window (both recorded) and shared affine map.
    t0 = max(human[0][0], robot[0][0])
    t1 = min(human[-1][0], robot[-1][0])
    print(f"overlap window {t1 - t0:.1f}s")

    allx = [p[1] for p in human] + [p[1] for p in robot]
    ally = [p[2] for p in human] + [p[2] for p in robot]
    xmin, xmax, ymin, ymax = min(allx), max(allx), min(ally), max(ally)
    sx = STORE_W / (xmax - xmin) if xmax > xmin else 1.0
    sy = STORE_H / (ymax - ymin) if ymax > ymin else 1.0
    s = min(sx, sy)                                  # uniform scale (no distort)
    cx0, cy0 = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0

    def to_store(x, y):
        return (STORE_CENTER[0] + (x - cx0) * s,
                STORE_CENTER[1] + (y - cy0) * s)

    def build(series, has_reach):
        frames = []
        prev = None
        t = t0
        while t <= t1:
            v = sample(series, t)
            x, y = to_store(v[0], v[1])
            reach = int(v[2]) if has_reach and len(v) > 2 else 0
            heading = prev[2] if prev else 0.0
            if prev:
                dx, dy = x - prev[0], y - prev[1]
                if math.hypot(dx, dy) > 0.005:
                    heading = math.atan2(dy, dx)
            frames.append([round(x, 3), round(y, 3), round(heading, 3), reach])
            prev = (x, y, heading)
            t += DT
        return frames

    os.makedirs(OUT_DIR, exist_ok=True)
    # The reaching human = the WORKER stocking shelves (rich, what the learner
    # watches and learns to become); the walking robot path = the SHOPPER.
    for name, series, has_reach in [("worker", human, True),
                                    ("shopper", robot, False)]:
        frames = build(series, has_reach)
        path = os.path.join(OUT_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump({"dt": DT, "n": len(frames), "frames": frames}, f)
        xs = [fr[0] for fr in frames]
        ys = [fr[1] for fr in frames]
        rr = sum(fr[3] for fr in frames)
        print(f"{name}: {len(frames)} frames  x[{min(xs):.1f},{max(xs):.1f}]"
              f" y[{min(ys):.1f},{max(ys):.1f}] reach_frames={rr} -> {path}")


if __name__ == "__main__":
    main()
