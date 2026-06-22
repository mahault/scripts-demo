"""Build a dashboard for a recorded retail social-layer demo run.

Parses the per-robot controller logs (Worker_T / Learner_L / Customer_1) from a
recorded run and renders a single, self-explanatory figure plus an HTML page:

  * store map with the three robots' routes and where the social acts happen
  * the learner's crystallisation curve (precision -> strong, from observation)
  * a tally of the social interactions (yield / greet / hand-over / wayfinding)
  * worker restock loops over time
  * a learning-milestones timeline and a summary panel

Usage:
    python scripts/build_dashboard.py [RUN_DIR] [OUT_DIR]

Defaults: RUN_DIR=dashboard/recorded_run, OUT_DIR=dashboard
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec


# ----------------------------------------------------------------------------
# Store geometry (from worlds/tiago_retail_demo.wbt) for the map panel.
# Each fixture is (label, x_centre, y_centre, width_x, depth_y, colour).
# ----------------------------------------------------------------------------
FIXTURES = [
    ("stock shelf", -6.0, -2.5, 0.5, 1.8, "#8d6e63"),
    ("shelf A",     -2.0, -5.0, 0.45, 1.8, "#7e57c2"),
    ("shelf B",      1.5, -5.0, 0.45, 1.8, "#7e57c2"),
    ("counter",      4.5, -7.5, 0.8, 2.0, "#5d4037"),
    ("divider",     -4.0, -1.5, 0.15, 5.0, "#9e9e9e"),
]
CUE_ZONES = [
    ("stock",   -5.5, -1.0, 2.0, 2.0, "#2f6fd0"),
    ("shelf A", -2.0, -5.0, 1.2, 0.8, "#e8911a"),
    ("shelf B",  1.5, -5.0, 1.2, 0.8, "#e8911a"),
    ("counter",  4.5, -7.5, 2.2, 1.0, "#d9b310"),
    ("queue",    4.5, -6.0, 1.5, 1.0, "#2bb6b6"),
]
HANDOVER_XY = (4.5, -7.0)
ASK_XY = (2.5, -7.0)


def _read(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def parse_worker(lines: list[str]) -> dict:
    route, loops_t = [], []
    rx = re.compile(
        r"Worker_T: t=([0-9.]+) pos=\(([0-9.\-]+),([0-9.\-]+)\) wp=\d+ "
        r"state=\S+ loops=(\d+)")
    for ln in lines:
        m = rx.search(ln)
        if m:
            t, x, y, lp = float(m[1]), float(m[2]), float(m[3]), int(m[4])
            route.append((x, y))
            loops_t.append((t, lp))
    return {
        "route": route,
        "loops_t": loops_t,
        "yield": sum(1 for ln in lines if "YIELD" in ln),
        "greet": sum(1 for ln in lines if "GREET" in ln),
        "handover": sum(1 for ln in lines if "HANDOVER" in ln),
        "wayfind": sum(1 for ln in lines if "DIRECTING" in ln),
    }


def parse_learner(lines: list[str]) -> dict:
    curve = []  # (observed_loop, precision, trajectory_count)
    rx = re.compile(r"OBSERVED loop (\d+).*prec=([0-9.]+) traj=(\d+)")
    for ln in lines:
        m = rx.search(ln)
        if m:
            curve.append((int(m[1]), float(m[2]), int(m[3])))
    cryst = None
    mc = re.search(r"CRYSTALLIZED.*prec=([0-9.]+)", "\n".join(lines))
    if mc:
        cryst = float(mc[1])
    steps = None
    ms = re.search(r"learned script installed \((\d+) steps\)", "\n".join(lines))
    if ms:
        steps = int(ms[1])
    route = []
    rr = re.compile(r"Learner_L: t=[0-9.]+ pos=\(([0-9.\-]+),([0-9.\-]+)\)")
    for ln in lines:
        m = rr.search(ln)
        if m:
            route.append((float(m[1]), float(m[2])))
    return {
        "curve": curve, "crystallized_prec": cryst,
        "learned_steps": steps, "route": route,
        "switched": any("SWITCHING TO EXECUTION" in ln for ln in lines),
        "nav_success": sum(1 for ln in lines if "[NAV] SUCCESS" in ln),
        "assists": sum(1 for ln in lines if "assist after timeout" in ln),
    }


def parse_customer(lines: list[str]) -> dict:
    route, collected = [], 0
    rx = re.compile(
        r"Customer_1: t=[0-9.]+ pos=\(([0-9.\-]+),([0-9.\-]+)\)"
        r".*collected=(\d+)")
    for ln in lines:
        m = rx.search(ln)
        if m:
            route.append((float(m[1]), float(m[2])))
            collected = max(collected, int(m[3]))
    return {"route": route, "collected": collected,
            "asks": sum(1 for ln in lines if "arrived at ASK_WORKER" in ln)}


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def _draw_store(ax, w, le, cu):
    ax.set_title("Store map — who went where", fontsize=12, weight="bold")
    for label, x, y, wx, dy, col in FIXTURES:
        ax.add_patch(Rectangle((x - wx / 2, y - dy / 2), wx, dy,
                               facecolor=col, edgecolor="black", alpha=0.85, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=6,
                color="white", zorder=4, rotation=90)
    for label, x, y, wx, dy, col in CUE_ZONES:
        ax.add_patch(Rectangle((x - wx / 2, y - dy / 2), wx, dy,
                               facecolor=col, edgecolor="none", alpha=0.16, zorder=1))

    def path(route, col, lbl):
        if route:
            xs = [p[0] for p in route]
            ys = [p[1] for p in route]
            ax.plot(xs, ys, color=col, lw=1.2, alpha=0.8, label=lbl, zorder=5)

    path(w["route"], "#1f77b4", "worker (teacher)")
    path(le["route"], "#2ca02c", "learner (replay)")
    path(cu["route"], "#ff7f0e", "customer")

    # Social-act markers.
    if w["handover"]:
        ax.scatter(*HANDOVER_XY, marker="*", s=240, color="#d62728",
                   edgecolor="black", zorder=6, label=f"hand-over ×{w['handover']}")
    if w["wayfind"]:
        ax.scatter(*ASK_XY, marker="P", s=180, color="#9467bd",
                   edgecolor="black", zorder=6, label=f"wayfinding ×{w['wayfind']}")

    ax.set_xlim(-7.5, 6.0)
    ax.set_ylim(-9.2, 1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.15)


def _draw_curve(ax, curve, cryst):
    ax.set_title("Learner crystallises a script from watching",
                 fontsize=12, weight="bold")
    if curve:
        loops = [c[0] for c in curve]
        prec = [c[1] for c in curve]
        ax.plot(loops, prec, "-o", color="#2ca02c", lw=2, ms=6)
        ax.axhline(0.8, ls="--", color="#d62728", lw=1)
        ax.text(loops[-1], 0.82, "strong threshold", color="#d62728",
                ha="right", fontsize=8)
        # Mark crystallisation (first loop precision >= 0.8).
        for lp, pr in zip(loops, prec):
            if pr >= 0.8:
                ax.scatter([lp], [pr], s=180, marker="*", color="#d4af37",
                           edgecolor="black", zorder=5)
                ax.annotate("crystallised!", (lp, pr), textcoords="offset points",
                            xytext=(-6, 10), fontsize=9, weight="bold")
                break
    ax.set_xlabel("observed restock loop")
    ax.set_ylabel("pattern precision")
    ax.set_ylim(0, 1.7)
    ax.grid(alpha=0.2)


def _draw_tally(ax, w):
    ax.set_title("Social interactions performed", fontsize=12, weight="bold")
    labels = ["give-way\n(yield)", "greet", "hand-over", "wayfinding\n(milk?)"]
    vals = [w["yield"], w["greet"], w["handover"], w["wayfind"]]
    cols = ["#1f77b4", "#17becf", "#d62728", "#9467bd"]
    bars = ax.bar(labels, vals, color=cols, edgecolor="black", alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center",
                va="bottom", fontsize=11, weight="bold")
    ax.set_ylabel("count")
    ax.set_ylim(0, max(vals + [1]) * 1.25)
    ax.grid(alpha=0.2, axis="y")


def _draw_loops(ax, loops_t):
    ax.set_title("Worker restock loops over time", fontsize=12, weight="bold")
    if loops_t:
        ts = [p[0] for p in loops_t]
        ls = [p[1] for p in loops_t]
        ax.plot(ts, ls, color="#1f77b4", lw=2)
        ax.fill_between(ts, ls, color="#1f77b4", alpha=0.15)
    ax.set_xlabel("sim time (s)")
    ax.set_ylabel("loops completed")
    ax.grid(alpha=0.2)


def _draw_summary(ax, w, le, cu):
    ax.axis("off")
    ax.set_title("Run summary", fontsize=12, weight="bold")
    rows = [
        ("Worker restock loops", f"{w['loops_t'][-1][1] if w['loops_t'] else 0}"),
        ("Give-ways (yield)", f"{w['yield']}"),
        ("Greets", f"{w['greet']}"),
        ("Hand-overs at counter", f"{w['handover']}"),
        ("Wayfinding answers", f"{w['wayfind']}"),
        ("Customer items collected", f"{cu['collected']}"),
        ("Learner observed loops", f"{le['curve'][-1][0] if le['curve'] else 0}"),
        ("Learner crystallised", "yes" if le["crystallized_prec"] else "no"),
        ("Learned script length", f"{le['learned_steps'] or '-'} steps"),
        ("Switched to execution", "yes" if le["switched"] else "no"),
        ("Learner nav goals reached", f"{le['nav_success']}"),
    ]
    y = 0.95
    for k, v in rows:
        ax.text(0.02, y, k, fontsize=10, va="top")
        ax.text(0.98, y, v, fontsize=10, va="top", ha="right", weight="bold")
        y -= 0.088


def build(run_dir: Path, out_dir: Path) -> Path:
    w = parse_worker(_read(run_dir / "Worker_T.log"))
    le = parse_learner(_read(run_dir / "Learner_L.log"))
    cu = parse_customer(_read(run_dir / "Customer_1.log"))

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Retail Social-Layer Demo — Recorded Run Dashboard",
                 fontsize=17, weight="bold")
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.15, 1.0],
                  hspace=0.28, wspace=0.22,
                  left=0.05, right=0.97, top=0.92, bottom=0.07)

    _draw_store(fig.add_subplot(gs[0, 0:2]), w, le, cu)
    _draw_summary(fig.add_subplot(gs[0, 2]), w, le, cu)
    _draw_curve(fig.add_subplot(gs[1, 0]), le["curve"], le["crystallized_prec"])
    _draw_tally(fig.add_subplot(gs[1, 1]), w)
    _draw_loops(fig.add_subplot(gs[1, 2]), w["loops_t"])

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "dashboard.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)

    _write_html(out_dir / "index.html", w, le, cu)
    return png


def _write_html(path: Path, w, le, cu) -> None:
    loops = w["loops_t"][-1][1] if w["loops_t"] else 0
    obs = le["curve"][-1][0] if le["curve"] else 0
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Retail Social-Layer Demo Dashboard</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0 auto;
        max-width: 1180px; padding: 24px; color: #1a1a1a; }}
 h1 {{ font-size: 26px; margin-bottom: 4px; }}
 .sub {{ color: #666; margin-bottom: 20px; }}
 img {{ width: 100%; border: 1px solid #ddd; border-radius: 8px; }}
 .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; }}
 .card {{ border: 1px solid #e3e3e3; border-radius: 8px; padding: 14px; }}
 .card .n {{ font-size: 30px; font-weight: 700; }}
 .card .l {{ color: #666; font-size: 13px; }}
 ol {{ line-height: 1.7; }} code {{ background:#f3f3f3; padding:1px 5px; border-radius:4px; }}
</style></head><body>
<h1>Retail Social-Layer Demo — Run Dashboard</h1>
<div class="sub">A TIAGo worker restocks a store while socially interacting with a
shopper; a second TIAGo learns the worker's routine purely by watching, then replays it.</div>
<div class="grid">
 <div class="card"><div class="n">{loops}</div><div class="l">worker restock loops</div></div>
 <div class="card"><div class="n">{w['handover']}</div><div class="l">hand-overs at counter</div></div>
 <div class="card"><div class="n">{w['wayfind']}</div><div class="l">wayfinding answers</div></div>
 <div class="card"><div class="n">{w['yield']}</div><div class="l">give-ways to shoppers</div></div>
 <div class="card"><div class="n">{obs}</div><div class="l">loops the learner observed</div></div>
 <div class="card"><div class="n">{'yes' if le['crystallized_prec'] else 'no'}</div><div class="l">script crystallised</div></div>
 <div class="card"><div class="n">{le['learned_steps'] or '-'}</div><div class="l">learned script steps</div></div>
 <div class="card"><div class="n">{cu['collected']}</div><div class="l">customer items collected</div></div>
</div>
<img src="dashboard.png" alt="dashboard">
<h2>What you are looking at</h2>
<ol>
 <li><b>Store map</b> — routes of the worker (blue), the learner replaying the
     learned route (green) and the customer (orange). Stars mark where the
     hand-over (counter) and wayfinding ("where's the milk?") happen.</li>
 <li><b>Crystallisation curve</b> — the learner's best pattern precision climbs
     with each observed restock loop; once it passes the strong threshold the
     routine <i>crystallises</i> into a script the learner then executes.</li>
 <li><b>Interaction tally</b> — the four social acts the worker performs:
     give-way, greet, hand-over, and wayfinding.</li>
 <li><b>Loops over time</b> — the worker keeps restocking steadily.</li>
</ol>
</body></html>"""
    path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "dashboard" / "recorded_run"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "dashboard"
    png = build(run, out)
    print(f"Dashboard written: {png}")
    print(f"HTML written:      {out / 'index.html'}")
