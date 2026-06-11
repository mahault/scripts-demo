#!/usr/bin/env python3
"""One-command launcher for the retail investor demo.

Usage:
    python launch_retail_demo.py

This sets up the environment and launches Webots with the retail world.
Requires Webots R2025a to be installed and on your PATH.
"""

from __future__ import annotations

import os
import sys
import subprocess
import platform


def find_webots() -> str | None:
    """Locate the Webots executable."""
    if platform.system() == "Windows":
        candidates = [
            r"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe",
            r"C:\Users\mahau\AppData\Local\Programs\Webots\msys64\mingw64\bin\webots.exe",
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
    else:
        for cmd in ["webots", "/usr/local/webots/webots"]:
            if subprocess.run(["which", cmd], capture_output=True).returncode == 0:
                return cmd
    return None


def main() -> int:
    project_dir = os.path.dirname(os.path.abspath(__file__))
    world_path = os.path.join(project_dir, "worlds", "tiago_retail_demo.wbt")

    if not os.path.isfile(world_path):
        print(f"ERROR: World file not found: {world_path}")
        return 1

    webots = find_webots()
    if webots is None:
        print("ERROR: Webots executable not found.")
        print("Please install Webots R2025a or add it to your PATH.")
        return 1

    # Ensure logs dir exists
    os.makedirs(os.path.join(project_dir, "logs"), exist_ok=True)

    # Environment for the demo
    env = os.environ.copy()
    env["RETAIL_DEMO"] = "1"
    env["PHASE_LEVEL"] = "9"
    env["WEBOTS_CONTROLLER"] = "1"

    # Add src/ to PYTHONPATH so controllers can import architecture_core
    src_path = os.path.join(project_dir, "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    print("=" * 60)
    print("Social Layer — Retail Script Learning Demo")
    print("=" * 60)
    print(f"Webots:  {webots}")
    print(f"World:   {world_path}")
    print(f"PHASE:   {env['PHASE_LEVEL']}")
    print("=" * 60)
    print("\nPress Ctrl+C to stop.\n")

    try:
        subprocess.run([webots, "--mode=realtime", world_path], env=env, check=False)
    except KeyboardInterrupt:
        print("\nDemo stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
