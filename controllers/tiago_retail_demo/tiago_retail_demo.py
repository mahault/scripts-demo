"""Webots controller entry point for the retail script learning demo.

Shim that bootstraps retail_controller.py from the plugins directory.
"""

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.normpath(os.path.join(_THIS_DIR, os.pardir, os.pardir, "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

os.environ["RETAIL_DEMO"] = "1"
os.environ["PHASE_LEVEL"] = "9"

from plugins.tiago_webots.retail_controller import main

main()
