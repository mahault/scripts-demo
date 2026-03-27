"""Webots controller entry point — thin shim that bootstraps the social-layer
controller from src/plugins/tiago_webots/controller.py.

Webots requires controllers at controllers/<name>/<name>.py.
"""

import sys
import os

# Add src/ to path so all architecture_core and plugins imports work
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.normpath(os.path.join(_THIS_DIR, os.pardir, os.pardir, "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# For the queue demo, Phase 10 (table-clearing) must be disabled —
# it replaces the navigate-to-goal ScriptManager with UnifiedTaskController
# which has nothing to do when there are no table objects.
os.environ["PHASE_LEVEL"] = "9"
os.environ["QUEUE_DEMO"] = "1"

from plugins.tiago_webots.controller import main

main()
