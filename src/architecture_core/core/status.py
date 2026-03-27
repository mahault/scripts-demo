"""architecture_core/core/status.py
TODOs:
- Keep status vocabulary stable; plugins depend on it.
"""

from typing import Literal

Status = Literal["IDLE", "RUNNING", "SUCCESS", "FAILURE", "PREEMPTED", "TIMEOUT"]
