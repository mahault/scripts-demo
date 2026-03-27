"""plugin_templates/example_robot_plugin/plugin.py
TODOs:
- Example plugin entry point.
- In a real package, you might expose a function like `register(registry)`.
"""

from architecture_core.core.registry import SkillEntry
from plugin_templates.example_robot_plugin.skills.nav_skill import ExampleNavSkill
from plugin_templates.example_robot_plugin.skills.nav_policy import ExampleNavIntentPolicy

def register(registry):
    registry.register(SkillEntry(skill=ExampleNavSkill(), policy=ExampleNavIntentPolicy()))
