import os
import yaml
from typing import Any, Dict


def load_config(path: str = None) -> Dict[str, Any]:
    """Load YAML config from given path or default location `config.yaml` in repo root."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_universe_config(config: Dict[str, Any], name: str = None) -> Dict[str, Any]:
    if not config:
        return {}
    if name is None:
        return config.get("universe", {})
    universes = config.get("universes", {})
    return universes.get(name, {})
