"""
YAML configuration loader utility.
"""
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Base config directory relative to this file
_CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_yaml(filename: str) -> dict[str, Any]:
    """
    Load a YAML file from the config directory.

    Args:
        filename: YAML filename (e.g., 'agents.yaml')

    Returns:
        Parsed YAML content as a dict.
    """
    path = _CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    logger.debug(f"Loaded config from {path}")
    return data or {}


def load_agents_config() -> dict[str, Any]:
    return load_yaml("agents.yaml")


def load_tasks_config() -> dict[str, Any]:
    return load_yaml("tasks.yaml")
