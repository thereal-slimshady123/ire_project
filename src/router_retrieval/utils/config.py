"""Configuration loader and schema validator."""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file.

    Args:
        config_path: Path to .yaml file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """Validate that required keys are present in config."""
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise KeyError(f"Config missing required keys: {missing}")
    return True
