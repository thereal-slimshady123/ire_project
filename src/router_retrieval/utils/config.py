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
    raise NotImplementedError


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """Validate that required keys are present in config."""
    raise NotImplementedError
