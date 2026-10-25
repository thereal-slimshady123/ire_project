"""Configuration loader and schema validator."""

from pathlib import Path
from typing import Any, Dict, List
import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file.

    Args:
        config_path: Path to .yaml file.

    Returns:
        Configuration dictionary.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate that required keys are present in config.

    Args:
        config: Configuration dictionary.
        required_keys: List of required top-level keys.

    Returns:
        True if all required keys are present, False otherwise.
    """
    return all(key in config for key in required_keys)
