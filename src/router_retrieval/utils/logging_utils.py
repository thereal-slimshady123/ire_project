"""Structured logging setup for experimental runs."""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str, log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Get or create a configured structured logger.

    Args:
        name: Logger module name.
        log_file: Optional path to write file log output.
        level: Logging verbosity level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
