"""Global seed control for deterministic reproducibility."""

import os
import random
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """
    Set seeds for Python random, numpy, and environment for reproducibility.

    Args:
        seed: Random seed integer.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
