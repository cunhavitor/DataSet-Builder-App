# core/types.py

from pathlib import Path
from typing import Union
import numpy as np

# Tipos alias usados em vários módulos

PathLike = Union[str, Path]
ImageArray = np.ndarray
