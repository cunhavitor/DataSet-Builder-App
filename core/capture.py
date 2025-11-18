# core/capture.py

from pathlib import Path
from typing import Optional, List, Union
import cv2
import numpy as np

# Alias de tipos
PathLike = Union[str, Path]
ImageArray = np.ndarray


def list_sheet_images(input_folder: PathLike, extensions: Optional[List[str]] = None) -> List[Path]:
    """
    Lista imagens de folha numa pasta.
    """
    folder = Path(input_folder)

    if extensions is None:
        extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    files = []
    for ext in extensions:
        files.extend(folder.glob(f"*{ext}"))

    return sorted(files)


def load_sheet_image(path: PathLike) -> ImageArray:
    """
    Carrega imagem BGR com cv2.imread.
    """
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem: {path}")

    return img
