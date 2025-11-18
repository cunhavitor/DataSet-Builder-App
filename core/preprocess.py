# core/preprocess.py

from typing import Dict
import numpy as np

from .types import ImageArray
from . import normalize_image as nz  # importa o teu módulo real


def normalize_sheet_image(
    img_bgr: ImageArray,
    normalization_profile: Dict,
) -> ImageArray:
    """
    Aplica o pipeline de normalização real da folha, usando o normalize_image
    já existente no teu outro projeto.

    Args:
        img_bgr: imagem BGR original da folha.
        normalization_profile: dicionário de opções (por agora podes ignorar ou
                               usar só 'size', por exemplo).

    Returns:
        Imagem normalizada (np.ndarray).
    """
    size = normalization_profile.get("size", 256)
    target_v = normalization_profile.get("target_v", 180.0)
    clahe_clip = normalization_profile.get("clahe_clip", 1.5)
    clahe_tile = normalization_profile.get("clahe_tile", (4, 4))
    # adapta isto ao signature real do teu normalize_image
    img_norm = nz.normalize_image(
        img_bgr,
        size=size,
        target_v=target_v,
        clahe_clip=clahe_clip,
        clahe_tile=clahe_tile,
    )
    return img_norm
