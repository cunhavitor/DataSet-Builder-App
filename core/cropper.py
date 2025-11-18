# core/cropper.py

from typing import List, Dict, Optional
import numpy as np
import cv2


def _letterbox_square(img: np.ndarray, size: int = 256, pad_color=(0, 0, 0)):
    """
    Resize com proporção, preenchendo para quadrado (letterbox).
    Só faz downscale se a imagem for maior que size; caso contrário, apenas pad.
    """
    h, w = img.shape[:2]

    if h == 0 or w == 0:
        raise ValueError("Imagem com dimensão zero em _letterbox_square.")

    # se já couber, não escalar para cima; apenas pad
    if h <= size and w <= size:
        canvas = np.full((size, size, 3), pad_color, dtype=np.uint8)
        y0 = (size - h) // 2
        x0 = (size - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = img
        return canvas

    scale = min(size / w, size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((size, size, 3), pad_color, dtype=np.uint8)

    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2

    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    return canvas


def _expand_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    img_w: int,
    img_h: int,
    expand_ratio: float = 0.15,
):
    """
    Expande o bbox em torno do centro, mantendo dentro da imagem.
    expand_ratio = 0.15 -> aumenta ~15% em cada direção.
    """
    cx = x + w / 2.0
    cy = y + h / 2.0

    new_w = w * (1.0 + expand_ratio)
    new_h = h * (1.0 + expand_ratio)

    x0 = int(round(cx - new_w / 2.0))
    y0 = int(round(cy - new_h / 2.0))
    x1 = int(round(cx + new_w / 2.0))
    y1 = int(round(cy + new_h / 2.0))

    # clamp aos limites da imagem
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img_w, x1)
    y1 = min(img_h, y1)

    if x1 <= x0 or y1 <= y0:
        # fallback para bbox original se algo correr mal
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(img_w, x + w)
        y1 = min(img_h, y + h)

    return x0, y0, x1, y1


def crop_cans_from_sheet(
    sheet_img_bgr: np.ndarray,
    detections: List[Dict],
    mask_template: Optional[np.ndarray] = None,
    size: int = 256,
    expand_ratio: float = 0.03,
) -> List[Dict]:
    """
    - Recebe a folha original + deteções (bbox em coords da folha)
    - Expande cada bbox para ter margem extra
    - Recorta
    - Faz letterbox com proporção para size x size

    Retorna lista de dicts:
      { "can_id": int, "image": np.ndarray (BGR size x size) }
    """
    h_img, w_img = sheet_img_bgr.shape[:2]

    results = []
    for det in detections:
        x, y, w, h = det["bbox"]

        # 1) expandir bbox para ter margem
        x0, y0, x1, y1 = _expand_bbox(x, y, w, h, w_img, h_img, expand_ratio=expand_ratio)

        crop = sheet_img_bgr[y0:y1, x0:x1].copy()

        # 2) resize mantendo proporção + padding
        crop = _letterbox_square(crop, size=size)

        results.append(
            {
                "can_id": det["can_id"],
                "image": crop,
            }
        )
    return results
