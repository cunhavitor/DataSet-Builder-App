"""
normalize_image.py

Pipeline de normalização de imagem usado pelo Dataset Builder / Calibrator.

Objetivo principal:
- Tornar as imagens o mais consistentes possível em termos de cor, contraste,
  brilho e tamanho, para reduzir variações no dataset.

Passos do pipeline (normalize_image):
    1) Validar e garantir BGR 3 canais.
    2) Gray World (balance de brancos simples).
    3) CLAHE no canal L (espaço LAB) para melhorar contraste.
    4) Letterbox para canvas quadrado size x size, preservando aspeto.
    5) Normalização de brilho global no canal V (HSV) para um alvo.
    6) Converter para float32 em [0, 1].

Função principal:
    normalize_image(img_bgr, size=256, pad_color=(0, 0, 0), target_v=180)

Retorno:
    np.ndarray float32 com shape (size, size, 3) e valores no intervalo [0, 1].
"""

from __future__ import annotations

from typing import Tuple
from pathlib import Path

import cv2
import numpy as np

PadColor = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _ensure_bgr_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Garante que a imagem é BGR com 3 canais.

    Raises:
        ValueError se a imagem não for válida.
    """
    if img_bgr is None:
        raise ValueError("Input image is None.")

    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(
            f"Expected BGR image with shape (H, W, 3), got {img_bgr.shape}."
        )

    if img_bgr.dtype != np.uint8:
        # Se vier em float [0,1] ou outro, convertemos para uint8
        img = img_bgr.astype(np.float32)
        img = np.clip(img, 0.0, 1.0) if img.max() <= 1.0 else np.clip(img, 0.0, 255.0)
        if img.max() <= 1.0:
            img = (img * 255.0).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        return img

    return img_bgr


def gray_world(img_bgr: np.ndarray) -> np.ndarray:
    """
    Aplica correção de balanço de brancos Gray World:
    - assume que a média de todas as cores deveria ser neutra
    - ajusta os canais B, G, R para aproximar esse cinza médio.
    """
    img = _ensure_bgr_image(img_bgr)
    img_f = img.astype(np.float32)

    # médias por canal
    b_mean, g_mean, r_mean = np.mean(img_f, axis=(0, 1))
    mean_gray = (b_mean + g_mean + r_mean) / 3.0 + 1e-6

    gain_b = mean_gray / (b_mean + 1e-6)
    gain_g = mean_gray / (g_mean + 1e-6)
    gain_r = mean_gray / (r_mean + 1e-6)

    img_f[:, :, 0] *= gain_b
    img_f[:, :, 1] *= gain_g
    img_f[:, :, 2] *= gain_r

    img_f = np.clip(img_f, 0, 255)
    return img_f.astype(np.uint8)


def clahe_L_in_LAB(
    img_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Aplica CLAHE apenas no canal L em LAB, para melhorar contraste sem destruir cor.
    """
    img = _ensure_bgr_image(img_bgr)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    L_eq = clahe.apply(L)

    lab_eq = cv2.merge([L_eq, a, b])
    bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return bgr_eq


def letterbox_square(
    img_bgr: np.ndarray,
    size: int = 256,
    pad_color: PadColor = (0, 0, 0),
) -> np.ndarray:
    """
    Redimensiona a imagem mantendo o aspeto para caber num quadrado size x size,
    preenchendo o resto com pad_color (letterbox).
    """
    img = _ensure_bgr_image(img_bgr)
    h, w = img.shape[:2]

    if h == 0 or w == 0:
        raise ValueError("Image has zero size.")

    # se já couber, apenas pad; evita upscaling desnecessário
    if h <= size and w <= size:
        canvas = np.full((size, size, 3), pad_color, dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        canvas[y_offset:y_offset + h, x_offset:x_offset + w, :] = img
        return canvas

    scale = float(size) / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # resize mantendo aspeto (apenas downscale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # criar canvas
    canvas = np.full((size, size, 3), pad_color, dtype=np.uint8)

    # colocar centrado
    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w, :] = resized

    return canvas


def normalize_brightness_hsv(
    img_bgr: np.ndarray,
    target_v: float = 180.0,
) -> np.ndarray:
    """
    Normaliza o brilho global usando o canal V (Value) em HSV.

    target_v:
        valor alvo para a média do canal V (0–255).
    """
    img = _ensure_bgr_image(img_bgr)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v_f = v.astype(np.float32)
    current_mean = float(np.mean(v_f)) + 1e-6
    gain = target_v / current_mean

    v_f *= gain
    v_f = np.clip(v_f, 0, 255)
    v_norm = v_f.astype(np.uint8)

    hsv_norm = cv2.merge([h, s, v_norm])
    bgr_norm = cv2.cvtColor(hsv_norm, cv2.COLOR_HSV2BGR)
    return bgr_norm


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def normalize_image(
    img_bgr: np.ndarray,
    size: int = 256,
    pad_color: PadColor = (0, 0, 0),
    target_v: float = 180.0,
    clahe_clip: float = 1.5,
    clahe_tile: Tuple[int, int] = (4, 4),
) -> np.ndarray:
    """
    Pipeline completo de normalização da imagem.

    Args:
        img_bgr: imagem original em BGR (uint8 ou float).
        size: tamanho do lado do quadrado de saída (size x size).
        pad_color: cor de preenchimento usada no letterbox.
        target_v: valor alvo médio do canal V (brilho) em HSV, para normalização.
        clahe_clip: clipLimit do CLAHE no canal L.
        clahe_tile: tileGridSize do CLAHE no canal L.

    Returns:
        Imagem normalizada em float32 com shape (size, size, 3) e valores em [0, 1].
    """
    # 1) Garantir formato
    img = _ensure_bgr_image(img_bgr)

    # 2) Gray world
    img = gray_world(img)

    # 3) CLAHE no L (LAB)
    img = clahe_L_in_LAB(img, clip_limit=clahe_clip, tile_grid_size=clahe_tile)

    # 4) Letterbox para quadrado
    img = letterbox_square(img, size=size, pad_color=pad_color)

    # 5) Normalizar brilho global
    img = normalize_brightness_hsv(img, target_v=target_v)

    # 6) Converter para float32 [0, 1]
    img_f = img.astype(np.float32) / 255.0

    return img_f
