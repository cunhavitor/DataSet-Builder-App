from __future__ import annotations

from typing import Dict, Any, Tuple
from pathlib import Path
import cv2
import numpy as np

# Parametros de detecao (compativeis com debug)
CANNY_LOW = 30
CANNY_HIGH = 140
GAUSS_KSIZE = 3  # blur (impar) - mais baixo para evitar suavizar em excesso
DILATE_ITERS = 2
CLOSE_ITERS = 2

# alvo de contorno: centro e area (a partir do debug, base size ~256)
TARGET_AREA = 10100.0
TARGET_CENTER_WEIGHT = 0.8
TARGET_AREA_WEIGHT = 0.2
TARGET_AREA_TOL = 0.15  # +-15%
MASK_BORDER_PAD = 1  # px de dilatacao extra na mascara


# ------------------------
#  Contornos
# ------------------------
def _find_contours(img_bgr: np.ndarray, *, strong: bool = False):
    """Canny + blur + morfologia; retorna contornos ordenados por area (maior->menor)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (GAUSS_KSIZE, GAUSS_KSIZE), 0)

    h, w = gray_blur.shape
    margin_y = int(0.10 * h)
    margin_x = int(0.10 * w)
    roi = gray_blur[margin_y:h - margin_y, margin_x:w - margin_x]

    edges_roi = cv2.Canny(roi, CANNY_LOW, CANNY_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3 if not strong else 5))
    edges_roi = cv2.dilate(edges_roi, kernel, iterations=1 if not strong else 3)
    edges_roi = cv2.morphologyEx(edges_roi, cv2.MORPH_CLOSE, kernel, iterations=2 if not strong else 4)

    contours, _ = cv2.findContours(edges_roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        c[:, 0, 0] += margin_x
        c[:, 0, 1] += margin_y

    contours_sorted = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
    return contours_sorted


def _score_contour(c, img_shape, target_area: float):
    h, w = img_shape[:2]
    img_center = np.array([w / 2.0, h / 2.0])
    area = cv2.contourArea(c)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return float("inf")
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    dist_center = np.linalg.norm(np.array([cx, cy]) - img_center)
    area_diff_rel = abs(area - target_area) / max(target_area, 1.0)
    return TARGET_CENTER_WEIGHT * dist_center + TARGET_AREA_WEIGHT * area_diff_rel * 1000


def _filter_band(contours, target_area: float, target_tol: float):
    area_tol = target_area * target_tol
    area_min = target_area - area_tol
    area_max = target_area + area_tol
    return [c for c in contours if area_min <= cv2.contourArea(c) <= area_max]


def _best_contour(contours, img_shape, target_area: float):
    if not contours:
        return None
    return min(contours, key=lambda c: _score_contour(c, img_shape, target_area))


def _vertical_delta_from_rect(rect):
    """
    Calcula delta (graus) para alinhar o lado maior do retangulo ao eixo Y (vertical).
    Retorna (delta_to_vertical, deviation_from_y).
    """
    (w, h) = rect[1]
    angle = rect[2]  # angulo do lado width em relacao ao eixo X ([-90,0))
    if w >= h:
        deviation_from_y = angle - 90.0
        delta_to_vertical = -deviation_from_y
    else:
        deviation_from_y = angle
        delta_to_vertical = -deviation_from_y
    return delta_to_vertical, deviation_from_y


def _center_on_contour(img_bgr: np.ndarray, contour: np.ndarray):
    """Translada imagem para colocar o centroide do contorno no centro da imagem."""
    h, w = img_bgr.shape[:2]
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return img_bgr, 0.0, 0.0
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    dx = w / 2.0 - cx
    dy = h / 2.0 - cy
    M_t = np.float32([[1, 0, dx], [0, 1, dy]])
    centered = cv2.warpAffine(img_bgr, M_t, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return centered, dx, dy


def _rotate_with_delta(img_bgr: np.ndarray, contour: np.ndarray, delta_deg: float):
    """Rotaciona imagem/contorno pelo delta indicado (graus)."""
    h, w = img_bgr.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -delta_deg/2, 1.0)
    rotated = cv2.warpAffine(img_bgr, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    contour_rot = cv2.transform(contour, rot_mat).astype(np.int32)
    contour_rot[:, 0, 0] = np.clip(contour_rot[:, 0, 0], 0, w - 1)
    contour_rot[:, 0, 1] = np.clip(contour_rot[:, 0, 1], 0, h - 1)
    return rotated, contour_rot


def _load_and_prepare_mask(target_shape) -> np.ndarray:
    """Carrega a mascara base e dilata alguns pixels para nao cortar a borda."""
    mask_path = Path(__file__).resolve().parent.parent / "data" / "mask" / "can_mask.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        # fallback: mascara cheia (nao corta nada)
        return np.ones(target_shape[:2], dtype=np.uint8) * 255

    if MASK_BORDER_PAD > 0:
        k = 2 * MASK_BORDER_PAD + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)

    mask_resized = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask_resized


# ------------------------
#  API principal
# ------------------------
def preprocess_can_image(
    can_img_bgr: np.ndarray,
) -> Tuple[np.ndarray | None, np.ndarray | None, Dict[str, Any]]:
    """
    Pipeline:
      1. Detecta contornos (Canny+morfologia) e filtra pela faixa de area alvo.
      2. Seleciona contorno com melhor score (centro + area).
      3. Rotaciona para alinhar lado maior ao eixo Y.
      4. Centra o contorno no frame.

    Retorna:
      - imagem_alinhada_centrada (BGR) ou None se nao encontrar contorno alvo
      - mascara aplicada (uint8) ou None se nao encontrar contorno alvo
      - info (found, area, angle, delta_vertical, dx, dy)
    """
    h, w = can_img_bgr.shape[:2]
    # adapta a area alvo ao tamanho da imagem (base 256)
    scale_area = (h / 256.0) ** 2
    target_area = TARGET_AREA * scale_area
    target_tol = TARGET_AREA_TOL
    info: Dict[str, Any] = {
        "label_found": False,
        "contour_area": 0.0,
        "angle_deg": 0.0,
        "delta_vertical": 0.0,
        "dx": 0.0,
        "dy": 0.0,
    }

    contours = _find_contours(can_img_bgr)
    contours = _filter_band(contours, target_area=target_area, target_tol=target_tol)
    best = _best_contour(contours, can_img_bgr.shape, target_area=target_area)

    if best is None or len(best) < 3:
        # nenhum contorno valido -> sinaliza para nao salvar
        return None, None, info

    info["label_found"] = True
    info["contour_area"] = float(cv2.contourArea(best))

    # rotacao para alinhar lado maior ao eixo Y
    rect = cv2.minAreaRect(np.asarray(best, dtype=np.float32))
    angle = rect[2]
    delta_vertical, deviation_y = _vertical_delta_from_rect(rect)
    info["angle_deg"] = float(angle)
    info["delta_vertical"] = float(delta_vertical)

    img_rot, contour_rot = _rotate_with_delta(can_img_bgr, best, delta_vertical)

    # centrar no contorno rotacionado
    img_centered, dx, dy = _center_on_contour(img_rot, contour_rot)
    info["dx"] = float(dx)
    info["dy"] = float(dy)

    # aplica mascara de lata (dilatada 20px) centrada no frame
    mask = _load_and_prepare_mask(img_centered.shape)
    img_masked = cv2.bitwise_and(img_centered, img_centered, mask=mask)

    return img_masked, mask, info
