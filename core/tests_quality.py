"""
core/tests_quality.py

MVP de testes de qualidade da folha.

Compara a folha atual com um template, usando:
- brilho (canal L do LAB)
- diferença de L entre folha e template
- diferença de cor (a,b) entre folha e template

Também devolve campos de 'rotation' e 'centering' como placeholders
(para no futuro implementares esses testes a sério).

A função principal é:
    run_all_tests(sheet_img_bgr, template_img_bgr, config)
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


def _compute_lab_means(img_bgr: np.ndarray) -> Tuple[float, float, float]:
    """Converte BGR -> LAB e devolve (mean_L, mean_a, mean_b)."""
    if img_bgr is None:
        raise ValueError("Imagem é None em _compute_lab_means.")

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    mean_L = float(np.mean(L))
    mean_a = float(np.mean(A))
    mean_b = float(np.mean(B))
    return mean_L, mean_a, mean_b


def run_all_tests(
    sheet_img_bgr: np.ndarray,
    template_img_bgr: np.ndarray,
    config: Dict,
) -> Tuple[bool, Dict[str, float]]:
    """
    Executa todos os testes de qualidade da folha.

    Parameters
    ----------
    sheet_img_bgr : np.ndarray
        Imagem da folha atual (BGR uint8).
    template_img_bgr : np.ndarray
        Imagem da folha template (BGR uint8).
    config : Dict
        Dicionário com:
            - max_lab_delta_L: float
            - max_lab_delta_ab: float
            - target_brightness: float   (no espaço L de 0..255)
            - max_brightness_delta: float

    Returns
    -------
    tests_pass : bool
        True se todos os testes (MVP) estiverem dentro dos limites.
    details : Dict[str, float]
        Métricas calculadas (para logging/metadata).
    """
    # Lê parâmetros com defaults razoáveis
    max_lab_delta_L = float(config.get("max_lab_delta_L", 10.0))
    max_lab_delta_ab = float(config.get("max_lab_delta_ab", 15.0))
    target_brightness = float(config.get("target_brightness", 180.0))
    max_brightness_delta = float(config.get("max_brightness_delta", 50.0))

    # --- calcular médias LAB da folha e do template ---
    sheet_L, sheet_a, sheet_b = _compute_lab_means(sheet_img_bgr)
    templ_L, templ_a, templ_b = _compute_lab_means(template_img_bgr)

    # --- brilho (diferença do target) ---
    brightness_delta = abs(sheet_L - target_brightness)

    # --- diferença em L entre folha e template ---
    lab_delta_L = abs(sheet_L - templ_L)

    # --- diferença de cor a/b entre folha e template ---
    delta_a = sheet_a - templ_a
    delta_b = sheet_b - templ_b
    # podemos usar norma Euclidiana como medida de "delta_ab"
    lab_delta_ab = float(np.sqrt(delta_a ** 2 + delta_b ** 2))

    # --- placeholders para testes futuros ---
    rotation_score = 1.0  # 1.0 = OK
    centering_score = 1.0  # 1.0 = OK

    # --- verificações ---
    brightness_ok = brightness_delta <= max_brightness_delta
    lab_L_ok = lab_delta_L <= max_lab_delta_L
    lab_ab_ok = lab_delta_ab <= max_lab_delta_ab

    # Por agora, assumimos rotação e centragem sempre OK
    rotation_ok = True
    centering_ok = True

    tests_pass = all([brightness_ok, lab_L_ok, lab_ab_ok, rotation_ok, centering_ok])

    details: Dict[str, float] = {
        "brightness": float(sheet_L),
        "brightness_delta": float(brightness_delta),
        "lab_delta_L": float(lab_delta_L),
        "lab_delta_ab": float(lab_delta_ab),
        "rotation": float(rotation_score),
        "centering": float(centering_score),
    }

    # Logging simples para veres os valores no terminal
    print(
        "[TESTS] L={:.1f}, ΔL(target)={:.1f}, ΔL(template)={:.1f}, Δab={:.2f} -> pass={}".format(
            sheet_L, brightness_delta, lab_delta_L, lab_delta_ab, tests_pass
        )
    )

    return tests_pass, details
