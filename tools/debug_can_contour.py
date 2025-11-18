from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# --------- PARÂMETROS DO CANNY / MORFO ---------
CANNY_LOW = 30
CANNY_HIGH = 140
GAUSS_KSIZE = 5          # blur (ímpar)
DILATE_ITERS = 2
CLOSE_ITERS = 2

# alvo de contorno: centro e área próxima de 10100
TARGET_AREA = 10100.0
TARGET_CENTER_WEIGHT = 0.6  # peso para distância do centro no score
TARGET_AREA_WEIGHT = 0.4    # peso para diferença de área no score
TARGET_AREA_TOL = 0.25      # tolerância relativa para aceitar a área alvo


def _find_contours(img_bgr: np.ndarray, *, strong: bool = False):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (GAUSS_KSIZE, GAUSS_KSIZE), 0)

    h, w = gray_blur.shape
    # corta uma região central (ajusta a margem a gosto)
    margin_y = int(0.10 * h)
    margin_x = int(0.10 * w)
    roi = gray_blur[margin_y:h - margin_y, margin_x:w - margin_x]

    edges_roi = cv2.Canny(roi, CANNY_LOW, CANNY_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_roi = cv2.dilate(edges_roi, kernel, iterations=1)
    edges_roi = cv2.morphologyEx(edges_roi, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges_roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # deslocar os contornos para as coords da imagem original
    for c in contours:
        c[:, 0, 0] += margin_x
        c[:, 0, 1] += margin_y

    contours_sorted = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)

    # para debug, podes reconstruir uma imagem de edges do tamanho original
    edges_full = np.zeros_like(gray, dtype=np.uint8)
    edges_full[margin_y:h - margin_y, margin_x:w - margin_x] = edges_roi

    return contours_sorted, edges_full

def _draw_center_cross(img_bgr: np.ndarray, color=(0, 0, 255)):
    """Só para referência visual do centro da imagem (se quiseres)."""
    h, w = img_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(img_bgr, (cx, 0), (cx, h - 1), color, 1)
    cv2.line(img_bgr, (0, cy), (w - 1, cy), color, 1)


def _center_image_on_contour(img_bgr: np.ndarray, contour: np.ndarray):
    """Translada a imagem para alinhar o centroide do contorno ao centro da imagem."""
    h, w = img_bgr.shape[:2]
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return img_bgr.copy(), contour, (0, 0)

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    shift_x = int(round(w / 2.0 - cx))
    shift_y = int(round(h / 2.0 - cy))

    warp_mat = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    centered = cv2.warpAffine(
        img_bgr,
        warp_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    contour_shifted = contour.copy()
    contour_shifted[:, 0, 0] = np.clip(contour_shifted[:, 0, 0] + shift_x, 0, w - 1)
    contour_shifted[:, 0, 1] = np.clip(contour_shifted[:, 0, 1] + shift_y, 0, h - 1)

    return centered, contour_shifted, (shift_x, shift_y)


def _vertical_delta_from_rect(rect):
    """
    Calcula o delta (graus) para alinhar o lado maior do retângulo ao eixo Y (vertical).
    Retorna também o desvio atual em relação ao eixo Y.
    """
    (w, h) = rect[1]
    angle = rect[2]  # ângulo do lado width em relação ao eixo X (OpenCV: [-90,0) )

    if w >= h:
        # lado maior é width, queremos esse lado apontando para o eixo Y (90°)
        deviation_from_y = angle - 90.0
        delta_to_vertical = -deviation_from_y  # 90 - angle
    else:
        # lado maior é height, que está em angle+90; queremos height em 90°
        deviation_from_y = (angle + 90.0) - 90.0  # = angle
        delta_to_vertical = -deviation_from_y  # -angle

    return delta_to_vertical, deviation_from_y


def _rotate_with_delta(img_bgr: np.ndarray, contour: np.ndarray, delta_deg: float):
    """Gira imagem/contorno pelo delta passado (graus)."""
    h, w = img_bgr.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -delta_deg/2, 1.0)

    rotated = cv2.warpAffine(
        img_bgr,
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    contour_rot = cv2.transform(contour, rot_mat).astype(np.int32)
    contour_rot[:, 0, 0] = np.clip(contour_rot[:, 0, 0], 0, w - 1)
    contour_rot[:, 0, 1] = np.clip(contour_rot[:, 0, 1], 0, h - 1)

    return rotated, contour_rot


def debug_one_image(img_path: Path):
    print(f"\n[INFO] Analisar: {img_path}")
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print("[ERRO] Não consegui ler a imagem.")
        return "n"

    contours, edges = _find_contours(img_bgr)
    num_contours = len(contours)
    print(f"[INFO] {num_contours} contornos encontrados.")

    # filtra contornos dentro da faixa de área alvo
    area_tol = TARGET_AREA * TARGET_AREA_TOL
    area_min = TARGET_AREA - area_tol
    area_max = TARGET_AREA + area_tol
    contours = [c for c in contours if area_min <= cv2.contourArea(c) <= area_max]
    num_contours = len(contours)
    found_target = num_contours > 0
    print(
        f"[INFO] Contornos na faixa [{area_min:.0f}, {area_max:.0f}] = "
        f"{num_contours} | found_target={found_target}"
    )

    if num_contours == 0:
        cv2.imshow("orig", img_bgr)
        cv2.imshow("edges", edges)
        # mostra imagem rotacionada em 90 com marcação de rejeição
        reject_img = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
        cv2.putText(
            reject_img,
            "Reject",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.imshow("rotated_reject", reject_img)

        print("[WARN] Nenhum contorno na faixa alvo. 'n' próxima imagem | 'q' sair")
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord("q"):
            return "q"
        return "n"

    # seleciona contorno mais próximo do centro e da área alvo
    h, w = img_bgr.shape[:2]
    img_center = np.array([w / 2.0, h / 2.0])

    def _score_contour(idx: int):
        c = contours[idx]
        area = cv2.contourArea(c)
        M = cv2.moments(c)
        if M["m00"] == 0:
            return float("inf")
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dist_center = np.linalg.norm(np.array([cx, cy]) - img_center)
        area_diff_rel = abs(area - TARGET_AREA) / max(TARGET_AREA, 1.0)
        # score pondera centro e área (área em escala 0-1000 para balancear)
        return TARGET_CENTER_WEIGHT * dist_center + TARGET_AREA_WEIGHT * area_diff_rel * 1000

    contour_idx = min(range(num_contours), key=_score_contour)
    best_area = cv2.contourArea(contours[contour_idx])
    area_ok = abs(best_area - TARGET_AREA) <= TARGET_AREA * TARGET_AREA_TOL
    print(
        f"[INFO] Melhor contorno inicial idx={contour_idx} area={best_area:.1f} "
        f"{'(area ok)' if area_ok else '(fora da tolerância)'}"
    )

    while True:
        # mostra original e edges sempre
        cv2.imshow("orig", img_bgr)
        cv2.imshow("edges", edges)

        # imagem para desenhar o contorno atual
        vis = img_bgr.copy()

        c = contours[contour_idx]
        # minAreaRect requer ao menos 3 pontos; se não tiver, segue para próximo
        if len(c) < 3:
            print(f"[WARN] Contorno {contour_idx} sem pontos suficientes ({len(c)}).")
            contour_idx = (contour_idx + 1) % num_contours
            
        # garante tipo/shape compatível
        c_proc = np.asarray(c, dtype=np.float32).reshape(-1, 1, 2)
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        rect = cv2.minAreaRect(c_proc)
        angle = rect[2]
        # normaliza angulo para faixa 0..180 (w<h -> soma 90) apenas para display
        if rect[1][0] < rect[1][1]:
            angle += 90.0
        delta_vertical, deviation_y = _vertical_delta_from_rect(rect)

        # desenha contorno atual bem grosso e vermelho
        cv2.drawContours(vis, [c], -1, (0, 0, 255), 2)

        # centroide do contorno
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(vis, (cx, cy), 5, (255, 0, 255), -1)  # magenta

        # texto com idx e área
        text = f"idx={contour_idx}/{num_contours-1}  area={area:.1f}  ang={angle:.1f}"
        if peri > 0:
            circ = 4.0 * np.pi * area / (peri * peri)
            text += f"  circ={circ:.3f}"
        cv2.putText(
            vis,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("contour_debug", vis)

        # imagem centralizada no contorno atual
        centered_img, centered_contour, (sx, sy) = _center_image_on_contour(img_bgr, c)
        centered_vis = centered_img.copy()
        cv2.drawContours(centered_vis, [centered_contour], -1, (0, 0, 255), 2)
        if M["m00"] != 0:
            cv2.circle(
                centered_vis,
                (int(cx + sx), int(cy + sy)),
                5,
                (255, 0, 255),
                -1,
            )
        cv2.putText(
            centered_vis,
            f"shift=({sx},{sy}) ang={angle:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("centered_on_contour", centered_vis)

        # rotaciona imagem para alinhar lado maior do contorno ao eixo Y
        rotated_img, rotated_contour = _rotate_with_delta(img_bgr, c, delta_vertical)
        rotated_vis = rotated_img.copy()
        cv2.drawContours(rotated_vis, [rotated_contour], -1, (0, 0, 255), 2)
        cv2.putText(
            rotated_vis,
            f"rot delta={delta_vertical:.1f} deg | desvioY={deviation_y:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("rotated_to_90", rotated_vis)

        print(
            f"[INFO] Contorno {contour_idx}/{num_contours-1} | "
            f"area={area:.1f}, peri={peri:.1f}, ang={angle:.1f}, desvioY={deviation_y:.1f}"
        )
        print("[INFO] Teclas: 'c' próximo contorno | 'n' próxima imagem | 'q' sair")

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            cv2.destroyAllWindows()
            return "q"

        elif key == ord("n"):
            cv2.destroyAllWindows()
            return "n"

        elif key == ord("c"):
            # próximo contorno (loop circular)
            contour_idx = (contour_idx + 1) % num_contours
            # não fechamos as janelas, só redesenhamos no próximo loop
            continue

        else:
            # qualquer outra tecla = ignora e espera outra
            continue


def main():
    """
    Uso:

      - Para uma pasta com várias latas:
          python -m tools.debug_can_contour data/cans/session_0001/sheet_0001

      - Para uma lata específica:
          python -m tools.debug_can_contour data/cans/session_0001/sheet_0001/can_01.png
    """
    if len(sys.argv) < 2:
        print("Uso: python -m tools.debug_can_contour <pasta_ou_imagem>")
        sys.exit(1)

    target = Path(sys.argv[1])

    if target.is_file():
        image_paths = [target]
    elif target.is_dir():
        image_paths = sorted(target.glob("*.png"))
        if not image_paths:
            print("[ERRO] Não encontrei imagens .png na pasta.")
            sys.exit(1)
    else:
        print("[ERRO] Caminho inválido.")
        sys.exit(1)

    for img_path in image_paths:
        cmd = debug_one_image(img_path)
        if cmd == "q":
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
