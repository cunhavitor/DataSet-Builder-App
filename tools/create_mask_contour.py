from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# --------- PARAMETROS DO CANNY / MORFO ---------
CANNY_LOW = 30
CANNY_HIGH = 140
GAUSS_KSIZE = 5          # blur (impar)
DILATE_ITERS = 2
CLOSE_ITERS = 2


def _find_contours(img_bgr: np.ndarray):
    """Retorna contornos do maior para o menor + edges para debug."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (GAUSS_KSIZE, GAUSS_KSIZE), 0)

    edges = cv2.Canny(gray_blur, CANNY_LOW, CANNY_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=DILATE_ITERS)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERS)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)

    return contours_sorted, edges


def _save_mask(img_shape, contour, mask_path: Path):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    cv2.imwrite(str(mask_path), mask)
    return mask_path


def debug_one_image(img_path: Path):
    print(f"\n[INFO] Analisar: {img_path}")
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print("[ERRO] Nao consegui ler a imagem.")
        return "n"

    contours, edges = _find_contours(img_bgr)
    num_contours = len(contours)
    print(f"[INFO] {num_contours} contornos encontrados (ordenados por area desc).")

    if num_contours == 0:
        cv2.imshow("orig", img_bgr)
        cv2.imshow("edges", edges)
        print("[WARN] Sem contornos. 'n' proxima imagem | 'q' sair")
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord("q"):
            return "q"
        return "n"

    contour_idx = 0

    while True:
        cv2.imshow("orig", img_bgr)
        cv2.imshow("edges", edges)

        vis = img_bgr.copy()
        c = contours[contour_idx]
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)

        cv2.drawContours(vis, [c], -1, (0, 0, 255), 2)

        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(vis, (cx, cy), 5, (255, 0, 255), -1)

        text = f"idx={contour_idx}/{num_contours-1}  area={area:.1f}"
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

        print(
            f"[INFO] Contorno {contour_idx}/{num_contours-1} | "
            f"area={area:.1f}, peri={peri:.1f}"
        )
        print("[INFO] Teclas: 'c'=proximo contorno | 'm'=gerar mascara | 'n'=proxima imagem | 'q'=sair")

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            cv2.destroyAllWindows()
            return "q"
        elif key == ord("n"):
            cv2.destroyAllWindows()
            return "n"
        elif key == ord("c"):
            contour_idx = (contour_idx + 1) % num_contours
            continue
        elif key == ord("m"):
            mask_path = img_path.with_name(f"{img_path.stem}_mask.png")
            _save_mask(img_bgr.shape, c, mask_path)
            print(f"[INFO] Mascara salva em {mask_path}")
            continue
        else:
            continue


def main():
    """
    Uso:
      python -m tools.create_mask_contour <pasta_ou_imagem>
    """
    if len(sys.argv) < 2:
        print("Uso: python -m tools.create_mask_contour <pasta_ou_imagem>")
        sys.exit(1)

    target = Path(sys.argv[1])

    if target.is_file():
        image_paths = [target]
    elif target.is_dir():
        image_paths = sorted(target.glob("*.png"))
        if not image_paths:
            print("[ERRO] Nao encontrei imagens .png na pasta.")
            sys.exit(1)
    else:
        print("[ERRO] Caminho invalido.")
        sys.exit(1)

    for img_path in image_paths:
        cmd = debug_one_image(img_path)
        if cmd == "q":
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
