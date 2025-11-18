"""
models/can_detector.py

Wrapper para o modelo YOLO de deteção de latas.

- Carrega o ficheiro de pesos can_detector.pt em models/weights/can_detector.pt
- Expõe a classe CanDetector com o método .detect(img_bgr)
- Saída: lista de dicts com "bbox", "confidence", "class_id"
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Union, Optional

import cv2
import numpy as np
from ultralytics import YOLO


class CanDetector:
    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        conf_thres: float = 0.10,   # antes 0.25
        iou_thres: float = 0.45,
        device: str = "cpu",
    ) -> None:
        """
        :param weights_path: caminho para o .pt; se None, assume models/weights/can_detector.pt
        :param conf_thres: limiar mínimo de confiança
        :param iou_thres: IoU threshold para NMS
        :param device: "cpu" ou "cuda"
        """
        if weights_path is None:
            weights_path = (
                Path(__file__).resolve().parent / "weights" / "can_detector.pt"
            )

        self.weights_path = Path(weights_path)
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"Pesos do detector não encontrados em: {self.weights_path}"
            )

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device

        # Carrega modelo YOLO
        self.model = YOLO(str(self.weights_path))

    @staticmethod
    def _ensure_bgr_image(img: np.ndarray) -> np.ndarray:
        if img is None:
            raise ValueError("Input image is None.")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected BGR image with shape (H, W, 3), got {img.shape}")
        return img

    def detect(self, img_bgr: np.ndarray) -> List[Dict]:
        """
        Deteta latas numa imagem BGR (folha completa).

        Returns
        -------
        detections : List[Dict]
            Cada dict tem:
            - "bbox": (x, y, w, h) em coordenadas de imagem
            - "confidence": float
            - "class_id": int (id de classe YOLO, se existir)
        """
        img_bgr = self._ensure_bgr_image(img_bgr)

        # Chamada ao modelo YOLO
        results = self.model(
            img_bgr,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
        )[0]

        # DEBUG: ver quantas boxes o modelo produziu
        num_boxes = 0 if results.boxes is None else len(results.boxes)
        print(f"[DEBUG] YOLO: {num_boxes} boxes brutas antes de filtragem.")

        detections: List[Dict] = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w = x2 - x1
            h = y2 - y1

            conf = float(box.conf[0].cpu().item())
            cls_id = int(box.cls[0].cpu().item()) if box.cls is not None else -1

            detections.append(
                {
                    "bbox": (int(x1), int(y1), int(w), int(h)),
                    "confidence": conf,
                    "class_id": cls_id,
                }
            )

        # DEBUG: mostrar top 5 por confiança
        detections_sorted = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        for d in detections_sorted[:5]:
            print(
                f"[DEBUG] box conf={d['confidence']:.3f}, bbox={d['bbox']}, class_id={d['class_id']}"
            )

        return detections
