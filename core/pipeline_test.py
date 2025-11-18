from pathlib import Path
import shutil
import numpy as np
import cv2

# --- MODELOS / PROCESSAMENTO ---
from models.can_detector import CanDetector          # <--- NOVO local do detector
from core.cropper import crop_cans_from_sheet
from core.normalize_image import normalize_image
from core.session_report import append_sheet_report_row
from core.can_preprocess import preprocess_can_image

from core.capture import list_sheet_images, load_sheet_image
from core.preprocess import normalize_sheet_image
from core.tests_quality import run_all_tests
from core.dataset_manager import (
    create_session,
    register_sheet,
    register_cans_for_sheet,
)

# NOTA:
# Removido:
#   from models.weights.can_detector import CanDetector
#   from models.weights.can_detector import detect_cans_on_sheet
# E removido código inválido no topo:
#   detector = CanDetector(...)
#   detections = detector.detect(sheet_img_bgr)

# perfis/configs mínimos (ajustáveis)
NORMALIZATION_PROFILE = {
    "size": 256,
    "target_v": 180.0,
}

TESTS_CONFIG = {
    "max_lab_delta_L": 15.0,
    "max_lab_delta_ab": 20.0,
    "target_brightness": 180.0,
    "max_brightness_delta": 60.0,
}

DETECTOR_CONFIG = {
    "weights_path": "models/weights/can_detector.pt",
    "conf_thres": 0.25,
    "iou_thres": 0.45,
    "device": "cpu",
}

# tamanho final das latas salvas (pode ser alterado pela UI)
CAN_OUTPUT_SIZE = 256

MODEL_CONFIG = {}  # já não é usado neste ficheiro, mas podes manter para futuro

DATA_DIR = Path("data")
RAW_SHEETS_DIR = DATA_DIR / "raw_sheets"
NORM_SHEETS_DIR = DATA_DIR / "normalized_sheets"
CANS_BASE_DIR = DATA_DIR / "cans"
METADATA_DIR = DATA_DIR / "metadata"


def process_session_from_folder(input_folder: str, sku: str, operator: str, notes: str = "") -> dict:
    input_folder = Path(input_folder)
    RAW_SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    NORM_SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    CANS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- instanciar o modelo de deteção de latas (YOLO) ---
    can_detector = CanDetector(
        weights_path=DETECTOR_CONFIG.get("weights_path", "models/weights/can_detector.pt"),
        conf_thres=float(DETECTOR_CONFIG.get("conf_thres", 0.25)),
        iou_thres=float(DETECTOR_CONFIG.get("iou_thres", 0.45)),
        device=str(DETECTOR_CONFIG.get("device", "cpu")),
    )

    summary = {
        "session_id": None,
        "sheets_total": 0,
        "sheets_passed": 0,
        "sheets_failed": 0,
        "cans_saved": 0,
    }

    # 1) criar sessão
    session_id = create_session(
        sku=sku,
        operator=operator,
        notes=notes,
        metadata_dir=METADATA_DIR,
    )
    summary["session_id"] = session_id
    print(f"[INFO] Criada sessão: {session_id}")

    # 2) listar folhas
    sheet_paths = list_sheet_images(input_folder)
    print(f"[INFO] Encontradas {len(sheet_paths)} folhas em {input_folder}")

    # 3) carregar template fake (por agora pode ser a primeira folha)
    template_img_norm = None

    for idx, sheet_path in enumerate(sheet_paths, start=1):
        summary["sheets_total"] += 1
        print(f"\n[INFO] Processar folha {idx}: {sheet_path}")

        img_bgr = load_sheet_image(sheet_path)

        # Normalizar folha (para testes + para deteção, por agora)
        sheet_norm = normalize_sheet_image(img_bgr, NORMALIZATION_PROFILE)

        # Inicializar template na primeira folha (dummy)
        if template_img_norm is None:
            template_img_norm = sheet_norm.copy()

        # Correr testes de qualidade
        tests_pass, tests_details = run_all_tests(sheet_norm, template_img_norm, TESTS_CONFIG)

        # Se vier em float [0,1], convertemos para uint8
        if sheet_norm.dtype != np.uint8 or sheet_norm.max() <= 1.0:
            sheet_norm_to_save = (sheet_norm * 255.0).clip(0, 255).astype(np.uint8)
        else:
            sheet_norm_to_save = sheet_norm
            
        # Guardar imagem normalizada
        sheet_id_str = f"sheet_{idx:04d}"
        norm_out_path = NORM_SHEETS_DIR / f"{session_id}_{sheet_id_str}.png"
        cv2.imwrite(str(norm_out_path), sheet_norm_to_save)

        raw_copy_path = RAW_SHEETS_DIR / f"{session_id}_{sheet_id_str}.png"
        # por agora copia só a original
        shutil.copy2(sheet_path, raw_copy_path)

        # Registar folha
        sheet_id = register_sheet(
            session_id=session_id,
            sheet_index=idx,
            tests_pass=tests_pass,
            tests_details=tests_details,
            raw_path=str(raw_copy_path),
            norm_path=str(norm_out_path),
            metadata_dir=METADATA_DIR,
        )
        print(f"[INFO] Registada folha como {sheet_id} (tests_pass={tests_pass})")

        # --- Atualizar relatório global de sessões (CSV) ---
        append_sheet_report_row(
            metadata_dir=METADATA_DIR,
            session_id=session_id,
            sheet_id=sheet_id,
            sheet_index=idx,
            sku=sku,
            operator=operator,
            notes=notes,
            tests_pass=tests_pass,
            tests_details=tests_details,
            raw_path=str(raw_copy_path),
            norm_path=str(norm_out_path),
        )

        if not tests_pass:
            print("[WARN] Folha reprovada. A detecção/crop de latas será ignorada.")
            summary["sheets_failed"] += 1
            # mesmo assim registamos a folha nos metadados
            register_sheet(
                session_id=session_id,
                sheet_index=idx,
                tests_pass=tests_pass,
                tests_details=tests_details,
                raw_path=str(raw_copy_path),
                norm_path=str(norm_out_path),
                metadata_dir=METADATA_DIR,
            )
            append_sheet_report_row(
                metadata_dir=METADATA_DIR,
                session_id=session_id,
                sheet_id=f"sheet_{idx:04d}",
                sheet_index=idx,
                sku=sku,
                operator=operator,
                notes=notes,
                tests_pass=tests_pass,
                tests_details=tests_details,
                raw_path=str(raw_copy_path),
                norm_path=str(norm_out_path),
            )
            continue
        else:
            summary["sheets_passed"] += 1

        # ------------------------------------------------------------------
        # Detetar latas com o modelo real (YOLO)
        # ------------------------------------------------------------------
        detections = can_detector.detect(img_bgr)
        print(f"[INFO] Detetadas {len(detections)} latas (modelo YOLO).")

        # Atribuir can_id sequencial por folha
        for i, det in enumerate(detections, start=1):
            det["can_id"] = i

        # ------------------------------------------------------------------
        # Crop das latas (a partir da folha normalizada)
        # ------------------------------------------------------------------
        crop_size = int(CAN_OUTPUT_SIZE)
        cans_crops = crop_cans_from_sheet(img_bgr, detections, mask_template=None, size=crop_size)


        # ------------------------------------------------------------------
        # Guardar cada lata (opcional: voltar a normalizar com normalize_image)
        # ------------------------------------------------------------------
        session_cans_dir = CANS_BASE_DIR / session_id / sheet_id
        session_cans_dir.mkdir(parents=True, exist_ok=True)

        cans_info = []
        for can_res in cans_crops:
            can_id = can_res["can_id"]
            can_img = can_res["image"]  # BGR 256x256 do cropper

            # alinhar + centrar pela label, sem máscara
            can_aligned, can_mask, info_align = preprocess_can_image(can_img)
            if can_aligned is None:
                print(f"[WARN] Lata {can_id} sem contorno alvo -> ignorada")
                continue

            # normalizar para o dataset usando tamanho definido (evitar múltiplos reescalonamentos)
            target_size = int(CAN_OUTPUT_SIZE)
            # já cortámos para target_size; normalizar mantendo esse tamanho
            can_img_norm = normalize_image(can_aligned, size=target_size)
            can_img_u8 = (can_img_norm * 255.0).clip(0, 255).astype(np.uint8)

            # salvar em PNG (sem perdas)
            can_filename = f"can_{can_id:02d}.png"
            can_path = session_cans_dir / can_filename
            cv2.imwrite(str(can_path), can_img_u8)

            cans_info.append(
                {
                    "can_id": can_id,
                    "image_path": str(can_path),
                    "quality_ok": tests_pass,
                    "brightness_score": float(tests_details.get("brightness", 0.0)),
                    "lab_delta_L": float(tests_details.get("lab_delta_L", 0.0)),
                    "lab_delta_ab": float(tests_details.get("lab_delta_ab", 0.0)),
                    "rotation_score": float(tests_details.get("rotation", 0.0)),
                    "centering_score": float(tests_details.get("centering", 0.0)),
                    # Debug extra útil:
                    "label_found": bool(info_align.get("label_found", False)),
                    "label_score": float(info_align.get("label_score", 0.0)),
                    "label_angle_deg": float(info_align.get("angle_deg", 0.0)),
                }
            )

        register_cans_for_sheet(
            session_id=session_id,
            sheet_id=sheet_id,
            cans_info=cans_info,
            metadata_dir=METADATA_DIR,
        )

        summary["cans_saved"] += len(cans_info)
        print(f"[INFO] Registadas {len(cans_info)} latas para {sheet_id}")

    print(f"\n[OK] Sessão completa: {session_id}")
    return summary


if __name__ == "__main__":
    # ajusta o input_folder para uma pasta onde tenhas 1–2 imagens de teste
    process_session_from_folder(
        input_folder="data/test_input_sheets",
        sku="SKU_TEST",
        operator="Vitor",
        notes="MVP pipeline",
    )
