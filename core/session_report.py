# core/session_report.py
from __future__ import annotations

from pathlib import Path
from typing import Dict
import csv


REPORT_FILENAME = "sessions_report.csv"


def append_sheet_report_row(
    metadata_dir: Path,
    session_id: str,
    sheet_id: str,
    sheet_index: int,
    sku: str,
    operator: str,
    notes: str,
    tests_pass: bool,
    tests_details: Dict[str, float],
    raw_path: str,
    norm_path: str,
) -> None:
    """
    Acrescenta (ou cria) uma linha no ficheiro sessions_report.csv com info
    de qualidade da folha (brightness, ΔL, Δab, etc).
    """
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    report_path = metadata_dir / REPORT_FILENAME

    # colunas fixas do relatório
    fieldnames = [
        "session_id",
        "sheet_id",
        "sheet_index",
        "sku",
        "operator",
        "notes",
        "tests_pass",
        "brightness",
        "brightness_delta",
        "lab_delta_L",
        "lab_delta_ab",
        "rotation",
        "centering",
        "raw_path",
        "norm_path",
    ]

    # determinar se precisamos de escrever o header
    write_header = not report_path.exists()

    with report_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        row = {
            "session_id": session_id,
            "sheet_id": sheet_id,
            "sheet_index": sheet_index,
            "sku": sku,
            "operator": operator,
            "notes": notes,
            "tests_pass": int(bool(tests_pass)),  # 1/0 para facilitar filtros
            "brightness": float(tests_details.get("brightness", 0.0)),
            "brightness_delta": float(tests_details.get("brightness_delta", 0.0)),
            "lab_delta_L": float(tests_details.get("lab_delta_L", 0.0)),
            "lab_delta_ab": float(tests_details.get("lab_delta_ab", 0.0)),
            "rotation": float(tests_details.get("rotation", 0.0)),
            "centering": float(tests_details.get("centering", 0.0)),
            "raw_path": raw_path,
            "norm_path": norm_path,
        }

        writer.writerow(row)
