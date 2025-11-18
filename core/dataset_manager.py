# core/dataset_manager.py

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any
import csv
from datetime import datetime

from .types import PathLike


def _ensure_metadata_dir(metadata_dir: PathLike) -> Path:
    md = Path(metadata_dir)
    md.mkdir(parents=True, exist_ok=True)
    return md


def _sessions_csv_path(metadata_dir: PathLike) -> Path:
    return _ensure_metadata_dir(metadata_dir) / "sessions.csv"


def _cans_csv_path(metadata_dir: PathLike) -> Path:
    return _ensure_metadata_dir(metadata_dir) / "cans.csv"


def _filter_csv(path: Path, keep_fn):
    """
    Helper: lê CSV, mantém linhas para as quais keep_fn(row_dict) é True,
    reescreve o ficheiro (preserva header).
    """
    if not path.exists():
        return
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        fieldnames = reader[0].keys() if reader else []
    rows_to_keep = [r for r in reader if keep_fn(r)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_to_keep:
            writer.writerow(r)


# ---------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------
def create_session(
    sku: str,
    operator: str,
    notes: str = "",
    metadata_dir: PathLike = "data/metadata",
) -> str:
    """
    Cria uma nova sessão no sessions.csv e devolve session_id (ex: 'session_0001').
    MVP: não atualiza num_sheets ainda, apenas cria o registo da sessão.
    """
    sessions_path = _sessions_csv_path(metadata_dir)

    # Descobrir próximo session_id
    next_idx = 1
    if sessions_path.exists():
        with sessions_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("session_id", "")
                if sid.startswith("session_"):
                    try:
                        idx = int(sid.split("_")[1])
                        next_idx = max(next_idx, idx + 1)
                    except ValueError:
                        pass

    session_id = f"session_{next_idx:04d}"
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Escrever / append no sessions.csv
    write_header = not sessions_path.exists()
    with sessions_path.open("a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "session_id",
            "created_at",
            "sku",
            "operator",
            "notes",
            "num_sheets",
            "num_sheets_pass",
            "num_sheets_fail",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        writer.writerow(
            {
                "session_id": session_id,
                "created_at": created_at,
                "sku": sku,
                "operator": operator,
                "notes": notes,
                "num_sheets": 0,
                "num_sheets_pass": 0,
                "num_sheets_fail": 0,
            }
        )


def delete_session(
    session_id: str,
    metadata_dir: PathLike = "data/metadata",
    delete_images: bool = False,
) -> None:
    """
    Remove registos de uma sessão (sessions.csv e cans.csv).
    Se delete_images=True, tenta apagar ficheiros de imagem referenciados dessa sessão.
    """
    md = _ensure_metadata_dir(metadata_dir)
    sessions_path = _sessions_csv_path(md)
    cans_path = _cans_csv_path(md)

    # remover da sessions.csv
    def keep_session(row):
        return row.get("session_id") != session_id

    _filter_csv(sessions_path, keep_session)

    # remover registos das latas e opcionalmente imagens
    image_paths_to_delete = []
    if cans_path.exists():
        with cans_path.open("r", newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            fieldnames = reader[0].keys() if reader else []
        rows_to_keep = []
        for r in reader:
            if r.get("session_id") != session_id:
                rows_to_keep.append(r)
            else:
                if delete_images and r.get("image_path"):
                    image_paths_to_delete.append(r["image_path"])
        with cans_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_to_keep)

    if delete_images:
        for img_path in image_paths_to_delete:
            try:
                Path(img_path).unlink(missing_ok=True)
            except Exception:
                # ignora falhas individuais
                pass
    return session_id


# ---------------------------------------------------------------------
# register_sheet
# ---------------------------------------------------------------------
def register_sheet(
    session_id: str,
    sheet_index: int,
    tests_pass: bool,
    tests_details: Dict[str, Any],
    raw_path: Optional[str],
    norm_path: str,
    metadata_dir: PathLike = "data/metadata",
) -> str:
    """
    Regista uma folha associada a uma sessão.
    MVP: apenas devolve um sheet_id coerente. Não guarda info num ficheiro separado,
    porque por enquanto sessões/folhas são registadas indiretamente via cans.csv.
    """
    sheet_id = f"sheet_{sheet_index:04d}"
    # No futuro podemos ter um sheets.csv, aqui não é obrigatório.
    # tests_pass e tests_details são usados depois nas latas.
    return sheet_id


# ---------------------------------------------------------------------
# register_cans_for_sheet
# ---------------------------------------------------------------------
def register_cans_for_sheet(
    session_id: str,
    sheet_id: str,
    cans_info: List[Dict[str, Any]],
    metadata_dir: PathLike = "data/metadata",
) -> None:
    """
    Regista as latas de uma folha no cans.csv.

    Cada elemento de cans_info deve conter pelo menos:
      - can_id (int)
      - image_path (str)
      - quality_ok (bool)
      - brightness_score, lab_delta_L, lab_delta_ab, rotation_score, centering_score (floats)
    """
    cans_path = _cans_csv_path(metadata_dir)

    write_header = not cans_path.exists()
    fieldnames = [
        "session_id",
        "sheet_id",
        "can_id",
        "image_path",
        "mask_path",
        "label",
        "quality_ok",
        "brightness_score",
        "lab_delta_L",
        "lab_delta_ab",
        "rotation_score",
        "centering_score",
        "created_at",
        "notes",
    ]

    with cans_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for info in cans_info:
            writer.writerow(
                {
                    "session_id": session_id,
                    "sheet_id": sheet_id,
                    "can_id": info.get("can_id"),
                    "image_path": info.get("image_path"),
                    "mask_path": info.get("mask_path", ""),
                    "label": "unknown",  # inicialmente sem curadoria
                    "quality_ok": info.get("quality_ok", True),
                    "brightness_score": info.get("brightness_score", 0.0),
                    "lab_delta_L": info.get("lab_delta_L", 0.0),
                    "lab_delta_ab": info.get("lab_delta_ab", 0.0),
                    "rotation_score": info.get("rotation_score", 0.0),
                    "centering_score": info.get("centering_score", 0.0),
                    "created_at": created_at,
                    "notes": "",
                }
            )
