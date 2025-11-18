# tools/export_dataset_for_colab.py

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pandas as pd


def main():
    # raiz do projeto (onde está a pasta data/)
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    metadata_dir = data_dir / "metadata"
    cans_base_dir = data_dir / "cans"

    report_path = metadata_dir / "sessions_report.csv"
    if not report_path.is_file():
        print(f"[ERRO] sessions_report.csv não encontrado em: {report_path}")
        sys.exit(1)

    print(f"[INFO] A ler relatório: {report_path}")
    df = pd.read_csv(report_path)

    if df.empty:
        print("[WARN] sessions_report.csv está vazio, nada para exportar.")
        sys.exit(0)

    # garantir coluna tests_pass como int
    if "tests_pass" not in df.columns:
        print("[ERRO] Coluna 'tests_pass' não existe no CSV.")
        sys.exit(1)

    df["tests_pass"] = df["tests_pass"].astype(int)

    # filtramos só folhas aprovadas
    good_sheets = df[df["tests_pass"] == 1].copy()

    if good_sheets.empty:
        print("[WARN] Não há folhas com tests_pass == 1. Nada para exportar.")
        sys.exit(0)

    print(f"[INFO] Folhas aprovadas para export: {len(good_sheets)}")

    # pasta de saída do dataset
    export_dir = data_dir / "export" / "cans_good"
    export_dir.mkdir(parents=True, exist_ok=True)

    # CSV de saída com metadados por lata
    export_csv_path = export_dir / "cans_good_meta.csv"
    rows = []

    # para cada folha aprovada, vamos buscar as latas em data/cans/session_id/sheet_id
    for _, row in good_sheets.iterrows():
        session_id = str(row["session_id"])
        sheet_id = str(row["sheet_id"])
        sheet_index = int(row["sheet_index"])
        sku = str(row.get("sku", ""))
        operator = str(row.get("operator", ""))
        notes = str(row.get("notes", ""))

        brightness = float(row.get("brightness", 0.0))
        brightness_delta = float(row.get("brightness_delta", 0.0))
        lab_delta_L = float(row.get("lab_delta_L", 0.0))
        lab_delta_ab = float(row.get("lab_delta_ab", 0.0))

        cans_dir = cans_base_dir / session_id / sheet_id
        if not cans_dir.is_dir():
            print(f"[WARN] Diretório de latas não encontrado: {cans_dir}")
            continue

        can_files = sorted(cans_dir.glob("can_*.png"))
        if not can_files:
            print(f"[WARN] Nenhuma lata encontrada em: {cans_dir}")
            continue

        print(f"[INFO] Exportar {len(can_files)} latas de {session_id}/{sheet_id}")

        for can_path in can_files:
            # extrair can_id do nome (assumindo can_XX.png ou can_XXX.png)
            stem = can_path.stem  # ex: "can_01"
            try:
                can_id_str = stem.split("_")[1]
                can_id = int(can_id_str)
            except Exception:
                can_id = -1

            # destino no dataset exportado
            # (renomeamos para incluir sessão+folha+lata -> único)
            dst_name = f"{session_id}_{sheet_id}_can_{can_id:03d}.png"
            dst_path = export_dir / dst_name

            shutil.copy2(can_path, dst_path)

            # caminho relativo (para usar no Colab)
            image_rel = dst_path.relative_to(project_root).as_posix()

            rows.append(
                {
                    "session_id": session_id,
                    "sheet_id": sheet_id,
                    "sheet_index": sheet_index,
                    "sku": sku,
                    "operator": operator,
                    "notes": notes,
                    "can_id": can_id,
                    "image_path": image_rel,
                    "brightness": brightness,
                    "brightness_delta": brightness_delta,
                    "lab_delta_L": lab_delta_L,
                    "lab_delta_ab": lab_delta_ab,
                }
            )

    if not rows:
        print("[WARN] Nenhuma lata exportada (lista de rows vazia).")
        sys.exit(0)

    export_df = pd.DataFrame(rows)
    export_df.to_csv(export_csv_path, index=False, encoding="utf-8")
    print(f"[OK] Exportadas {len(rows)} latas para: {export_dir}")
    print(f"[OK] Metadados guardados em: {export_csv_path}")


if __name__ == "__main__":
    main()
