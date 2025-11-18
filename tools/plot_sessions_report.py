# tools/plot_sessions_report.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px


def load_sessions_report(metadata_dir: str | Path) -> pd.DataFrame:
    metadata_dir = Path(metadata_dir)
    csv_path = metadata_dir / "sessions_report.csv"

    if not csv_path.is_file():
        print(f"[ERRO] Ficheiro não encontrado: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # garantir colunas esperadas (caso falte alguma)
    expected_cols = [
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
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Faltam colunas no CSV: {missing} (vou continuar na mesma).")

    # converter types básicos
    if "tests_pass" in df.columns:
        df["tests_pass"] = df["tests_pass"].astype(int)

    return df


def plot_metric_vs_sheet(df: pd.DataFrame, metric: str, output_html: Path | None = None):
    """
    Cria um gráfico interativo de uma métrica vs sheet_index, colorido por tests_pass.
    """
    if metric not in df.columns:
        print(f"[WARN] Métrica '{metric}' não existe no dataframe.")
        return None

    # criar label mais amigável
    metric_labels = {
        "brightness": "Brightness (L médio)",
        "brightness_delta": "ΔL (target)",
        "lab_delta_L": "ΔL (template)",
        "lab_delta_ab": "Δab (template)",
    }
    y_label = metric_labels.get(metric, metric)

    fig = px.scatter(
        df,
        x="sheet_index",
        y=metric,
        color=df["tests_pass"].map({1: "OK", 0: "FAIL"}),
        symbol="session_id",
        hover_data=["session_id", "sheet_id", "sku", "operator", "notes"],
        title=f"{y_label} por folha",
        labels={
            "sheet_index": "Índice da folha (sheet_index)",
            "color": "Resultado testes",
            metric: y_label,
        },
    )

    fig.update_traces(marker=dict(size=10, opacity=0.8))
    fig.update_layout(
        legend_title_text="tests_pass",
        xaxis=dict(dtick=1),
        template="plotly_dark",
    )

    if output_html is not None:
        import plotly.io as pio

        pio.write_html(fig, file=str(output_html), auto_open=False)
        print(f"[OK] Gráfico '{metric}' guardado em: {output_html}")

    return fig


def main():
    # diretório base do projeto (onde está a pasta data/)
    project_root = Path(__file__).resolve().parents[1]
    metadata_dir = project_root / "data" / "metadata"

    print(f"[INFO] A ler sessions_report em: {metadata_dir}")
    df = load_sessions_report(metadata_dir)

    if df.empty:
        print("[WARN] DataFrame vazio. Nada para plotar.")
        return

    # ordenar por session_id + sheet_index para facilitar leitura
    df = df.sort_values(by=["session_id", "sheet_index"]).reset_index(drop=True)

    # diretório onde vamos guardar os HTMLs
    plots_dir = metadata_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) Brightness (L médio)
    plot_metric_vs_sheet(
        df,
        metric="brightness",
        output_html=plots_dir / "brightness_vs_sheet.html",
    )

    # 2) ΔL em relação ao target (brightness_delta)
    plot_metric_vs_sheet(
        df,
        metric="brightness_delta",
        output_html=plots_dir / "brightness_delta_vs_sheet.html",
    )

    # 3) ΔL vs template
    plot_metric_vs_sheet(
        df,
        metric="lab_delta_L",
        output_html=plots_dir / "lab_delta_L_vs_sheet.html",
    )

    # 4) Δab vs template
    plot_metric_vs_sheet(
        df,
        metric="lab_delta_ab",
        output_html=plots_dir / "lab_delta_ab_vs_sheet.html",
    )

    print("\n[OK] Plots gerados em:")
    print(f"  {plots_dir / 'brightness_vs_sheet.html'}")
    print(f"  {plots_dir / 'brightness_delta_vs_sheet.html'}")
    print(f"  {plots_dir / 'lab_delta_L_vs_sheet.html'}")
    print(f"  {plots_dir / 'lab_delta_ab_vs_sheet.html'}")
    print("\nAbre estes ficheiros no browser para veres os gráficos interativos.")


if __name__ == "__main__":
    main()
