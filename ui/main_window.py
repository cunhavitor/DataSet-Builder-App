from __future__ import annotations

import sys
import traceback
from pathlib import Path
import csv
import json
import cv2
from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import Qt

from ui.session_window import SessionWindow
from ui.review_window import ReviewWindow
from ui.export_window import ExportWindow
from ui.calibration_window import CalibrationWindow
from core import pipeline_test
from core import can_preprocess

APP_STYLE = """
QMainWindow {
    background: #0f141a;
    color: #e0e6ed;
}
QWidget {
    background: #0f141a;
    color: #e0e6ed;
}
QTabWidget::pane {
    border: 1px solid #22303c;
    background: #0f141a;
}
QTabBar::tab {
    background: #18222d;
    color: #e0e6ed;
    padding: 8px 14px;
    margin: 2px;
    border: 1px solid #22303c;
    border-radius: 4px;
}
QTabBar::tab:selected {
    background: #1f2a36;
    color: #6ab0ff;
}
QPushButton {
    background-color: #1f2a36;
    color: #e0e6ed;
    border: 1px solid #276fbf;
    padding: 6px 12px;
    border-radius: 4px;
}
QPushButton:hover { background-color: #233041; }
QPushButton:pressed { background-color: #1a2431; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QTextBrowser {
    background: #1a2029;
    color: #e0e6ed;
    border: 1px solid #22303c;
    border-radius: 4px;
    padding: 4px;
}
QLabel {
    color: #e0e6ed;
}
QGroupBox {
    border: 1px solid #22303c;
    border-radius: 4px;
    margin-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QScrollArea {
    background: #0f141a;
}
QScrollArea > QWidget > QWidget {
    background: #0f141a;
}
QTextEdit#logWidget {
    background: #0c1117;
    color: #a7b6c7;
    border: 1px solid #22303c;
}
QStatusBar {
    background: #0c1117;
    color: #e0e6ed;
}
"""


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DataSet Builder")
        self.resize(1100, 720)
        self.worker: QtCore.QThread | None = None

        self.tabs = QtWidgets.QTabWidget()
        self.session_tab = SessionWindow()
        self.review_tab = ReviewWindow()
        self.export_tab = ExportWindow()
        self.calibration_tab = CalibrationWindow()

        self.tabs.addTab(self.session_tab, "SessÃ£o")
        self.tabs.addTab(self.review_tab, "RevisÃ£o")
        self.tabs.addTab(self.export_tab, "Exportar")
        self.tabs.addTab(self.calibration_tab, "CalibraÃ§Ã£o")

        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage("Pronto")

        # dock de log persistente
        self.log_widget = QtWidgets.QTextEdit()
        self.log_widget.setObjectName("logWidget")
        self.log_widget.setReadOnly(True)
        dock = QtWidgets.QDockWidget("Log", self)
        dock.setWidget(self.log_widget)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

        self._connect_signals()

    def _connect_signals(self) -> None:
        # liga sinais das tabs para status/log
        for tab in [self.session_tab, self.review_tab, self.export_tab, self.calibration_tab]:
            tab.status_message.connect(self.statusBar().showMessage)
            tab.status_message.connect(self._append_log)
        self.session_tab.start_processing.connect(self._run_session)
        self.export_tab.start_export.connect(self._run_export)
        self.export_tab.start_export.connect(self._run_export)

    def _append_log(self, msg: str) -> None:
        self.log_widget.append(msg)

    # ----------------- processamento da sessÃ£o -----------------
    def _run_session(self, cfg: dict) -> None:
        if self.worker is not None:
            self._append_log("JÃ¡ existe um processamento em curso.")
            return

        # aplicar configs aos mÃ³dulos globais
        pipeline_test.NORMALIZATION_PROFILE = {
            "size": int(cfg["normalize"]["size"]),
            "target_v": float(cfg["normalize"]["target_v"]),
        }
        pipeline_test.TESTS_CONFIG = cfg["tests"]
        pipeline_test.DETECTOR_CONFIG.update(
            {
                "weights_path": cfg["detector"]["weights"],
                "conf_thres": cfg["detector"]["conf"],
                "iou_thres": cfg["detector"]["iou"],
                "device": cfg["detector"]["device"],
            }
        )
        pipeline_test.CAN_OUTPUT_SIZE = int(cfg["normalize"]["size"])
        can_preprocess.CANNY_LOW = int(cfg["preprocess"]["canny_low"])
        can_preprocess.CANNY_HIGH = int(cfg["preprocess"]["canny_high"])
        can_preprocess.GAUSS_KSIZE = int(cfg["preprocess"]["gauss"])
        can_preprocess.DILATE_ITERS = int(cfg["preprocess"]["dilate"])
        can_preprocess.CLOSE_ITERS = int(cfg["preprocess"]["close"])
        can_preprocess.TARGET_AREA = float(cfg["preprocess"]["target_area"])
        can_preprocess.TARGET_AREA_TOL = float(cfg["preprocess"]["target_tol"])
        can_preprocess.TARGET_CENTER_WEIGHT = float(cfg["preprocess"]["center_w"])
        can_preprocess.TARGET_AREA_WEIGHT = float(cfg["preprocess"]["area_w"])
        try:
            can_preprocess.MASK_BORDER_PAD = int(cfg["preprocess"]["mask_pad"])
        except Exception:
            pass

        input_folder = cfg["paths"]["input"]
        sku = cfg["session"]["sku"]
        operator = cfg["session"]["operator"]
        notes = cfg["session"]["notes"]

        self.thread = QtCore.QThread()
        self.worker = _SessionWorker(input_folder, sku, operator, notes)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_worker_done)
        self.worker.failed.connect(self._on_worker_fail)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.statusBar().showMessage("Processamento em curso...")
        self.thread.start()

    def _on_worker_done(self, session_id: str):
        self._append_log(f"[OK] SessÃ£o completa: {session_id}")
        self.statusBar().showMessage(f"SessÃ£o completa: {session_id}")
        self.worker = None

    def _on_worker_fail(self, msg: str):
        self._append_log(f"[ERRO] {msg}")
        self.statusBar().showMessage("Erro no processamento")
        self.worker = None

    # ----------------- exportação -----------------
    def _run_export(self, cfg: dict) -> None:
        try:
            stats = self._do_export(cfg)
            self._append_log(
                f"[EXPORT] {stats['exported']} imagens para {cfg['dir']} "
                f"(tag={cfg['tag_filter']}, sessions={cfg['sessions'] or 'todas'})"
            )
            self.statusBar().showMessage(f"Export concluída ({stats['exported']} imagens)")
        except Exception as e:
            tb = traceback.format_exc()
            self._append_log(f"[ERRO] Exportação falhou: {e}\n{tb}")
            self.statusBar().showMessage("Erro na exportação")

    def _do_export(self, cfg: dict) -> dict:
        cans_path = Path("data/metadata/cans.csv")
        if not cans_path.exists():
            raise FileNotFoundError("data/metadata/cans.csv não encontrado")

        sessions_filter = {s.strip() for s in cfg.get("sessions", "").split(",") if s.strip()}
        tag_filter = cfg.get("tag_filter", "todos")
        only_pass = bool(cfg.get("only_pass", False))
        size = int(cfg.get("size", 256))
        fmt = cfg.get("format", "png").lower()

        out_dir = Path(cfg["dir"])
        (out_dir / "train").mkdir(parents=True, exist_ok=True)

        exported = 0
        manifest_rows = []

        with cans_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if sessions_filter and row.get("session_id") not in sessions_filter:
                    continue
                label = row.get("label", "unknown")
                if tag_filter != "todos" and label != tag_filter:
                    continue
                if only_pass:
                    quality_ok = str(row.get("quality_ok", "True")).lower() in ("true", "1", "yes")
                    if not quality_ok:
                        continue
                src = Path(row.get("image_path", ""))
                if not src.exists():
                    continue

                dest_dir = out_dir / "train" / label
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_name = f"{row.get('session_id')}_{row.get('sheet_id')}_{row.get('can_id')}.{fmt}"
                dest_path = dest_dir / dest_name

                img = cv2.imread(str(src))
                if img is None:
                    continue
                if size and (img.shape[0] != size or img.shape[1] != size):
                    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                if fmt == "png":
                    cv2.imwrite(str(dest_path), img)
                else:
                    cv2.imwrite(str(dest_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                exported += 1
                manifest_rows.append(
                    {
                        "path": str(dest_path),
                        "session_id": row.get("session_id"),
                        "sheet_id": row.get("sheet_id"),
                        "can_id": row.get("can_id"),
                        "label": label,
                        "quality_ok": row.get("quality_ok"),
                        "brightness_score": row.get("brightness_score"),
                        "lab_delta_L": row.get("lab_delta_L"),
                        "lab_delta_ab": row.get("lab_delta_ab"),
                    }
                )

        if cfg.get("include_manifest", True):
            manifest_path = out_dir / "train_manifest.json"
            manifest_path.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
            csv_path = out_dir / "train_manifest.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                if manifest_rows:
                    fieldnames = manifest_rows[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(manifest_rows)

        return {"exported": exported, "output": str(out_dir)}


class _SessionWorker(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, input_folder: str, sku: str, operator: str, notes: str) -> None:
        super().__init__()
        self.input_folder = input_folder
        self.sku = sku
        self.operator = operator
        self.notes = notes

    @QtCore.Slot()
    def run(self):
        try:
            session_id = pipeline_test.process_session_from_folder(
                input_folder=self.input_folder,
                sku=self.sku,
                operator=self.operator,
                notes=self.notes,
            )
            self.finished.emit(session_id)
        except Exception as e:
            tb = traceback.format_exc()
            self.failed.emit(f"{e}\n{tb}")


def launch():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch()
