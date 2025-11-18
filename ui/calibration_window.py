from __future__ import annotations

from pathlib import Path
from PySide6 import QtCore, QtWidgets


class CalibrationWindow(QtWidgets.QWidget):
    status_message = QtCore.Signal(str)
    calibration_loaded = QtCore.Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self.calib_path = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("...")
        browse.setFixedWidth(32)
        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(self.calib_path)
        path_row.addWidget(browse)

        self.percentil_low = self._spin_double(0, 1, 0.01, 0.001)
        self.percentil_high = self._spin_double(0, 1, 0.99, 0.001)
        self.threshold = self._spin_double(0, 1000, 50.0, 0.5)

        form = QtWidgets.QFormLayout()
        form.addRow("Ficheiro calibração (JSON/CSV)", path_row)
        form.addRow("Percentil baixo", self.percentil_low)
        form.addRow("Percentil alto", self.percentil_high)
        form.addRow("Threshold", self.threshold)

        self.btn_load = QtWidgets.QPushButton("Importar calibração")

        layout.addWidget(QtWidgets.QLabel("Importar parâmetros de calibração do Colab"))
        layout.addLayout(form)
        layout.addWidget(self.btn_load)
        layout.addStretch(1)

        browse.clicked.connect(self._choose_file)
        self.btn_load.clicked.connect(self._emit_calibration)

    def _spin_double(self, min_v, max_v, val, step):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(min_v, max_v)
        s.setDecimals(4)
        s.setSingleStep(step)
        s.setValue(val)
        return s

    def _choose_file(self) -> None:
        sel, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Escolher calibração", str(Path.cwd()), "JSON/CSV (*.json *.csv)"
        )
        if sel:
            self.calib_path.setText(sel)

    def _emit_calibration(self) -> None:
        cfg = {
            "path": self.calib_path.text(),
            "percentil_low": self.percentil_low.value(),
            "percentil_high": self.percentil_high.value(),
            "threshold": self.threshold.value(),
        }
        self.calibration_loaded.emit(cfg)
        self.status_message.emit("Calibração carregada (stub) – ligar ao backend")
