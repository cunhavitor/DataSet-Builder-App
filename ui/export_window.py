from __future__ import annotations

from pathlib import Path
from PySide6 import QtCore, QtWidgets


class ExportWindow(QtWidgets.QWidget):
    status_message = QtCore.Signal(str)
    start_export = QtCore.Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self.export_dir = QtWidgets.QLineEdit("export")
        self.btn_browse = QtWidgets.QPushButton("...")
        self.btn_browse.setFixedWidth(32)
        path_layout = QtWidgets.QHBoxLayout()
        path_layout.addWidget(self.export_dir)
        path_layout.addWidget(self.btn_browse)

        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItems(["png", "jpg"])
        self.include_mask = QtWidgets.QCheckBox("Incluir máscara")
        self.include_mask.setChecked(True)
        self.include_manifest = QtWidgets.QCheckBox("Gerar manifest (CSV/JSON)")
        self.include_manifest.setChecked(True)
        self.tag_filter = QtWidgets.QComboBox()
        self.tag_filter.addItems(["todos", "good", "defect", "duvidosa"])
        self.size_spin = QtWidgets.QSpinBox()
        self.size_spin.setRange(64, 2048)
        self.size_spin.setValue(256)
        self.session_list = QtWidgets.QListWidget()
        self.session_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.only_pass = QtWidgets.QCheckBox("Apenas testes aprovados (quality_ok)")

        form = QtWidgets.QFormLayout()
        form.addRow("Pasta de exportação", path_layout)
        form.addRow("Formato", self.format_combo)
        form.addRow("Tamanho (px)", self.size_spin)
        form.addRow("Sessões (seleção múltipla)", self.session_list)
        form.addRow("Filtrar tag", self.tag_filter)
        form.addRow("", self.only_pass)
        form.addRow("", self.include_mask)
        form.addRow("", self.include_manifest)

        self.btn_export = QtWidgets.QPushButton("Exportar dataset")

        layout.addWidget(QtWidgets.QLabel("Exportar dataset e manifest"))
        layout.addLayout(form)
        layout.addWidget(self.btn_export)
        layout.addStretch(1)

        self.btn_browse.clicked.connect(self._choose_dir)
        self.btn_export.clicked.connect(self._emit_export)
        self._load_sessions()

    def _load_sessions(self) -> None:
        sessions_path = Path("data/metadata/sessions.csv")
        self.session_list.clear()
        if not sessions_path.exists():
            return
        try:
            import csv

            with sessions_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid = row.get("session_id")
                    if sid:
                        item = QtWidgets.QListWidgetItem(sid)
                        self.session_list.addItem(item)
        except Exception:
            pass

    def _choose_dir(self) -> None:
        sel = QtWidgets.QFileDialog.getExistingDirectory(self, "Escolher pasta", str(Path.cwd()))
        if sel:
            self.export_dir.setText(sel)

    def _emit_export(self) -> None:
        selected_sessions = [i.text() for i in self.session_list.selectedItems()]
        cfg = {
            "dir": self.export_dir.text(),
            "format": self.format_combo.currentText(),
            "size": self.size_spin.value(),
            "include_mask": self.include_mask.isChecked(),
            "include_manifest": self.include_manifest.isChecked(),
            "tag_filter": self.tag_filter.currentText(),
            "sessions": ",".join(selected_sessions),
            "only_pass": self.only_pass.isChecked(),
        }
        self.start_export.emit(cfg)
        self.status_message.emit("Exportação solicitada…")
