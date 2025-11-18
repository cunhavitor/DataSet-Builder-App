from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
from core.dataset_manager import delete_session


class ReviewWindow(QtWidgets.QWidget):
    status_message = QtCore.Signal(str)

    def __init__(self, metadata_dir: str = "data/metadata") -> None:
        super().__init__()
        self.metadata_dir = Path(metadata_dir)
        self.cans_data: List[Dict] = []
        self.filtered: List[Dict] = []
        self._build_ui()
        self._load_data()

    # ---------------- UI ----------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # filtros
        filt_layout = QtWidgets.QHBoxLayout()
        self.session_filter = QtWidgets.QComboBox()
        self.sheet_filter = QtWidgets.QComboBox()
        self.label_filter = QtWidgets.QComboBox()
        self.tests_filter = QtWidgets.QComboBox()
        self.btn_reload = QtWidgets.QPushButton("Recarregar")
        self.btn_delete_session = QtWidgets.QPushButton("Eliminar sessão")

        for cb, title in [
            (self.session_filter, "Sessão"),
            (self.sheet_filter, "Folha"),
            (self.label_filter, "Label"),
            (self.tests_filter, "Teste"),
        ]:
            cb.addItem("todos", None)
            cb.setToolTip(f"Filtrar por {title.lower()}")

        self.label_filter.addItems(["good", "defect", "duvidosa", "unknown", "excluded"])
        self.tests_filter.addItems(["aprovado", "reprovado"])

        filt_layout.addWidget(QtWidgets.QLabel("Sessão"))
        filt_layout.addWidget(self.session_filter)
        filt_layout.addWidget(QtWidgets.QLabel("Folha"))
        filt_layout.addWidget(self.sheet_filter)
        filt_layout.addWidget(QtWidgets.QLabel("Label"))
        filt_layout.addWidget(self.label_filter)
        filt_layout.addWidget(QtWidgets.QLabel("Testes"))
        filt_layout.addWidget(self.tests_filter)
        filt_layout.addWidget(self.btn_reload)
        filt_layout.addWidget(self.btn_delete_session)
        self.btn_select_all = QtWidgets.QPushButton("Selecionar tudo")
        self.btn_mark_good = QtWidgets.QPushButton("Marcar como good")
        self.btn_mark_defect = QtWidgets.QPushButton("Marcar como defect")
        self.btn_mark_doubt = QtWidgets.QPushButton("Marcar como duvidosa")
        self.btn_mark_excluded = QtWidgets.QPushButton("Marcar como excluded")
        self.btn_select_all.setToolTip("Seleciona todos os itens filtrados.")
        self.btn_mark_good.setToolTip("Aplica label 'good' a todos os itens selecionados.")
        self.btn_mark_defect.setToolTip("Aplica label 'defect' a todos os itens selecionados.")
        self.btn_mark_doubt.setToolTip("Aplica label 'duvidosa' a todos os itens selecionados.")
        self.btn_mark_excluded.setToolTip("Aplica label 'excluded' a todos os itens selecionados.")
        layout.addLayout(filt_layout)

        bulk_layout = QtWidgets.QHBoxLayout()
        bulk_layout.addWidget(self.btn_select_all)
        bulk_layout.addWidget(self.btn_mark_good)
        bulk_layout.addWidget(self.btn_mark_defect)
        bulk_layout.addWidget(self.btn_mark_doubt)
        bulk_layout.addWidget(self.btn_mark_excluded)
        layout.addLayout(bulk_layout)

        # grid de thumbnails
        self.list = QtWidgets.QListWidget()
        self.list.setViewMode(QtWidgets.QListView.IconMode)
        self.list.setResizeMode(QtWidgets.QListView.Adjust)
        self.list.setIconSize(QtCore.QSize(128, 128))
        self.list.setGridSize(QtCore.QSize(160, 180))
        self.list.setMovement(QtWidgets.QListView.Static)
        self.list.setSpacing(8)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.list)

        self.list.itemClicked.connect(self._on_item_clicked)
        self.btn_reload.clicked.connect(self._load_data)
        self.session_filter.currentIndexChanged.connect(self._apply_filters)
        self.sheet_filter.currentIndexChanged.connect(self._apply_filters)
        self.label_filter.currentIndexChanged.connect(self._apply_filters)
        self.tests_filter.currentIndexChanged.connect(self._apply_filters)
        self.btn_delete_session.clicked.connect(self._delete_selected_session)
        self.btn_select_all.clicked.connect(self._select_all)
        self.btn_mark_good.clicked.connect(lambda: self._bulk_label("good"))
        self.btn_mark_defect.clicked.connect(lambda: self._bulk_label("defect"))
        self.btn_mark_doubt.clicked.connect(lambda: self._bulk_label("duvidosa"))
        self.btn_mark_excluded.clicked.connect(lambda: self._bulk_label("excluded"))

    # ---------------- Data ----------------
    def _cans_csv_path(self) -> Path:
        return self.metadata_dir / "cans.csv"

    def _load_data(self) -> None:
        path = self._cans_csv_path()
        if not path.exists():
            self.status_message.emit(f"Nenhum cans.csv em {path.parent}")
            self.cans_data = []
            self._refresh_list()
            return
        try:
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.cans_data = [row for row in reader]
        except Exception as e:
            self.status_message.emit(f"Erro a ler cans.csv: {e}")
            self.cans_data = []
        self._populate_filters()
        self._apply_filters()
        self.status_message.emit(f"Carregado cans.csv ({len(self.cans_data)} registos)")

    def _populate_filters(self) -> None:
        def fill(combo: QtWidgets.QComboBox, values):
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("todos", None)
            for v in sorted(values):
                combo.addItem(v, v)
            # restore if possible
            if current is not None:
                idx = combo.findData(current)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.blockSignals(False)

        sessions = {row.get("session_id", "") for row in self.cans_data if row.get("session_id")}
        sheets = {row.get("sheet_id", "") for row in self.cans_data if row.get("sheet_id")}
        labels = {row.get("label", "unknown") for row in self.cans_data}
        fill(self.session_filter, sessions)
        fill(self.sheet_filter, sheets)
        # label filter already prefilled; update with seen labels
        self.label_filter.blockSignals(True)
        self.label_filter.clear()
        self.label_filter.addItem("todos", None)
        for v in sorted(labels | {"good", "defect", "duvidosa", "unknown", "excluded"}):
            self.label_filter.addItem(v, v)
        self.label_filter.blockSignals(False)

    def _apply_filters(self) -> None:
        sess = self.session_filter.currentData()
        sheet = self.sheet_filter.currentData()
        label = self.label_filter.currentData()
        tests = self.tests_filter.currentData()

        def ok(row):
            if sess and row.get("session_id") != sess:
                return False
            if sheet and row.get("sheet_id") != sheet:
                return False
            if label and row.get("label") != label:
                return False
            if tests:
                quality_ok = str(row.get("quality_ok", True)).lower() in ("true", "1", "yes")
                if tests == "aprovado" and not quality_ok:
                    return False
                if tests == "reprovado" and quality_ok:
                    return False
            return True

        self.filtered = [r for r in self.cans_data if ok(r)]
        self._refresh_list()
        self.status_message.emit(f"{len(self.filtered)} latas filtradas")

    def _refresh_list(self) -> None:
        self.list.clear()
        for row in self.filtered:
            path = row.get("image_path", "")
            pix = self._load_thumb(path)
            item = QtWidgets.QListWidgetItem()
            item.setIcon(QtGui.QIcon(pix))
            label = row.get("label", "unknown")
            item.setText(f"{row.get('can_id','?')} | {label}")
            item.setData(QtCore.Qt.UserRole, row)
            self.list.addItem(item)

    # ---------------- Actions ----------------
    def _on_item_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        row = item.data(QtCore.Qt.UserRole)
        dlg = _DetailDialog(row, parent=self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            new_label = dlg.selected_label
            self._update_label(row, new_label)
            self._apply_filters()

    def _update_label(self, row: Dict, new_label: str) -> None:
        # atualiza o registo correspondente em self.cans_data
        key = (row.get("session_id"), row.get("sheet_id"), row.get("can_id"))
        for r in self.cans_data:
            if (r.get("session_id"), r.get("sheet_id"), r.get("can_id")) == key:
                r["label"] = new_label
                break
        # persist all rows
        path = self._cans_csv_path()
        if self.cans_data:
            fieldnames = list(self.cans_data[0].keys())
        else:
            fieldnames = [
                "session_id",
                "sheet_id",
                "can_id",
                "image_path",
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
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.cans_data:
                    writer.writerow(r)
            self.status_message.emit(f"Label atualizada para {new_label}")
        except Exception as e:
            self.status_message.emit(f"Erro ao gravar cans.csv: {e}")

    # ---------------- Helpers ----------------
    def _load_thumb(self, path: str) -> QtGui.QPixmap:
        p = Path(path)
        if not p.exists():
            pix = QtGui.QPixmap(128, 128)
            pix.fill(QtGui.QColor("#333"))
            return pix
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            pix = QtGui.QPixmap(128, 128)
            pix.fill(QtGui.QColor("#333"))
            return pix
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(128, 128, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        return pix

    def _select_all(self) -> None:
        self.list.selectAll()

    def _bulk_label(self, label: str) -> None:
        selected_items = self.list.selectedItems()
        if not selected_items:
            self.status_message.emit("Nenhum item selecionado.")
            return
        for item in selected_items:
            row = item.data(QtCore.Qt.UserRole)
            self._update_label(row, label)
        self._apply_filters()
        self.status_message.emit(f"{len(selected_items)} itens marcados como {label}")

    def _delete_selected_session(self) -> None:
        sess = self.session_filter.currentData()
        if not sess:
            QtWidgets.QMessageBox.information(self, "Eliminar sessão", "Selecione uma sessão para eliminar.")
            return
        ret = QtWidgets.QMessageBox.question(
            self,
            "Confirmar eliminação",
            f"Eliminar sessão {sess} dos registos? (imagens não serão apagadas)",
        )
        if ret != QtWidgets.QMessageBox.Yes:
            return
        try:
            delete_session(sess, metadata_dir=str(self.metadata_dir), delete_images=False)
            self.status_message.emit(f"Sessão {sess} eliminada de sessions.csv e cans.csv (imagens mantidas).")
            self._load_data()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erro", f"Falha ao eliminar sessão: {e}")
            self.status_message.emit(f"Erro ao eliminar sessão: {e}")

class _DetailDialog(QtWidgets.QDialog):
    def __init__(self, row: Dict, parent=None) -> None:
        super().__init__(parent)
        self.selected_label = row.get("label", "unknown")
        self.setWindowTitle(f"Lata {row.get('can_id')}")
        self.resize(640, 480)
        v = QtWidgets.QVBoxLayout(self)

        # imagem
        img_label = QtWidgets.QLabel()
        img_label.setAlignment(QtCore.Qt.AlignCenter)
        pix = parent._load_thumb(row.get("image_path", "")) if isinstance(parent, ReviewWindow) else QtGui.QPixmap()
        img_label.setPixmap(pix.scaled(320, 320, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        v.addWidget(img_label)

        # metadados
        meta = QtWidgets.QTextBrowser()
        meta.setReadOnly(True)
        meta.setMinimumHeight(120)
        meta.setHtml(
            "<br>".join(
                f"<b>{k}</b>: {row.get(k,'')}"
                for k in ["session_id", "sheet_id", "can_id", "label", "quality_ok", "brightness_score", "lab_delta_L", "lab_delta_ab"]
            )
        )
        v.addWidget(meta)

        # botoes de label
        btns = QtWidgets.QHBoxLayout()
        for lbl in ["good", "defect", "duvidosa", "excluded", "unknown"]:
            b = QtWidgets.QPushButton(lbl)
            b.clicked.connect(lambda _, l=lbl: self._set_label(l))
            btns.addWidget(b)
        v.addLayout(btns)

        # OK/Cancel
        ok_cancel = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        ok_cancel.accepted.connect(self.accept)
        ok_cancel.rejected.connect(self.reject)
        v.addWidget(ok_cancel)

    def _set_label(self, lbl: str) -> None:
        self.selected_label = lbl
        # fecha o diálogo imediatamente após escolher
        self.accept()
