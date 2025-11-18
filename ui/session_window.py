from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui

from core import can_preprocess
from core.preprocess import normalize_sheet_image
from core.tests_quality import run_all_tests
from core.cropper import crop_cans_from_sheet
from core.normalize_image import normalize_image
from models.can_detector import CanDetector


class SessionWindow(QtWidgets.QWidget):
    status_message = QtCore.Signal(str)
    start_processing = QtCore.Signal(dict)
    run_tests_only = QtCore.Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self._preview_img: np.ndarray | None = None
        self._build_ui()
        self._load_params()

    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)

        # painel esquerdo: tabs de configuração e botões
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)

        self.tabs = QtWidgets.QTabWidget()
        self.tab_paths = self._wrap_scroll(self._paths_group())
        self.tab_session = self._wrap_scroll(self._session_group())
        self.tab_detector = self._wrap_scroll(self._detector_group())
        self.tab_normalize = self._wrap_scroll(self._normalize_group())
        self.tab_preprocess = self._wrap_scroll(self._preprocess_group())
        self.tab_tests = self._wrap_scroll(self._tests_group())

        self.tabs.addTab(self.tab_session, "Sessão")
        self.tabs.addTab(self.tab_paths, "Pastas")
        self.tabs.addTab(self.tab_detector, "Detector YOLO")
        self.tabs.addTab(self.tab_normalize, "Normalização")
        self.tabs.addTab(self.tab_preprocess, "Pré-processamento")
        self.tabs.addTab(self.tab_tests, "Testes qualidade")

        left_layout.addWidget(self.tabs)

        btns = QtWidgets.QHBoxLayout()
        self.btn_run_tests = QtWidgets.QPushButton("Testar folhas")
        self.btn_run_all = QtWidgets.QPushButton("Processar sessão completa")
        self.btn_save_params = QtWidgets.QPushButton("Guardar parâmetros")
        self.btn_run_tests.setToolTip("Executa apenas os testes de qualidade nas folhas.")
        self.btn_run_all.setToolTip("Processa a sessão completa: normaliza, deteta, alinha e salva.")
        self.btn_save_params.setToolTip("Guarda os parâmetros atuais em data/config/ui_settings.json.")
        btns.addStretch(1)
        btns.addWidget(self.btn_save_params)
        btns.addWidget(self.btn_run_tests)
        btns.addWidget(self.btn_run_all)
        left_layout.addLayout(btns)

        # painel direito: previews e relatório
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.addWidget(self._preview_group())
        right_layout.addWidget(self._tests_report_group())
        right_layout.addStretch(1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([480, 720])

        layout.addWidget(splitter)

        # ligações
        self.btn_run_tests.clicked.connect(self._emit_run_tests)
        self.btn_run_all.clicked.connect(self._emit_run_all)
        self.btn_load_preview.clicked.connect(self._load_preview_image)
        self.btn_apply_preview.clicked.connect(self._apply_preview)
        self.btn_preview_norm.clicked.connect(self._preview_normalize)
        self.btn_preview_flow.clicked.connect(self._run_full_preview)
        self.btn_save_params.clicked.connect(self._save_params)

    # ---------------- grupos ----------------
    def _paths_group(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(box)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self.input_folder = QtWidgets.QLineEdit()
        self.raw_dir = QtWidgets.QLineEdit("data/raw_sheets")
        self.norm_dir = QtWidgets.QLineEdit("data/normalized_sheets")
        self.cans_dir = QtWidgets.QLineEdit("data/cans")
        self.meta_dir = QtWidgets.QLineEdit("data/metadata")
        for w in (self.input_folder, self.raw_dir, self.norm_dir, self.cans_dir, self.meta_dir):
            w.setToolTip("Pasta de entrada/saída.")

        rows = [
            ("Folhas (input)", self._with_browse(self.input_folder, True)),
            ("Raw sheets", self._with_browse(self.raw_dir, False)),
            ("Normalized", self._with_browse(self.norm_dir, False)),
            ("Cans", self._with_browse(self.cans_dir, False)),
            ("Metadata", self._with_browse(self.meta_dir, False)),
        ]
        for label, widget in rows:
            form.addRow(label, widget)
        return box

    def _session_group(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(box)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.sku = QtWidgets.QLineEdit("SKU_TEST")
        self.operator = QtWidgets.QLineEdit("Operador")
        self.notes = QtWidgets.QLineEdit()
        form.addRow("SKU", self.sku)
        form.addRow("Operador", self.operator)
        form.addRow("Notas", self.notes)
        return box

    def _detector_group(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(box)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.weights_path = QtWidgets.QLineEdit("models/weights/can_detector.pt")
        self.weights_path.setToolTip("Caminho para os pesos YOLO.")
        self.conf_spin = QtWidgets.QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setToolTip("Confiança mínima das deteções.")
        self.iou_spin = QtWidgets.QDoubleSpinBox()
        self.iou_spin.setRange(0.05, 1.0)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setToolTip("IoU para NMS das deteções.")
        self.device = QtWidgets.QComboBox()
        self.device.addItems(["cpu", "cuda"])
        self.device.setToolTip("Dispositivo para inferência YOLO.")
        form.addRow("Weights", self._with_browse(self.weights_path, True, file_mode=True))
        form.addRow("Conf", self.conf_spin)
        form.addRow("IoU", self.iou_spin)
        form.addRow("Device", self.device)
        return box

    def _normalize_group(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(box)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.norm_size = QtWidgets.QComboBox()
        for v in [128, 256, 384, 512, 640]:
            self.norm_size.addItem(f"{v}", v)
        self.norm_size.setCurrentIndex(1)  # 256 by default
        self.norm_target_v = self._spin_double(0, 255, 180.0, 1.0)
        self.clahe_clip = self._spin_double(0.1, 5.0, 1.5, 0.1)
        self.clahe_tile_x = self._spin_int(2, 32, 4, 2)
        self.clahe_tile_y = self._spin_int(2, 32, 4, 2)
        self.norm_size.setToolTip("Tamanho final (letterbox) das imagens.")
        self.norm_target_v.setToolTip("Brilho alvo no canal V (HSV) para normalização.")
        self.clahe_clip.setToolTip("ClipLimit do CLAHE no canal L (contraste).")
        self.clahe_tile_x.setToolTip("TileSize X do CLAHE.")
        self.clahe_tile_y.setToolTip("TileSize Y do CLAHE.")
        form.addRow("Tamanho saída (px)", self.norm_size)
        form.addRow("Target V (brilho)", self.norm_target_v)
        form.addRow("CLAHE clip", self.clahe_clip)
        form.addRow("CLAHE tile X", self.clahe_tile_x)
        form.addRow("CLAHE tile Y", self.clahe_tile_y)
        return box

    def _preprocess_group(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(box)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        self.canny_low = self._spin_int(0, 500, 30)
        self.canny_high = self._spin_int(0, 500, 140)
        self.canny_low.setToolTip("Limite inferior do Canny.")
        self.canny_high.setToolTip("Limite superior do Canny.")
        self.gauss = self._spin_int(1, 21, 5, 2)
        self.gauss.setToolTip("Kernel do blur Gaussiano (ímpar).")
        self.dilate = self._spin_int(0, 10, 2)
        self.dilate.setToolTip("Iterações de dilatação para fechar bordas.")
        self.close = self._spin_int(0, 10, 2)
        self.close.setToolTip("Iterações de fechamento morfológico.")
        self.target_area = self._spin_double(0, 50000, 10100, 10)
        self.target_area.setToolTip("Área alvo do contorno da label.")
        self.target_tol = self._spin_double(0, 1, 0.15, 0.01)
        self.target_tol.setToolTip("Tolerância relativa da área alvo.")
        self.center_w = self._spin_double(0, 20, 0.8, 0.1)
        self.center_w.setToolTip("Peso da distância ao centro no score.")
        self.area_w = self._spin_double(0, 20, 0.2, 0.1)
        self.area_w.setToolTip("Peso da diferença de área no score.")
        self.mask_pad = self._spin_int(0, 100, 20)
        self.mask_pad.setToolTip("Pixels adicionais na máscara para não cortar borda.")
        form.addRow("Canny low", self.canny_low)
        form.addRow("Canny high", self.canny_high)
        form.addRow("Gauss ksize", self.gauss)
        form.addRow("Dilate iters", self.dilate)
        form.addRow("Close iters", self.close)
        form.addRow("Área alvo", self.target_area)
        form.addRow("Tolerância área", self.target_tol)
        form.addRow("Peso centro", self.center_w)
        form.addRow("Peso área", self.area_w)
        form.addRow("Mask pad (px)", self.mask_pad)
        return box

    def _tests_group(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(box)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.max_lab_L = self._spin_double(0, 100, 15.0, 0.5)
        self.max_lab_ab = self._spin_double(0, 200, 20.0, 0.5)
        self.target_bright = self._spin_double(0, 255, 180.0, 1)
        self.max_bright_delta = self._spin_double(0, 255, 60.0, 1)
        self.max_lab_L.setToolTip("ΔL máximo permitido.")
        self.max_lab_ab.setToolTip("Δab máximo permitido.")
        self.target_bright.setToolTip("Brilho alvo (V em HSV).")
        self.max_bright_delta.setToolTip("Desvio máximo do brilho alvo.")
        form.addRow("ΔL máximo", self.max_lab_L)
        form.addRow("Δab máximo", self.max_lab_ab)
        form.addRow("Brilho alvo", self.target_bright)
        form.addRow("Δ brilho máximo", self.max_bright_delta)
        return box

    def _preview_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Pré-visualização")
        h = QtWidgets.QHBoxLayout(box)
        self.preview_input = QtWidgets.QLabel("Input")
        self.preview_output = QtWidgets.QLabel("Output")
        for lbl in (self.preview_input, self.preview_output):
            lbl.setMinimumSize(320, 320)
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet("border: 1px solid #666; background: #222; color: #ccc;")
        btns = QtWidgets.QVBoxLayout()
        self.btn_load_preview = QtWidgets.QPushButton("Carregar exemplo")
        self.btn_apply_preview = QtWidgets.QPushButton("Aplicar pré-processamento")
        self.btn_preview_norm = QtWidgets.QPushButton("Pré-visualizar normalização")
        self.btn_preview_flow = QtWidgets.QPushButton("Pré-visualizar fluxo completo")
        self.btn_load_preview.setToolTip("Abrir imagem de exemplo para pré-visualização.")
        self.btn_apply_preview.setToolTip("Aplicar apenas o pré-processamento de contorno/máscara.")
        self.btn_preview_norm.setToolTip("Mostrar resultado do passo de normalização (brilho/tamanho).")
        self.btn_preview_flow.setToolTip("Pipeline completo em modo preview (normalize -> YOLO -> crop -> alinhar -> normalizar).")
        btns.addWidget(self.btn_load_preview)
        btns.addWidget(self.btn_apply_preview)
        btns.addWidget(self.btn_preview_norm)
        btns.addWidget(self.btn_preview_flow)
        btns.addStretch(1)
        h.addLayout(btns)
        h.addWidget(self.preview_input)
        h.addWidget(self.preview_output)
        return box

    def _tests_report_group(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Relatório de testes (preview)")
        v = QtWidgets.QVBoxLayout(box)
        self.tests_report = QtWidgets.QTextBrowser()
        self.tests_report.setReadOnly(True)
        self.tests_report.setMinimumHeight(140)
        v.addWidget(self.tests_report)
        return box

    # ---------------- utils UI ----------------
    def _spin_int(self, min_v: int, max_v: int, val: int, step: int = 1) -> QtWidgets.QSpinBox:
        s = QtWidgets.QSpinBox()
        s.setRange(min_v, max_v)
        s.setSingleStep(step)
        s.setValue(val)
        s.setMinimumWidth(80)
        return s

    def _spin_double(self, min_v: float, max_v: float, val: float, step: float) -> QtWidgets.QDoubleSpinBox:
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(min_v, max_v)
        s.setSingleStep(step)
        s.setDecimals(3)
        s.setValue(val)
        s.setMinimumWidth(80)
        return s

    def _with_browse(self, line: QtWidgets.QLineEdit, dir_mode: bool, file_mode: bool = False) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        btn = QtWidgets.QPushButton("...")
        btn.setFixedWidth(32)
        h.addWidget(line)
        h.addWidget(btn)

        def choose():
            if file_mode:
                sel, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Escolher ficheiro", str(Path.cwd()))
            elif dir_mode:
                sel = QtWidgets.QFileDialog.getExistingDirectory(self, "Escolher pasta", str(Path.cwd()))
            else:
                sel = QtWidgets.QFileDialog.getExistingDirectory(self, "Escolher pasta", str(Path.cwd()))
            if sel:
                line.setText(sel)

        btn.clicked.connect(choose)
        return w

    def _wrap_scroll(self, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        return scroll

    # ---------------- sinais ----------------
    def _emit_run_tests(self) -> None:
        cfg = self._collect_config()
        self.run_tests_only.emit(cfg)
        self.status_message.emit("Testes iniciados…")

    def _emit_run_all(self) -> None:
        cfg = self._collect_config()
        self.start_processing.emit(cfg)
        self.status_message.emit("Processamento iniciado…")

    def _save_params(self) -> None:
        cfg = self._collect_config()
        config_dir = Path("data/config")
        config_dir.mkdir(parents=True, exist_ok=True)
        out_path = config_dir / "ui_settings.json"
        try:
            out_path.write_text(json.dumps(cfg, indent=2))
            self.status_message.emit(f"Parâmetros guardados em {out_path}")
        except Exception as e:
            self.status_message.emit(f"Erro ao guardar parâmetros: {e}")

    def _load_params(self) -> None:
        config_path = Path("data/config/ui_settings.json")
        if not config_path.exists():
            return
        try:
            cfg = json.loads(config_path.read_text())
        except Exception as e:
            self.status_message.emit(f"Erro ao ler parâmetros salvos: {e}")
            return

        try:
            paths = cfg.get("paths", {})
            session = cfg.get("session", {})
            det = cfg.get("detector", {})
            norm = cfg.get("normalize", {})
            prep = cfg.get("preprocess", {})
            tests = cfg.get("tests", {})

            self.input_folder.setText(paths.get("input", self.input_folder.text()))
            self.raw_dir.setText(paths.get("raw", self.raw_dir.text()))
            self.norm_dir.setText(paths.get("normalized", self.norm_dir.text()))
            self.cans_dir.setText(paths.get("cans", self.cans_dir.text()))
            self.meta_dir.setText(paths.get("metadata", self.meta_dir.text()))

            self.sku.setText(session.get("sku", self.sku.text()))
            self.operator.setText(session.get("operator", self.operator.text()))
            self.notes.setText(session.get("notes", self.notes.text()))

            self.weights_path.setText(det.get("weights", self.weights_path.text()))
            if "conf" in det:
                self.conf_spin.setValue(float(det["conf"]))
            if "iou" in det:
                self.iou_spin.setValue(float(det["iou"]))
            if "device" in det:
                idx = self.device.findText(str(det["device"]))
                if idx >= 0:
                    self.device.setCurrentIndex(idx)

            if "size" in norm:
                idx = self.norm_size.findData(norm["size"])
                if idx >= 0:
                    self.norm_size.setCurrentIndex(idx)
            if "target_v" in norm:
                self.norm_target_v.setValue(float(norm["target_v"]))

            if "canny_low" in prep:
                self.canny_low.setValue(int(prep["canny_low"]))
            if "canny_high" in prep:
                self.canny_high.setValue(int(prep["canny_high"]))
            if "gauss" in prep:
                self.gauss.setValue(int(prep["gauss"]))
            if "dilate" in prep:
                self.dilate.setValue(int(prep["dilate"]))
            if "close" in prep:
                self.close.setValue(int(prep["close"]))
            if "target_area" in prep:
                self.target_area.setValue(float(prep["target_area"]))
            if "target_tol" in prep:
                self.target_tol.setValue(float(prep["target_tol"]))
            if "center_w" in prep:
                self.center_w.setValue(float(prep["center_w"]))
            if "area_w" in prep:
                self.area_w.setValue(float(prep["area_w"]))
            if "mask_pad" in prep:
                self.mask_pad.setValue(int(prep["mask_pad"]))

            if "max_lab_delta_L" in tests:
                self.max_lab_L.setValue(float(tests["max_lab_delta_L"]))
            if "max_lab_delta_ab" in tests:
                self.max_lab_ab.setValue(float(tests["max_lab_delta_ab"]))
            if "target_brightness" in tests:
                self.target_bright.setValue(float(tests["target_brightness"]))
            if "max_brightness_delta" in tests:
                self.max_bright_delta.setValue(float(tests["max_brightness_delta"]))
            self.status_message.emit(f"Parâmetros carregados de {config_path}")
        except Exception as e:
            self.status_message.emit(f"Erro ao aplicar parâmetros salvos: {e}")

    # ---------------- preview helpers ----------------
    def _load_preview_image(self) -> None:
        sel, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Escolher imagem", str(Path.cwd()), "Imagens (*.png *.jpg *.jpeg)"
        )
        if not sel:
            return
        img = self._imread_unicode(sel)
        if img is None:
            self.status_message.emit(f"Erro ao carregar imagem de pré-visualização: {sel}")
            return
        self._preview_img = img
        self.preview_input.setPixmap(self._to_pixmap(img))
        self.preview_output.clear()
        self.status_message.emit(f"Imagem carregada: {sel}")

    def _apply_preview(self) -> None:
        if self._preview_img is None:
            self.status_message.emit("Carregue uma imagem antes de aplicar.")
            return
        cfg = self._collect_config()
        self._apply_preprocess_params(cfg["preprocess"])
        try:
            img_out, mask, info = can_preprocess.preprocess_can_image(self._preview_img)
        except Exception as e:
            self.status_message.emit(f"Erro no pré-processamento: {e}")
            return
        if img_out is None:
            self.status_message.emit("Nenhum contorno alvo encontrado; parâmetros talvez restritos demais.")
            return
        self.preview_output.setPixmap(self._to_pixmap(img_out))
        self.status_message.emit(
            f"Pré-processado: area={info.get('contour_area',0):.1f}, delta={info.get('delta_vertical',0):.1f}, dx={info.get('dx',0):.1f}, dy={info.get('dy',0):.1f}"
        )

    def _run_full_preview(self) -> None:
        if self._preview_img is None:
            self.status_message.emit("Carregue uma imagem antes de pré-visualizar.")
            return

        cfg = self._collect_config()
        img = self._preview_img.copy()

        # 1) normalizar folha
        sheet_norm = normalize_sheet_image(
            img,
            {
                "size": int(cfg["normalize"]["size"]),
                "target_v": cfg["normalize"]["target_v"],
                "clahe_clip": cfg["normalize"]["clahe_clip"],
                "clahe_tile": cfg["normalize"]["clahe_tile"],
            },
        )
        self.preview_output.setPixmap(self._to_pixmap(sheet_norm))
        self.status_message.emit("Folha normalizada.")

        # 2) testes
        tests_pass, tests_details = run_all_tests(sheet_norm, sheet_norm, cfg["tests"])
        self._render_tests_report(tests_pass, tests_details, cfg["tests"])

        # 3) detector YOLO
        detector = CanDetector(
            weights_path=cfg["detector"]["weights"],
            conf_thres=cfg["detector"]["conf"],
            iou_thres=cfg["detector"]["iou"],
            device=cfg["detector"]["device"],
        )
        detections = detector.detect(img)
        if not detections:
            self.status_message.emit("Nenhuma lata detetada pelo YOLO.")
            return
        det_vis = img.copy()
        for d in detections:
            x, y, w, h = d["bbox"]
            cv2.rectangle(det_vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        self.preview_output.setPixmap(self._to_pixmap(det_vis))
        self.status_message.emit(f"YOLO detetou {len(detections)} latas; usando a primeira para prévia.")

        # 4) crop primeira lata
        det0 = dict(detections[0])
        det0.setdefault("can_id", 1)
        crop_size = int(cfg["normalize"]["size"])
        crops = crop_cans_from_sheet(img, [det0], mask_template=None, size=crop_size)
        if not crops:
            self.status_message.emit("Falha ao fazer crop da lata.")
            return
        can_crop = crops[0]["image"]

        # 5) alinhar + máscara
        self._apply_preprocess_params(cfg["preprocess"])
        img_aligned, mask, info = can_preprocess.preprocess_can_image(can_crop)
        if img_aligned is None:
            self.status_message.emit("Pré-processamento falhou (sem contorno alvo).")
            return

        # 6) normalizar lata
        img_norm = normalize_image(img_aligned, size=256)
        img_norm_u8 = (img_norm * 255.0).clip(0, 255).astype(np.uint8)
        self.preview_output.setPixmap(self._to_pixmap(img_norm_u8))
        self.status_message.emit(
            f"Lata pronta. Área={info.get('contour_area',0):.1f}, ΔY={info.get('delta_vertical',0):.1f}, dx={info.get('dx',0):.1f}, dy={info.get('dy',0):.1f}"
        )

    # ---------------- helpers ----------------
    def _apply_preprocess_params(self, params: dict) -> None:
        cp = can_preprocess
        cp.CANNY_LOW = int(params["canny_low"])
        cp.CANNY_HIGH = int(params["canny_high"])
        cp.GAUSS_KSIZE = int(params["gauss"])
        cp.DILATE_ITERS = int(params["dilate"])
        cp.CLOSE_ITERS = int(params["close"])
        cp.TARGET_AREA = float(params["target_area"])
        cp.TARGET_AREA_TOL = float(params["target_tol"])
        cp.TARGET_CENTER_WEIGHT = float(params["center_w"])
        cp.TARGET_AREA_WEIGHT = float(params["area_w"])
        try:
            cp.MASK_BORDER_PAD = int(params["mask_pad"])
        except Exception:
            pass

    def _preview_normalize(self) -> None:
        if self._preview_img is None:
            self.status_message.emit("Carregue uma imagem antes de pré-visualizar.")
            return
        cfg = self._collect_config()
        norm_img = normalize_sheet_image(
            self._preview_img,
            {
                "size": int(cfg["normalize"]["size"]),
                "target_v": cfg["normalize"]["target_v"],
                "clahe_clip": cfg["normalize"]["clahe_clip"],
                "clahe_tile": cfg["normalize"]["clahe_tile"],
            },
        )
        self.preview_output.setPixmap(self._to_pixmap(norm_img))
        self.status_message.emit("Imagem normalizada (prévia).")

    # ---------------- estado ----------------
    def set_running(self, running: bool) -> None:
        for btn in [self.btn_run_tests, self.btn_run_all, self.btn_save_params]:
            btn.setEnabled(not running)

    def _to_pixmap(self, img_bgr: np.ndarray) -> QtGui.QPixmap:
        img = img_bgr
        if img.dtype != np.uint8:
            img_disp = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img_disp = img
        rgb = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        target_w = max(self.preview_input.width(), 320)
        target_h = max(self.preview_input.height(), 320)
        return pix.scaled(target_w, target_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

    def _imread_unicode(self, path: str) -> np.ndarray | None:
        p = Path(path)
        if not p.exists():
            return None
        data = np.fromfile(str(p), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def _render_tests_report(self, passed: bool, details: dict, cfg: dict) -> None:
        msgs = []
        lab_L = details.get("lab_delta_L", 0.0)
        lab_ab = details.get("lab_delta_ab", 0.0)
        bright = details.get("brightness", 0.0)
        msgs.append(f"ΔL = {lab_L:.2f} (max {cfg.get('max_lab_delta_L')})")
        msgs.append(f"Δab = {lab_ab:.2f} (max {cfg.get('max_lab_delta_ab')})")
        msgs.append(f"Brilho = {bright:.1f} (alvo {cfg.get('target_brightness')}, Δmax {cfg.get('max_brightness_delta')})")

        tips = []
        if lab_L > cfg.get("max_lab_delta_L", lab_L + 1):
            tips.append("Ajuste iluminação ou target_v para reduzir ΔL.")
        if lab_ab > cfg.get("max_lab_delta_ab", lab_ab + 1):
            tips.append("Revise iluminação/calibração LAB para Δab.")
        if abs(bright - cfg.get("target_brightness", bright)) > cfg.get("max_brightness_delta", 0):
            tips.append("Corrija exposição/ganho ou target_brightness.")

        status = "PASSOU" if passed else "FALHOU"
        color = "#4CAF50" if passed else "#E53935"
        tips_html = "<br>".join(tips) if tips else "Sem recomendações."
        html = f"""
        <b>Status: <span style='color:{color}'>{status}</span></b><br>
        {'<br>'.join(msgs)}<br>
        <b>Dicas:</b> {tips_html}
        """
        self.tests_report.setHtml(html)

    def _collect_config(self) -> dict:
        return {
            "paths": {
                "input": self.input_folder.text(),
                "raw": self.raw_dir.text(),
                "normalized": self.norm_dir.text(),
                "cans": self.cans_dir.text(),
                "metadata": self.meta_dir.text(),
            },
            "session": {
                "sku": self.sku.text(),
                "operator": self.operator.text(),
                "notes": self.notes.text(),
            },
            "detector": {
                "weights": self.weights_path.text(),
                "conf": self.conf_spin.value(),
                "iou": self.iou_spin.value(),
                "device": self.device.currentText(),
            },
            "normalize": {
                "size": self.norm_size.currentData(),
                "target_v": self.norm_target_v.value(),
                "clahe_clip": self.clahe_clip.value(),
                "clahe_tile": (self.clahe_tile_x.value(), self.clahe_tile_y.value()),
            },
            "preprocess": {
                "canny_low": self.canny_low.value(),
                "canny_high": self.canny_high.value(),
                "gauss": self.gauss.value(),
                "dilate": self.dilate.value(),
                "close": self.close.value(),
                "target_area": self.target_area.value(),
                "target_tol": self.target_tol.value(),
                "center_w": self.center_w.value(),
                "area_w": self.area_w.value(),
                "mask_pad": self.mask_pad.value(),
            },
            "tests": {
                "max_lab_delta_L": self.max_lab_L.value(),
                "max_lab_delta_ab": self.max_lab_ab.value(),
                "target_brightness": self.target_bright.value(),
                "max_brightness_delta": self.max_bright_delta.value(),
            },
        }
