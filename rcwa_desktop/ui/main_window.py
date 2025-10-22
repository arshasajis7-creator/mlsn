from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QWidget,
)

from ..models.configuration import (
    CellSpec,
    Configuration,
    FrequencySpec,
    LayerSpec,
    MaskHole,
    MaskSpec,
)
from ..services import rcwa_runner
from ..services.optimizer import OptimizationResult, OptimizationSettings, optimize_configuration
from ..services.run_logger import RunContext, RunLogger
from ..services.settings_store import SettingsStore


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(object)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self) -> None:  # pragma: no cover - requires Qt event loop
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:  # pragma: no cover - forwarded to UI
            self.signals.error.emit(traceback.format_exc())
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self, base_dir: Path) -> None:
        super().__init__()
        self.base_dir = base_dir
        self.repo_root = self.base_dir.parent
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(max(self.thread_pool.maxThreadCount(), 4))

        self.settings_store = SettingsStore(self.base_dir)
        self.run_logger = RunLogger(self.repo_root)
        self.active_run_context: Optional[RunContext] = None

        self.current_config = Configuration()

        self._build_ui()
        self._apply_configuration(self.current_config)

        last_config = self.settings_store.load_last()
        if last_config is not None:
            self.current_config = last_config
            self._apply_configuration(last_config)
            self._append_log("آخرین تنظیمات بازیابی شد.")

    # ------------------------------------------------------------------
    # UI construction helpers

    def _build_ui(self) -> None:
        self.setWindowTitle("RCWA Desktop Optimizer")
        self.resize(1200, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.general_tab = QWidget()
        self.materials_tab = QWidget()
        self.geometry_tab = QWidget()
        self.execution_tab = QWidget()

        self.tabs.addTab(self.general_tab, "عمومی")
        self.tabs.addTab(self.materials_tab, "مواد")
        self.tabs.addTab(self.geometry_tab, "هندسه ماسک")
        self.tabs.addTab(self.execution_tab, "اجرا")

        self._init_general_tab()
        self._init_materials_tab()
        self._init_geometry_tab()
        self._init_execution_tab()

        toolbar = self.addToolBar("Configuration")
        toolbar.setMovable(False)

        self.load_button = QPushButton("بارگذاری تنظیمات")
        self.load_button.clicked.connect(self._load_configuration_dialog)
        toolbar.addWidget(self.load_button)

        self.save_button = QPushButton("ذخیره تنظیمات")
        self.save_button.clicked.connect(self._save_configuration_dialog)
        toolbar.addWidget(self.save_button)

        self.load_last_button = QPushButton("بازیابی آخرین ران")
        self.load_last_button.clicked.connect(self._load_last_configuration)
        toolbar.addWidget(self.load_last_button)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("آماده.")

    def _init_general_tab(self) -> None:
        layout = QFormLayout()

        self.freq_start = self._create_double_spin(0.1, 110.0, 0.1, suffix=" GHz")
        self.freq_stop = self._create_double_spin(0.1, 110.0, 0.1, suffix=" GHz")
        self.freq_points = QSpinBox()
        self.freq_points.setRange(2, 2001)
        self.freq_points.setSingleStep(1)

        self.cell_lx = self._create_double_spin(0.1, 100.0, 0.1, suffix=" mm")
        self.cell_ly = self._create_double_spin(0.1, 100.0, 0.1, suffix=" mm")

        self.theta_spin = self._create_double_spin(-90.0, 90.0, 0.5, suffix=" °")
        self.phi_spin = self._create_double_spin(-90.0, 90.0, 0.5, suffix=" °")

        self.polarization_combo = QtWidgets.QComboBox()
        self.polarization_combo.addItems(["TE", "TM"])

        self.harmonics_x = QSpinBox()
        self.harmonics_x.setRange(1, 51)
        self.harmonics_x.setSingleStep(2)

        self.harmonics_y = QSpinBox()
        self.harmonics_y.setRange(1, 51)
        self.harmonics_y.setSingleStep(2)

        self.output_prefix = QLineEdit()

        layout.addRow("شروع فرکانس", self.freq_start)
        layout.addRow("پایان فرکانس", self.freq_stop)
        layout.addRow("نقاط نمونه", self.freq_points)
        layout.addRow("طول سلول X", self.cell_lx)
        layout.addRow("طول سلول Y", self.cell_ly)
        layout.addRow("زاویه تتا", self.theta_spin)
        layout.addRow("زاویه فی", self.phi_spin)
        layout.addRow("پلاریزاسیون", self.polarization_combo)
        layout.addRow("هارمونیک X", self.harmonics_x)
        layout.addRow("هارمونیک Y", self.harmonics_y)
        layout.addRow("پیشوند خروجی", self.output_prefix)

        self.general_tab.setLayout(layout)

    def _init_materials_tab(self) -> None:
        layout = QVBoxLayout()

        self.layer_top_group = self._create_material_group("لایه بالا", thickness_suffix=" mm")
        self.mask_group = self._create_mask_group()
        self.layer_bottom_group = self._create_material_group("لایه پایین", thickness_suffix=" mm")

        layout.addWidget(self.layer_top_group)
        layout.addWidget(self.mask_group)
        layout.addWidget(self.layer_bottom_group)
        layout.addStretch(1)

        self.materials_tab.setLayout(layout)

    def _init_geometry_tab(self) -> None:
        layout = QVBoxLayout()
        controls_layout = QHBoxLayout()

        self.add_circle_btn = QPushButton("افزودن حفره")
        self.add_circle_btn.clicked.connect(lambda: self._add_hole_row("circle"))
        controls_layout.addWidget(self.add_circle_btn)

        self.remove_hole_btn = QPushButton("حذف انتخاب شده")
        self.remove_hole_btn.clicked.connect(self._remove_selected_holes)
        controls_layout.addWidget(self.remove_hole_btn)

        self.clear_holes_btn = QPushButton("پاک کردن همه")
        self.clear_holes_btn.clicked.connect(lambda: self.holes_table.setRowCount(0))
        controls_layout.addWidget(self.clear_holes_btn)

        controls_layout.addStretch(1)

        self.holes_table = QTableWidget(0, 4)
        self.holes_table.setHorizontalHeaderLabels([
            "شکل",
            "X (mm)",
            "Y (mm)",
            "قطر (mm)",
        ])
        self.holes_table.horizontalHeader().setStretchLastSection(True)
        self.holes_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.holes_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.holes_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.holes_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.AllEditTriggers)

        layout.addLayout(controls_layout)
        layout.addWidget(self.holes_table)
        self.geometry_tab.setLayout(layout)

    def _init_execution_tab(self) -> None:
        layout = QVBoxLayout()

        buttons_layout = QHBoxLayout()

        self.run_button = QPushButton("اجرای شبیه‌سازی")
        self.run_button.clicked.connect(self._run_simulation)
        buttons_layout.addWidget(self.run_button)

        self.optimize_button = QPushButton("بهینه‌سازی ماسک")
        self.optimize_button.clicked.connect(self._run_optimization)
        buttons_layout.addWidget(self.optimize_button)

        buttons_layout.addStretch(1)

        layout.addLayout(buttons_layout)

        optimize_layout = QGridLayout()
        optimize_layout.addWidget(QLabel("فرکانس هدف (GHz)"), 0, 0)
        self.optimize_target = self._create_double_spin(0.1, 110.0, 0.1)
        self.optimize_target.setValue(10.0)
        optimize_layout.addWidget(self.optimize_target, 0, 1)

        optimize_layout.addWidget(QLabel("تعداد تکرار"), 0, 2)
        self.optimize_iterations = QSpinBox()
        self.optimize_iterations.setRange(5, 500)
        self.optimize_iterations.setValue(60)
        optimize_layout.addWidget(self.optimize_iterations, 0, 3)

        optimize_layout.addWidget(QLabel("تعداد سوراخ مطلوب"), 1, 0)
        self.optimize_hole_count = QSpinBox()
        self.optimize_hole_count.setRange(1, 20)
        self.optimize_hole_count.setValue(4)
        optimize_layout.addWidget(self.optimize_hole_count, 1, 1)

        layout.addLayout(optimize_layout)

        self.optimize_history = QPlainTextEdit()
        self.optimize_history.setReadOnly(True)
        self.optimize_history.setPlaceholderText("گزارش بهینه‌سازی اینجا نمایش داده می‌شود.")
        layout.addWidget(self.optimize_history, stretch=1)

        plot_group = QGroupBox("نتیجه بازتاب")
        plot_layout = QVBoxLayout()
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvasQTAgg(self.figure)
        plot_layout.addWidget(self.canvas)
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group, stretch=2)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("لاگ رویدادهای رابط کاربری و شبیه‌ساز.")
        layout.addWidget(self.log_output, stretch=1)

        self.execution_tab.setLayout(layout)

    # ------------------------------------------------------------------
    # Configuration helpers

    def _create_double_spin(self, minimum: float, maximum: float, step: float, suffix: str = ""):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setDecimals(4)
        if suffix:
            widget.setSuffix(suffix)
        return widget

    def _create_material_group(self, title: str, thickness_suffix: str) -> QGroupBox:
        group = QGroupBox(title)
        layout = QGridLayout()

        csv_label = QLabel("فایل ماده")
        csv_edit = QLineEdit()
        csv_button = QPushButton("انتخاب")
        csv_button.clicked.connect(lambda: self._browse_csv(csv_edit))

        thickness_label = QLabel("ضخامت")
        thickness_spin = self._create_double_spin(0.01, 50.0, 0.05, suffix=thickness_suffix)

        layout.addWidget(csv_label, 0, 0)
        layout.addWidget(csv_edit, 0, 1)
        layout.addWidget(csv_button, 0, 2)
        layout.addWidget(thickness_label, 1, 0)
        layout.addWidget(thickness_spin, 1, 1)

        group.setLayout(layout)
        group.csv_edit = csv_edit  # type: ignore[attr-defined]
        group.thickness_spin = thickness_spin  # type: ignore[attr-defined]
        return group

    def _create_mask_group(self) -> QGroupBox:
        group = QGroupBox("ماسک")
        layout = QGridLayout()

        solid_label = QLabel("ماده پر")
        solid_edit = QLineEdit()
        solid_button = QPushButton("انتخاب")
        solid_button.clicked.connect(lambda: self._browse_csv(solid_edit))

        hole_label = QLabel("ماده سوراخ")
        hole_edit = QLineEdit()
        hole_button = QPushButton("انتخاب")
        hole_button.clicked.connect(lambda: self._browse_csv(hole_edit))

        thickness_label = QLabel("ضخامت")
        thickness_spin = self._create_double_spin(0.01, 50.0, 0.05, suffix=" mm")

        grid_nx_label = QLabel("Grid Nx")
        grid_nx_spin = QSpinBox()
        grid_nx_spin.setRange(3, 256)
        grid_nx_spin.setValue(48)

        grid_ny_label = QLabel("Grid Ny")
        grid_ny_spin = QSpinBox()
        grid_ny_spin.setRange(3, 256)
        grid_ny_spin.setValue(48)

        layout.addWidget(solid_label, 0, 0)
        layout.addWidget(solid_edit, 0, 1)
        layout.addWidget(solid_button, 0, 2)
        layout.addWidget(hole_label, 1, 0)
        layout.addWidget(hole_edit, 1, 1)
        layout.addWidget(hole_button, 1, 2)
        layout.addWidget(thickness_label, 2, 0)
        layout.addWidget(thickness_spin, 2, 1)
        layout.addWidget(grid_nx_label, 3, 0)
        layout.addWidget(grid_nx_spin, 3, 1)
        layout.addWidget(grid_ny_label, 3, 2)
        layout.addWidget(grid_ny_spin, 3, 3)

        group.setLayout(layout)
        group.solid_edit = solid_edit  # type: ignore[attr-defined]
        group.hole_edit = hole_edit  # type: ignore[attr-defined]
        group.thickness_spin = thickness_spin  # type: ignore[attr-defined]
        group.grid_nx_spin = grid_nx_spin  # type: ignore[attr-defined]
        group.grid_ny_spin = grid_ny_spin  # type: ignore[attr-defined]
        return group

    # ------------------------------------------------------------------
    # Configuration <-> UI conversion

    def _apply_configuration(self, config: Configuration) -> None:
        self.freq_start.setValue(config.freq.start)
        self.freq_stop.setValue(config.freq.stop)
        self.freq_points.setValue(config.freq.points)
        self.cell_lx.setValue(config.cell.Lx_m * 1000.0)
        self.cell_ly.setValue(config.cell.Ly_m * 1000.0)
        self.theta_spin.setValue(config.theta_deg)
        self.phi_spin.setValue(config.phi_deg)
        self.polarization_combo.setCurrentText(config.polarization)
        if isinstance(config.n_harmonics, list) and len(config.n_harmonics) == 2:
            self.harmonics_x.setValue(int(config.n_harmonics[0]))
            self.harmonics_y.setValue(int(config.n_harmonics[1]))
        self.output_prefix.setText(config.output_prefix)

        self.layer_top_group.csv_edit.setText(config.layer_top.material_csv)  # type: ignore[attr-defined]
        self.layer_top_group.thickness_spin.setValue(config.layer_top.thickness_m * 1000.0)  # type: ignore[attr-defined]

        self.mask_group.solid_edit.setText(config.mask.solid_csv)  # type: ignore[attr-defined]
        self.mask_group.hole_edit.setText(config.mask.hole_csv)  # type: ignore[attr-defined]
        self.mask_group.thickness_spin.setValue(config.mask.thickness_m * 1000.0)  # type: ignore[attr-defined]
        self.mask_group.grid_nx_spin.setValue(config.mask.grid_nx)  # type: ignore[attr-defined]
        self.mask_group.grid_ny_spin.setValue(config.mask.grid_ny)  # type: ignore[attr-defined]

        self.layer_bottom_group.csv_edit.setText(config.layer_bottom.material_csv)  # type: ignore[attr-defined]
        self.layer_bottom_group.thickness_spin.setValue(config.layer_bottom.thickness_m * 1000.0)  # type: ignore[attr-defined]

        self._populate_mask_table(config.mask.holes)

    def _collect_configuration(self) -> Configuration:
        freq = FrequencySpec(
            start=self.freq_start.value(),
            stop=self.freq_stop.value(),
            points=self.freq_points.value(),
        )

        cell = CellSpec(
            Lx_m=self.cell_lx.value() / 1000.0,
            Ly_m=self.cell_ly.value() / 1000.0,
        )

        layer_top = LayerSpec(
            material_csv=self.layer_top_group.csv_edit.text(),  # type: ignore[attr-defined]
            thickness_m=self.layer_top_group.thickness_spin.value() / 1000.0,  # type: ignore[attr-defined]
        )

        mask = MaskSpec(
            solid_csv=self.mask_group.solid_edit.text(),  # type: ignore[attr-defined]
            hole_csv=self.mask_group.hole_edit.text(),  # type: ignore[attr-defined]
            thickness_m=self.mask_group.thickness_spin.value() / 1000.0,  # type: ignore[attr-defined]
            grid_nx=self.mask_group.grid_nx_spin.value(),  # type: ignore[attr-defined]
            grid_ny=self.mask_group.grid_ny_spin.value(),  # type: ignore[attr-defined]
            holes=self._collect_mask_holes(),
        )

        layer_bottom = LayerSpec(
            material_csv=self.layer_bottom_group.csv_edit.text(),  # type: ignore[attr-defined]
            thickness_m=self.layer_bottom_group.thickness_spin.value() / 1000.0,  # type: ignore[attr-defined]
        )

        config = Configuration(
            freq=freq,
            cell=cell,
            layer_top=layer_top,
            mask=mask,
            layer_bottom=layer_bottom,
            polarization=self.polarization_combo.currentText(),
            n_harmonics=[self.harmonics_x.value(), self.harmonics_y.value()],
            theta_deg=self.theta_spin.value(),
            phi_deg=self.phi_spin.value(),
            output_prefix=self.output_prefix.text() or "desktop_config",
            backing=self.current_config.backing,
            solver=self.current_config.solver,
            tolerances=self.current_config.tolerances,
        )

        return config

    def _populate_mask_table(self, holes: List[MaskHole]) -> None:
        self.holes_table.setRowCount(0)
        for hole in holes:
            self._add_hole_row(hole.shape, hole)

    def _add_hole_row(self, shape: str, hole: Optional[MaskHole] = None) -> None:
        row = self.holes_table.rowCount()
        self.holes_table.insertRow(row)

        default = MaskHole(shape="circle", x_m=0.0, y_m=0.0, size1=0.001, size2=None)
        hole = hole or default

        display_shape = "circle" if hole.shape not in {"circle"} else hole.shape

        self.holes_table.setItem(row, 0, QTableWidgetItem(display_shape))
        self.holes_table.setItem(row, 1, QTableWidgetItem(f"{hole.x_m * 1000.0:.4f}"))
        self.holes_table.setItem(row, 2, QTableWidgetItem(f"{hole.y_m * 1000.0:.4f}"))
        self.holes_table.setItem(row, 3, QTableWidgetItem(f"{hole.adapter_diameter() * 1000.0:.4f}"))

    def _remove_selected_holes(self) -> None:
        selected = self.holes_table.selectionModel().selectedRows()
        for model_index in reversed(selected):
            self.holes_table.removeRow(model_index.row())

    def _collect_mask_holes(self) -> List[MaskHole]:
        holes: List[MaskHole] = []
        for row in range(self.holes_table.rowCount()):
            shape_item = self.holes_table.item(row, 0)
            if shape_item is None:
                continue
            x = self._parse_table_float(row, 1) / 1000.0
            y = self._parse_table_float(row, 2) / 1000.0
            diameter = max(self._parse_table_float(row, 3) / 1000.0, 1e-6)
            holes.append(MaskHole(shape="circle", x_m=x, y_m=y, size1=diameter, size2=None))
        return holes

    def _parse_table_float(self, row: int, column: int) -> float:
        item = self.holes_table.item(row, column)
        if item is None:
            return 0.0
        try:
            return float(item.text())
        except ValueError:
            return 0.0

    # ------------------------------------------------------------------
    # Actions

    def _run_simulation(self) -> None:
        config = self._collect_configuration()
        self.current_config = config
        self.settings_store.save_last(config)

        run_context = self.run_logger.start_run(config.output_prefix)
        run_context.record_configuration(config)
        self.active_run_context = run_context

        self._append_log("اجرای شبیه‌سازی آغاز شد...")
        self.run_button.setEnabled(False)
        self.optimize_button.setEnabled(False)
        self.status_bar.showMessage("در حال اجرای شبیه‌سازی...")

        worker = Worker(rcwa_runner.run_simulation, config, self.repo_root, log_dir=run_context.directory)
        worker.signals.result.connect(lambda result: self._handle_simulation_result(result, run_context))
        worker.signals.error.connect(lambda err: self._handle_simulation_error(err, run_context))
        worker.signals.finished.connect(self._on_worker_finished)
        self.thread_pool.start(worker)

    def _handle_simulation_result(self, result: rcwa_runner.SimulationResult, run_context: RunContext) -> None:
        self._append_log("شبیه‌سازی با موفقیت پایان یافت.")
        if result.warnings:
            self.status_bar.showMessage("شبیه‌سازی کامل شد (با هشدار).", 7000)
        else:
            self.status_bar.showMessage("شبیه‌سازی کامل شد.", 5000)

        self._update_plot(result.freq_GHz, result.RL_dB)

        run_context.record_adapter_output(result.stdout, result.stderr)

        log_section = "\n".join([
            "--- STDOUT ---",
            result.stdout.strip(),
            "--- STDERR ---",
            result.stderr.strip() or "(خالی)",
        ])
        self.log_output.appendPlainText(log_section)
        run_context.append_gui(log_section)

        if result.warnings:
            warning_block = "\n".join(["--- WARNINGS ---", *result.warnings])
            self.log_output.appendPlainText(warning_block)
            run_context.append_gui(warning_block)
            QMessageBox.warning(
                self,
                "Simulation warnings",
                "\n".join(result.warnings),
            )

    def _handle_simulation_error(self, error: str, run_context: RunContext) -> None:
        self._append_log("شبیه‌سازی با خطا مواجه شد.")
        self.status_bar.showMessage("خطای اجرای شبیه‌سازی.", 10000)
        run_context.append_gui(error)
        QMessageBox.critical(self, "Simulation Error", error)

    def _on_worker_finished(self) -> None:
        self.run_button.setEnabled(True)
        self.optimize_button.setEnabled(True)
        self.active_run_context = None

    def _run_optimization(self) -> None:
        config = self._collect_configuration()
        settings = OptimizationSettings(
            target_frequency_GHz=self.optimize_target.value(),
            iterations=self.optimize_iterations.value(),
            desired_hole_count=self.optimize_hole_count.value(),
        )

        self._append_log("بهینه‌سازی ماسک آغاز شد...")
        self.optimize_button.setEnabled(False)
        self.status_bar.showMessage("در حال بهینه‌سازی ماسک...")

        worker = Worker(optimize_configuration, config, settings)
        worker.signals.result.connect(self._handle_optimization_result)
        worker.signals.error.connect(self._handle_optimization_error)
        worker.signals.finished.connect(lambda: self.optimize_button.setEnabled(True))
        self.thread_pool.start(worker)

    def _handle_optimization_result(self, result: OptimizationResult) -> None:
        self._append_log("بهینه‌سازی به پایان رسید.")
        self.status_bar.showMessage("بهینه‌سازی کامل شد.", 5000)
        self.current_config = result.configuration
        self._apply_configuration(result.configuration)
        self.settings_store.save_last(result.configuration)

        history_text = "\n".join(result.history + [f"Best score: {result.score:.4f}"])
        self.optimize_history.setPlainText(history_text)

    def _handle_optimization_error(self, error: str) -> None:
        self._append_log("خطا در بهینه‌سازی.")
        self.status_bar.showMessage("بهینه‌سازی ناموفق بود.", 8000)
        QMessageBox.critical(self, "Optimization Error", error)

    def _update_plot(self, freq: List[float], rl: List[float]) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if freq and rl:
            ax.plot(freq, rl, marker="o")
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Reflection Loss (dB)")
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Persistence helpers

    def _load_configuration_dialog(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load configuration",
            str(self.repo_root),
            "JSON files (*.json)",
        )
        if not filename:
            return
        config = self.settings_store.load_from_path(Path(filename))
        self.current_config = config
        self._apply_configuration(config)
        self._append_log(f"تنظیمات از {filename} بارگذاری شد.")

    def _save_configuration_dialog(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save configuration",
            str(self.repo_root / "config_store"),
            "JSON files (*.json)",
        )
        if not filename:
            return
        config = self._collect_configuration()
        self.settings_store.save_to_path(config, Path(filename))
        self._append_log(f"تنظیمات در {filename} ذخیره شد.")

    def _load_last_configuration(self) -> None:
        config = self.settings_store.load_last()
        if config is None:
            QMessageBox.information(self, "اطلاعات", "تنظیمات اخیر یافت نشد.")
            return
        self.current_config = config
        self._apply_configuration(config)
        self._append_log("آخرین تنظیمات اعمال شد.")

    # ------------------------------------------------------------------
    # Misc helpers

    def _browse_csv(self, target_edit: QLineEdit) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select material table",
            str(self.repo_root),
            "CSV files (*.csv)",
        )
        if filename:
            target_edit.setText(filename)

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        self.log_output.appendPlainText(line)
        if self.active_run_context is not None:
            self.active_run_context.append_gui(line)

