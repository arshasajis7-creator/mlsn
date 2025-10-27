from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import QObject, QThread, QRectF, QPointF, Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QColor, QFont, QPainter, QPalette, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QGroupBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
try:
    from ..models.configuration import (
        CellSpec,
        Configuration,
        FrequencySpec,
        LayerSpec,
        MaskHole,
        MaskSpec,
        load_configuration,
        save_configuration,
    )
    from ..services.rcwa_runner import SimulationResult, run_simulation
    from ..services.run_logger import RunContext, RunLogger
    from ..services.optimizer import (
        AlgorithmSettings,
        OptimizationConstraint,
        OptimizationJob,
        OptimizationObjective,
        OptimizationVariable,
        OptimizerService,
    )
except ImportError:  # pragma: no cover - fallback when run as loose scripts
    from models.configuration import (  # type: ignore[no-redef]
        CellSpec,
        Configuration,
        FrequencySpec,
        LayerSpec,
        MaskHole,
        MaskSpec,
        load_configuration,
        save_configuration,
    )
    from services.rcwa_runner import SimulationResult, run_simulation  # type: ignore[no-redef]
    from services.run_logger import RunContext, RunLogger  # type: ignore[no-redef]
    from services.optimizer import (  # type: ignore[no-redef]
        AlgorithmSettings,
        OptimizationConstraint,
        OptimizationJob,
        OptimizationObjective,
        OptimizationVariable,
        OptimizerService,
    )


class SimulationWorker(QObject):
    finished = pyqtSignal(SimulationResult)
    error = pyqtSignal(str)

    def __init__(self, config: Configuration, repo_root: Path, log_dir: Path | None = None) -> None:
        super().__init__()
        self.config = config
        self.repo_root = repo_root
        self.log_dir = log_dir

    def run(self) -> None:
        try:
            result = run_simulation(self.config, self.repo_root, log_dir=self.log_dir)
        except Exception as exc:  # pragma: no cover - user I/O
            self.error.emit(str(exc))
        else:
            self.finished.emit(result)


class LossPlotWidget(QWidget):
    """Lightweight plotting widget that avoids the matplotlib dependency."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._freq: List[float] = []
        self._rl: List[float] = []
        self.setMinimumHeight(220)
        self._title_font = QFont()
        self._title_font.setPointSize(10)

    def set_data(self, freq: List[float], rl: List[float]) -> None:
        self._freq = list(freq)
        self._rl = list(rl)
        self.update()

    def clear(self) -> None:
        self._freq.clear()
        self._rl.clear()
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        painter.fillRect(rect, self.palette().color(QPalette.ColorRole.Base))

        if not self._freq or not self._rl:
            painter.setPen(self.palette().color(QPalette.ColorRole.Text))
            painter.drawText(
                rect,
                Qt.AlignmentFlag.AlignCenter,
                "Reflection Loss plot will appear here after a run.",
            )
            return

        left_margin = 60
        right_margin = 20
        top_margin = 24
        bottom_margin = 50

        plot_rect = QRectF(
            rect.left() + left_margin,
            rect.top() + top_margin,
            rect.width() - left_margin - right_margin,
            rect.height() - top_margin - bottom_margin,
        )

        if plot_rect.width() <= 0 or plot_rect.height() <= 0:
            return

        min_x = min(self._freq)
        max_x = max(self._freq)
        min_y = min(self._rl)
        max_y = max(self._rl)

        if math.isclose(max_x, min_x):
            max_x += 1.0
        if math.isclose(max_y, min_y):
            max_y += 1.0

        def x_to_px(x: float) -> float:
            return plot_rect.left() + (x - min_x) / (max_x - min_x) * plot_rect.width()

        def y_to_px(y: float) -> float:
            return plot_rect.bottom() - (y - min_y) / (max_y - min_y) * plot_rect.height()

        axis_color = self.palette().color(QPalette.ColorRole.Mid)
        painter.setPen(QPen(axis_color, 1))
        painter.drawRect(plot_rect)

        grid_pen = QPen(axis_color, 1, Qt.PenStyle.DashLine)
        painter.setPen(grid_pen)
        steps = 4
        for i in range(1, steps):
            x = plot_rect.left() + i * plot_rect.width() / steps
            painter.drawLine(QPointF(x, plot_rect.top()), QPointF(x, plot_rect.bottom()))
            y = plot_rect.top() + i * plot_rect.height() / steps
            painter.drawLine(QPointF(plot_rect.left(), y), QPointF(plot_rect.right(), y))

        painter.setPen(QPen(self.palette().color(QPalette.ColorRole.Text), 2))
        points = [
            (x_to_px(x), y_to_px(y))
            for x, y in zip(self._freq, self._rl)
        ]
        if points:
            previous = points[0]
            for current in points[1:]:
                painter.drawLine(QPointF(previous[0], previous[1]), QPointF(current[0], current[1]))
                previous = current

        marker_radius = 4
        marker_color = QColor(33, 150, 243)
        painter.setBrush(marker_color)
        painter.setPen(QPen(marker_color.darker(), 1))
        for x, y in points:
            painter.drawEllipse(QPointF(x, y), marker_radius / 2, marker_radius / 2)

        label_pen = QPen(self.palette().color(QPalette.ColorRole.Text))
        painter.setPen(label_pen)
        painter.setFont(self._title_font)
        painter.drawText(
            QRectF(rect.left(), rect.top() + 16, rect.width(), 16),
            Qt.AlignmentFlag.AlignCenter,
            "Reflection Loss vs Frequency",
        )

        painter.drawText(
            QRectF(plot_rect.left(), plot_rect.bottom() + 28, plot_rect.width(), 20),
            Qt.AlignmentFlag.AlignCenter,
            f"Frequency (GHz) [{min_x:.2f} – {max_x:.2f}]",
        )

        painter.save()
        painter.translate(rect.left() + 18, rect.center().y())
        painter.rotate(-90)
        painter.drawText(
            QRectF(-rect.height() / 2, -12, rect.height(), 20),
            Qt.AlignmentFlag.AlignCenter,
            f"Reflection Loss (dB) [{min_y:.2f} – {max_y:.2f}]",
        )
        painter.restore()

class MainWindow(QMainWindow):
    """Standalone desktop controller for RCWA simulations."""

    def __init__(self, base_dir: Path) -> None:
        super().__init__()
        self.base_dir = base_dir
        self.repo_root = base_dir.parent

        self.current_config_path: Optional[Path] = None
        self.config = Configuration()

        self.general_fields: Dict[str, QLineEdit] = {}
        self.material_fields: Dict[str, QLineEdit] = {}
        self.geometry_table: Optional[QTableWidget] = None
        self.log_view: Optional[QTextEdit] = None
        self.plot_widget: Optional["LossPlotWidget"] = None
        self.run_button: Optional[QPushButton] = None

        self.run_logger = RunLogger(self.repo_root)
        self.current_run_context: Optional[RunContext] = None
        self.current_config_snapshot: Optional[Configuration] = None

        self.optimizer_service = OptimizerService(self.repo_root)
        self.optimizer_timer = QTimer(self)
        self.optimizer_timer.setInterval(1000)
        self.optimizer_timer.timeout.connect(self._poll_optimizer_status)
        self.optimizer_job_id: Optional[str] = None
        self.optimizer_job_dir: Optional[Path] = None
        self.optimizer_last_state: Optional[str] = None
        self.optimizer_best_eval: Optional[Dict[str, Any]] = None

        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[SimulationWorker] = None

        self.setWindowTitle("RCWA Desktop Controller")
        self.resize(1120, 720)

        self._init_menu()
        self._init_tabs()
        self._populate_fields()

    # ------------------------------------------------------------------ setup
    def _init_menu(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        open_action = QAction("Open Config JSON…", self)
        open_action.triggered.connect(self._select_config_file)
        file_menu.addAction(open_action)

        save_action = QAction("Save Config As…", self)
        save_action.triggered.connect(self._save_config_as)
        file_menu.addAction(save_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _init_tabs(self) -> None:
        tabs = QTabWidget(self)
        tabs.addTab(self._build_general_tab(), "General")
        tabs.addTab(self._build_materials_tab(), "Materials")
        tabs.addTab(self._build_geometry_tab(), "Geometry")
        tabs.addTab(self._build_execution_tab(), "Execution")
        self.setCentralWidget(tabs)

    # ------------------------------------------------------------------ tabs
    def _build_general_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("پارامترهای عمومی شبیه‌سازی"))

        form = QFormLayout()
        layout.addLayout(form)

        self.general_fields["freq_start"] = QLineEdit()
        self.general_fields["freq_stop"] = QLineEdit()
        self.general_fields["freq_points"] = QLineEdit()
        self.general_fields["cell_lx"] = QLineEdit()
        self.general_fields["cell_ly"] = QLineEdit()
        self.general_fields["output_prefix"] = QLineEdit()
        self.general_fields["polarization"] = QLineEdit()
        self.general_fields["n_harmonics"] = QLineEdit()
        self.general_fields["theta_deg"] = QLineEdit()
        self.general_fields["phi_deg"] = QLineEdit()

        form.addRow("فرکانس شروع (GHz)", self.general_fields["freq_start"])
        form.addRow("فرکانس پایان (GHz)", self.general_fields["freq_stop"])
        form.addRow("تعداد نقاط", self.general_fields["freq_points"])
        form.addRow("ابعاد سلول X (m)", self.general_fields["cell_lx"])
        form.addRow("ابعاد سلول Y (m)", self.general_fields["cell_ly"])
        form.addRow("Output Prefix", self.general_fields["output_prefix"])
        form.addRow("Polarization (TE/TM/AVG)", self.general_fields["polarization"])
        form.addRow("Harmonics (مثلاً 11,11)", self.general_fields["n_harmonics"])
        form.addRow("زاویهٔ θ (deg)", self.general_fields["theta_deg"])
        form.addRow("زاویهٔ φ (deg)", self.general_fields["phi_deg"])

        save_btn = QPushButton("ذخیرهٔ تغییرات در حافظه")
        save_btn.clicked.connect(self._handle_save_to_memory)
        layout.addWidget(save_btn)

        notes = QTextEdit()
        notes.setPlaceholderText("یادداشت‌های طراحی یا TODO ها…")
        layout.addWidget(notes)
        return widget

    def _build_materials_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = QFormLayout()
        layout.addLayout(form)

        self._add_file_row(form, "لایهٔ بالایی (m1.csv)", "layer_top_csv")
        self._add_number_row(form, "ضخامت لایهٔ بالا (mm)", "layer_top_thickness")

        self._add_file_row(form, "ماسک - مادهٔ صلب (m2.csv)", "mask_solid_csv")
        self._add_file_row(form, "ماسک - مادهٔ سوراخ (mhole.csv)", "mask_hole_csv")
        self._add_number_row(form, "ضخامت ماسک (mm)", "mask_thickness")
        self._add_number_row(form, "Grid Nx", "mask_grid_nx", integer=True)
        self._add_number_row(form, "Grid Ny", "mask_grid_ny", integer=True)

        self._add_file_row(form, "لایهٔ پایینی (m3.csv)", "layer_bottom_csv")
        self._add_number_row(form, "ضخامت لایهٔ پایین (mm)", "layer_bottom_thickness")

        hint = QLabel("مسیر فایل‌های مواد را دقیقاً مطابق نام موجود در rcwa_adaptor انتخاب کنید.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return widget

    def _build_geometry_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.geometry_table = QTableWidget(0, 6)
        self.geometry_table.setHorizontalHeaderLabels(
            ['Shape', 'X (mm)', 'Y (mm)', 'Size1 (mm)', 'Size2 (mm)', 'Rotation (deg)']
        )
        self.geometry_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.geometry_table)

        button_row = QHBoxLayout()
        add_circle_btn = QPushButton('Add Circle')
        add_circle_btn.clicked.connect(self._add_circle_row)
        button_row.addWidget(add_circle_btn)

        add_square_btn = QPushButton('Add Square')
        add_square_btn.clicked.connect(self._add_square_row)
        button_row.addWidget(add_square_btn)

        add_rectangle_btn = QPushButton('Add Rectangle')
        add_rectangle_btn.clicked.connect(self._add_rectangle_row)
        button_row.addWidget(add_rectangle_btn)

        add_ellipse_btn = QPushButton('Add Ellipse')
        add_ellipse_btn.clicked.connect(self._add_ellipse_row)
        button_row.addWidget(add_ellipse_btn)

        remove_btn = QPushButton('Remove Selected')
        remove_btn.clicked.connect(self._remove_selected_row)
        button_row.addWidget(remove_btn)

        layout.addLayout(button_row)

        info_text = (
            'Shape can be one of circle, square, rectangle or ellipse. ' 
            'Size1 and Size2 capture the primary dimensions (diameter, width/height or axes) and Rotation sets the angle in degrees. ' 
            'At the moment the solver treats non-circular holes as equivalent discs, but every parameter is preserved in the configuration and logs.'
        )
        info = QLabel(info_text)
        info.setWordWrap(True)
        layout.addWidget(info)
        return widget

    def _build_execution_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self._on_run_clicked)
        layout.addWidget(self.run_button)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)

        self.plot_widget = LossPlotWidget(widget)
        layout.addWidget(self.plot_widget)


        layout.addWidget(QLabel("پس از اجرای موفق، نمودار Reflection Loss در اینجا نمایش داده می‌شود."))
        return widget

    # ------------------------------------------------------------------ actions
    def _handle_save_to_memory(self) -> None:
        if self._update_config_from_fields():
            QMessageBox.information(self, "Saved", "پیکربندی در حافظه به‌روزرسانی شد.")

    def _select_config_file(self) -> None:
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilters(["JSON files (*.json)", "All files (*)"])
        if dialog.exec():
            selected = dialog.selectedFiles()
            if selected:
                path = Path(selected[0])
                try:
                    self.config = load_configuration(path)
                except Exception as exc:  # pragma: no cover
                    QMessageBox.critical(
                        self,
                        "Load failed",
                        f"Could not load configuration:\n{exc}",
                    )
                    return
                self.current_config_path = path
                self._populate_fields()
                QMessageBox.information(
                    self,
                    "Config loaded",
                    f"Configuration loaded from:\n{path}",
                )

    def _save_config_as(self) -> None:
        if not self._update_config_from_fields():
            return
        dialog = QFileDialog(self)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilters(["JSON files (*.json)", "All files (*)"])
        dialog.setDefaultSuffix("json")
        if dialog.exec():
            selected = dialog.selectedFiles()
            if selected:
                path = Path(selected[0])
                try:
                    save_configuration(self.config, path)
                except Exception as exc:  # pragma: no cover
                    QMessageBox.critical(
                        self,
                        "Save failed",
                        f"Could not save configuration:\n{exc}",
                    )
                    return
                self.current_config_path = path
                QMessageBox.information(
                    self,
                    "Saved",
                    f"Configuration saved to:\n{path}",
                )

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "About",
            "RCWA Desktop Controller\n"
            "این نسخه امکان تنظیم پارامترها، ذخیره/بارگذاری پیکربندی، اجرای شبیه‌سازی و مشاهدهٔ نمودار را فراهم می‌کند.",
        )

    # --------------------------------------------------------------- helpers
    def _populate_fields(self) -> None:
        self.general_fields["freq_start"].setText(f"{self.config.freq.start}")
        self.general_fields["freq_stop"].setText(f"{self.config.freq.stop}")
        self.general_fields["freq_points"].setText(f"{self.config.freq.points}")
        self.general_fields["cell_lx"].setText(f"{self.config.cell.Lx_m}")
        self.general_fields["cell_ly"].setText(f"{self.config.cell.Ly_m}")
        self.general_fields["output_prefix"].setText(self.config.output_prefix)
        self.general_fields["polarization"].setText(self.config.polarization)
        self.general_fields["n_harmonics"].setText(",".join(str(h) for h in self.config.n_harmonics))
        self.general_fields["theta_deg"].setText(f"{self.config.theta_deg}")
        self.general_fields["phi_deg"].setText(f"{self.config.phi_deg}")

        self._populate_material_fields()
        self._populate_geometry_table()
        self._clear_plot()
        if self.log_view:
            self.log_view.clear()

    def _populate_material_fields(self) -> None:
        if not self.material_fields:
            return
        self.material_fields["layer_top_csv"].setText(self.config.layer_top.material_csv)
        self.material_fields["layer_top_thickness"].setText(f"{self.config.layer_top.thickness_m * 1000:.3f}")
        self.material_fields["mask_solid_csv"].setText(self.config.mask.solid_csv)
        self.material_fields["mask_hole_csv"].setText(self.config.mask.hole_csv)
        self.material_fields["mask_thickness"].setText(f"{self.config.mask.thickness_m * 1000:.3f}")
        self.material_fields["mask_grid_nx"].setText(str(self.config.mask.grid_nx))
        self.material_fields["mask_grid_ny"].setText(str(self.config.mask.grid_ny))
        self.material_fields["layer_bottom_csv"].setText(self.config.layer_bottom.material_csv)
        self.material_fields["layer_bottom_thickness"].setText(
            f"{self.config.layer_bottom.thickness_m * 1000:.3f}"
        )

    def _populate_geometry_table(self) -> None:
        if self.geometry_table is None:
            return
        self.geometry_table.setRowCount(0)
        for hole in self.config.mask.holes:
            row = self.geometry_table.rowCount()
            self.geometry_table.insertRow(row)
            self.geometry_table.setItem(row, 0, QTableWidgetItem(hole.shape))
            self.geometry_table.setItem(row, 1, QTableWidgetItem(f"{hole.x_m * 1000:.3f}"))
            self.geometry_table.setItem(row, 2, QTableWidgetItem(f"{hole.y_m * 1000:.3f}"))
            self.geometry_table.setItem(row, 3, QTableWidgetItem(f"{hole.size1 * 1000:.3f}"))
            size2_value = hole.size2 if hole.size2 is not None else (hole.size1 if hole.shape in {"square", "rectangle", "ellipse"} else None)
            self.geometry_table.setItem(row, 4, QTableWidgetItem(f"{(size2_value or 0) * 1000:.3f}" if size2_value else ""))
            self.geometry_table.setItem(row, 5, QTableWidgetItem(f"{getattr(hole, 'rotation_deg', 0.0):.3f}"))

    def _clear_plot(self) -> None:
        if self.plot_widget:
            self.plot_widget.clear()


    def _add_file_row(self, form: QFormLayout, label: str, key: str) -> None:
        line = QLineEdit()
        browse = QPushButton("Browse")
        browse.clicked.connect(lambda: self._browse_material_file(key))
        container = QWidget()
        row_layout = QHBoxLayout(container)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(line)
        row_layout.addWidget(browse)
        form.addRow(label, container)
        self.material_fields[key] = line

    def _add_number_row(self, form: QFormLayout, label: str, key: str, integer: bool = False) -> None:
        line = QLineEdit()
        if integer:
            line.setPlaceholderText("integer")
        self.material_fields[key] = line
        form.addRow(label, line)

    def _browse_material_file(self, key: str) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV",
            str(self.base_dir),
            "CSV files (*.csv);;All files (*)",
        )
        if filename:
            self.material_fields[key].setText(filename)

    def _add_circle_row(self) -> None:
        if self.geometry_table is None:
            return
        row = self.geometry_table.rowCount()
        self.geometry_table.insertRow(row)
        defaults = ["circle", "0.0", "0.0", "6.0", "", "0.0"]
        for col, value in enumerate(defaults):
            self.geometry_table.setItem(row, col, QTableWidgetItem(value))

    def _add_square_row(self) -> None:
        if self.geometry_table is None:
            return
        row = self.geometry_table.rowCount()
        self.geometry_table.insertRow(row)
        defaults = ["square", "0.0", "0.0", "4.0", "4.0", "0.0"]
        for col, value in enumerate(defaults):
            self.geometry_table.setItem(row, col, QTableWidgetItem(value))

    def _add_rectangle_row(self) -> None:
        if self.geometry_table is None:
            return
        row = self.geometry_table.rowCount()
        self.geometry_table.insertRow(row)
        defaults = ["rectangle", "0.0", "0.0", "6.0", "3.0", "0.0"]
        for col, value in enumerate(defaults):
            self.geometry_table.setItem(row, col, QTableWidgetItem(value))

    def _add_ellipse_row(self) -> None:
        if self.geometry_table is None:
            return
        row = self.geometry_table.rowCount()
        self.geometry_table.insertRow(row)
        defaults = ["ellipse", "0.0", "0.0", "6.0", "3.0", "0.0"]
        for col, value in enumerate(defaults):
            self.geometry_table.setItem(row, col, QTableWidgetItem(value))

    def _remove_selected_row(self) -> None:
        if self.geometry_table is None:
            return
        row = self.geometry_table.currentRow()
        if row >= 0:
            self.geometry_table.removeRow(row)

    # ----------------------------------------------------------- update logic
    def _update_config_from_fields(self) -> bool:
        try:
            freq = FrequencySpec(
                start=float(self.general_fields["freq_start"].text()),
                stop=float(self.general_fields["freq_stop"].text()),
                points=int(self.general_fields["freq_points"].text()),
            )
            cell = CellSpec(
                Lx_m=float(self.general_fields["cell_lx"].text()),
                Ly_m=float(self.general_fields["cell_ly"].text()),
            )
            prefix = self.general_fields["output_prefix"].text().strip() or "desktop_config"
        except ValueError:
            QMessageBox.warning(self, "Invalid input", "مقادیر عمومی معتبر نیستند.")
            return False

        if freq.start >= freq.stop:
            QMessageBox.warning(self, "Invalid frequency", "فرکانس شروع باید کمتر از پایان باشد.")
            return False
        if freq.points < 2:
            QMessageBox.warning(self, "Invalid points", "تعداد نقاط باید >= 2 باشد.")
            return False
        if cell.Lx_m <= 0 or cell.Ly_m <= 0:
            QMessageBox.warning(self, "Invalid cell", "ابعاد سلول باید مثبت باشند.")
            return False

        polarization = self.general_fields["polarization"].text().strip().upper() or "TE"
        if polarization not in {"TE", "TM", "AVG"}:
            QMessageBox.warning(self, "Invalid polarization", "قطبش باید TE یا TM یا AVG باشد.")
            return False

        harmonics_text = self.general_fields["n_harmonics"].text().strip() or "11,11"
        try:
            harmonics = [int(part.strip()) for part in harmonics_text.split(",")]
            if not (1 <= len(harmonics) <= 2):
                raise ValueError
            for h in harmonics:
                if h < 1 or h % 2 == 0:
                    raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Invalid harmonics", "مثال معتبر: 11 یا 11,11 (اعداد فرد مثبت).")
            return False

        try:
            theta_deg = float(self.general_fields["theta_deg"].text() or "0")
            phi_deg = float(self.general_fields["phi_deg"].text() or "0")
        except ValueError:
            QMessageBox.warning(self, "Invalid angles", "زاویه‌ها باید عددی باشند.")
            return False

        if not self._collect_material_fields():
            return False
        if not self._collect_geometry_from_table():
            return False

        self.config.freq = freq
        self.config.cell = cell
        self.config.output_prefix = prefix
        self.config.polarization = polarization
        self.config.n_harmonics = harmonics
        self.config.theta_deg = theta_deg
        self.config.phi_deg = phi_deg
        return True

    def _collect_material_fields(self) -> bool:
        try:
            layer_top_thickness = float(self.material_fields["layer_top_thickness"].text()) / 1000
            mask_thickness = float(self.material_fields["mask_thickness"].text()) / 1000
            layer_bottom_thickness = float(self.material_fields["layer_bottom_thickness"].text()) / 1000
            grid_nx = int(self.material_fields["mask_grid_nx"].text())
            grid_ny = int(self.material_fields["mask_grid_ny"].text())
        except (ValueError, KeyError):
            QMessageBox.warning(self, "Invalid materials", "ورودی‌های مواد یا ماسک معتبر نیستند.")
            return False

        if layer_top_thickness <= 0 or mask_thickness <= 0 or layer_bottom_thickness <= 0:
            QMessageBox.warning(self, "Thickness error", "ضخامت‌ها باید مثبت باشند.")
            return False
        if grid_nx <= 0 or grid_ny <= 0:
            QMessageBox.warning(self, "Grid error", "Grid Nx/Ny باید مثبت باشند.")
            return False

        self.config.layer_top = LayerSpec(
            material_csv=self.material_fields["layer_top_csv"].text().strip(),
            thickness_m=layer_top_thickness,
        )
        self.config.mask = MaskSpec(
            solid_csv=self.material_fields["mask_solid_csv"].text().strip(),
            hole_csv=self.material_fields["mask_hole_csv"].text().strip(),
            thickness_m=mask_thickness,
            grid_nx=grid_nx,
            grid_ny=grid_ny,
            holes=self.config.mask.holes,  # temporary until geometry table updates
        )
        self.config.layer_bottom = LayerSpec(
            material_csv=self.material_fields["layer_bottom_csv"].text().strip(),
            thickness_m=layer_bottom_thickness,
        )
        return True

    def _collect_geometry_from_table(self) -> bool:
        if self.geometry_table is None:
            return True
        holes: List[MaskHole] = []
        valid_shapes = {"circle", "square", "rectangle", "ellipse"}
        for row in range(self.geometry_table.rowCount()):
            items = [self.geometry_table.item(row, col) for col in range(6)]
            if not items or items[0] is None or not (items[0].text() and items[0].text().strip()):
                continue

            shape = items[0].text().strip().lower()
            if shape not in valid_shapes:
                QMessageBox.warning(self, "Geometry error", f"Unsupported shape at row {row + 1}: {shape}")
                return False

            try:
                x_mm = float(items[1].text()) if items[1] and items[1].text().strip() else 0.0
                y_mm = float(items[2].text()) if items[2] and items[2].text().strip() else 0.0
                size1_mm = float(items[3].text()) if items[3] and items[3].text().strip() else 0.0
            except ValueError:
                QMessageBox.warning(self, "Geometry error", f"Invalid numeric entry for row {row + 1}.")
                return False

            if size1_mm <= 0.0:
                QMessageBox.warning(self, "Geometry error", f"Size1 must be positive (row {row + 1}).")
                return False

            size2_mm: float | None = None
            if shape in {"square", "rectangle", "ellipse"}:
                text_val = items[4].text().strip() if items[4] and items[4].text() else ""
                try:
                    size2_mm = float(text_val) if text_val else size1_mm
                except ValueError:
                    QMessageBox.warning(self, "Geometry error", f"Invalid Size2 value (row {row + 1}).")
                    return False
                if size2_mm <= 0.0:
                    QMessageBox.warning(self, "Geometry error", f"Size2 must be positive (row {row + 1}).")
                    return False
            else:
                size2_mm = None

            rotation_deg = 0.0
            if items[5] and items[5].text() and items[5].text().strip():
                try:
                    rotation_deg = float(items[5].text())
                except ValueError:
                    QMessageBox.warning(self, "Geometry error", f"Invalid rotation value (row {row + 1}).")
                    return False

            holes.append(
                MaskHole(
                    shape=shape,
                    x_m=x_mm / 1000.0,
                    y_m=y_mm / 1000.0,
                    size1=size1_mm / 1000.0,
                    size2=(size2_mm / 1000.0) if size2_mm is not None else None,
                    rotation_deg=rotation_deg,
                )
            )
        self.config.mask.holes = holes
        return True


    def _build_optimizer_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        return widget

    # ------------------------------------------------------------ execution
    def _append_log(self, message: str) -> None:
        if self.log_view:
            self.log_view.append(message)
        if self.current_run_context:
            self.current_run_context.append_gui(message)

    def _on_run_clicked(self) -> None:
        if not self._update_config_from_fields():
            return
        if self.run_button:
            self.run_button.setEnabled(False)
        if self.log_view:
            self.log_view.clear()
        self._clear_plot()

        config_snapshot = Configuration.from_json(self.config.to_json())
        self.current_config_snapshot = config_snapshot
        run_context = self.run_logger.start_run(config_snapshot.output_prefix)
        self.current_run_context = run_context
        config_path = run_context.record_configuration(config_snapshot)

        self._append_log(f"Run directory: {run_context.directory}")
        self._append_log(f"Saved configuration to: {config_path}")
        self._append_log("Running simulation...")

        self.worker = SimulationWorker(
            config_snapshot,
            self.repo_root,
            log_dir=run_context.directory,
        )
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_simulation_finished)
        self.worker.error.connect(self._on_simulation_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self._cleanup_thread)
        self.worker_thread.start()

    def _on_simulation_finished(self, result: SimulationResult) -> None:
        if self.run_button:
            self.run_button.setEnabled(True)
        self._append_log("Simulation completed successfully.")
        if result.stdout.strip():
            self._append_log("STDOUT:\n" + result.stdout.strip())
        if result.stderr.strip():
            self._append_log("STDERR:\n" + result.stderr.strip())

        summary_lines: List[str] = [f"Results CSV: {result.output_csv}"]
        self._append_log(summary_lines[0])

        if result.freq_GHz and result.RL_dB:
            samples = len(result.freq_GHz)
            min_rl = min(result.RL_dB)
            max_rl = max(result.RL_dB)
            min_idx = result.RL_dB.index(min_rl)
            max_idx = result.RL_dB.index(max_rl)
            freq_min = result.freq_GHz[min_idx]
            freq_max = result.freq_GHz[max_idx]
            summary_lines.append(f"Samples: {samples}")
            summary_lines.append(f"Min RL: {min_rl:.3f} dB @ {freq_min:.3f} GHz")
            summary_lines.append(f"Max RL: {max_rl:.3f} dB @ {freq_max:.3f} GHz")
            for line in summary_lines[1:]:
                self._append_log(line)
        else:
            self._append_log("No reflection loss samples recorded.")

        shape_lines: List[str] = []
        if self.current_config_snapshot:
            for idx, hole in enumerate(self.current_config_snapshot.mask.holes, start=1):
                parts = [
                    f"Hole {idx}: shape={hole.shape}",
                    f"x={hole.x_m * 1e3:.3f} mm",
                    f"y={hole.y_m * 1e3:.3f} mm",
                    f"size1={hole.size1 * 1e3:.3f} mm",
                ]
                if hole.size2 is not None or hole.shape in {"square", "rectangle", "ellipse"}:
                    size2_display = hole.size2 if hole.size2 is not None else hole.size1
                    parts.append(f"size2={size2_display * 1e3:.3f} mm")
                rotation = getattr(hole, "rotation_deg", 0.0) or 0.0
                if rotation:
                    parts.append(f"rotation={rotation:.2f} deg")
                line = ", ".join(parts)
                shape_lines.append(line)
                self._append_log(line)
        summary_lines.extend(shape_lines)

        artefact_lines: List[str] = []
        if result.loss_plot_image:
            artefact_lines.append(f"Reflection loss plot: {result.loss_plot_image}")
        if result.mask_layout_image:
            artefact_lines.append(f"Mask layout image: {result.mask_layout_image}")
        for line in artefact_lines:
            self._append_log(line)
        summary_lines.extend(artefact_lines)

        if result.warnings:
            for warning in result.warnings:
                message = f"Warning: {warning}"
                self._append_log(message)
                summary_lines.append(message)

        if self.current_run_context:
            summary_path = self.current_run_context.directory / "summary.txt"
            summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

        if self.plot_widget:
            self.plot_widget.set_data(result.freq_GHz, result.RL_dB)


    def _on_simulation_error(self, message: str) -> None:
        if self.run_button:
            self.run_button.setEnabled(True)
        self._append_log("Simulation failed:")
        self._append_log(message)
        if self.current_run_context:
            summary_path = self.current_run_context.directory / "summary.txt"
            summary_path.write_text(f"Simulation failed:\n{message}", encoding="utf-8")
        QMessageBox.critical(self, "Simulation failed", message)

    def _cleanup_thread(self) -> None:
        if self.worker_thread:
            self.worker_thread.deleteLater()
        self.worker_thread = None
        self.worker = None
