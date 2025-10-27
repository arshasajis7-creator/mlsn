import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

try:  # pragma: no cover - exercised differently when packaged vs script
    from .ui.main_window import MainWindow
except ImportError:  # pragma: no cover - fallback for running from source tree
    from ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    # Ensure application has a deterministic working directory
    base_dir = Path(__file__).resolve().parent
    window = MainWindow(base_dir=base_dir)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
