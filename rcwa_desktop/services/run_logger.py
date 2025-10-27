"""Centralised logging utilities for UI and adapter runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..models.configuration import Configuration, save_configuration


def _sanitize_prefix(prefix: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "-" for ch in prefix)
    return cleaned.strip("-") or "run"


@dataclass
class RunContext:
    directory: Path

    @property
    def gui_log_path(self) -> Path:
        return self.directory / "gui.log"

    def append_gui(self, message: str) -> None:
        with self.gui_log_path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")

    def record_configuration(self, config: Configuration) -> Path:
        config_path = self.directory / "config.json"
        save_configuration(config, config_path)
        return config_path

    def record_adapter_output(self, stdout: str, stderr: str) -> None:
        (self.directory / "adapter_stdout.txt").write_text(stdout, encoding="utf-8")
        (self.directory / "adapter_stderr.txt").write_text(stderr, encoding="utf-8")


class RunLogger:
    """Creates per-run folders beneath ``logs/`` at the project root."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.logs_root = self.project_root / "logs"
        self.logs_root.mkdir(parents=True, exist_ok=True)

    def start_run(self, prefix: str) -> RunContext:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_prefix = _sanitize_prefix(prefix)
        directory = self.logs_root / f"{safe_prefix}_{timestamp}"
        directory.mkdir(parents=True, exist_ok=True)
        return RunContext(directory=directory)
