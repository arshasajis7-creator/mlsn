"""Persistence helpers for saving and restoring desktop configurations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..models.configuration import Configuration, load_configuration, save_configuration


@dataclass
class SettingsStore:
    base_dir: Path

    def __post_init__(self) -> None:
        self.store_dir = self.base_dir / "config_store"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.last_run_path = self.store_dir / "last_run.json"

    def save_last(self, config: Configuration) -> Path:
        """Persist *config* as the most recently used configuration."""

        save_configuration(config, self.last_run_path)
        return self.last_run_path

    def load_last(self) -> Optional[Configuration]:
        if not self.last_run_path.exists():
            return None
        return load_configuration(self.last_run_path)

    def save_to_path(self, config: Configuration, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_configuration(config, path)

    def load_from_path(self, path: Path) -> Configuration:
        return load_configuration(path)
