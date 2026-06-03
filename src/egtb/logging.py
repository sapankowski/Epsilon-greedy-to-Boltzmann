"""CSV and JSON helpers for experiment outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class CSVLogger:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        cleaned = {field: row.get(field, "") for field in self.fieldnames}
        self.writer.writerow(cleaned)
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
