"""Small file-backed local run history. Designed for single-user development."""
from __future__ import annotations
import json
import os
import threading
from pathlib import Path
from typing import Any

_lock = threading.RLock()


def read_history(path: Path) -> list[dict[str, Any]]:
    with _lock:
        if not path.exists():
            return []
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
            return content if isinstance(content, list) else []
        except (ValueError, OSError):
            return []


def add_run(path: Path, record: dict[str, Any], limit: int) -> None:
    with _lock:
        history = read_history(path)
        history.insert(0, record)
        history = history[:limit]
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(history, indent=2), encoding="utf-8")
        os.replace(temp, path)


def clear_history(path: Path, media_dir: Path) -> None:
    with _lock:
        if path.exists():
            path.unlink()
        if media_dir.exists():
            for item in media_dir.iterdir():
                if item.is_file() and item.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    item.unlink()
