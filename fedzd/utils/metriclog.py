"""Structured metric logging.

Two sinks per run:
  * `events.jsonl`  -- append-only stream of every logged record (raw, per-round
    metrics kept so we can plot convergence curves, not just endpoints -- Stage 6).
  * `<name>.csv`    -- tidy tabular exports for specific metric families.

Also a plain-text run log for human-readable narration of what happened.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self.run_dir / "events.jsonl"
        self._py = logging.getLogger(f"fedzd.{self.run_dir.name}")
        if not self._py.handlers:
            self._py.setLevel(logging.INFO)
            fh = logging.FileHandler(self.run_dir / "run.log")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("%(message)s"))
            self._py.addHandler(fh)
            self._py.addHandler(sh)
            self._py.propagate = False

    # ---- human-readable ----
    def info(self, msg: str) -> None:
        self._py.info(msg)

    def warn(self, msg: str) -> None:
        self._py.warning("WARN: " + msg)

    # ---- machine-readable ----
    def event(self, kind: str, **fields: Any) -> None:
        rec = {"t": time.time(), "kind": kind, **fields}
        with open(self._events_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def table(self, name: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> Path:
        path = self.run_dir / f"{name}.csv"
        if not rows:
            path.write_text("")
            return path
        fieldnames = fieldnames or list({k for r in rows for k in r.keys()})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return path

    def dump_json(self, name: str, obj: Any) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        return path
