"""Config system: layered YAML configs -> a frozen, hashable run config that is
snapshotted into each run directory.

Design goals (Stage 6 rigor):
  * No magic numbers in scripts: every hyperparameter comes from a config file.
  * Every run gets its own directory holding (a) the fully-resolved config it ran
    with, (b) a content hash of that config, and (c) all metric logs.
  * Configs compose: a base config + one or more overlays + CLI dotted overrides.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_dotted(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    # best-effort type coercion from string CLI values
    if isinstance(value, str):
        v = value.strip()
        if v[:1] in "[{":                       # JSON list/dict override
            try:
                cur[keys[-1]] = json.loads(v); return
            except ValueError:
                pass
        for caster in (int, float):
            try:
                value = caster(value)
                break
            except ValueError:
                continue
        else:
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
    cur[keys[-1]] = value


def load_config(
    paths: List[str],
    overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Merge a list of YAML files (later overrides earlier), then apply
    `key.subkey=value` CLI overrides."""
    cfg: Dict[str, Any] = {}
    for p in paths:
        with open(p, "r") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got: {ov}")
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k.strip(), v.strip())
    return cfg


def config_hash(cfg: Dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


@dataclass
class RunContext:
    """A single experiment run: its resolved config, its output directory, and
    the config hash that ties every logged number back to exact settings."""

    cfg: Dict[str, Any]
    run_dir: Path
    cfg_hash: str

    @property
    def seed(self) -> int:
        return int(self.cfg.get("seed", 0))

    def path(self, *parts: str) -> Path:
        p = self.run_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def make_run_context(
    cfg: Dict[str, Any],
    runs_root: str = "runs",
    tag: Optional[str] = None,
) -> RunContext:
    h = config_hash(cfg)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{tag or cfg.get('experiment', 'run')}_seed{cfg.get('seed', 0)}_{h}_{stamp}"
    run_dir = Path(runs_root) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Snapshot the exact config used.
    with open(run_dir / "config.resolved.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    with open(run_dir / "config.hash", "w") as f:
        f.write(h + "\n")
    return RunContext(cfg=cfg, run_dir=run_dir, cfg_hash=h)
