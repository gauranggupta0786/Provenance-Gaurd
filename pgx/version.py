"""The v2 defense version: one digest over ``pg/`` (v1 code), ``pgx/`` (v2 code) and the v2 data the runs depend on."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2_DATA_FILES = ("data/v2/agentdojo_tasks.json", "data/v2/split.json", "data/v2/injections.jsonl", "data/v2/plan.json",
                 "data/v2/adaptive_targets.json")
V2_SCRIPT_FILES = ("scripts/run_v2_agentdojo.py", "scripts/v2_prepare.py", "scripts/run_v2_adaptive.py", "scripts/stats_v2.py",
                   "scripts/attack_curves.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pgx_files(root: Path = ROOT) -> list[Path]:
    files = sorted((root / "pg").glob("*.py")) + sorted((root / "pgx").rglob("*.py"))
    files += [root / n for n in (*V2_DATA_FILES, *V2_SCRIPT_FILES) if (root / n).exists()]
    return [f for f in files if "__pycache__" not in f.parts]


def pgx_digest(root: Path = ROOT) -> tuple[str, dict[str, str]]:
    hashes = {str(p.relative_to(root)).replace("\\", "/"): _sha(p) for p in pgx_files(root)}
    blob = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest(), hashes
