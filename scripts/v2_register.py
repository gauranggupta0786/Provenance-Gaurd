"""Record the v2 preregistration: freezes the preregistration text and the v2 code digest before any test-split episode.

    python scripts/v2_register.py            # refuses while docs/PREREGISTRATION_V2.md still has [CONFIRM: ...] markers
    python scripts/v2_register.py --show     # print what is registered

After this, ``run_v2_agentdojo.py --split test`` runs only while the code (``pgx_digest``: pg/ + pgx/ + v2 data + v2 runners) and
the preregistration are byte-identical to what was registered.  Changing either afterwards means a new registration, which the
paper must report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgx.version import pgx_digest  # noqa: E402

PREREG = ROOT / "docs" / "PREREGISTRATION_V2.md"
OUT = ROOT / "data" / "v2" / "registration.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)
    if args.show:
        print(OUT.read_text() if OUT.exists() else "nothing registered yet")
        return 0
    text = PREREG.read_text(encoding="utf-8")
    pending = [line.strip() for line in text.splitlines() if "[CONFIRM:" in line]
    if pending:
        print(f"REFUSED: {len(pending)} line(s) still marked [CONFIRM: ...] in {PREREG.relative_to(ROOT)}; decide each, remove the "
              f"marker, then register.  First: {pending[0][:120]}")
        return 2
    if OUT.exists():
        print(f"REFUSED: {OUT.relative_to(ROOT)} already exists.  A second registration must be deliberate: move the old file to "
              f"data/v2/registration_<date>.json, say why in the preregistration's change log, then rerun.")
        return 2
    digest, files = pgx_digest(ROOT)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    record = {"schema": "pgx-v2-registration-v1", "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "preregistration_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(), "pgx_digest": digest,
              "files": len(files), "git_commit": commit}
    OUT.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"registered: preregistration {record['preregistration_sha256'][:12]}, v2 code {digest[:12]} -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
