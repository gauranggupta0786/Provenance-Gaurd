"""Build the ProvenanceGuard hand-off package: one zip a mentor can unpack and run.

    python scripts/build_handoff_package.py                       # writes dist/provenanceguard-handoff-<digest8>.zip
    python scripts/build_handoff_package.py --out /some/dir
    python scripts/build_handoff_package.py --selftest            # also run the mock-server self-test from the unpacked zip

The package is assembled from an explicit allow-list, never "everything in the folder":

* IN   pg/ (the guard), frontier/ (the OpenAI-compatible adapter, probes, campaign runner, mock server), scripts/, tests/,
       data/ (every frozen dataset and manifest), schemas/, docs/, paper/ (if present), README, MENTOR_QUICKSTART, the pinned
       requirements, demo.py, pytest.ini.
* OUT  results/ (hundreds of MB of local-model episode logs -- not needed to run anything, and the hand-off is a *method*
       package), .venv, .git, caches, scratch folders, deliverables/ and output/ (slide decks and PDFs), and anything else not
       named above.

It then writes ``SHA256SUMS`` (the hash of every file) and ``HANDOFF_MANIFEST.json`` (the digests that define "this is the
defense that was measured"), zips deterministically (sorted names, fixed timestamps: the same tree always gives the same zip
hash), and finally unpacks the zip into a scratch folder and runs ``scripts/handoff_verify.py`` there.  A package that does not
verify after being unpacked is not written to the output folder.

Nothing in the working tree is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOP_FILES = ("README.md", "MENTOR_QUICKSTART.md", "demo.py", "pytest.ini", "requirements.lock", "requirements.txt")
TOP_DIRS = ("pg", "frontier", "scripts", "tests", "data", "schemas", "docs", "paper")
EXCLUDE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".test-tmp", ".venv", "venv", ".git", "results", "output", "deliverables",
                     ".mypy_cache", ".ruff_cache", "dist", "node_modules"}
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".log", ".tmp", ".swp")
EXCLUDE_NAMES = {".DS_Store", "Thumbs.db"}
GENERATED = ("SHA256SUMS", "HANDOFF_MANIFEST.json")
FIXED_TIME = (2026, 1, 1, 0, 0, 0)          # zip timestamps are fixed so the same tree always gives the same archive
EXECUTABLE_SUFFIXES = (".sh",)
PACKAGE_DIR = "provenanceguard-handoff"
SCHEMA = "provenanceguard-handoff-v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect(root: Path) -> list[str]:
    """Relative POSIX paths of every file that goes into the package, sorted."""

    found: list[str] = []
    missing = [name for name in TOP_FILES if not (root / name).is_file()]
    if missing:
        raise SystemExit(f"cannot build: required top-level file(s) missing: {missing}")
    found += list(TOP_FILES)
    for directory in TOP_DIRS:
        base = root / directory
        if not base.is_dir():
            if directory == "paper":
                continue
            raise SystemExit(f"cannot build: required directory missing: {directory}/")
        for current, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIR_NAMES)
            for name in sorted(files):
                if name in EXCLUDE_NAMES or name.endswith(EXCLUDE_SUFFIXES):
                    continue
                found.append(Path(current, name).relative_to(root).as_posix())
    return sorted(set(found))


def _pinned_datasets() -> dict:
    from frontier.suites import DATASETS

    return DATASETS


def _suite_sizes() -> dict:
    """Episodes per model for every other experiment (the attacker suite counts objectives of up to 15 attempts)."""

    return {"adaptive_evasion": 648, "baselines": 864, "w11": 504, "w11b": 420, "w11c": 600, "week8": 600, "week8_adaptive_objectives": 10}


def guard_digests(root: Path) -> tuple[str, str]:
    from pg.week8_protocol import code_digest
    from scripts.week11_lib import week11_code_digest

    return code_digest(root)[0], week11_code_digest(root)[0]


def classify(rel: str) -> str:
    if rel.endswith((".md", ".txt", ".mmd")):
        return "docs"
    if rel.startswith("data/"):
        return "data"
    return "code"


def build_stage(root: Path, stage: Path, files: list[str]) -> dict:
    """Copy the files into ``stage`` and write SHA256SUMS and HANDOFF_MANIFEST.json there.  Returns the manifest."""

    for rel in files:
        target = stage / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, target)
    sums = "".join(f"{sha256_bytes((stage / rel).read_bytes())}  {rel}\n" for rel in files)
    (stage / "SHA256SUMS").write_text(sums, encoding="utf-8", newline="\n")

    guard, week11 = guard_digests(root)
    kinds = {"code": 0, "data": 0, "docs": 0}
    for rel in files:
        kinds[classify(rel)] += 1
    manifest = {
        "schema_version": SCHEMA,
        "what_this_is": ("The ProvenanceGuard evaluation method: guard code, frozen datasets, an OpenAI-compatible runner for every "
                         "experiment in the report, probes, report and re-score tools, and a mock-server self-test. It contains NO results from any hosted model."),
        "guard_code_digest": guard,
        "week11_code_digest": week11,
        "datasets": {
            **{f"{key}": {"path": spec["path"], "sha256": spec["sha256"]} for key, spec in _pinned_datasets().items()},
            "tuning_48_scenarios": {"path": "data/week11e/scenarios.jsonl", "sha256": sha256_bytes((root / "data/week11e/scenarios.jsonl").read_bytes())},
            "heldout_72_scenarios": {"path": "data/heldout/scenarios.jsonl", "sha256": sha256_bytes((root / "data/heldout/scenarios.jsonl").read_bytes())},
        },
        "requirements_lock_sha256": sha256_bytes((root / "requirements.lock").read_bytes()),
        "seeds": [42, 314, 2718],
        "configurations": ["C0", "SC-v3", "SCP-v3"],
        "episodes_per_model": {"tuning": 432, "heldout": 648, **_suite_sizes()},
        "file_counts": {**kinds, "total": len(files)},
        "sha256sums_sha256": sha256_bytes((stage / "SHA256SUMS").read_bytes()),
        "excluded_on_purpose": sorted(EXCLUDE_DIR_NAMES - {"__pycache__", ".pytest_cache", ".test-tmp", ".mypy_cache", ".ruff_cache"}),
    }
    (stage / "HANDOFF_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def write_zip(stage: Path, files: list[str], target: Path) -> None:
    names = sorted([*files, *GENERATED])
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".zip.partial")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for rel in names:
            info = zipfile.ZipInfo(f"{PACKAGE_DIR}/{rel}", date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if rel.endswith(EXECUTABLE_SUFFIXES) else 0o100644) << 16
            archive.writestr(info, (stage / rel).read_bytes())
    tmp.replace(target)


def verify_unpacked(archive: Path, run_selftest: bool) -> tuple[bool, str]:
    """Unpack the finished zip somewhere else and run the verifier (and optionally the self-test) from there."""

    with tempfile.TemporaryDirectory(prefix="pg-unpack-") as scratch:
        with zipfile.ZipFile(archive) as z:
            bad = z.testzip()
            if bad:
                return False, f"zip is corrupt at {bad}"
            z.extractall(scratch)
        tree = Path(scratch) / PACKAGE_DIR
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.run([sys.executable, "-B", "scripts/handoff_verify.py"], cwd=tree, env=env, capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        if proc.returncode != 0:
            return False, "handoff_verify failed in the unpacked tree:\n" + out
        if run_selftest:
            proc = subprocess.run([sys.executable, "-B", "scripts/frontier_selftest.py"], cwd=tree, env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                return False, "frontier_selftest failed in the unpacked tree:\n" + proc.stdout[-2500:] + proc.stderr[-800:]
            out += "\n" + proc.stdout.strip().splitlines()[-2]
        return True, out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "dist", help="folder for the zip (default: dist/)")
    ap.add_argument("--selftest", action="store_true", help="also run the mock-server self-test from the unpacked zip (about a minute)")
    ap.add_argument("--list", action="store_true", help="print the files that would be packaged and stop")
    args = ap.parse_args(argv)

    files = collect(ROOT)
    if args.list:
        print("\n".join(files))
        print(f"\n{len(files)} files")
        return 0

    with tempfile.TemporaryDirectory(prefix="pg-stage-") as scratch:
        stage = Path(scratch)
        manifest = build_stage(ROOT, stage, files)
        name = f"provenanceguard-handoff-{manifest['guard_code_digest'][:8]}.zip"
        candidate = Path(scratch + ".zip.d") / name
        write_zip(stage, files, candidate)
        ok, report = verify_unpacked(candidate, args.selftest)
        if not ok:
            print("BUILD FAILED -- the package does not verify after being unpacked; nothing was written.\n" + report)
            shutil.rmtree(candidate.parent, ignore_errors=True)
            return 1
        args.out.mkdir(parents=True, exist_ok=True)
        final = args.out / name
        shutil.move(str(candidate), final)
        shutil.rmtree(candidate.parent, ignore_errors=True)

    digest = sha256_bytes(final.read_bytes())
    print(report)
    print(f"\nwrote {final}  ({final.stat().st_size / 1024:.0f} KB, {len(files) + 2} files)")
    print(f"sha256 {digest}")
    print(f"guard code digest {manifest['guard_code_digest']}")
    print("The sha256 above is for the mentor to check the download; the zip is reproducible (same tree -> same hash).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
