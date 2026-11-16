"""Check that this copy of the ProvenanceGuard package is intact and its environment can reproduce the measured defense.

    python scripts/handoff_verify.py            # human-readable, exit 0 if nothing FAILs
    python scripts/handoff_verify.py --json out.json

Run it first, before any model is involved.  It needs no network, no key and no GPU, and writes nothing except a probe file
that it deletes again.  Each line is PASS, WARN, FAIL or INFO:

* FAIL  something that would make a result not comparable to the ones already measured, or that would stop the run.
* WARN  a difference that probably does not matter (a documentation file edited, a non-core package at another version) but
        that you should know about.
* INFO  context, nothing to act on.

What it checks, and why each matters
------------------------------------
1. Python version and the pinned packages in ``requirements.lock``.  The core packages (agentdojo, openai, pydantic,
   langgraph, httpx, PyYAML, cryptography, pytest) FAIL when they differ: a different AgentDojo changes the benchmark itself.
   Every other pin only WARNs.
2. The guard code digest.  This is the hash ``pg.week8_protocol.code_digest`` computes over every ``pg/*.py`` file plus the
   pinned protocol files; the six local models were measured under exactly one value of it.  A different value means a
   different defense, so the runner refuses to start.  The ``week11`` digest additionally covers the ladder's own scripts.
3. The datasets by content hash: the 48-scenario tuning set and the 72-scenario held-out set (and the held-out manifest).
4. ``SHA256SUMS`` -- the checksum of every file in the package, written when the package was built.  A code or data file that
   differs FAILs; a documentation file that differs only WARNs.  (A missing file FAILs for code and data, WARNs for docs.)
5. That the AgentDojo workspace environment actually loads, that the ``frontier`` modules import, and that the results
   directory is writable and has room.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as md
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIN_PYTHON = (3, 11)
MEASURED_ON = "Python 3.12 (Windows, original freeze); re-installed and re-tested on Python 3.13 (Linux)"
CORE_PACKAGES = ("agentdojo", "openai", "pydantic", "pydantic-core", "langgraph", "langgraph-checkpoint", "httpx", "pyyaml",
                 "cryptography", "pytest")
GUARD_DIGEST = "dca1a798694935c90886af0f2a535b24251de7e6a5ceef5448f2442ae1e0648b"
WEEK11_DIGEST = "22a4622b386181ff127e3d67ba452feb241efdb517811f71c903f204f2e58010"
DATASET_HASHES = {
    "data/week11e/scenarios.jsonl": "4a01d47d18675ea0bfcb587cf895ba37a25481071cba7a3bc61c6ed3c679dbf3",   # tuning, 48 scenarios
    "data/heldout/scenarios.jsonl": "785c474078879f5456fb69030760b431fbd095c81aa26ae3492c68561777d830",   # held-out, 72 scenarios
    "data/adaptive_evasion/scenarios.jsonl": "c5a593e5d3aae4efea4e4315a912d62f83e61cceb37f071c77cc02f71aae64f7",   # evasion suite, 72
    "data/week11/policy_passing.jsonl": "6daadfd459d00d37cb649928dd485e059bf0494e78f455ff2a7d82bc7455390c",      # Week 11 stage tables
    "data/week11c/scenarios.jsonl": "56a67cc0433d6d1d26e27deb35e78b516d67c3d2e765b4fe1555daa2a57c4dcb",          # Week 11c stage table
    "data/week5/heldout.jsonl": "80e9d21dc9620f42d5d76cd3cef1e628f2ca84b944bc8ed6bda6ff05b6374daa",              # Week 8, 100 scenarios
    "data/week6/adaptive_objectives.jsonl": "d263783156014a2f8bc8390f74189001707b8b339c2a84a9c1b9bef9848d2908",   # Week 8 attacker, 10
}
MANIFEST_PREFIXES = {"data/heldout/manifest.json": "3308c3afef363ad682b5"}
DOC_SUFFIXES = (".md", ".txt", ".pdf", ".mmd")
CODE_ROOTS = ("pg/", "frontier/", "scripts/", "data/", "schemas/", "tests/", "requirements.lock", "requirements.txt", "pytest.ini")
MIN_FREE_MB = 500


class Report:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append({"status": status, "check": name, "detail": detail})
        mark = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL", "INFO": "info"}[status]
        print(f"  {mark:4s}  {name}" + (f"  -- {detail}" if detail else ""))

    def count(self, status: str) -> int:
        return sum(1 for r in self.rows if r["status"] == status)


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[normalise(name.strip())] = version.strip()
    return pins


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_python(r: Report) -> None:
    v = sys.version_info
    label = f"Python {v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < MIN_PYTHON:
        r.add("FAIL", "python version", f"{label}; need {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer")
    else:
        r.add("PASS", "python version", f"{label}  (measured on: {MEASURED_ON})")


def check_packages(r: Report, root: Path) -> None:
    lock = root / "requirements.lock"
    if not lock.exists():
        r.add("FAIL", "requirements.lock present", "missing")
        return
    pins = parse_lock(lock)
    installed = {normalise(d.metadata["Name"]): d.version for d in md.distributions() if d.metadata["Name"]}
    core_bad: list[str] = []
    other_bad: list[str] = []
    for name, want in sorted(pins.items()):
        have = installed.get(name)
        if have == want:
            continue
        text = f"{name} {have or 'NOT INSTALLED'} (pinned {want})"
        (core_bad if name in CORE_PACKAGES else other_bad).append(text)
    core_ok = [p for p in CORE_PACKAGES if p in pins and installed.get(p) == pins[p]]
    if core_bad:
        r.add("FAIL", "core packages match requirements.lock", "; ".join(core_bad) +
              "   fix: pip install -r requirements.lock")
    else:
        r.add("PASS", f"core packages match requirements.lock ({len(core_ok)} checked)")
    if other_bad:
        r.add("WARN", f"{len(other_bad)} other pinned package(s) differ", "; ".join(other_bad[:6]) + (" ..." if len(other_bad) > 6 else ""))
    else:
        r.add("PASS", f"all {len(pins)} pins in requirements.lock are installed at the pinned version")


def check_digests(r: Report, root: Path) -> None:
    try:
        from pg.week8_protocol import code_digest
        from scripts.week11_lib import week11_code_digest
        guard, _ = code_digest(root)
        week11, _ = week11_code_digest(root)
    except Exception as exc:  # a missing file is itself the finding
        r.add("FAIL", "guard code digest computable", f"{type(exc).__name__}: {str(exc)[:200]}")
        return
    if guard == GUARD_DIGEST:
        r.add("PASS", "guard code digest equals the measured value", guard[:16] + "...")
    else:
        r.add("FAIL", "guard code digest equals the measured value",
              f"got {guard[:16]}..., measured {GUARD_DIGEST[:16]}...: pg/ or a pinned protocol file differs, so this is a different "
              f"defense and results would not be comparable (see SHA256SUMS check for the file)")
    if week11 == WEEK11_DIGEST:
        r.add("PASS", "ladder code digest (week 11) equals the measured value", week11[:16] + "...")
    else:
        r.add("FAIL", "ladder code digest (week 11) equals the measured value", f"got {week11[:16]}..., measured {WEEK11_DIGEST[:16]}...")


def check_datasets(r: Report, root: Path) -> None:
    for rel, want in DATASET_HASHES.items():
        path = root / rel
        if not path.exists():
            r.add("FAIL", f"dataset {rel}", "missing")
            continue
        got = sha256_file(path)
        r.add("PASS" if got == want else "FAIL", f"dataset {rel} matches its frozen hash",
              got[:16] + "..." if got == want else f"got {got[:16]}..., frozen {want[:16]}...")
    for rel, prefix in MANIFEST_PREFIXES.items():
        path = root / rel
        got = sha256_file(path) if path.exists() else ""
        r.add("PASS" if got.startswith(prefix) else "FAIL", f"{rel} matches its recorded hash", got[:16] + "..." if got else "missing")
    try:
        from frontier.suites import DATASETS
        from scripts.week11e_lib import WEEK11D_DATASET_SHA256
        agree = (all(DATASET_HASHES[v["path"]] == v["sha256"] for v in DATASETS.values())
                 and {v["path"] for v in DATASETS.values()} == set(DATASET_HASHES)
                 and DATASETS["tuning"]["sha256"] == WEEK11D_DATASET_SHA256)
        r.add("PASS" if agree else "FAIL", "the runner, the frozen libraries and this checker pin the same dataset hashes")
    except Exception as exc:
        r.add("FAIL", "dataset pins agree", f"{type(exc).__name__}: {exc}")


def is_code(rel: str) -> bool:
    return rel.startswith(CODE_ROOTS) and not rel.endswith(DOC_SUFFIXES)


def check_sums(r: Report, root: Path) -> None:
    sums = root / "SHA256SUMS"
    if not sums.exists():
        r.add("INFO", "SHA256SUMS", "not present (a source checkout rather than the built package); skipped")
        return
    listed: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, _, rel = line.partition("  ")
            listed[rel.strip()] = digest.strip()
    code_bad: list[str] = []
    doc_bad: list[str] = []
    for rel, want in sorted(listed.items()):
        path = root / rel
        if not path.exists():
            problem = f"{rel} missing"
        elif sha256_file(path) != want:
            problem = f"{rel} differs"
        else:
            continue
        (code_bad if is_code(rel) else doc_bad).append(problem)
    if code_bad:
        r.add("FAIL", "code and data files match SHA256SUMS", "; ".join(code_bad[:8]) + (" ..." if len(code_bad) > 8 else ""))
    else:
        r.add("PASS", f"code and data files match SHA256SUMS ({sum(1 for k in listed if is_code(k))} files)")
    if doc_bad:
        r.add("WARN", "documentation files differ from SHA256SUMS", "; ".join(doc_bad[:6]))
    else:
        r.add("PASS", f"documentation files match SHA256SUMS ({sum(1 for k in listed if not is_code(k))} files)")
    skip = {"SHA256SUMS", "HANDOFF_MANIFEST.json"}
    extra = [str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts
             and not str(p.relative_to(root)).replace("\\", "/").startswith(("results/", ".venv/", "venv/", "output/"))
             and str(p.relative_to(root)).replace("\\", "/") not in listed and p.name not in skip]
    code_extra = [e for e in extra if is_code(e)]
    if code_extra:
        r.add("WARN", "code/data files present that are not in SHA256SUMS", "; ".join(code_extra[:6]))
    elif extra:
        r.add("INFO", f"{len(extra)} extra non-code file(s) present (your own notes or label files are fine)")


def check_environment(r: Report, root: Path) -> None:
    try:
        from agentdojo.task_suite.load_suites import get_suite
        from pg.agentdojo_adapter import BENCHMARK_VERSION, SUITE_NAME
        env = get_suite(BENCHMARK_VERSION, SUITE_NAME).load_and_inject_default_environment({})
        r.add("PASS", f"AgentDojo {SUITE_NAME} {BENCHMARK_VERSION} environment loads", type(env).__name__)
    except Exception as exc:
        r.add("FAIL", "AgentDojo workspace environment loads", f"{type(exc).__name__}: {str(exc)[:200]}")
    for module in ("frontier.openai_adapter", "frontier.frontier_probes", "frontier.campaign", "frontier.mock_server",
                   "scripts.frontier_report", "scripts.content_rescore"):
        try:
            importlib.import_module(module)
        except Exception as exc:
            r.add("FAIL", f"import {module}", f"{type(exc).__name__}: {str(exc)[:160]}")
            return
    r.add("PASS", "frontier modules import")
    results = root / "results"
    try:
        results.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=results, prefix=".verify-", delete=True) as handle:
            handle.write(b"ok")
        free_mb = shutil.disk_usage(results).free // (1024 * 1024)
        r.add("PASS" if free_mb >= MIN_FREE_MB else "WARN", "results/ is writable", f"{free_mb:,} MB free" +
              ("" if free_mb >= MIN_FREE_MB else f" (a full held-out run writes well under 200 MB, but {MIN_FREE_MB} MB is the margin)"))
    except OSError as exc:
        r.add("FAIL", "results/ is writable", str(exc))


def check_endpoint_variables(r: Report) -> None:
    import os
    names = ("PROVENANCEGUARD_API_BASE", "PROVENANCEGUARD_API_KEY", "PROVENANCEGUARD_MODEL")
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        r.add("INFO", "model endpoint variables", f"not set here: {', '.join(missing)} (needed only for the real run, not for this check)")
    else:
        r.add("INFO", "model endpoint variables", f"set for model {os.environ['PROVENANCEGUARD_MODEL']} (key not shown)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, help="also write the checks as JSON")
    ap.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    print(f"ProvenanceGuard hand-off verification  ({root})\n")
    r = Report()
    check_python(r)
    check_packages(r, root)
    check_digests(r, root)
    check_datasets(r, root)
    check_sums(r, root)
    check_environment(r, root)
    check_endpoint_variables(r)
    fails, warns = r.count("FAIL"), r.count("WARN")
    print()
    if fails:
        print(f"NOT READY: {fails} check(s) FAILED. Fix those before running anything; results from this tree would not be comparable.")
    elif warns:
        print(f"READY, with {warns} warning(s) above. None affects comparability, but read them.")
    else:
        print("READY: the package is intact and the environment matches what was measured.")
    if args.json:
        args.json.write_text(json.dumps({"ready": not fails, "failures": fails, "warnings": warns, "checks": r.rows}, indent=2) + "\n",
                             encoding="utf-8")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
