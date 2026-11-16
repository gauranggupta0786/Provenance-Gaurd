"""Week 8 -- the frozen cross-model protocol.

Everything a measured campaign must not change is verified here before a single episode runs:

* the Week 7 matrix is frozen and untampered, and the chosen model's profile is the frozen file;
* the live server serves *that* model (digest), *that* template (hash), and *that* engine version;
* the code that produces measurements is byte-identical to the code digest frozen in the Week 8
  protocol manifest (so a patch installed mid-campaign is refused instead of silently mixed in);
* both arms of every scenario/seed pair were built from identical requests (arm-parity audit).

Server ports are deliberately NOT part of a profile's identity here: every Slurm job starts its own
server on a different port, so identity uses ``portable_fingerprint`` (endpoint masked).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx

from .agentdojo_adapter import AGENTDOJO_VERSION, BENCHMARK_VERSION, SUITE_NAME
from .diagnostics import RuntimeLimits
from .formal_evaluation import EVALUATOR_VERSION
from .trusted_models import ModelProfile, schema_manifest

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "data" / "week7" / "final_model_matrix.json"
PROFILES_DIR = ROOT / "data" / "week7" / "profiles_final"
APPROVAL_PATH = ROOT / "data" / "week7" / "gpu_approval.json"
PROTOCOL_PATH = ROOT / "data" / "week8" / "protocol_manifest.json"
HELDOUT_PATH = ROOT / "data" / "week5" / "heldout.jsonl"
WEEK5_MANIFEST = ROOT / "data" / "week5" / "manifest.json"
ADAPTIVE_DATASET = ROOT / "data" / "week6" / "adaptive_objectives.jsonl"
ADAPTIVE_MANIFEST = ROOT / "data" / "week6" / "adaptive_manifest.json"
WEEK6_PROTOCOL = ROOT / "data" / "week6" / "frozen_protocol" / "manifest.json"
RESULTS_ROOT = ROOT / "results" / "week8"

SEEDS = (42, 314, 2718)
ARMS = ("baseline", "protected")
ADAPTIVE_SEED = 42
LIMITS = RuntimeLimits(model_seconds=300.0, tool_seconds=30.0, approval_seconds=30.0, episode_seconds=1800.0)

# Files whose bytes determine what is measured. A change after the protocol is frozen is refused.
CODE_FILES = (
    "requirements.lock", "data/week5/manifest.json", "data/week6/adaptive_manifest.json",
    "data/week6/frozen_protocol/manifest.json", "scripts/run_week8_frozen.py", "scripts/run_week8_adaptive.py",
    "scripts/run_week6_adaptive.py",
)


class ProtocolViolation(RuntimeError):
    """A frozen-protocol check failed. The campaign must not start or continue."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def portable_fingerprint(profile: ModelProfile) -> str:
    """Profile identity without the per-job server port."""

    payload = profile.to_dict()
    payload["endpoint"] = "<runtime>"
    return canonical_sha256(payload)


# ------------------------------------------------------------------ matrix, approval, profiles

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def matrix_digest(matrix: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        {k: v for k, v in matrix.items() if k not in ("generated_at", "matrix_sha256")}, sort_keys=True).encode()).hexdigest()


def load_frozen_matrix(path: Path = MATRIX_PATH) -> dict[str, Any]:
    if not Path(path).exists():
        raise ProtocolViolation(f"{path} not found: freeze the Week 7 matrix first")
    matrix = load_json(path)
    if matrix.get("frozen") is not True:
        raise ProtocolViolation("the Week 7 matrix is not frozen")
    if matrix.get("matrix_sha256") != matrix_digest(matrix):
        raise ProtocolViolation("matrix_sha256 does not match the matrix content: the file was edited after freezing")
    if matrix.get("family_duplicates_in_selected"):
        raise ProtocolViolation(f"more than one model per family: {matrix['family_duplicates_in_selected']}")
    if not matrix.get("selected"):
        raise ProtocolViolation("no models selected in the frozen matrix")
    return matrix


def load_approval(matrix: Mapping[str, Any], path: Path = APPROVAL_PATH) -> dict[str, Any]:
    if not Path(path).exists():
        raise ProtocolViolation(f"{path} not found: run scripts/record_week7_approval.py (Week 7 completion gate)")
    approval = load_json(path)
    if approval.get("matrix_sha256") != matrix["matrix_sha256"]:
        raise ProtocolViolation("the recorded approval refers to a different matrix than the frozen one")
    if not str(approval.get("approved_by") or "").strip():
        raise ProtocolViolation("the recorded approval has no approver name")
    return approval


def matrix_entry(matrix: Mapping[str, Any], slug: str) -> dict[str, Any]:
    for item in matrix["selected"]:
        if item["slug"] == slug:
            return dict(item)
    raise ProtocolViolation(f"{slug!r} is not a selected model in the frozen matrix "
                            f"(selected: {[m['slug'] for m in matrix['selected']]})")


def load_frozen_profile(slug: str, matrix: Mapping[str, Any], profiles_dir: Path = PROFILES_DIR) -> tuple[ModelProfile, dict[str, Any]]:
    entry = matrix_entry(matrix, slug)
    path = Path(profiles_dir) / f"{slug}.json"
    if not path.exists():
        raise ProtocolViolation(f"{path} not found")
    profile = ModelProfile.from_json(path)
    if profile.fingerprint != entry["profile_sha256"]:
        raise ProtocolViolation(f"{path.name} does not match the frozen matrix (profile_sha256): the profile was edited")
    for field, expected in (("model_id", entry["ollama_tag"]), ("model_digest", entry["model_digest"]),
                            ("chat_template_sha256", entry["template_sha256"])):
        if getattr(profile, field) != expected:
            raise ProtocolViolation(f"{slug}: profile {field} differs from the frozen matrix")
    return profile, entry


# ------------------------------------------------------------------ live server

def _directives(modelfile: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in (modelfile or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("RENDERER", "PARSER"):
            found[parts[0].lower()] = parts[1]
    return found


def verify_live_server(profile: ModelProfile, entry: Mapping[str, Any], http: httpx.Client) -> dict[str, Any]:
    """The server answering at the endpoint must serve exactly the frozen model."""

    version = http.get("/api/version").raise_for_status().json().get("version")
    if str(version) != str(profile.serving_engine_version):
        raise ProtocolViolation(f"engine version {version!r} differs from the frozen {profile.serving_engine_version!r}")
    tags = http.get("/api/tags").raise_for_status().json().get("models", [])
    accepted = {profile.model_id, f"{profile.model_id}:latest"}
    listed = next((m for m in tags if m.get("name") in accepted or m.get("model") in accepted), None)
    if listed is None:
        raise ProtocolViolation(f"{profile.model_id} is not installed on the server (ollama pull it)")
    live_digest = str(listed.get("digest") or "").removeprefix("sha256:")
    if live_digest != profile.model_digest:
        raise ProtocolViolation(f"{profile.model_id}: installed digest {live_digest[:12]} != frozen {profile.model_digest[:12]}")
    show = http.post("/api/show", json={"model": profile.model_id}).raise_for_status().json()
    template = str(show.get("template") or "") or "{{ .Prompt }}"
    if hashlib.sha256(template.encode("utf-8")).hexdigest() != profile.chat_template_sha256:
        raise ProtocolViolation(f"{profile.model_id}: chat template differs from the frozen template")
    directives = _directives(str(show.get("modelfile") or ""))
    for key in ("renderer", "parser"):
        if (entry.get(key) or None) != (directives.get(key) or None):
            raise ProtocolViolation(f"{profile.model_id}: {key} {directives.get(key)!r} differs from frozen {entry.get(key)!r}")
    if "tools" not in (show.get("capabilities") or []):
        raise ProtocolViolation(f"{profile.model_id}: server no longer reports the tools capability")
    return {"engine_version": str(version), "digest": live_digest, "template_sha256": profile.chat_template_sha256,
            "renderer": directives.get("renderer"), "parser": directives.get("parser"),
            "capabilities": sorted(show.get("capabilities") or []), "verified_at": utc_now()}


def runtime_profile(profile: ModelProfile, endpoint: str) -> ModelProfile:
    return profile.revalidated(endpoint=endpoint.rstrip("/"))


# ------------------------------------------------------------------ run context and code identity

def run_context() -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": platform.node(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": platform.python_version(), "platform": platform.platform(),
        "ollama_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("OLLAMA_")},
    }
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=15, check=False).stdout
            ctx["gpus"] = [line.strip() for line in out.splitlines() if line.strip()]
        except Exception as exc:
            ctx["gpus_error"] = f"{type(exc).__name__}: {exc}"
    return ctx


# Analysis-only modules: fixing a table after a campaign must not invalidate the measurements that produced it.
ANALYSIS_ONLY_MODULES = ("week7_analysis.py", "week7_probes.py", "week8_analysis.py")


def code_digest(root: Path = ROOT) -> tuple[str, dict[str, str]]:
    files = sorted(p for p in (root / "pg").glob("*.py") if p.name not in ANALYSIS_ONLY_MODULES) + [root / name for name in CODE_FILES]
    missing = [str(p.relative_to(root)) for p in files if not p.exists()]
    if missing:
        raise ProtocolViolation(f"code files missing: {missing}")
    hashes = {str(p.relative_to(root)).replace("\\", "/"): file_sha256(p) for p in files}
    return canonical_sha256(hashes), hashes


# ------------------------------------------------------------------ protocol manifest

def build_protocol_manifest(
    *, root: Path = ROOT, matrix_path: Path = MATRIX_PATH, profiles_dir: Path = PROFILES_DIR,
    approval_path: Path = APPROVAL_PATH,
) -> dict[str, Any]:
    matrix = load_frozen_matrix(matrix_path)
    approval = load_approval(matrix, approval_path)
    week5 = load_json(root / "data" / "week5" / "manifest.json")
    dataset = root / "data" / "week5" / "heldout.jsonl"
    if file_sha256(dataset) != week5["heldout"]["sha256"]:
        raise ProtocolViolation("held-out dataset checksum mismatch")
    adaptive_expected = load_json(root / "data" / "week6" / "adaptive_manifest.json")
    if file_sha256(root / "data" / "week6" / "adaptive_objectives.jsonl") != adaptive_expected["sha256"]:
        raise ProtocolViolation("adaptive objective checksum mismatch")
    week6 = load_json(root / "data" / "week6" / "frozen_protocol" / "manifest.json")
    models: dict[str, Any] = {}
    for item in matrix["selected"]:
        profile, entry = load_frozen_profile(item["slug"], matrix, profiles_dir)
        models[item["slug"]] = {
            "ollama_tag": profile.model_id, "family": entry["family"], "model_digest": profile.model_digest,
            "profile_file_sha256": profile.fingerprint, "portable_profile_sha256": portable_fingerprint(profile),
            "template_sha256": profile.chat_template_sha256, "renderer": entry.get("renderer"), "parser": entry.get("parser"),
            "quantization": profile.quantization, "serving_engine": f"{profile.serving_engine} {profile.serving_engine_version}",
            "selected_context_length": profile.selected_context_length, "max_output_tokens": profile.max_output_tokens,
            "thinking_mode": profile.thinking_mode, "temperature": profile.temperature, "top_p": profile.top_p,
            "top_k": profile.top_k, "min_p": profile.min_p,
        }
    digest, hashes = code_digest(root)
    schemas = schema_manifest()
    manifest = {
        "schema_version": "week8-protocol-v1", "frozen": True, "created_at": utc_now(),
        "week7_matrix": {"path": str(matrix_path.relative_to(root)).replace("\\", "/") if matrix_path.is_relative_to(root) else str(matrix_path),
                         "matrix_sha256": matrix["matrix_sha256"], "selected": [m["slug"] for m in matrix["selected"]],
                         "context_bucket": matrix["context"]["common_bucket"]},
        "approval": {"approved_by": approval["approved_by"], "approved_at": approval.get("approved_at"),
                     "note": approval.get("note"), "decisions": approval.get("decisions"),
                     "approval_file_sha256": file_sha256(approval_path)},
        "models": models,
        "campaign": {
            "dataset": "data/week5/heldout.jsonl", "dataset_sha256": week5["heldout"]["sha256"], "scenarios": 100,
            "arms": list(ARMS), "seeds": list(SEEDS), "episodes_per_model": 100 * len(ARMS) * len(SEEDS),
            "adaptive": {"dataset_sha256": adaptive_expected["sha256"], "objectives": adaptive_expected["count"],
                         "max_attempts": adaptive_expected["attempts_per_objective"], "seed": ADAPTIVE_SEED},
        },
        "limits": {"model_seconds": LIMITS.model_seconds, "tool_seconds": LIMITS.tool_seconds,
                   "approval_seconds": LIMITS.approval_seconds, "episode_seconds": LIMITS.episode_seconds,
                   "infrastructure_retries": LIMITS.infrastructure_retries},
        "evaluator": {"version": EVALUATOR_VERSION, "week6_protocol_manifest_sha256": week6["manifest_sha256"],
                      "policy_version": week6["policy_version"], "trusted_schema_bundle_sha256": schemas["bundle_sha256"],
                      "agentdojo_version": AGENTDOJO_VERSION, "benchmark": f"{BENCHMARK_VERSION} {SUITE_NAME}"},
        "code": {"defense_version": digest, "component_hashes": hashes},
        "rules": [
            "baseline and protected arms use identical dataset, seeds, tool schemas, limits, profile, and evaluator",
            "every scheduled episode stays in every denominator; runtime, truncation, malformed-call and overflow failures are recorded, never dropped",
            "no setting, dataset, profile, or code file may change after this manifest is frozen; a change requires a new manifest and output directory",
            "a campaign resumes by skipping finished (scenario, arm, seed) keys only",
        ],
    }
    manifest["protocol_sha256"] = canonical_sha256({k: v for k, v in manifest.items() if k not in ("created_at", "protocol_sha256")})
    return manifest


def verify_protocol(path: Path = PROTOCOL_PATH, *, root: Path = ROOT, matrix_path: Path = MATRIX_PATH,
                    check_code: bool = True) -> dict[str, Any]:
    """``check_code=False`` skips only the code-digest comparison.  Week 10 changed the guard code on purpose and
    freezes its own digest (``scripts/week10_lib.verify_week10_code``); every other Week 8 check still runs."""

    if not Path(path).exists():
        raise ProtocolViolation(f"{path} not found: run scripts/week8_prepare.py after installing the final code")
    manifest = load_json(path)
    if manifest.get("protocol_sha256") != canonical_sha256(
            {k: v for k, v in manifest.items() if k not in ("created_at", "protocol_sha256")}):
        raise ProtocolViolation("protocol_sha256 does not match the manifest content: the manifest was edited")
    matrix = load_frozen_matrix(matrix_path)
    if matrix["matrix_sha256"] != manifest["week7_matrix"]["matrix_sha256"]:
        raise ProtocolViolation("the Week 7 matrix changed after the Week 8 protocol was frozen")
    if check_code:
        digest, hashes = code_digest(root)
        if digest != manifest["code"]["defense_version"]:
            changed = sorted(k for k in hashes if hashes[k] != manifest["code"]["component_hashes"].get(k))
            raise ProtocolViolation(f"code changed after the protocol was frozen ({changed[:6]}); "
                                    "re-freeze with scripts/week8_prepare.py and use a NEW results directory")
    return manifest


def ensure_run_manifest(path: Path, candidate: dict[str, Any], immutable: Sequence[str]) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        existing = load_json(path)
        changed = [f for f in immutable if existing.get(f) != candidate.get(f)]
        if changed:
            raise ProtocolViolation(f"refusing to mix runs: frozen settings changed {changed}; choose a new output directory")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return candidate


# ------------------------------------------------------------------ arm parity

def audit_arm_parity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Both arms of a (scenario, seed) pair must have been built from identical inputs.

    Checked from recorded hashes, not assumed: scenario checksum, model digest, portable profile, defense
    version, protocol, the hash of the FIRST native request (same prompt + tools + options), and a single
    tool-schema hash across every call of both arms.
    """

    pairs: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        pairs[(str(row["scenario_id"]), int(row["seed"]))][str(row["arm"])] = row
    checked = inconclusive = 0
    mismatches: list[dict[str, Any]] = []
    for key, pair in sorted(pairs.items()):
        if set(pair) != set(ARMS):
            continue
        a, b = pair["baseline"], pair["protected"]
        for field in ("scenario_checksum", "model_digest", "portable_profile_sha256", "defense_version", "protocol_sha256", "model_id"):
            if a.get(field) != b.get(field):
                mismatches.append({"pair": key, "field": field, "baseline": a.get(field), "protected": b.get(field)})
        logs = [[e for e in (r.get("request_log") or []) if "request_sha256" in e] for r in (a, b)]
        if not logs[0] or not logs[1]:
            inconclusive += 1
            continue
        checked += 1
        if logs[0][0]["request_sha256"] != logs[1][0]["request_sha256"]:
            mismatches.append({"pair": key, "field": "first_request_sha256", "baseline": logs[0][0]["request_sha256"],
                               "protected": logs[1][0]["request_sha256"]})
        tools = {e["tools_sha256"] for log in logs for e in log}
        if len(tools) != 1:
            mismatches.append({"pair": key, "field": "tools_sha256", "distinct": len(tools)})
    return {"pairs_checked": checked, "pairs_inconclusive_no_request_recorded": inconclusive,
            "mismatches": len(mismatches), "examples": mismatches[:8], "passed": checked > 0 and not mismatches}


# ------------------------------------------------------------------ one model, one session

def week8_environment(profile: ModelProfile, live: Mapping[str, Any]) -> dict[str, Any]:
    """Environment manifest for a Week 8 run (the Week 5 helper describes only the Week 2 Qwen profile)."""

    schemas = schema_manifest()
    masked = profile.to_dict()
    masked["endpoint"] = "<runtime>"
    return {
        "agentdojo_version": AGENTDOJO_VERSION, "benchmark_version": BENCHMARK_VERSION, "suite": SUITE_NAME,
        "model": profile.model_id, "model_profile": masked, "model_profile_portable_sha256": portable_fingerprint(profile),
        "provider_adapter": "ollama-native-chat-v1", "state_evaluator": "pg.week5_runner.evaluate_state/v1",
        "output_guard": "pg.confidentiality.ConfidentialityGuard/v1", "live_server": dict(live),
        "trusted_schema_bundle_sha256": schemas["bundle_sha256"],
        "trusted_schema_hashes": {item["model"]: item["sha256"] for item in schemas["schemas"]},
    }


def prepare_model_session(
    slug: str, endpoint: str, *, protocol_path: Path = PROTOCOL_PATH, matrix_path: Path = MATRIX_PATH,
    profiles_dir: Path = PROFILES_DIR, root: Path = ROOT, http: httpx.Client | None = None,
    check_code: bool = True,
) -> dict[str, Any]:
    """Run every identity check for ``slug`` against the live server; returns what a campaign needs."""

    protocol = verify_protocol(protocol_path, root=root, matrix_path=matrix_path, check_code=check_code)
    if slug not in protocol["models"]:
        raise ProtocolViolation(f"{slug!r} is not in the frozen Week 8 protocol ({list(protocol['models'])})")
    matrix = load_frozen_matrix(matrix_path)
    profile, entry = load_frozen_profile(slug, matrix, profiles_dir)
    frozen = protocol["models"][slug]
    if profile.fingerprint != frozen["profile_file_sha256"] or portable_fingerprint(profile) != frozen["portable_profile_sha256"]:
        raise ProtocolViolation(f"{slug}: profile differs from the one frozen in the Week 8 protocol")
    run_profile = runtime_profile(profile, endpoint)
    owns = http is None
    client = http or httpx.Client(base_url=endpoint.rstrip("/"), timeout=60.0)
    try:
        live = verify_live_server(run_profile, entry, client)
    finally:
        if owns:
            client.close()
    identity = {
        "slug": slug, "model_id": run_profile.model_id, "model_digest": run_profile.model_digest,
        "portable_profile_sha256": frozen["portable_profile_sha256"], "protocol_sha256": protocol["protocol_sha256"],
        "defense_version": protocol["code"]["defense_version"],
    }
    return {"protocol": protocol, "profile": run_profile, "entry": entry, "live": live, "identity": identity,
            "environment": week8_environment(run_profile, live)}


def quarantine_stale_evidence(path: Path) -> Path | None:
    """An interrupted episode leaves a half-written event store. Rerunning into it would carry stale rejected-
    attempt counts into the policy budget, so the leftovers are moved aside (kept, never deleted)."""

    path = Path(path)
    if not path.exists():
        return None
    target = path.parent / "_interrupted" / f"{path.name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
    target.parent.mkdir(parents=True, exist_ok=True)
    path.rename(target)
    return target


def evidence_token_dir(evidence_root: Path, scenario_id: str, seed: int, arm: str) -> Path:
    """Same naming rule as pg.week5_runner.compact_episode_evidence_dir (without creating it)."""

    token = hashlib.sha256(f"{scenario_id}|{seed}|{arm}".encode("utf-8")).hexdigest()[:12]
    return Path(evidence_root) / f"r-{token}"
