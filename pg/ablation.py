"""Week 10: explicit on/off switches for each ProvenanceGuard layer.

``FULL`` (every layer on) is the default everywhere, and with it the code
paths are byte-for-byte the ones Weeks 1-9 exercised.  Turning a layer off
replaces that layer's *enforcement* with the most naive plausible
alternative; it never removes the surrounding plumbing, because the
plumbing (mediated call -> proposal -> approval -> gateway -> adapter) is
what lets the other layers be measured in isolation.

What each switch turns off
--------------------------
schema               Tool-contract validation/normalisation of arguments
                     (types, lengths, extra/missing fields, e-mail syntax,
                     editable-field allowlist).  The strict Pydantic *record*
                     models that guard persistence stay on: they are the
                     storage format, not a defence.
provenance           Source records and source digests.  Proposals carry one
                     fixed placeholder digest and no source rows are stored,
                     so an executed action cannot be traced to the content
                     that influenced it.
policy               Allowlist / ownership decisions.  Routing by tool class
                     is kept (read tools: ALLOW, everything else:
                     REQUIRE_APPROVAL) so a human-approval path still exists,
                     but nothing is ever DENY.
signed_proposal      Ed25519 signing of proposals and signature verification
                     whenever a proposal is reloaded.  (The stored-digest vs
                     payload recomputation is content addressing and stays.)
grounded_approval    The trusted approval card.  Off: the card is built from
                     the model's own ``claimed_fields`` / explanation and
                     mismatches are not surfaced.
gateway_revalidation Everything the gateway re-checks before executing:
                     approval signature, decision, expiry, digest / proposal /
                     session / audience binding, policy-version and live
                     policy recheck, source drift, repeated-probe freeze.
atomic_consume       Atomic consume-and-reserve of the approval.  Off: a
                     check-then-act flag test (with an optional race window)
                     and bookkeeping written only *after* the side effect,
                     with no execution-level lock.
event_chain          Hash-chained, signed audit events and chain verification.
checkpoints          Signed checkpoints with an anchor and checkpoint
                     verification.
output_guard         Read-derived sensitivity tracking and the final-output
                     disclosure block.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Iterable

LAYER_NAMES: tuple[str, ...] = (
    "schema",
    "provenance",
    "policy",
    "signed_proposal",
    "grounded_approval",
    "gateway_revalidation",
    "atomic_consume",
    "event_chain",
    "checkpoints",
    "output_guard",
)


@dataclass(frozen=True)
class Layers:
    schema: bool = True
    provenance: bool = True
    policy: bool = True
    signed_proposal: bool = True
    grounded_approval: bool = True
    gateway_revalidation: bool = True
    atomic_consume: bool = True
    event_chain: bool = True
    checkpoints: bool = True
    output_guard: bool = True

    def __post_init__(self) -> None:
        for name in LAYER_NAMES:
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"layer switch {name!r} must be a bool")

    def enabled(self) -> tuple[str, ...]:
        return tuple(name for name in LAYER_NAMES if getattr(self, name))

    def disabled(self) -> tuple[str, ...]:
        return tuple(name for name in LAYER_NAMES if not getattr(self, name))

    def without(self, *names: str) -> "Layers":
        _check_names(names)
        return replace(self, **{name: False for name in names})

    def with_(self, *names: str) -> "Layers":
        _check_names(names)
        return replace(self, **{name: True for name in names})

    @classmethod
    def only(cls, *names: str) -> "Layers":
        _check_names(names)
        return cls(**{name: name in names for name in LAYER_NAMES})

    def to_dict(self) -> dict[str, bool]:
        return {name: getattr(self, name) for name in LAYER_NAMES}

    @classmethod
    def from_dict(cls, data: dict[str, bool]) -> "Layers":
        unknown = set(data) - set(LAYER_NAMES)
        if unknown:
            raise ValueError(f"unknown layer switches: {sorted(unknown)}")
        return cls(**{name: bool(data.get(name, True)) for name in LAYER_NAMES})

    @property
    def is_full(self) -> bool:
        return all(getattr(self, f.name) for f in fields(self))


def _check_names(names: Iterable[str]) -> None:
    unknown = [name for name in names if name not in LAYER_NAMES]
    if unknown:
        raise ValueError(f"unknown layer(s) {unknown}; valid: {list(LAYER_NAMES)}")


FULL = Layers()
NONE = Layers.only()  # mediation plumbing only

# Cumulative ladder.  A0 is the stock AgentDojo baseline (no pipeline at
# all) and is not a Layers value; A0m is the pipeline with every layer off.
LADDER_STEPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("A1", ("schema",)),
    ("A2", ("provenance",)),
    ("A3", ("policy",)),
    ("A4", ("signed_proposal",)),
    ("A5", ("grounded_approval",)),
    ("A6", ("gateway_revalidation", "atomic_consume")),
    ("A7", ("event_chain", "checkpoints")),
    ("A8", ("output_guard",)),
)


def ladder() -> dict[str, Layers]:
    configs: dict[str, Layers] = {}
    enabled: list[str] = []
    for name, added in LADDER_STEPS:
        enabled.extend(added)
        configs[name] = Layers.only(*enabled)
    assert configs["A8"] == FULL
    return configs


def ladder_added(name: str) -> tuple[str, ...]:
    for step, added in LADDER_STEPS:
        if step == name:
            return added
    raise KeyError(name)


def leave_one_out() -> dict[str, Layers]:
    return {f"FULL-{name}": FULL.without(name) for name in LAYER_NAMES}


def only_one() -> dict[str, Layers]:
    return {f"ONLY-{name}": Layers.only(name) for name in LAYER_NAMES}


def part_a_configs() -> dict[str, Layers]:
    """Every configuration the deterministic protocol-attack suite runs."""
    configs: dict[str, Layers] = {"A0m": NONE}
    configs.update(ladder())
    configs.update(leave_one_out())
    configs.update(only_one())
    return configs


# Model-driven configurations (Part B).  A0 is the stock baseline arm.  The
# leave-one-out run is limited to the policy layer, the only one expected to
# change model-driven outcomes; the remaining leave-one-out and single-layer
# comparisons are exhaustive in the deterministic suite.
MODEL_DRIVEN_ORDER: tuple[str, ...] = (
    "A0", "A8", "A3", "A2", "A0m", "A1", "A4", "A5", "A6", "A7", "FULL-policy",
)


def model_driven_configs() -> dict[str, Layers | None]:
    configs: dict[str, Layers | None] = {"A0": None, "A0m": NONE}
    configs.update(ladder())
    configs["FULL-policy"] = FULL.without("policy")
    return {name: configs[name] for name in MODEL_DRIVEN_ORDER}


def describe(layers: Layers) -> str:
    off = layers.disabled()
    if not off:
        return "all layers on"
    if len(off) == len(LAYER_NAMES):
        return "all layers off (mediation plumbing only)"
    if len(off) <= 3:
        return "off: " + ", ".join(off)
    return "on: " + ", ".join(layers.enabled())
