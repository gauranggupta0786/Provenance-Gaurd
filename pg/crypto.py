"""Week 2 Day 3: public-key signatures with a vetted library.

Design point from the plan: signing and verification are *separate
interfaces*. A Signer holds the private key and must never be reachable
from agent code. A Verifier holds only the public key and key registry,
and is what the policy/verifier/gateway components use.

HMAC is kept only as a clearly labeled local-demo option (not the
production path) per "Retain HMAC only as a clearly labeled local-demo
option."
"""
from __future__ import annotations

import hmac
import hashlib
import base64
import json
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature


class CryptoError(Exception):
    pass


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


@dataclass
class KeyPair:
    key_id: str
    private_key: Ed25519PrivateKey

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self.private_key.public_key()

    def public_bytes_b64(self) -> str:
        raw = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return _b64(raw)


def generate_keypair(key_id: str) -> KeyPair:
    return KeyPair(key_id=key_id, private_key=Ed25519PrivateKey.generate())


class DevelopmentKeyStore:
    """Persistent local-development Ed25519 keys with rotation metadata.

    This is deliberately not a production secret store. Deployments must use
    a KMS/HSM-backed signer and an independently retained public-key registry.
    """

    SCHEMA_VERSION = "development-keystore-v1"

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "manifest.json"

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _validate_key_id(key_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", key_id):
            raise CryptoError("key_id must be a safe 1-128 character identifier")

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, path)

    def _manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {
                "schema_version": self.SCHEMA_VERSION,
                "active_key_id": None,
                "rotation_counter": 0,
                "keys": [],
                "deployment_requirement": "KMS/HSM-backed non-exportable signing key",
            }
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != self.SCHEMA_VERSION:
            raise CryptoError("unsupported development key-store version")
        return manifest

    def _key_path(self, key_id: str) -> Path:
        self._validate_key_id(key_id)
        return self.directory / f"{key_id}.key.json"

    def _create(self, key_id: str, *, rotated_from: str | None) -> KeyPair:
        self._validate_key_id(key_id)
        path = self._key_path(key_id)
        if path.exists():
            raise CryptoError(f"key {key_id!r} already exists")
        pair = generate_keypair(key_id)
        private_raw = pair.private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        created_at = self._now()
        self._atomic_json(path, {
            "schema_version": "development-private-key-v1",
            "key_id": key_id,
            "algorithm": "Ed25519",
            "private_key_b64": _b64(private_raw),
            "public_key_b64": pair.public_bytes_b64(),
            "created_at": created_at,
            "rotated_from": rotated_from,
        })
        manifest = self._manifest()
        manifest["active_key_id"] = key_id
        manifest["rotation_counter"] = int(manifest["rotation_counter"]) + 1
        manifest["keys"].append({
            "key_id": key_id,
            "algorithm": "Ed25519",
            "public_key_b64": pair.public_bytes_b64(),
            "created_at": created_at,
            "rotated_from": rotated_from,
            "status": "active",
            "retired_at": None,
        })
        self._atomic_json(self.manifest_path, manifest)
        return pair

    def load_or_create(self, initial_key_id: str = "provenanceguard-dev-v1") -> KeyPair:
        manifest = self._manifest()
        active = manifest.get("active_key_id")
        if active is None:
            return self._create(initial_key_id, rotated_from=None)
        return self.load(active)

    def load(self, key_id: str) -> KeyPair:
        payload = json.loads(self._key_path(key_id).read_text(encoding="utf-8"))
        if payload.get("schema_version") != "development-private-key-v1" or payload.get("key_id") != key_id:
            raise CryptoError("invalid development private-key record")
        pair = KeyPair(
            key_id=key_id,
            private_key=Ed25519PrivateKey.from_private_bytes(_unb64(payload["private_key_b64"])),
        )
        if pair.public_bytes_b64() != payload["public_key_b64"]:
            raise CryptoError("development key public/private material does not match")
        return pair

    def rotate(self, new_key_id: str) -> KeyPair:
        manifest = self._manifest()
        previous = manifest.get("active_key_id")
        if previous is None:
            return self._create(new_key_id, rotated_from=None)
        now = self._now()
        for item in manifest["keys"]:
            if item["key_id"] == previous:
                item["status"] = "retired"
                item["retired_at"] = now
        self._atomic_json(self.manifest_path, manifest)
        return self._create(new_key_id, rotated_from=previous)

    def verifier(self) -> "Verifier":
        verifier = Verifier()
        for item in self._manifest()["keys"]:
            verifier.register(item["key_id"], item["public_key_b64"])
        return verifier

    def metadata(self) -> dict[str, Any]:
        return self._manifest()


class Signer:
    """Holds a private key. Never import this module's Signer into agent code."""

    def __init__(self, keypair: KeyPair):
        self._keypair = keypair

    @property
    def key_id(self) -> str:
        return self._keypair.key_id

    def sign(self, message: bytes) -> str:
        sig = self._keypair.private_key.sign(message)
        return _b64(sig)


class Verifier:
    """Holds only public keys. Safe to give to the policy/verifier/gateway."""

    def __init__(self):
        self._public_keys: dict[str, Ed25519PublicKey] = {}

    def register(self, key_id: str, public_key_b64: str) -> None:
        raw = _unb64(public_key_b64)
        self._public_keys[key_id] = Ed25519PublicKey.from_public_bytes(raw)

    def register_keypair(self, keypair: KeyPair) -> None:
        self._public_keys[keypair.key_id] = keypair.public_key

    def known_key_ids(self) -> list[str]:
        return sorted(self._public_keys)

    def verify(self, key_id: str, message: bytes, signature_b64: str) -> bool:
        pub = self._public_keys.get(key_id)
        if pub is None:
            return False
        try:
            pub.verify(_unb64(signature_b64), message)
            return True
        except (InvalidSignature, ValueError):
            return False


# ---------------------------------------------------------------------------
# HMAC -- local-demo option only. Not used by the default pipeline.
# ---------------------------------------------------------------------------

def hmac_sign_DEMO_ONLY(shared_secret: bytes, message: bytes) -> str:
    return _b64(hmac.new(shared_secret, message, hashlib.sha256).digest())


def hmac_verify_DEMO_ONLY(shared_secret: bytes, message: bytes, signature_b64: str) -> bool:
    expected = hmac.new(shared_secret, message, hashlib.sha256).digest()
    try:
        return hmac.compare_digest(expected, _unb64(signature_b64))
    except Exception:
        return False
