import os
import sys
import shutil
from pathlib import Path
from uuid import uuid4
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pg.crypto import generate_keypair, Signer, Verifier
from pg.store import EventStore
from pg.builder import ProposalService
from pg.agent import DeterministicAgent, make_propose_channel
from pg.approval import ApprovalService
from pg.gateway import ExecutionGateway
from pg.tools import WorldState


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".test-tmp"


@pytest.fixture()
def tmp_path():
    """Workspace-local temp path that remains writable in Windows sandboxes.

    ``tempfile.mkdtemp`` and pytest's Windows temp fixture create mode-0700
    directories. That ACL can exclude the separate Codex sandbox identity even
    though the project itself is writable.
    """

    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEST_TMP_ROOT / f"case-{uuid4().hex}"
    path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def tmp_dirs(tmp_path):
    yield str(tmp_path)


@pytest.fixture()
def keypair():
    return generate_keypair("key_test_1")


@pytest.fixture()
def signer(keypair):
    return Signer(keypair)


@pytest.fixture()
def verifier(keypair):
    v = Verifier()
    v.register_keypair(keypair)
    return v


@pytest.fixture()
def store(tmp_dirs):
    db_path = os.path.join(tmp_dirs, "events.db")
    cp_dir = os.path.join(tmp_dirs, "checkpoints")
    s = EventStore(db_path, cp_dir)
    yield s
    s.close()


@pytest.fixture()
def service(store, signer, verifier):
    return ProposalService(store, signer, verifier)


@pytest.fixture()
def agent(service):
    channel = make_propose_channel(service)
    return DeterministicAgent(
        session_id="sess_1", principal_id="user_gaurang", model_id="stub-model-v1",
        propose_channel=channel,
    )


@pytest.fixture()
def approvals(store, signer, verifier):
    return ApprovalService(store, signer, verifier)


@pytest.fixture()
def world():
    return WorldState()


@pytest.fixture()
def gateway(store, signer, verifier, world):
    return ExecutionGateway(store, signer, verifier, world)
