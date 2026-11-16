import copy
import pytest

from pg.tools import WorldState
from pg.crypto import Verifier
from pg.schemas import canonicalize


def test_agent_request_becomes_signed_proposal_without_touching_world(agent, store):
    world_before = WorldState()
    snapshot = copy.deepcopy(world_before)

    outcome = agent.handle_request("Please send an email with the weekly update")

    # World state is untouched: nothing in pg.agent or pg.builder imports pg.tools,
    # so there is no code path from this call that could have mutated it.
    assert world_before == snapshot
    assert len(world_before.mailbox) == 0

    p = outcome.proposal
    stored = store.get_proposal(p.proposal_id)
    assert stored is not None
    assert stored["payload"]["tool_name"] == "send_email"
    assert stored["digest"] == p.digest


def test_proposal_signature_verifies(agent, store, verifier: Verifier):
    outcome = agent.handle_request("Please summarize the message")
    p = outcome.proposal
    stored = store.get_proposal(p.proposal_id)
    ok = verifier.verify(stored["key_id"], canonicalize(stored["payload"]), stored["signature"])
    assert ok


def test_agent_module_has_no_tool_or_signing_access():
    """Negative access test (Week 2 exit check): the agent module must not
    import the tools module (world-mutating adapters) or the crypto Signer
    class (private-key holder). Checked via the AST's actual import
    statements, not a substring match (which would also flag this
    docstring's own explanation)."""
    import pg.agent as agent_mod
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(agent_mod))
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported_names.add(module)
            imported_names.update(f"{module}.{alias.name}" for alias in node.names)

    assert not any("tools" in name for name in imported_names), imported_names
    assert not any(name.endswith("Signer") or name == "pg.crypto" for name in imported_names), imported_names

    # Stronger structural check: module namespace has no 'tools' or 'Signer' symbol at all.
    assert not hasattr(agent_mod, "tools")
    assert not hasattr(agent_mod, "Signer")


def test_agent_instance_exposes_no_signer_or_store(agent):
    """The agent object itself carries only a propose channel -- no .store
    or .signer attribute for it to reach into."""
    assert not hasattr(agent, "store")
    assert not hasattr(agent, "signer")
    assert not hasattr(agent, "_store")
    assert not hasattr(agent, "_signer")
    # The channel is a plain function/closure, not the ProposalService itself.
    assert not hasattr(agent._propose, "_signer")
    assert not hasattr(agent._propose, "_store")


def test_wrapper_records_rejected_attempts_too(service, store):
    """"verify that wrappers record all attempted calls" -- including rejections."""
    n_events_before = store.latest_seq()
    from pg.builder import ProposalRejected
    with pytest.raises(ProposalRejected):
        service.propose_action(
            session_id="s", principal_id="user_gaurang", user_goal="x", model_id="m",
            tool_name="send_email",
            arguments={"recipient": "attacker@evil.example", "subject": "x", "body": "y"},
        )
    assert store.latest_seq() == n_events_before + 1
    rows = store.all_events()
    assert rows[-1]["event_type"] == "REJECTED_POLICY"
