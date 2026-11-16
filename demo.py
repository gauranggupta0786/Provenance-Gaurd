"""Runs the full Week 1-4 propose -> approve -> execute pipeline:
valid requests, a deceptive model explanation, a forged/expired/replayed
approval, and an end-to-end integrity check. This is the "concise demo"
deliverable referenced by the execution plan (valid approved action;
forged summary; expired/replayed approval; audit investigation).

Run: python3 demo.py
"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from pg.crypto import generate_keypair, Signer, Verifier
from pg.store import EventStore
from pg.builder import ProposalService, ProposalRejected
from pg.agent import DeterministicAgent, make_propose_channel
from pg.approval import ApprovalService, ApprovalError, ReviewerAuthError, render_card
from pg.gateway import ExecutionGateway, ForgedTokenError, TokenExpiredError, ReplayRejected
from pg.tools import WorldState
from pg import audit


def hr(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data", "demo_run")
    shutil.rmtree(data_dir, ignore_errors=True)
    os.makedirs(data_dir, exist_ok=True)

    keypair = generate_keypair("key_demo_1")
    signer = Signer(keypair)
    verifier = Verifier()
    verifier.register_keypair(keypair)

    store = EventStore(os.path.join(data_dir, "events.db"), os.path.join(data_dir, "checkpoints"))
    service = ProposalService(store, signer, verifier)
    approvals = ApprovalService(store, signer, verifier)
    world = WorldState()
    gateway = ExecutionGateway(store, signer, verifier, world)

    channel = make_propose_channel(service)
    agent = DeterministicAgent(session_id="demo_session", principal_id="user_gaurang",
                                model_id="stub-model-v1", propose_channel=channel)

    # -- 1. Valid, low-risk action: auto-approved (policy ALLOW), executed --
    hr("1. Valid low-risk action -- summarize (auto-approved, executed)")
    outcome = agent.handle_request("Please summarize the latest message")
    p = outcome.proposal
    print(f"proposal_id={p.proposal_id[:8]}...  policy={outcome.policy_decision.value}")
    ap = approvals.auto_approve(p.proposal_id)
    res = gateway.execute(p.proposal_id, ap.approval_id)
    print(f"executed: status={res.receipt.result_status}  result={res.tool_result}")

    # -- 2. Valid, high-risk action: human-approved, executed ---------------
    hr("2. Valid high-risk action -- send_email (human-approved, executed)")
    o2 = service.propose_action(
        session_id="demo_session", principal_id="user_gaurang", user_goal="send an email",
        model_id="stub-model-v1", tool_name="send_email",
        arguments={"recipient": "mentor@example.com", "subject": "Weekly update",
                   "body": "Summary of this week's progress is attached."},
        model_explanation="Sending the routine weekly status update to my mentor.",
    )
    card = render_card(store, verifier, o2.proposal.proposal_id)
    print("Approval card shown to reviewer:")
    print(card.render_text())
    r2 = approvals.decide(o2.proposal.proposal_id, reviewer_id="mentor",
                           session_token="sess-tok-mentor-001", decision="approve")
    res2 = gateway.execute(o2.proposal.proposal_id, r2.approval_id)
    print(f"\nexecuted: status={res2.receipt.result_status}  mailbox={world.mailbox}")

    # -- 3. Forged approval summary: model's own explanation lies ------------
    hr("3. Forged approval summary -- model explanation lies about the action")
    o3 = service.propose_action(
        session_id="demo_session", principal_id="user_gaurang", user_goal="update a record",
        model_id="stub-model-v1", tool_name="update_record",
        arguments={"record_id": "rec_1001", "field": "status", "new_value": "closed"},
        model_explanation="Refunded the office coffee vendor $10. Status: Verified.",
        claimed_fields={"field": "notes", "new_value": "no change"},
    )
    card3 = render_card(store, verifier, o3.proposal.proposal_id)
    print("Card shows the REAL action despite the misleading explanation:")
    print(card3.render_text())
    print(f"\n-> A human approving this card sees the true target/field/value, "
          f"not the model's prose. {len(card3.claim_mismatches)} mismatch(es) flagged.")

    # -- 4. Replay: reuse a consumed approval --------------------------------
    hr("4. Replay rejected -- reusing an already-consumed approval")
    try:
        gateway.execute(o2.proposal.proposal_id, r2.approval_id)
    except ReplayRejected as e:
        print(f"REJECTED as expected: {e}")

    # -- 5. Expired token ------------------------------------------------------
    hr("5. Expired approval rejected")
    o5 = service.propose_action(
        session_id="demo_session", principal_id="user_gaurang", user_goal="send an email",
        model_id="stub-model-v1", tool_name="send_email",
        arguments={"recipient": "mentor@example.com", "subject": "late", "body": "x"},
    )
    r5 = approvals.decide(o5.proposal.proposal_id, reviewer_id="mentor",
                           session_token="sess-tok-mentor-001", decision="approve", ttl_seconds=1)
    time.sleep(1.2)
    try:
        gateway.execute(o5.proposal.proposal_id, r5.approval_id)
    except TokenExpiredError as e:
        print(f"REJECTED as expected: {e}")

    # -- 6. Rejected: unauthorized recipient (TIP-style attempt) --------------
    hr("6. Rejected -- unauthorized recipient (simulated TIP/compromised-agent attempt)")
    try:
        service.propose_action(
            session_id="demo_session", principal_id="user_gaurang", user_goal="send an email",
            model_id="stub-model-v1", tool_name="send_email",
            arguments={"recipient": "attacker@evil.example", "subject": "wire details", "body": "..."},
        )
    except ProposalRejected as e:
        print(f"REJECTED as expected: {e}")

    # -- 7. Audit investigation ------------------------------------------------
    hr("7. Audit console / incident investigation")
    verifier2 = Verifier()
    verifier2.register_keypair(keypair)
    store.verify_chain(verifier2)
    print(f"Chain verified OK. {store.latest_seq()} events.")
    cp_path = store.export_checkpoint(signer)
    store.verify_against_checkpoint(verifier2)
    print(f"Checkpoint exported and verified: {cp_path}")

    snap = audit.summary(store, verifier2, principal_ids=["user_gaurang"])
    print(f"pending_approvals={snap['pending_approvals']}")
    print(f"blocked_or_denied={snap['blocked_or_denied']}")
    print(f"missing_receipts={len(snap['missing_receipts'])}")
    print(f"integrity={'OK' if snap['integrity'] is None else snap['integrity']}")
    print(f"alerts={snap['alerts']}")

    print("\nFull trace for the email action (proposal -> approval -> receipt):")
    t = audit.trace(store, verifier2, o2.proposal.proposal_id)
    print(json.dumps({
        "current_state": t["current_state"],
        "proposal_signature_valid": t["proposal_signature_valid"],
        "receipt": t["receipt"],
        "state_history": [h["state"] for h in t["state_history"]],
    }, indent=2, sort_keys=True))

    store.close()
    print(f"\nDemo artifacts written under: {data_dir}")


if __name__ == "__main__":
    main()
