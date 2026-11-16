import os
import sqlite3
import threading
import pytest

from pg.store import EventStore
from pg.builder import ProposalService, ProposalRejected


def test_controlled_restart_retains_proposals_and_chain(agent, store, signer, verifier, tmp_dirs):
    outcome = agent.handle_request("summarize the message")
    pid = outcome.proposal.proposal_id
    store.export_checkpoint(signer)

    db_path, cp_dir = store.db_path, store.checkpoint_dir
    store.close()  # simulate process stop

    reopened = EventStore(db_path, cp_dir)  # simulate restart
    assert reopened.get_proposal(pid) is not None
    reopened.verify_chain(verifier)
    reopened.verify_against_checkpoint(verifier)
    reopened.close()


def test_concurrent_submissions_produce_no_gaps_or_dupes(signer, verifier, tmp_dirs):
    db_path = os.path.join(tmp_dirs, "concurrent.db")
    cp_dir = os.path.join(tmp_dirs, "concurrent_cp")
    EventStore(db_path, cp_dir).close()  # create schema once up front

    errors = []

    def worker(i):
        try:
            store = EventStore(db_path, cp_dir)
            service = ProposalService(store, signer, verifier)
            service.propose_action(
                session_id=f"sess_{i}", principal_id="user_gaurang", user_goal="summarize please",
                model_id="stub-model-v1", tool_name="summarize",
                arguments={"message_id": "msg_1", "max_words": 5},
            )
            store.close()
        except Exception as e:  # pragma: no cover - failure path surfaced via `errors`
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent submissions raised: {errors}"

    final_store = EventStore(db_path, cp_dir)
    rows = final_store.all_events()
    seqs = [r["seq"] for r in rows]
    assert seqs == list(range(1, len(seqs) + 1)), "sequence must be gap-free and strictly increasing"
    final_store.verify_chain(verifier)  # no interleaved/corrupted writes
    final_store.close()


def test_unavailable_event_store_prevents_new_proposals(agent, store, signer, verifier, tmp_dirs):
    """A corrupt or unavailable event store must prevent action execution
    (fail closed), not silently proceed."""
    store.close()  # simulate the store becoming unavailable

    service = ProposalService(store, signer, verifier)
    with pytest.raises(sqlite3.Error):
        service.propose_action(
            session_id="sess_1", principal_id="user_gaurang", user_goal="summarize please",
            model_id="stub-model-v1", tool_name="summarize",
            arguments={"message_id": "msg_1", "max_words": 5},
        )


def test_corrupted_db_file_fails_closed(signer, tmp_dirs):
    db_path = os.path.join(tmp_dirs, "corrupt.db")
    cp_dir = os.path.join(tmp_dirs, "corrupt_cp")
    with open(db_path, "wb") as f:
        f.write(b"not a real sqlite file, this is garbage bytes")

    with pytest.raises(sqlite3.DatabaseError):
        EventStore(db_path, cp_dir)
