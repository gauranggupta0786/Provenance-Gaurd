"""Week 2 Day 5 exit checks: record mutation, deletion, reordering, wrong
keys, missing records, partial writes, log-tail deletion vs checkpoint.
"""
import sqlite3
import pytest

from pg.store import IntegrityError


def _seed_events(agent, n=3):
    outcomes = []
    goals = ["summarize the message", "send an email update", "update the record status"]
    for i in range(n):
        outcomes.append(agent.handle_request(goals[i % len(goals)]))
    return outcomes


# -- normal path: triggers block ordinary UPDATE/DELETE ---------------------

def test_trigger_blocks_direct_update_on_events(agent, store):
    _seed_events(agent, 2)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store._conn.execute("UPDATE events SET event_type='HACKED' WHERE seq=1")


def test_trigger_blocks_direct_delete_on_events(agent, store):
    _seed_events(agent, 2)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store._conn.execute("DELETE FROM events WHERE seq=1")


def test_trigger_blocks_direct_update_on_proposals(agent, store):
    outcome = agent.handle_request("summarize the message")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store._conn.execute(
            "UPDATE proposals SET payload='{}' WHERE proposal_id=?",
            (outcome.proposal.proposal_id,),
        )


# -- attacker with raw file/DB access bypasses triggers: hash chain catches it --

def _drop_event_triggers(store):
    store._conn.executescript(
        "DROP TRIGGER IF EXISTS events_no_update; DROP TRIGGER IF EXISTS events_no_delete;"
    )


def test_mutated_event_content_detected(agent, store, verifier):
    _seed_events(agent, 3)
    _drop_event_triggers(store)
    store._conn.execute("UPDATE events SET record_ids='{\"attempt_id\": \"forged\"}' WHERE seq=2")

    with pytest.raises(IntegrityError, match="altered"):
        store.verify_chain(verifier)


def test_deleted_middle_event_detected(agent, store, verifier):
    _seed_events(agent, 3)
    _drop_event_triggers(store)
    store._conn.execute("DELETE FROM events WHERE seq=2")

    with pytest.raises(IntegrityError, match="gap|link"):
        store.verify_chain(verifier)


def test_reordered_prev_hash_detected(agent, store, verifier):
    _seed_events(agent, 3)
    _drop_event_triggers(store)
    # Corrupt the hash link directly -- simulates reordering/splicing.
    store._conn.execute("UPDATE events SET prev_hash='deadbeef' || prev_hash WHERE seq=3")

    with pytest.raises(IntegrityError, match="link"):
        store.verify_chain(verifier)


def test_forged_signature_detected(agent, store, verifier):
    _seed_events(agent, 2)
    _drop_event_triggers(store)
    store._conn.execute("UPDATE events SET signature='AAAAforged' WHERE seq=1")

    with pytest.raises(IntegrityError, match="signature"):
        store.verify_chain(verifier)


# -- checkpoint catches tail deletion / whole-log replacement that a bare hash chain misses --

def test_tail_deletion_after_checkpoint_detected(agent, store, signer, verifier):
    _seed_events(agent, 3)
    store.export_checkpoint(signer)  # checkpoint pinned at seq=3
    _seed_events(agent, 2)  # 2 more events past the checkpoint (seq=4,5)

    _drop_event_triggers(store)
    # Truncate below the checkpoint's seq (only seq=1 survives).
    store._conn.execute("DELETE FROM events WHERE seq > 1")

    # The local hash chain among the *remaining* rows can still look internally
    # consistent -- the checkpoint is what catches the truncation.
    with pytest.raises(IntegrityError, match="truncated"):
        store.verify_against_checkpoint(verifier)


def test_whole_log_replacement_detected(agent, store, signer, verifier, tmp_dirs):
    import os
    from pg.store import EventStore

    _seed_events(agent, 3)
    store.export_checkpoint(signer)

    # Build a *different*, internally-consistent, same-length replacement log
    # in a fresh DB file, then swap it in -- simulating full log replacement.
    replacement_path = os.path.join(tmp_dirs, "replacement.db")
    other = EventStore(replacement_path, os.path.join(tmp_dirs, "other_checkpoints"))
    # Match the checkpoint's seq exactly (proposal + PROPOSED + VERIFIED
    # state-transition events per handle_request as of Week 3) so this
    # test exercises "same length, different content" divergence rather
    # than tripping the (also-correct) truncation check first.
    for i in range(store.latest_seq()):
        other.append_event(
            event_type="PROPOSAL_STORED",
            record_ids={"proposal_id": f"forged-{i}"},
            content_digest="0" * 64,
            signer=signer,
        )
    other.close()

    store.close()
    os.replace(replacement_path, store.db_path)
    store2 = EventStore(store.db_path, store.checkpoint_dir)
    with pytest.raises(IntegrityError, match="diverges"):
        store2.verify_against_checkpoint(verifier)
    store2.close()


# -- missing records / wrong keys / partial writes ---------------------------

def test_missing_record_returns_none(store):
    assert store.get_proposal("does-not-exist") is None


def test_duplicate_proposal_id_rejected_no_partial_write(agent, store):
    from pg.schemas import SchemaError
    outcome = agent.handle_request("summarize the message")
    before = store.latest_seq()
    dup = outcome.proposal
    with pytest.raises(SchemaError):
        store.append_proposal(dup)  # same proposal_id again
    # No partial/duplicate row, no stray event from the failed re-insert attempt.
    assert store.latest_seq() == before
