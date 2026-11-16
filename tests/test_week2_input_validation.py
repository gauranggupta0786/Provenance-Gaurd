import pytest

from pg.schemas import (
    strict_json_loads, normalize_arguments, SchemaError, MAX_ARG_COUNT,
)


def test_duplicate_json_keys_rejected_at_input_boundary():
    raw = '{"recipient": "a@example.com", "recipient": "attacker@evil.example", "subject": "x", "body": "y"}'
    with pytest.raises(SchemaError, match="duplicate JSON key"):
        strict_json_loads(raw)


def test_wrong_type_rejected():
    with pytest.raises(SchemaError, match="must be a string"):
        normalize_arguments("send_email", {"recipient": 12345, "subject": "x", "body": "y"})


def test_bool_not_accepted_as_int_or_str():
    # bool is a subclass of int in Python; must not silently pass as either.
    with pytest.raises(SchemaError):
        normalize_arguments("summarize", {"message_id": "msg_1", "max_words": True})
    with pytest.raises(SchemaError):
        normalize_arguments("summarize", {"message_id": True, "max_words": 5})


def test_oversized_field_rejected():
    with pytest.raises(SchemaError, match="exceeds max length"):
        normalize_arguments("send_email", {
            "recipient": "a@example.com", "subject": "x", "body": "y" * 100_000,
        })


def test_out_of_range_int_rejected():
    with pytest.raises(SchemaError, match="out of allowed range"):
        normalize_arguments("summarize", {"message_id": "msg_1", "max_words": 999_999})
    with pytest.raises(SchemaError, match="out of allowed range"):
        normalize_arguments("summarize", {"message_id": "msg_1", "max_words": -1})


def test_ambiguous_identifier_rejected_empty_string():
    with pytest.raises(SchemaError, match="must not be empty"):
        normalize_arguments("update_record", {"record_id": "", "field": "status", "new_value": "x"})


def test_ambiguous_identifier_rejected_whitespace_padding_stripped_not_bypassed():
    # A recipient with embedded control characters shouldn't sneak past
    # the format check via odd whitespace.
    with pytest.raises(SchemaError):
        normalize_arguments("send_email", {
            "recipient": " a@example.com extra ", "subject": "x", "body": "y",
        })


def test_non_editable_field_rejected():
    with pytest.raises(SchemaError, match="not editable"):
        normalize_arguments("update_record", {
            "record_id": "rec_1001", "field": "owner_password", "new_value": "hunter2",
        })


def test_too_many_argument_fields_rejected():
    args = {f"extra_{i}": "x" for i in range(MAX_ARG_COUNT + 5)}
    with pytest.raises(SchemaError, match="too many argument"):
        normalize_arguments("summarize", args)


def test_unknown_tool_rejected():
    with pytest.raises(SchemaError, match="unknown tool"):
        normalize_arguments("wire_transfer", {"amount": 100})


def test_missing_field_rejected():
    with pytest.raises(SchemaError, match="missing required"):
        normalize_arguments("update_record", {"record_id": "rec_1001", "field": "status"})


def test_extra_field_rejected_including_model_supplied_flags():
    with pytest.raises(SchemaError, match="unexpected extra fields"):
        normalize_arguments("update_record", {
            "record_id": "rec_1001", "field": "status", "new_value": "x",
            "trusted": True, "priority": "urgent",
        })
