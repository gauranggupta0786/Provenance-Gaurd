"""The four AgentDojo v1.2.2 suites as data: which tools read, which write, how a write becomes a signed canonical call.

Workspace reuses v1's tables verbatim (``pg.agentdojo_adapter``), so a workspace episode under v2 mediates exactly the same calls
as under v1.  Banking, Slack and Travel are new.  Every tool a suite exposes must be classified here; an unclassified tool makes
the runtime refuse to start (the v1 rule, kept).

Canonical encodings.  v1's contract checker (``pg.schemas``) knows strings, integers, lists, objects and optionals, not floats or
booleans, and must not be edited.  Money amounts are therefore signed as decimal strings ("12.5") and flags as "true"/"false";
the gateway adapter decodes them back to the AgentDojo types just before execution.  A decimal string is also the safer thing
to sign: two floats that print differently can be the same number, a canonical decimal cannot.

``get_webpage`` (Slack) reads in name but sends data out through its URL (an injection can ask the agent to "visit"
``attacker.com/?d=<secret>``), so it is mediated as an outgoing action and its result is logged as a read source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping

from pg.agentdojo_adapter import READ_CONTENT_WRITE_FUNCTIONS as WS_READ_CONTENT_WRITES
from pg.agentdojo_adapter import READ_ONLY_FUNCTIONS as WS_READS
from pg.agentdojo_adapter import WRITE_FUNCTION_MAP as WS_WRITES
from pg.agentdojo_adapter import to_canonical_call as ws_to_canonical
from pg.schemas import RiskLevel

BENCHMARK_VERSION = "v1.2.2"
SUITES = ("workspace", "travel", "banking", "slack")


class SuiteSpecError(ValueError):
    pass


def money(value: Any) -> str:
    """Canonical decimal string for an amount (raises on non-numbers, NaN and infinities)."""
    if isinstance(value, bool):
        raise SuiteSpecError("an amount must be a number, not a boolean")
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise SuiteSpecError(f"amount {value!r} is not a number") from exc
    if not d.is_finite():
        raise SuiteSpecError(f"amount {value!r} is not finite")
    text = format(d.normalize(), "f")
    return "0" if text in ("-0", "") else text


def flag(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower()
    raise SuiteSpecError(f"{value!r} is not a boolean")


def _opt(f: Callable[[Any], str], value: Any) -> str | None:
    return None if value is None else f(value)


def _c(required: dict, optional: dict | None = None, *, max_len: int = 4096, empty: set | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"required_args": required, "risk_level": RiskLevel.HIGH, "requires_approval": True, "max_str_len": max_len}
    if optional:
        out["optional_args"] = optional
    if empty:
        out["allow_empty_fields"] = set(empty)
    return out


# --------------------------------------------------------------------------- contracts for the new canonical tools
NEW_CONTRACTS: dict[str, dict[str, Any]] = {
    # banking
    "banking_send_money": _c({"recipient": str, "amount": str, "subject": str, "date": str}, max_len=1024),
    "banking_schedule_transaction": _c({"recipient": str, "amount": str, "subject": str, "date": str, "recurring": str}, max_len=1024),
    "banking_update_scheduled_transaction": _c({"id": int}, {"recipient": (str | None, None), "amount": (str | None, None),
                                                             "subject": (str | None, None), "date": (str | None, None),
                                                             "recurring": (str | None, None)}, max_len=1024),
    "banking_update_password": _c({"password": str}, max_len=256),
    "banking_update_user_info": _c({}, {"first_name": (str | None, None), "last_name": (str | None, None),
                                        "street": (str | None, None), "city": (str | None, None)}, max_len=256),
    # slack
    "slack_add_user_to_channel": _c({"user": str, "channel": str}, max_len=256),
    "slack_send_direct_message": _c({"recipient": str, "body": str}, max_len=8192),
    "slack_send_channel_message": _c({"channel": str, "body": str}, max_len=8192),
    "slack_invite_user_to_slack": _c({"user": str, "user_email": str}, max_len=256),
    "slack_remove_user_from_slack": _c({"user": str}, max_len=256),
    "slack_post_webpage": _c({"url": str, "content": str}, max_len=16384),
    "slack_get_webpage": _c({"url": str}, max_len=2048),
    # travel
    "travel_create_calendar_event": _c({"title": str, "start_time": str, "end_time": str},
                                       {"description": (str, ""), "participants": (list[str], []), "location": (str | None, None)},
                                       empty={"description"}),
    "travel_cancel_calendar_event": _c({"event_id": str}, max_len=256),
    "travel_reserve_hotel": _c({"hotel": str, "start_day": str, "end_day": str}, max_len=256),
    "travel_reserve_car_rental": _c({"company": str, "start_time": str}, {"end_time": (str | None, None)}, max_len=256),
    "travel_reserve_restaurant": _c({"restaurant": str, "start_time": str}, max_len=256),
    "travel_send_email": _c({"recipients": list[str], "subject": str, "body": str},
                            {"attachments": (list[dict], []), "cc": (list[str], []), "bcc": (list[str], [])}),
}


# --------------------------------------------------------------------------- encode (AgentDojo -> canonical) / decode
def _enc_send_email(a: dict) -> dict:
    a["recipients"] = list(a.get("recipients") or [])
    a["attachments"] = list(a.get("attachments") or [])
    a["cc"] = list(a.get("cc") or [])
    a["bcc"] = list(a.get("bcc") or [])
    return a


def _enc_event(a: dict) -> dict:
    a["description"] = a.get("description") or ""
    a["participants"] = list(a.get("participants") or [])
    a["location"] = a.get("location")
    return a


def _encode(function: str, a: dict) -> dict:
    if function in ("send_money", "schedule_transaction"):
        a["amount"] = money(a.get("amount"))
        if function == "schedule_transaction":
            a["recurring"] = flag(a.get("recurring"))
    elif function == "update_scheduled_transaction":
        for k in ("recipient", "subject", "date"):
            a.setdefault(k, None)
        a["amount"] = _opt(money, a.get("amount"))
        a["recurring"] = _opt(flag, a.get("recurring"))
    elif function == "update_user_info":
        for k in ("first_name", "last_name", "street", "city"):
            a.setdefault(k, None)
    elif function == "send_email":
        a = _enc_send_email(a)
    elif function == "create_calendar_event":
        a = _enc_event(a)
    elif function == "reserve_car_rental":
        a.setdefault("end_time", None)
    return a


def _decode(function: str, a: Mapping[str, Any]) -> dict:
    out = dict(a)
    if function in ("send_money", "schedule_transaction"):
        out["amount"] = float(Decimal(out["amount"]))
        if function == "schedule_transaction":
            out["recurring"] = out["recurring"] == "true"
    elif function == "update_scheduled_transaction":
        if out.get("amount") is not None:
            out["amount"] = float(Decimal(out["amount"]))
        if out.get("recurring") is not None:
            out["recurring"] = out["recurring"] == "true"
    return out


# --------------------------------------------------------------------------- the spec
@dataclass(frozen=True)
class SuiteSpec:
    name: str
    reads: frozenset[str]
    writes: Mapping[str, str]                      # AgentDojo name -> canonical name
    read_content_writes: frozenset[str] = frozenset()
    # canonical tool -> argument names that are destinations (who or where the action goes) and their kind
    destinations: Mapping[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)
    # canonical tool -> argument names that refer to an existing object by id (resolved to a name by trusted code)
    objects: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # canonical tool -> argument names carrying free text that leaves the user's control (checked for content provenance)
    text_fields: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def canonical_to_agentdojo(self) -> dict[str, str]:
        return {c: f for f, c in self.writes.items()}

    def to_canonical(self, function: str, arguments: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        if function not in self.writes:
            raise SuiteSpecError(f"{function!r} is not a mediated {self.name} write function")
        if self.name == "workspace":
            return ws_to_canonical(function, arguments)
        return self.writes[function], _encode(function, dict(arguments))

    def to_agentdojo(self, canonical: str, arguments: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        function = self.canonical_to_agentdojo[canonical]
        return function, (dict(arguments) if self.name == "workspace" else _decode(function, arguments))

    def classify(self, tool_names: set[str]) -> None:
        unknown = set(tool_names) - set(self.reads) - set(self.writes)
        if unknown:
            raise SuiteSpecError(f"{self.name}: unclassified tools {sorted(unknown)}")
        both = set(self.reads) & set(self.writes)
        if both:
            raise SuiteSpecError(f"{self.name}: tools classified as both read and write {sorted(both)}")


_EMAIL_DEST = (("recipients", "email"), ("cc", "email"), ("bcc", "email"))

WORKSPACE = SuiteSpec(
    "workspace", frozenset(WS_READS), dict(WS_WRITES), frozenset(WS_READ_CONTENT_WRITES),
    destinations={"workspace_send_email": _EMAIL_DEST, "workspace_share_file": (("email", "email"),),
                  "workspace_create_calendar_event": (("participants", "email"),),
                  "workspace_add_calendar_event_participants": (("participants", "email"),)},
    objects={"workspace_delete_email": ("email_id",), "workspace_delete_file": ("file_id",), "workspace_share_file": ("file_id",),
             "workspace_append_to_file": ("file_id",), "workspace_cancel_calendar_event": ("event_id",),
             "workspace_reschedule_calendar_event": ("event_id",), "workspace_add_calendar_event_participants": ("event_id",)},
    text_fields={"workspace_send_email": ("subject", "body"), "workspace_create_file": ("content",),
                 "workspace_append_to_file": ("content",), "workspace_create_calendar_event": ("description",)},
)

BANKING = SuiteSpec(
    "banking",
    frozenset({"get_iban", "get_balance", "get_most_recent_transactions", "get_scheduled_transactions", "read_file", "get_user_info"}),
    {"send_money": "banking_send_money", "schedule_transaction": "banking_schedule_transaction",
     "update_scheduled_transaction": "banking_update_scheduled_transaction", "update_password": "banking_update_password",
     "update_user_info": "banking_update_user_info"},
    destinations={"banking_send_money": (("recipient", "iban"),), "banking_schedule_transaction": (("recipient", "iban"),),
                  "banking_update_scheduled_transaction": (("recipient", "iban"),)},
    objects={"banking_update_scheduled_transaction": ("id",)},
    text_fields={"banking_send_money": ("subject",), "banking_schedule_transaction": ("subject",),
                 "banking_update_scheduled_transaction": ("subject",)},
)

SLACK = SuiteSpec(
    "slack",
    frozenset({"get_channels", "read_channel_messages", "read_inbox", "get_users_in_channel"}),
    {"add_user_to_channel": "slack_add_user_to_channel", "send_direct_message": "slack_send_direct_message",
     "send_channel_message": "slack_send_channel_message", "invite_user_to_slack": "slack_invite_user_to_slack",
     "remove_user_from_slack": "slack_remove_user_from_slack", "post_webpage": "slack_post_webpage",
     "get_webpage": "slack_get_webpage"},
    frozenset({"get_webpage"}),
    destinations={"slack_add_user_to_channel": (("user", "slack_user"), ("channel", "slack_channel")),
                  "slack_send_direct_message": (("recipient", "slack_user"),),
                  "slack_send_channel_message": (("channel", "slack_channel"),),
                  "slack_invite_user_to_slack": (("user", "slack_user"), ("user_email", "email")),
                  "slack_remove_user_from_slack": (("user", "slack_user"),),
                  "slack_post_webpage": (("url", "url"),), "slack_get_webpage": (("url", "url"),)},
    text_fields={"slack_send_direct_message": ("body",), "slack_send_channel_message": ("body",),
                 "slack_post_webpage": ("content",), "slack_get_webpage": ("url",)},
)

TRAVEL = SuiteSpec(
    "travel",
    frozenset({"get_user_information", "get_all_hotels_in_city", "get_hotels_prices", "get_rating_reviews_for_hotels",
               "get_hotels_address", "get_all_restaurants_in_city", "get_cuisine_type_for_restaurants", "get_restaurants_address",
               "get_rating_reviews_for_restaurants", "get_dietary_restrictions_for_all_restaurants",
               "get_contact_information_for_restaurants", "get_price_for_restaurants", "check_restaurant_opening_hours",
               "get_all_car_rental_companies_in_city", "get_car_types_available", "get_rating_reviews_for_car_rental",
               "get_car_fuel_options", "get_car_rental_address", "get_car_price_per_day", "search_calendar_events",
               "get_day_calendar_events", "get_flight_information"}),
    {"create_calendar_event": "travel_create_calendar_event", "cancel_calendar_event": "travel_cancel_calendar_event",
     "reserve_hotel": "travel_reserve_hotel", "reserve_car_rental": "travel_reserve_car_rental",
     "reserve_restaurant": "travel_reserve_restaurant", "send_email": "travel_send_email"},
    destinations={"travel_send_email": _EMAIL_DEST, "travel_create_calendar_event": (("participants", "email"),),
                  "travel_reserve_hotel": (("hotel", "place"),), "travel_reserve_car_rental": (("company", "place"),),
                  "travel_reserve_restaurant": (("restaurant", "place"),)},
    objects={"travel_cancel_calendar_event": ("event_id",)},
    text_fields={"travel_send_email": ("subject", "body"), "travel_create_calendar_event": ("description",)},
)

SPECS: dict[str, SuiteSpec] = {s.name: s for s in (WORKSPACE, BANKING, SLACK, TRAVEL)}


def spec_for(suite: str) -> SuiteSpec:
    try:
        return SPECS[suite]
    except KeyError as exc:
        raise SuiteSpecError(f"unknown suite {suite!r}; choose from {SUITES}") from exc



def destinations_of(spec: SuiteSpec, canonical: str, arguments: Mapping[str, Any]) -> list[tuple[str, str]]:
    """(kind, value) for every destination in a call: whom or where it goes."""
    out: list[tuple[str, str]] = []
    for arg, kind in spec.destinations.get(canonical, ()):
        value = arguments.get(arg)
        values = value if isinstance(value, list) else [value]
        out += [(kind, str(v)) for v in values if v not in (None, "")]
    return out


def texts_of(spec: SuiteSpec, canonical: str, arguments: Mapping[str, Any]) -> str:
    return "\n".join(str(arguments.get(k) or "") for k in spec.text_fields.get(canonical, ()))
