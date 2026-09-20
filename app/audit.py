"""Append-only adjudication audit trail with a per-merge SHA-256 chain.

Every request that successfully changes a merge order through one of the
adjudication paths — explicit resolutions, rule resolutions, historical
reuse or a draft commit — appends exactly one event here, inside the same
SQLite transaction as the merge update itself: a state change and its
audit event commit or roll back together. Failed requests (400/409/422)
and no-op requests that change nothing (a rule set or reuse plan without
any match) write no merge state and therefore leave no event.

Each event records the revision before/after the change, the adjudication
method and the JSON Pointers that actually took effect, and is chained to
the merge's previous event: ``hash`` is the SHA-256 of the event payload's
canonical JSON (sorted keys, no whitespace), and the payload includes the
previous event's hash (null for the first event of a merge). Verification
replays the stored events in sequence order, recomputes every hash and
checks the linkage, reporting the first sequence number where the chain
breaks.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone

from . import db
from .patch import canonical_json

METHOD_RESOLUTIONS = "resolutions"
METHOD_RULES = "rule-resolutions"
METHOD_REUSE = "reuse-resolutions"
METHOD_DRAFT_COMMIT = "draft-commit"

# Kept in sync with the CHECK constraint on merge_audit_events.method.
METHODS = (METHOD_RESOLUTIONS, METHOD_RULES, METHOD_REUSE, METHOD_DRAFT_COMMIT)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _payload(event: dict) -> dict:
    """The canonical hash input: every stored field except the hash itself."""
    return {
        "merge_id": event["merge_id"],
        "seq": event["seq"],
        "method": event["method"],
        "revision_before": event["revision_before"],
        "revision_after": event["revision_after"],
        "pointers": event["pointers"],
        "detail": event["detail"],
        "prev_hash": event["prev_hash"],
        "created_at": event["created_at"],
    }


def compute_hash(event: dict) -> str:
    """SHA-256 hex digest of the event payload's canonical JSON."""
    return hashlib.sha256(canonical_json(_payload(event))).hexdigest()


def append_event(
    conn: sqlite3.Connection,
    merge_id: str,
    method: str,
    revision_before: int,
    revision_after: int,
    pointers: list[str],
    detail: dict | None = None,
) -> None:
    """Append one event to the merge's chain inside ``conn``'s transaction.

    The caller runs the transaction that also updates the merge row, so
    both writes commit or roll back together. ``pointers`` are the JSON
    Pointers that actually took effect, in application order.
    """
    last = conn.execute(
        "SELECT seq, hash FROM merge_audit_events WHERE merge_id = ? "
        "ORDER BY seq DESC LIMIT 1",
        (merge_id,),
    ).fetchone()
    event = {
        "merge_id": merge_id,
        "seq": 1 if last is None else last["seq"] + 1,
        "method": method,
        "revision_before": revision_before,
        "revision_after": revision_after,
        "pointers": list(pointers),
        "detail": detail if detail is not None else {},
        "prev_hash": None if last is None else last["hash"],
        "created_at": _now(),
    }
    conn.execute(
        """
        INSERT INTO merge_audit_events (
            merge_id, seq, method, revision_before, revision_after,
            pointers, detail, prev_hash, hash, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event["merge_id"],
            event["seq"],
            event["method"],
            event["revision_before"],
            event["revision_after"],
            json.dumps(event["pointers"]),
            json.dumps(event["detail"]),
            event["prev_hash"],
            compute_hash(event),
            event["created_at"],
        ),
    )


def list_events(merge_id: str) -> list[dict]:
    """All stored events of a merge in chain order, JSON fields parsed."""
    rows = db.query_all(
        "SELECT * FROM merge_audit_events WHERE merge_id = ? ORDER BY seq",
        (merge_id,),
    )
    return [
        {
            "merge_id": row["merge_id"],
            "seq": row["seq"],
            "method": row["method"],
            "revision_before": row["revision_before"],
            "revision_after": row["revision_after"],
            "pointers": json.loads(row["pointers"]),
            "detail": json.loads(row["detail"]),
            "prev_hash": row["prev_hash"],
            "hash": row["hash"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def verify_chain(events: list[dict]) -> tuple[bool, int | None]:
    """Replay the chain, returning (chain_valid, first_bad_seq).

    An event is anomalous when its sequence number is not the successor of
    the previous event's, when its ``prev_hash`` does not point at the
    previous event's hash (null for the first event), or when its stored
    hash differs from the recomputed one. The first anomalous event's
    ``seq`` is reported; an empty trail verifies as valid.
    """
    prev_hash = None
    expected_seq = 1
    for event in events:
        if (
            event["seq"] != expected_seq
            or event["prev_hash"] != prev_hash
            or event["hash"] != compute_hash(event)
        ):
            return False, event["seq"]
        prev_hash = event["hash"]
        expected_seq += 1
    return True, None


def event_view(event: dict) -> dict:
    """Serialize one event for the API response (merge_id is top-level)."""
    return {
        "seq": event["seq"],
        "method": event["method"],
        "revision_before": event["revision_before"],
        "revision_after": event["revision_after"],
        "pointers": event["pointers"],
        "detail": event["detail"],
        "prev_hash": event["prev_hash"],
        "hash": event["hash"],
        "created_at": event["created_at"],
    }
