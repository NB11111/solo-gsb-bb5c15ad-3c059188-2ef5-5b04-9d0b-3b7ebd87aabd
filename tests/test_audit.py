"""End-to-end tests for the adjudication audit trail.

Every successful resolutions / rule-resolutions / reuse-resolutions /
draft-commit request appends exactly one event to the merge's audit chain
inside the same SQLite transaction as the merge update; failed requests
and zero-match no-ops leave no event. GET /merges/{merge_id}/audit returns
the events in chain order together with the SHA-256 chain verification,
naming the first anomalous sequence number when the chain is broken.
"""

import hashlib
import json as _json
import os
import tempfile

os.environ["DATABASE_PATH"] = os.path.join(tempfile.mkdtemp(), "test_audit.db")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import db  # noqa: E402
from app.main import app  # noqa: E402
from app.patch import canonical_json  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # Context-manager form runs the lifespan, which creates the tables.
    with TestClient(app) as c:
        yield c


def make_merge(client, base, left, right, schema=None):
    payload = {"base": base, "left": left, "right": right}
    if schema is not None:
        payload["result_schema"] = schema
    resp = client.post("/merges", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


def simple_merge(client):
    """One conflict at /a."""
    return make_merge(client, base={"a": 1}, left={"a": 2}, right={"a": 3})


def get_audit(client, merge_id):
    resp = client.get(f"/merges/{merge_id}/audit")
    assert resp.status_code == 200, resp.text
    return resp.json()


def expected_hash(merge_id, event):
    """Recompute an event's chain hash from the API-visible fields."""
    payload = {
        "merge_id": merge_id,
        "seq": event["seq"],
        "method": event["method"],
        "revision_before": event["revision_before"],
        "revision_after": event["revision_after"],
        "pointers": event["pointers"],
        "detail": event["detail"],
        "prev_hash": event["prev_hash"],
        "created_at": event["created_at"],
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()


# ---------------------------------------------------------------------------
# Recording events per adjudication method
# ---------------------------------------------------------------------------


def test_resolution_batches_append_chained_events(client):
    merge = make_merge(
        client,
        base={"a": 1, "b": 1},
        left={"a": 2, "b": 2},
        right={"a": 3, "b": 3},
    )
    mid = merge["merge_id"]

    resp = client.post(
        f"/merges/{mid}/resolutions",
        json={"revision": 1, "resolutions": [{"pointer": "/a", "choice": "left"}]},
    )
    assert resp.status_code == 200
    resp = client.post(
        f"/merges/{mid}/resolutions",
        json={"revision": 2, "resolutions": [{"pointer": "/b", "value": 9}]},
    )
    assert resp.status_code == 200

    audit = get_audit(client, mid)
    assert audit["merge_id"] == mid
    assert audit["event_count"] == 2
    assert audit["chain_valid"] is True
    assert audit["first_bad_seq"] is None

    first, second = audit["events"]
    assert first["seq"] == 1
    assert first["method"] == "resolutions"
    assert first["revision_before"] == 1
    assert first["revision_after"] == 2
    assert first["pointers"] == ["/a"]
    assert first["detail"] == {}
    assert first["prev_hash"] is None
    assert len(first["hash"]) == 64

    assert second["seq"] == 2
    assert second["revision_before"] == 2
    assert second["revision_after"] == 3
    assert second["pointers"] == ["/b"]
    # The chain links each event to its predecessor's hash.
    assert second["prev_hash"] == first["hash"]

    # Every stored hash recomputes from the canonical-JSON payload.
    for event in audit["events"]:
        assert event["hash"] == expected_hash(mid, event)


def test_rule_resolutions_event_records_matched_pointers_only(client):
    merge = make_merge(
        client,
        base={"a": 1, "b": 1, "c": 1},
        left={"a": 2, "b": 2, "c": 2},
        right={"a": 3, "b": 3, "c": 3},
    )
    mid = merge["merge_id"]

    resp = client.post(
        f"/merges/{mid}/rule-resolutions",
        json={"revision": 1, "rules": [{"right": "/a"}, {"left": "/c"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["applied"] is True

    audit = get_audit(client, mid)
    assert audit["event_count"] == 1
    event = audit["events"][0]
    assert event["method"] == "rule-resolutions"
    assert event["revision_before"] == 1
    assert event["revision_after"] == 2
    # Only the matched conflicts took effect; /b stays open and unrecorded.
    assert event["pointers"] == ["/a", "/c"]
    assert audit["chain_valid"] is True


def test_rule_resolutions_without_match_leaves_no_event(client):
    merge = simple_merge(client)
    mid = merge["merge_id"]
    resp = client.post(
        f"/merges/{mid}/rule-resolutions",
        json={"revision": 1, "rules": [{"left": "/somewhere/else"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["applied"] is False
    assert resp.json()["revision"] == 1

    audit = get_audit(client, mid)
    assert audit["event_count"] == 0
    assert audit["events"] == []
    assert audit["chain_valid"] is True
    assert audit["first_bad_seq"] is None


def test_reuse_event_and_zero_match_noop(client):
    source = simple_merge(client)
    resp = client.post(
        f"/merges/{source['merge_id']}/resolutions",
        json={"revision": 1, "resolutions": [{"pointer": "/a", "choice": "right"}]},
    )
    assert resp.status_code == 200

    # Zero matches: a successful no-op that must not leave an event.
    other = make_merge(client, base={"x": 1}, left={"x": 2}, right={"x": 3})
    resp = client.post(
        f"/merges/{other['merge_id']}/reuse-resolutions",
        json={"revision": 1, "source_merge_id": source["merge_id"]},
    )
    assert resp.status_code == 200
    assert resp.json()["applied"] is False
    assert get_audit(client, other["merge_id"])["event_count"] == 0

    # A matching reuse appends one event naming the source merge.
    target = simple_merge(client)
    tid = target["merge_id"]
    resp = client.post(
        f"/merges/{tid}/reuse-resolutions",
        json={"revision": 1, "source_merge_id": source["merge_id"]},
    )
    assert resp.status_code == 200
    assert resp.json()["applied"] is True

    audit = get_audit(client, tid)
    assert audit["event_count"] == 1
    event = audit["events"][0]
    assert event["method"] == "reuse-resolutions"
    assert event["revision_before"] == 1
    assert event["revision_after"] == 2
    assert event["pointers"] == ["/a"]
    assert event["detail"] == {"source_merge_id": source["merge_id"]}
    assert event["hash"] == expected_hash(tid, event)
    assert audit["chain_valid"] is True


def test_draft_commit_event(client):
    merge = make_merge(
        client,
        base={"a": 1, "b": 1},
        left={"a": 2, "b": 2},
        right={"a": 3, "b": 3},
    )
    mid = merge["merge_id"]
    draft = client.post(f"/merges/{mid}/drafts").json()
    did = draft["draft_id"]
    resp = client.post(
        f"/merges/{mid}/drafts/{did}/resolutions",
        json={"resolutions": [{"pointer": "/b", "value": "B"}]},
    )
    assert resp.status_code == 200

    # Draft edits do not touch the merge: no audit event yet.
    assert get_audit(client, mid)["event_count"] == 0

    resp = client.post(f"/merges/{mid}/drafts/{did}/commit")
    assert resp.status_code == 200, resp.text

    audit = get_audit(client, mid)
    assert audit["event_count"] == 1
    event = audit["events"][0]
    assert event["method"] == "draft-commit"
    assert event["revision_before"] == 1
    assert event["revision_after"] == 2
    assert event["pointers"] == ["/b"]
    assert event["detail"] == {"draft_id": did}
    assert audit["chain_valid"] is True


def test_mixed_methods_form_one_chain(client):
    # A source merge for the reuse step: identical /c conflict.
    source = make_merge(client, base={"c": 1}, left={"c": 2}, right={"c": 3})
    resp = client.post(
        f"/merges/{source['merge_id']}/resolutions",
        json={"revision": 1, "resolutions": [{"pointer": "/c", "choice": "right"}]},
    )
    assert resp.status_code == 200

    merge = make_merge(
        client,
        base={"a": 1, "b": 1, "c": 1, "d": 1},
        left={"a": 2, "b": 2, "c": 2, "d": 2},
        right={"a": 3, "b": 3, "c": 3, "d": 3},
    )
    mid = merge["merge_id"]

    resp = client.post(
        f"/merges/{mid}/resolutions",
        json={"revision": 1, "resolutions": [{"pointer": "/a", "choice": "left"}]},
    )
    assert resp.status_code == 200
    resp = client.post(
        f"/merges/{mid}/rule-resolutions",
        json={"revision": 2, "rules": [{"right": "/b"}]},
    )
    assert resp.status_code == 200
    resp = client.post(
        f"/merges/{mid}/reuse-resolutions",
        json={"revision": 3, "source_merge_id": source["merge_id"]},
    )
    assert resp.status_code == 200
    assert resp.json()["applied"] is True
    draft = client.post(f"/merges/{mid}/drafts").json()
    resp = client.post(
        f"/merges/{mid}/drafts/{draft['draft_id']}/resolutions",
        json={"resolutions": [{"pointer": "/d", "value": 4}]},
    )
    assert resp.status_code == 200
    resp = client.post(f"/merges/{mid}/drafts/{draft['draft_id']}/commit")
    assert resp.status_code == 200
    assert resp.json()["status"] == "resolved"

    audit = get_audit(client, mid)
    assert audit["event_count"] == 4
    assert [e["method"] for e in audit["events"]] == [
        "resolutions",
        "rule-resolutions",
        "reuse-resolutions",
        "draft-commit",
    ]
    assert [e["seq"] for e in audit["events"]] == [1, 2, 3, 4]
    assert [(e["revision_before"], e["revision_after"]) for e in audit["events"]] == [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    ]
    assert [e["pointers"] for e in audit["events"]] == [
        ["/a"],
        ["/b"],
        ["/c"],
        ["/d"],
    ]
    assert audit["chain_valid"] is True
    assert audit["first_bad_seq"] is None
    for prev, event in zip([None] + audit["events"], audit["events"]):
        expected_prev = None if prev is None else prev["hash"]
        assert event["prev_hash"] == expected_prev
        assert event["hash"] == expected_hash(mid, event)


# ---------------------------------------------------------------------------
# Requests that must not leave events
# ---------------------------------------------------------------------------


def test_failed_requests_leave_no_events(client):
    merge = simple_merge(client)
    mid = merge["merge_id"]

    # Unknown conflict pointer -> 400.
    resp = client.post(
        f"/merges/{mid}/resolutions",
        json={"revision": 1, "resolutions": [{"pointer": "/nope", "value": 1}]},
    )
    assert resp.status_code == 400
    # Stale revision -> 409.
    resp = client.post(
        f"/merges/{mid}/resolutions",
        json={"revision": 999, "resolutions": [{"pointer": "/a", "choice": "left"}]},
    )
    assert resp.status_code == 409
    # Previews never write anything.
    resp = client.post(
        f"/merges/{mid}/resolution-preview",
        json={"revision": 1, "resolutions": [{"pointer": "/a", "choice": "left"}]},
    )
    assert resp.status_code == 200
    # Invalid reuse source -> 400.
    resp = client.post(
        f"/merges/{mid}/reuse-resolutions",
        json={"revision": 1, "source_merge_id": "no-such-source"},
    )
    assert resp.status_code == 400
    # Empty draft commit -> 400.
    draft = client.post(f"/merges/{mid}/drafts").json()
    resp = client.post(f"/merges/{mid}/drafts/{draft['draft_id']}/commit")
    assert resp.status_code == 400

    assert get_audit(client, mid)["event_count"] == 0

    # Resolve for real; further requests against the resolved merge -> 409.
    resp = client.post(
        f"/merges/{mid}/resolutions",
        json={"revision": 1, "resolutions": [{"pointer": "/a", "choice": "left"}]},
    )
    assert resp.status_code == 200
    resp = client.post(
        f"/merges/{mid}/resolutions",
        json={"revision": 2, "resolutions": [{"pointer": "/a", "choice": "left"}]},
    )
    assert resp.status_code == 409

    # Exactly one event: the single successful batch.
    audit = get_audit(client, mid)
    assert audit["event_count"] == 1
    assert audit["events"][0]["method"] == "resolutions"
    assert audit["chain_valid"] is True


def test_schema_gate_failure_leaves_no_event(client):
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer", "minimum": 5}},
        "required": ["a"],
    }
    merge = make_merge(
        client, base={"a": 1}, left={"a": 2}, right={"a": 3}, schema=schema
    )
    mid = merge["merge_id"]

    # The final batch violates result_schema -> 422, everything rolls back.
    resp = client.post(
        f"/merges/{mid}/resolutions",
        json={"revision": 1, "resolutions": [{"pointer": "/a", "value": 2}]},
    )
    assert resp.status_code == 422
    assert get_audit(client, mid)["event_count"] == 0

    # A passing batch then commits normally and leaves exactly one event.
    resp = client.post(
        f"/merges/{mid}/resolutions",
        json={"revision": 1, "resolutions": [{"pointer": "/a", "value": 6}]},
    )
    assert resp.status_code == 200
    audit = get_audit(client, mid)
    assert audit["event_count"] == 1
    assert audit["events"][0]["revision_after"] == 2
    assert audit["chain_valid"] is True


def test_immediately_resolved_merge_has_empty_audit(client):
    # No conflicts: creation alone resolves the merge; creation is not an
    # adjudication and leaves no event.
    merge = make_merge(client, base={"a": 1}, left={"a": 2}, right={"a": 2})
    assert merge["status"] == "resolved"
    audit = get_audit(client, merge["merge_id"])
    assert audit["event_count"] == 0
    assert audit["chain_valid"] is True
    assert audit["first_bad_seq"] is None


def test_audit_unknown_merge_returns_404(client):
    resp = client.get("/merges/does-not-exist/audit")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "merge not found"


# ---------------------------------------------------------------------------
# Chain verification
# ---------------------------------------------------------------------------


def resolved_merge_with_events(client):
    """A merge with two chained audit events; returns its id."""
    merge = make_merge(
        client,
        base={"a": 1, "b": 1},
        left={"a": 2, "b": 2},
        right={"a": 3, "b": 3},
    )
    mid = merge["merge_id"]
    for revision, pointer, value in ((1, "/a", 10), (2, "/b", 20)):
        resp = client.post(
            f"/merges/{mid}/resolutions",
            json={
                "revision": revision,
                "resolutions": [{"pointer": pointer, "value": value}],
            },
        )
        assert resp.status_code == 200
    assert get_audit(client, mid)["event_count"] == 2
    return mid


def test_chain_detects_tampered_event_payload(client):
    mid = resolved_merge_with_events(client)

    # Tamper with the first event's recorded pointers; its stored hash no
    # longer matches the recomputed one.
    db.execute(
        "UPDATE merge_audit_events SET pointers = ? WHERE merge_id = ? AND seq = 1",
        (_json.dumps(["/tampered"]), mid),
    )
    audit = get_audit(client, mid)
    assert audit["chain_valid"] is False
    assert audit["first_bad_seq"] == 1


def test_chain_detects_tampered_later_event(client):
    mid = resolved_merge_with_events(client)

    # Corrupt the second event's revision record: the first event still
    # verifies, the second is reported as the first anomaly.
    db.execute(
        "UPDATE merge_audit_events SET revision_after = 99 "
        "WHERE merge_id = ? AND seq = 2",
        (mid,),
    )
    audit = get_audit(client, mid)
    assert audit["chain_valid"] is False
    assert audit["first_bad_seq"] == 2


def test_chain_detects_deleted_first_event(client):
    mid = resolved_merge_with_events(client)

    # Removing the head of the chain breaks both the sequence (starts at 2)
    # and the linkage (prev_hash of a first event must be null).
    db.execute(
        "DELETE FROM merge_audit_events WHERE merge_id = ? AND seq = 1", (mid,)
    )
    audit = get_audit(client, mid)
    assert audit["chain_valid"] is False
    assert audit["first_bad_seq"] == 2


def test_chain_detects_relinked_prev_hash(client):
    mid = resolved_merge_with_events(client)

    # Point the second event at a bogus predecessor hash.
    db.execute(
        "UPDATE merge_audit_events SET prev_hash = ? WHERE merge_id = ? AND seq = 2",
        ("0" * 64, mid),
    )
    audit = get_audit(client, mid)
    assert audit["chain_valid"] is False
    assert audit["first_bad_seq"] == 2


def test_chains_are_independent_per_merge(client):
    first = resolved_merge_with_events(client)
    second = resolved_merge_with_events(client)

    # Same payloads, but the chains are separate: each merge's first event
    # has a null prev_hash and seq restarts at 1.
    audit_first = get_audit(client, first)
    audit_second = get_audit(client, second)
    assert audit_first["chain_valid"] is True
    assert audit_second["chain_valid"] is True
    assert [e["seq"] for e in audit_first["events"]] == [1, 2]
    assert [e["seq"] for e in audit_second["events"]] == [1, 2]
    assert audit_first["events"][0]["prev_hash"] is None
    assert audit_second["events"][0]["prev_hash"] is None
    # The merge_id is part of the hashed payload, so identical resolutions
    # on different merges still produce different event hashes.
    assert audit_first["events"][0]["hash"] != audit_second["events"][0]["hash"]
