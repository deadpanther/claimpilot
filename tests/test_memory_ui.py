"""Tests for the memory review panel.

Two things are new in this task:

- `memory.record_policy_note` now enforces a per-`(scope, merchant_id)`
  write-time cap of `settings.policy_note_cap_per_partition` (10) rows,
  evicting the oldest row(s) in that partition once it's exceeded --
  `test_cap_*` tests below.
- `claimpilot.web.app` gains a `POST /memory/notes/{id}/delete` route and
  `queue.html` gains a `<details>` section listing every policy note with
  provenance and a delete button -- `test_queue_page_*` /
  `test_delete_route_*` tests below.

Uses the same `create_app(client=..., transport=..., db_path=...)` /
`TestClient` convention as `tests/test_web.py`, but doesn't need a real
processed case for most of these -- policy notes are written directly via
`memory.record_policy_note`, independent of the case pipeline.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from claimpilot.clients.fixtures import FixtureClient
from claimpilot.config import settings
from claimpilot.memory import list_policy_notes, record_policy_note
from claimpilot.web.app import create_app
from tests.test_llm import FakeTransport


def _app(db_path: Path):
    return create_app(client=FixtureClient(include_synthetic=True), transport=FakeTransport([]), db_path=db_path)


# --- write-time cap + eviction ---------------------------------------------


def test_eleventh_note_in_merchant_partition_evicts_the_oldest(tmp_path: Path):
    db_path = tmp_path / "t.db"
    for i in range(11):
        record_policy_note(
            f"note #{i}", scope="merchant", merchant_id="MERCH-A", db_path=db_path
        )

    notes = [n for n in list_policy_notes(db_path=db_path) if n.merchant_id == "MERCH-A"]
    assert len(notes) == settings.policy_note_cap_per_partition == 10

    contents = {n.content for n in notes}
    assert "note #0" not in contents  # oldest, evicted
    assert "note #1" in contents  # second-oldest, survives
    assert "note #10" in contents  # newest, survives


def test_cap_is_partitioned_by_merchant_not_global(tmp_path: Path):
    """11 notes for merchant A must not evict merchant B's or global's
    notes -- each `(scope, merchant_id)` pair is its own independent
    partition (memory.py module docstring point 8).
    """
    db_path = tmp_path / "t.db"
    record_policy_note("merchant B's only note", scope="merchant", merchant_id="MERCH-B", db_path=db_path)
    record_policy_note("the only global note", scope="global", db_path=db_path)

    for i in range(11):
        record_policy_note(f"A note #{i}", scope="merchant", merchant_id="MERCH-A", db_path=db_path)

    all_notes = list_policy_notes(db_path=db_path)
    a_notes = [n for n in all_notes if n.merchant_id == "MERCH-A"]
    b_notes = [n for n in all_notes if n.merchant_id == "MERCH-B"]
    global_notes = [n for n in all_notes if n.scope == "global"]

    assert len(a_notes) == 10
    assert len(b_notes) == 1
    assert b_notes[0].content == "merchant B's only note"
    assert len(global_notes) == 1
    assert global_notes[0].content == "the only global note"


def test_global_partition_has_its_own_cap(tmp_path: Path):
    db_path = tmp_path / "t.db"
    for i in range(11):
        record_policy_note(f"global note #{i}", scope="global", db_path=db_path)

    global_notes = [n for n in list_policy_notes(db_path=db_path) if n.scope == "global"]
    assert len(global_notes) == 10
    contents = {n.content for n in global_notes}
    assert "global note #0" not in contents
    assert "global note #10" in contents


# --- queue page rendering ---------------------------------------------------


def test_queue_page_lists_policy_notes_with_provenance(tmp_path: Path):
    db_path = tmp_path / "t.db"
    record_policy_note(
        "Always confirm SKU before refunding",
        scope="merchant",
        merchant_id="MERCH-XYZ",
        source_case_id="CASE-777",
        db_path=db_path,
    )
    record_policy_note("Mention warranty terms for mug SKUs", scope="global", db_path=db_path)

    with TestClient(_app(db_path)) as tc:
        resp = tc.get("/cases")

    notes = list_policy_notes(db_path=db_path)
    merchant_note = next(n for n in notes if n.scope == "merchant")
    global_note = next(n for n in notes if n.scope == "global")

    assert resp.status_code == 200
    assert "<details>" in resp.text  # collapsible element the task specifies
    assert "Always confirm SKU before refunding" in resp.text
    assert "MERCH-XYZ" in resp.text
    assert "CASE-777" in resp.text
    assert "Mention warranty terms for mug SKUs" in resp.text
    # Template's form action must actually point at the delete route --
    # pins the template<->route seam so a template typo can't silently
    # 404 the delete button while text-content assertions still pass.
    assert f'action="/memory/notes/{merchant_note.id}/delete"' in resp.text
    assert f'action="/memory/notes/{global_note.id}/delete"' in resp.text


def test_queue_page_renders_cleanly_with_zero_policy_notes(tmp_path: Path):
    db_path = tmp_path / "t.db"

    with TestClient(_app(db_path)) as tc:
        resp = tc.get("/cases")

    assert resp.status_code == 200
    assert "No policy notes recorded yet." in resp.text


# --- delete route ------------------------------------------------------------


def test_delete_route_removes_the_note_and_redirects_to_queue(tmp_path: Path):
    db_path = tmp_path / "t.db"
    record_policy_note("delete me", scope="global", db_path=db_path)
    note = list_policy_notes(db_path=db_path)[0]

    with TestClient(_app(db_path)) as tc:
        resp = tc.post(f"/memory/notes/{note.id}/delete", follow_redirects=False)

    assert resp.status_code == 303
    assert resp.headers["location"] == "/cases"
    assert list_policy_notes(db_path=db_path) == []


def test_delete_route_is_a_no_op_on_nonexistent_id(tmp_path: Path):
    db_path = tmp_path / "t.db"

    with TestClient(_app(db_path)) as tc:
        resp = tc.post("/memory/notes/999999/delete", follow_redirects=False)

    assert resp.status_code == 303
    assert resp.headers["location"] == "/cases"
