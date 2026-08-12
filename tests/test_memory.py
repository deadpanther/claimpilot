"""Tests for the memory store.

Exercises `claimpilot.memory`'s public API directly against a `tmp_path`
SQLite database -- same test-injection convention (`db_path=`) as
`tests/test_store.py`. A few tests reach into the raw `memory`/`cases`
tables via `claimpilot.db.get_connection` to seed rows `memory.py`'s own
public API has no way to construct (a malformed correction row; a `cases`
row with a specific historical `created_at`, since `claimpilot.store`
always stamps `created_at` with the real wall clock, not an injectable
`now`) -- same pattern `tests/test_store.py` uses to verify the `actions`
table's UNIQUE constraint directly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from claimpilot import store
from claimpilot.config import settings
from claimpilot.db import ensure_schema, get_connection
from claimpilot.memory import (
    NO_MERCHANT_ID_MEMORY_CONTEXT,
    MemoryContext,
    PolicyNote,
    delete_note,
    global_policies,
    list_policy_notes,
    merchant_context,
    record_correction,
    record_policy_note,
)
from claimpilot.models import Case

NOW = datetime(2026, 3, 25, tzinfo=timezone.utc)


def _db(tmp_path: Path) -> Path:
    return tmp_path / "t.db"


def _case(case_id: str = "CASE-1", user_id: str | None = "M1") -> Case:
    return Case(case_id=case_id, status="New", user_id=user_id)


def _set_created_at(db_path: Path, case_id: str, created_at_iso: str) -> None:
    """Directly overwrite a `cases` row's `created_at` -- `store.create_case`
    always stamps the real wall clock (see `store._now()`), so this is the
    only way to seed a specific historical timestamp for deterministic
    90-day-window boundary testing.
    """
    conn = get_connection(db_path)
    try:
        conn.execute("UPDATE cases SET created_at = ? WHERE case_id = ?", (created_at_iso, case_id))
        conn.commit()
    finally:
        conn.close()


def _insert_raw_memory_row(
    db_path: Path,
    *,
    scope: str,
    merchant_id: str | None,
    kind: str,
    content: str,
    source_case_id: str | None = None,
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> None:
    conn = get_connection(db_path)
    try:
        ensure_schema(conn)
        conn.execute(
            "INSERT INTO memory (scope, merchant_id, kind, content, source_case_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (scope, merchant_id, kind, content, source_case_id, created_at),
        )
        conn.commit()
    finally:
        conn.close()


# --- record_correction + merchant_context round-trip -------------------------


def test_record_correction_then_merchant_context_shows_recent_note(tmp_path: Path):
    db_path = _db(tmp_path)
    case = _case(user_id="M1")

    record_correction(
        case,
        original_draft="Dear customer, original.",
        final_draft="Dear customer, edited to be warmer.",
        feedback="Please be warmer in tone.",
        db_path=db_path,
    )

    ctx = merchant_context("M1", db_path=db_path)
    assert ctx.recent_notes == ["Rep correction with feedback: Please be warmer in tone."]
    # No `cases` row was created for M1 in this test -- frequency stays 0.
    assert ctx.claim_frequency_90d == 0


def test_record_correction_without_feedback_renders_generic_note(tmp_path: Path):
    db_path = _db(tmp_path)
    case = _case(user_id="M1")

    record_correction(case, "original text", "final text", db_path=db_path)

    ctx = merchant_context("M1", db_path=db_path)
    assert ctx.recent_notes == [
        "Rep edited the drafted email before sending (no separate feedback text given)."
    ]


def test_record_correction_stores_structured_json_content(tmp_path: Path):
    """`record_correction`'s `content` column round-trips as a JSON object,
    not opaque prose -- the feedback distiller needs to parse this
    back out. Verified directly against the raw column, mirroring
    `test_store.py`'s "raw column really is a JSON string" style assertions.
    """
    import json

    db_path = _db(tmp_path)
    case = _case(case_id="CASE-42", user_id="M1")
    record_correction(case, "orig", "final", feedback="fix the tone", db_path=db_path)

    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM memory WHERE kind = 'correction'").fetchone()
    finally:
        conn.close()

    assert row["scope"] == "merchant"
    assert row["merchant_id"] == "M1"
    assert row["source_case_id"] == "CASE-42"
    payload = json.loads(row["content"])
    assert payload == {"original_draft": "orig", "final_draft": "final", "feedback": "fix the tone"}


def test_record_correction_raises_when_case_has_no_user_id(tmp_path: Path):
    db_path = _db(tmp_path)
    case = _case(user_id=None)

    with pytest.raises(ValueError):
        record_correction(case, "orig", "final", db_path=db_path)


def test_merchant_context_caps_recent_notes_at_max_and_orders_most_recent_first(tmp_path: Path):
    db_path = _db(tmp_path)
    case = _case(user_id="M1")

    total = settings.max_recent_notes + 3
    for i in range(total):
        record_correction(case, f"orig{i}", f"final{i}", feedback=f"fb{i}", db_path=db_path)

    ctx = merchant_context("M1", db_path=db_path)
    assert len(ctx.recent_notes) == settings.max_recent_notes
    # Most recent (highest i, last written) first.
    assert ctx.recent_notes[0] == f"Rep correction with feedback: fb{total - 1}"


def test_merchant_context_note_kind_content_is_plain_text(tmp_path: Path):
    db_path = _db(tmp_path)
    _insert_raw_memory_row(
        db_path, scope="merchant", merchant_id="M1", kind="note", content="Prefers phone contact over email."
    )

    ctx = merchant_context("M1", db_path=db_path)
    assert ctx.recent_notes == ["Prefers phone contact over email."]


def test_merchant_context_malformed_correction_json_falls_back_to_raw_text(tmp_path: Path):
    db_path = _db(tmp_path)
    _insert_raw_memory_row(db_path, scope="merchant", merchant_id="M1", kind="correction", content="not valid json")

    ctx = merchant_context("M1", db_path=db_path)
    assert ctx.recent_notes == ["not valid json"]


def test_merchant_context_empty_for_unknown_merchant(tmp_path: Path):
    db_path = _db(tmp_path)
    assert merchant_context("NO-SUCH-MERCHANT", db_path=db_path) == MemoryContext()


# --- claims_last_90_days boundary correctness ---------------------------------


def test_claims_last_90_days_counts_within_bounded_window_only(tmp_path: Path):
    """Both a lower AND upper bound matter (module docstring point 6): a
    case created exactly 90 days before `now` counts; one created one
    second earlier does not. A case created after `now` (the pipeline's own
    just-created row moments after case creation, in production; here
    simulated directly) must not count either -- confirms the window is
    bounded on both sides, not just "created_at >= cutoff".
    """
    db_path = _db(tmp_path)
    window_start = NOW - timedelta(days=90)

    store.create_case("CASE-IN-LOWER", merchant_id="M1", db_path=db_path)
    _set_created_at(db_path, "CASE-IN-LOWER", window_start.isoformat())

    store.create_case("CASE-OUT-LOWER", merchant_id="M1", db_path=db_path)
    _set_created_at(db_path, "CASE-OUT-LOWER", (window_start - timedelta(seconds=1)).isoformat())

    store.create_case("CASE-IN-UPPER", merchant_id="M1", db_path=db_path)
    _set_created_at(db_path, "CASE-IN-UPPER", NOW.isoformat())

    store.create_case("CASE-OUT-UPPER", merchant_id="M1", db_path=db_path)
    _set_created_at(db_path, "CASE-OUT-UPPER", (NOW + timedelta(seconds=1)).isoformat())

    # Different merchant entirely -- must never be counted for M1.
    store.create_case("CASE-OTHER-MERCHANT", merchant_id="M2", db_path=db_path)
    _set_created_at(db_path, "CASE-OTHER-MERCHANT", NOW.isoformat())

    ctx = merchant_context("M1", now=NOW, db_path=db_path)
    assert ctx.claim_frequency_90d == 2  # CASE-IN-LOWER + CASE-IN-UPPER only


def test_claim_frequency_window_days_override_actually_changes_the_window(monkeypatch, tmp_path: Path):
    """`settings.claim_frequency_window_days` (config.py) drives the actual
    window math, not just a cosmetic default -- a case 45 days back is
    counted at the default (90) but excluded once the window is narrowed to
    30, confirming the override is read live, not baked in at import time.
    """
    from claimpilot.config import settings

    db_path = _db(tmp_path)
    store.create_case("CASE-45-DAYS-BACK", merchant_id="M1", db_path=db_path)
    _set_created_at(db_path, "CASE-45-DAYS-BACK", (NOW - timedelta(days=45)).isoformat())

    assert merchant_context("M1", now=NOW, db_path=db_path).claim_frequency_90d == 1

    monkeypatch.setattr(settings, "claim_frequency_window_days", 30)
    assert merchant_context("M1", now=NOW, db_path=db_path).claim_frequency_90d == 0

    # Rendered prompt text follows the override too, not just the count.
    ctx = merchant_context("M1", now=NOW, db_path=db_path)
    assert "last 30 days" in ctx.to_prompt_text()


def test_merchant_context_exclude_case_id_excludes_itself_from_frequency(tmp_path: Path):
    """`exclude_case_id` is how `pipeline.process_case` keeps a case's own
    just-created `cases` row from inflating its own claim-frequency count
    (module docstring point 6) -- verified directly here.
    """
    db_path = _db(tmp_path)
    store.create_case("CASE-A", merchant_id="M1", db_path=db_path)
    _set_created_at(db_path, "CASE-A", NOW.isoformat())
    store.create_case("CASE-B", merchant_id="M1", db_path=db_path)
    _set_created_at(db_path, "CASE-B", NOW.isoformat())

    without_exclusion = merchant_context("M1", now=NOW, db_path=db_path)
    assert without_exclusion.claim_frequency_90d == 2

    with_exclusion = merchant_context("M1", now=NOW, exclude_case_id="CASE-B", db_path=db_path)
    assert with_exclusion.claim_frequency_90d == 1


def test_merchant_context_accepts_naive_now_without_raising(tmp_path: Path):
    db_path = _db(tmp_path)
    naive_now = datetime(2026, 3, 25)  # no tzinfo

    ctx = merchant_context("M1", now=naive_now, db_path=db_path)

    assert ctx.claim_frequency_90d == 0


# --- global vs merchant-scoped policy notes never leak ------------------------


def test_global_and_merchant_policy_notes_do_not_leak_into_each_other(tmp_path: Path):
    db_path = _db(tmp_path)
    record_policy_note("Always confirm SKU before approving.", scope="global", db_path=db_path)
    record_policy_note("This merchant prefers formal tone.", scope="merchant", merchant_id="M1", db_path=db_path)
    record_policy_note("A different merchant's note.", scope="merchant", merchant_id="M2", db_path=db_path)

    assert global_policies(db_path=db_path) == ["Always confirm SKU before approving."]

    ctx_m1 = merchant_context("M1", db_path=db_path)
    assert ctx_m1.policy_notes == ["This merchant prefers formal tone."]
    # Global notes surface as their own field, informational-only -- never
    # merged into `policy_notes` (see module docstring point 2 for why:
    # `policy_notes` feeds risk-tiering flags, and a global note is not
    # evidence this merchant specifically is risky).
    assert ctx_m1.global_policy_notes == ["Always confirm SKU before approving."]

    ctx_m2 = merchant_context("M2", db_path=db_path)
    assert ctx_m2.policy_notes == ["A different merchant's note."]
    assert "This merchant prefers formal tone." not in ctx_m2.policy_notes


def test_record_policy_note_requires_merchant_id_for_merchant_scope(tmp_path: Path):
    db_path = _db(tmp_path)
    with pytest.raises(ValueError):
        record_policy_note("some note", scope="merchant", db_path=db_path)


def test_record_policy_note_rejects_merchant_id_for_global_scope(tmp_path: Path):
    db_path = _db(tmp_path)
    with pytest.raises(ValueError):
        record_policy_note("some note", scope="global", merchant_id="M1", db_path=db_path)


# --- list_policy_notes / delete_note (review-panel groundwork) --------------


def test_list_policy_notes_returns_both_scopes_with_provenance(tmp_path: Path):
    db_path = _db(tmp_path)
    record_policy_note("Global note", scope="global", db_path=db_path)
    record_policy_note(
        "Merchant note", scope="merchant", merchant_id="M1", source_case_id="CASE-1", db_path=db_path
    )

    notes = list_policy_notes(db_path=db_path)

    assert len(notes) == 2
    assert all(isinstance(n, PolicyNote) for n in notes)
    assert {n.scope for n in notes} == {"global", "merchant"}
    merchant_note = next(n for n in notes if n.scope == "merchant")
    assert merchant_note.merchant_id == "M1"
    assert merchant_note.source_case_id == "CASE-1"
    assert merchant_note.content == "Merchant note"
    assert merchant_note.created_at
    global_note = next(n for n in notes if n.scope == "global")
    assert global_note.merchant_id is None
    assert global_note.source_case_id is None


def test_delete_note_removes_only_the_targeted_row(tmp_path: Path):
    db_path = _db(tmp_path)
    record_policy_note("Global note", scope="global", db_path=db_path)
    record_policy_note("Merchant note", scope="merchant", merchant_id="M1", db_path=db_path)
    notes = list_policy_notes(db_path=db_path)
    merchant_note_id = next(n.id for n in notes if n.scope == "merchant")

    delete_note(merchant_note_id, db_path=db_path)

    remaining = list_policy_notes(db_path=db_path)
    assert len(remaining) == 1
    assert remaining[0].scope == "global"


def test_delete_note_missing_id_is_a_noop(tmp_path: Path):
    db_path = _db(tmp_path)
    record_policy_note("Global note", scope="global", db_path=db_path)

    delete_note(999999, db_path=db_path)  # does not raise

    assert len(list_policy_notes(db_path=db_path)) == 1


# --- record_correction rows never surface via list_policy_notes --------------


def test_corrections_are_not_policy_notes(tmp_path: Path):
    """`kind="correction"` rows must never appear in `list_policy_notes`/
    `global_policies` -- only `kind="policy"` rows are policy notes.
    """
    db_path = _db(tmp_path)
    record_correction(_case(user_id="M1"), "orig", "final", feedback="fb", db_path=db_path)

    assert list_policy_notes(db_path=db_path) == []
    assert global_policies(db_path=db_path) == []


# --- MemoryContext.to_prompt_text() -------------------------------------------


def test_to_prompt_text_empty_context_is_reasonable():
    text = MemoryContext().to_prompt_text()

    assert "Claim frequency (last 90 days): 0" in text
    assert "(none)" in text


def test_to_prompt_text_includes_all_populated_sections():
    ctx = MemoryContext(
        recent_notes=["Rep correction with feedback: be warmer"],
        claim_frequency_90d=2,
        policy_notes=["Merchant prefers formal tone."],
        global_policy_notes=["Always confirm SKU before approving."],
    )

    text = ctx.to_prompt_text()

    assert "Claim frequency (last 90 days): 2" in text
    assert "Merchant prefers formal tone." in text
    assert "Always confirm SKU before approving." in text
    assert "Rep correction with feedback: be warmer" in text


def test_no_merchant_id_memory_context_constant_is_distinct_from_zero_frequency_text():
    """The shared fallback text used when a case has no `user_id` must not
    look like "confirmed zero prior claims" -- it must be a distinguishable
    "unknown, couldn't compute" message.
    """
    assert "Claim frequency" not in NO_MERCHANT_ID_MEMORY_CONTEXT
    assert "merchant" in NO_MERCHANT_ID_MEMORY_CONTEXT.lower()
