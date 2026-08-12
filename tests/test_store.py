"""Tests for the SQLite store + audit trail."""

from __future__ import annotations

import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from claimpilot.calc import CalcResult
from claimpilot.db import ensure_llm_calls_table, get_connection
from claimpilot.gates.eligibility import EligibilityResult
from claimpilot.gates.evidence import Gap
from claimpilot.gates.invoice_audit import (
    Discrepancy,
    ExtractedInvoice,
    ExtractedLine,
    InvoiceAudit,
    Severity,
)
from claimpilot.gates.validation import Judgment, ValidationResult
from claimpilot.models import CaseState, EvidenceItem, Recommendation, RecommendationLineItem
from claimpilot.store import (
    CaseAlreadyExistsError,
    CaseNotFoundError,
    DuplicateActionError,
    GateResults,
    IllegalTransitionError,
    create_case,
    get_audit_log,
    get_case,
    get_recommendation,
    get_status,
    list_cases_by_status,
    load_gate_results,
    log_event,
    record_action,
    save_gate_results,
    save_recommendation,
    stats,
    transition,
)

CASE_ID = "CASE-1"


def _db(tmp_path: Path) -> Path:
    return tmp_path / "t.db"


# --- case creation / reads ----------------------------------------------------


def test_get_connection_enables_wal_mode_and_busy_timeout(tmp_path: Path):
    """Found in a full-codebase concurrency audit: this app opens a fresh
    connection per operation, so SQLite's default (non-WAL) journal mode
    would let one rep's write briefly block another rep's read of the same
    file. `get_connection()` now sets WAL mode + a busy_timeout on every
    connection -- this test locks that in as an explicit, named behavior
    rather than leaving it as an implicit side effect other tests happen to
    exercise.
    """
    db_path = tmp_path / "t.db"
    conn = get_connection(db_path)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        conn.close()


def test_create_case_sets_initial_intake_status(tmp_path: Path):
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)

    assert get_status(CASE_ID, db_path=db_path) == CaseState.INTAKE
    row = get_case(CASE_ID, db_path=db_path)
    assert row["case_id"] == CASE_ID
    assert row["recommendation_json"] is None
    assert row["created_at"]
    assert row["updated_at"]


def test_create_case_twice_raises(tmp_path: Path):
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)
    with pytest.raises(CaseAlreadyExistsError):
        create_case(CASE_ID, db_path=db_path)


def test_get_status_missing_case_raises(tmp_path: Path):
    db_path = _db(tmp_path)
    with pytest.raises(CaseNotFoundError):
        get_status("no-such-case", db_path=db_path)


# --- valid transition path + audit rows ---------------------------------------


def test_valid_transition_path_writes_audit_row_per_transition(tmp_path: Path):
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)

    path = [
        (CaseState.ELIGIBILITY, "system", "gate:eligibility_passed"),
        (CaseState.EVIDENCE, "system", "gate:eligibility_ok"),
        (CaseState.VALIDATION, "system", "gate:evidence_ok"),
        (CaseState.CALC, "system", "gate:validation_ok"),
        (CaseState.PENDING_REVIEW, "system", "calc_complete"),
        (CaseState.APPROVED, "rep", "rep_approved"),
        (CaseState.SENT, "system", "email_sent"),
        (CaseState.CLOSED, "system", "case_closed"),
    ]

    for to_state, actor, event in path:
        transition(CASE_ID, to_state, actor=actor, event=event, payload={"note": event}, db_path=db_path)

    assert get_status(CASE_ID, db_path=db_path) == CaseState.CLOSED

    rows = get_audit_log(CASE_ID, db_path=db_path)
    assert len(rows) == len(path)
    for row, (to_state, actor, event) in zip(rows, path):
        assert row["case_id"] == CASE_ID
        assert row["actor"] == actor
        assert row["event"] == event
        assert json.loads(row["payload_json"]) == {"note": event}
        assert row["ts"]


def test_transition_updates_case_updated_at(tmp_path: Path):
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)
    before = get_case(CASE_ID, db_path=db_path)["updated_at"]

    transition(CASE_ID, CaseState.ELIGIBILITY, actor="system", event="advance", db_path=db_path)

    after = get_case(CASE_ID, db_path=db_path)
    assert after["status"] == CaseState.ELIGIBILITY.value
    assert after["updated_at"] >= before


# --- double-approve raises (explicit plan requirement) ------------------------


def test_double_approve_raises(tmp_path: Path):
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)
    for to_state in (
        CaseState.ELIGIBILITY,
        CaseState.EVIDENCE,
        CaseState.VALIDATION,
        CaseState.CALC,
        CaseState.PENDING_REVIEW,
    ):
        transition(CASE_ID, to_state, actor="system", event="advance", db_path=db_path)

    # First approve: legal (PENDING_REVIEW -> APPROVED).
    transition(CASE_ID, CaseState.APPROVED, actor="rep", event="rep_approved", db_path=db_path)
    assert get_status(CASE_ID, db_path=db_path) == CaseState.APPROVED

    # Second approve: APPROVED is not a legal source state for -> APPROVED.
    with pytest.raises(IllegalTransitionError):
        transition(CASE_ID, CaseState.APPROVED, actor="rep", event="rep_approved_again", db_path=db_path)

    # Status unchanged, and no audit row for the rejected second attempt.
    assert get_status(CASE_ID, db_path=db_path) == CaseState.APPROVED
    rows = get_audit_log(CASE_ID, db_path=db_path)
    events = [r["event"] for r in rows]
    assert events.count("rep_approved_again") == 0


# --- illegal transition raises (explicit plan requirement) --------------------


def test_illegal_transition_raises_and_leaves_no_trace(tmp_path: Path):
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)

    with pytest.raises(IllegalTransitionError):
        transition(CASE_ID, CaseState.APPROVED, actor="system", event="skip_ahead", db_path=db_path)

    # Status unchanged (still INTAKE).
    assert get_status(CASE_ID, db_path=db_path) == CaseState.INTAKE

    # Atomicity: rejected transition wrote no audit_log row.
    rows = get_audit_log(CASE_ID, db_path=db_path)
    assert rows == []


def test_illegal_transition_missing_case_raises_not_found(tmp_path: Path):
    db_path = _db(tmp_path)
    with pytest.raises(CaseNotFoundError):
        transition("ghost-case", CaseState.ELIGIBILITY, actor="system", event="x", db_path=db_path)


class _Unstringifiable:
    """Object whose `str()` raises -- used to force a failure during
    `json.dumps(..., default=str)` *after* `transition()`'s UPDATE has
    already executed, so we can prove the UPDATE gets rolled back too.
    """

    def __str__(self) -> str:  # pragma: no cover - exercised via json.dumps
        raise ValueError("cannot stringify")


def test_transition_rolls_back_status_update_when_audit_payload_fails(tmp_path: Path):
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)

    with pytest.raises(ValueError):
        transition(
            CASE_ID,
            CaseState.ELIGIBILITY,
            actor="system",
            event="advance",
            payload={"bad": _Unstringifiable()},
            db_path=db_path,
        )

    # The UPDATE ran before the payload serialization failed -- confirm it
    # was rolled back, not left partially applied.
    assert get_status(CASE_ID, db_path=db_path) == CaseState.INTAKE
    assert get_audit_log(CASE_ID, db_path=db_path) == []


# --- terminal state CLOSED has no legal outgoing transitions ------------------


def test_closed_is_terminal(tmp_path: Path):
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)
    for to_state in (
        CaseState.ELIGIBILITY,
        CaseState.EVIDENCE,
        CaseState.VALIDATION,
        CaseState.CALC,
        CaseState.PENDING_REVIEW,
        CaseState.APPROVED,
        CaseState.SENT,
        CaseState.CLOSED,
    ):
        transition(CASE_ID, to_state, actor="system", event="advance", db_path=db_path)

    assert get_status(CASE_ID, db_path=db_path) == CaseState.CLOSED

    for candidate in CaseState:
        with pytest.raises(IllegalTransitionError):
            transition(CASE_ID, candidate, actor="system", event="attempt", db_path=db_path)


# --- record_action idempotency backstop ---------------------------------------


def test_record_action_succeeds_once_then_raises_duplicate(tmp_path: Path):
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)

    record_action(CASE_ID, "reimbursement", {"amount": "42.00"}, db_path=db_path)

    with pytest.raises(DuplicateActionError):
        record_action(CASE_ID, "reimbursement", {"amount": "42.00"}, db_path=db_path)

    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM actions WHERE case_id = ? AND action = ?",
            (CASE_ID, "reimbursement"),
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert json.loads(rows[0]["payload_json"]) == {"amount": "42.00"}


def test_record_action_different_action_types_both_succeed(tmp_path: Path):
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)

    record_action(CASE_ID, "email", {"subject": "hi"}, db_path=db_path)
    record_action(CASE_ID, "reimbursement", {"amount": "10.00"}, db_path=db_path)

    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT action FROM actions WHERE case_id = ? ORDER BY action", (CASE_ID,)).fetchall()
    finally:
        conn.close()
    assert [r["action"] for r in rows] == ["email", "reimbursement"]


# --- recommendation JSON round-trip (Decimal precision) -----------------------


def test_recommendation_round_trips_with_exact_decimal_precision(tmp_path: Path):
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)

    rec = Recommendation(
        decision="approve",
        amount=Decimal("123.456789"),
        line_items=[
            RecommendationLineItem(sku="SKU-1", quantity=3, unit_price=Decimal("1.005"), subtotal=Decimal("3.015"))
        ],
        rationale="clear damage evidence",
        email_draft="Dear customer, ...",
        confidence=0.87,
        risk_tier="low",
    )

    save_recommendation(CASE_ID, rec, db_path=db_path)
    loaded = get_recommendation(CASE_ID, db_path=db_path)

    assert loaded == rec
    assert loaded.amount == Decimal("123.456789")
    assert isinstance(loaded.amount, Decimal)
    assert loaded.line_items[0].unit_price == Decimal("1.005")

    # Raw column really is a JSON string with Decimal serialized as text,
    # never a float -- guards against silent precision loss in storage.
    row = get_case(CASE_ID, db_path=db_path)
    raw = json.loads(row["recommendation_json"])
    assert raw["amount"] == "123.456789"
    assert isinstance(raw["amount"], str)


def test_get_recommendation_none_before_saved(tmp_path: Path):
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)
    assert get_recommendation(CASE_ID, db_path=db_path) is None


def test_save_recommendation_missing_case_raises(tmp_path: Path):
    db_path = _db(tmp_path)
    rec = Recommendation(
        decision="deny",
        amount=Decimal("0"),
        rationale="no evidence",
        email_draft="Dear customer, ...",
        confidence=0.1,
        risk_tier="high",
    )
    with pytest.raises(CaseNotFoundError):
        save_recommendation("ghost-case", rec, db_path=db_path)


# --- schema sanity -------------------------------------------------------------


# --- list_cases_by_status (review queue) ---------------------------


def test_list_cases_by_status_returns_only_matching_status_oldest_first(tmp_path: Path):
    db_path = _db(tmp_path)
    create_case("CASE-A", db_path=db_path)
    create_case("CASE-B", db_path=db_path)
    create_case("CASE-C", db_path=db_path)

    for case_id in ("CASE-A", "CASE-B", "CASE-C"):
        transition(case_id, CaseState.ELIGIBILITY, actor="system", event="advance", db_path=db_path)
        transition(case_id, CaseState.EVIDENCE, actor="system", event="advance", db_path=db_path)
        transition(case_id, CaseState.VALIDATION, actor="system", event="advance", db_path=db_path)
        transition(case_id, CaseState.CALC, actor="system", event="advance", db_path=db_path)
        # Each case's final transition into PENDING_REVIEW happens in a
        # separate call, so `updated_at` timestamps are monotonically
        # increasing in creation order (A, then B, then C).
        transition(case_id, CaseState.PENDING_REVIEW, actor="system", event="advance", db_path=db_path)

    # CASE-B advances further, to APPROVED -- must not show up as
    # PENDING_REVIEW anymore.
    transition("CASE-B", CaseState.APPROVED, actor="rep", event="rep_approved", db_path=db_path)

    rows = list_cases_by_status(CaseState.PENDING_REVIEW, db_path=db_path)
    assert [r["case_id"] for r in rows] == ["CASE-A", "CASE-C"]

    approved_rows = list_cases_by_status(CaseState.APPROVED, db_path=db_path)
    assert [r["case_id"] for r in approved_rows] == ["CASE-B"]


def test_list_cases_by_status_empty_when_no_matches(tmp_path: Path):
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)  # sits in INTAKE

    assert list_cases_by_status(CaseState.PENDING_REVIEW, db_path=db_path) == []


# --- log_event (audit-only logging, no status change) --------------


def test_log_event_writes_audit_row_without_changing_status(tmp_path: Path):
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)
    transition(CASE_ID, CaseState.ELIGIBILITY, actor="system", event="advance", db_path=db_path)
    before_status = get_status(CASE_ID, db_path=db_path)

    log_event(CASE_ID, actor="rep", event="pushback", payload={"feedback": "add more warmth"}, db_path=db_path)

    assert get_status(CASE_ID, db_path=db_path) == before_status  # unchanged

    rows = get_audit_log(CASE_ID, db_path=db_path)
    pushback_rows = [r for r in rows if r["event"] == "pushback"]
    assert len(pushback_rows) == 1
    assert pushback_rows[0]["actor"] == "rep"
    assert json.loads(pushback_rows[0]["payload_json"]) == {"feedback": "add more warmth"}


def test_log_event_missing_case_raises(tmp_path: Path):
    db_path = _db(tmp_path)
    with pytest.raises(CaseNotFoundError):
        log_event("ghost-case", actor="rep", event="pushback", db_path=db_path)


def test_log_event_payload_is_optional(tmp_path: Path):
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)

    log_event(CASE_ID, actor="system", event="note", db_path=db_path)

    row = get_audit_log(CASE_ID, db_path=db_path)[0]
    assert row["payload_json"] is None


# --- save_gate_results / load_gate_results (persistence-gap fix) ----


def _judgment(passed: bool = True, confidence: float = 0.9, note: str = "looks consistent") -> Judgment:
    return Judgment(passed=passed, confidence=confidence, note=note)


def _validation_result(**overrides) -> ValidationResult:
    defaults = dict(
        damage_visible=_judgment(),
        product_identifiable=_judgment(),
        product_on_invoice=_judgment(),
        packaging_documented=_judgment(),
        matched_skus=["A00360"],
    )
    defaults.update(overrides)
    return ValidationResult(**defaults)


def test_load_gate_results_returns_empty_bundle_when_none_saved(tmp_path: Path):
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)

    results = load_gate_results(CASE_ID, db_path=db_path)

    assert results == GateResults()
    assert results.eligibility is None
    assert results.evidence_gaps == []
    assert results.validation is None
    assert results.calc is None


def test_save_and_load_gate_results_round_trips_all_four_objects_with_decimal_precision(tmp_path: Path):
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)

    eligibility = EligibilityResult(eligible=True, reason=None, route="process")
    gaps = [Gap(item=EvidenceItem.CUSTOMER_CONFIRMATION, reason="LOW_CONFIDENCE", detail="a bit blurry")]
    validation = _validation_result(matched_skus=["A00360", "A00360"])
    calc = CalcResult(
        amount=Decimal("123.456789"),
        line_items=[
            RecommendationLineItem(sku="A00360", quantity=2, unit_price=Decimal("1.005"), subtotal=Decimal("2.01"))
        ],
        capped=True,
    )

    save_gate_results(CASE_ID, eligibility=eligibility, evidence_gaps=gaps, validation=validation, calc=calc, db_path=db_path)
    loaded = load_gate_results(CASE_ID, db_path=db_path)

    assert loaded.eligibility == eligibility
    assert loaded.evidence_gaps == gaps
    assert loaded.validation == validation
    assert loaded.calc == calc
    assert loaded.calc.amount == Decimal("123.456789")
    assert isinstance(loaded.calc.amount, Decimal)
    assert loaded.calc.line_items[0].unit_price == Decimal("1.005")

    # Raw column really stores Decimal as a JSON string, never a float.
    row = get_case(CASE_ID, db_path=db_path)
    raw = json.loads(row["gate_results_json"])
    assert raw["calc"]["amount"] == "123.456789"
    assert isinstance(raw["calc"]["amount"], str)


def test_save_gate_results_partial_matches_deny_on_ineligible_path(tmp_path: Path):
    """`pipeline.py`'s deny-on-ineligible exit only ever has `eligibility` --
    the other three fields must round-trip as their empty/None defaults, not
    error out or silently invent placeholder values.
    """
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)

    eligibility = EligibilityResult(eligible=False, reason="TOO_OLD", route="close")
    save_gate_results(CASE_ID, eligibility=eligibility, db_path=db_path)

    loaded = load_gate_results(CASE_ID, db_path=db_path)
    assert loaded.eligibility == eligibility
    assert loaded.evidence_gaps == []
    assert loaded.validation is None
    assert loaded.calc is None
    assert loaded.invoice_audit is None


def test_invoice_audit_round_trips_including_severity_and_extracted_lines(tmp_path: Path):
    """`InvoiceAudit` is the most awkward thing in the bundle to persist -- a
    pydantic model nested inside a frozen dataclass alongside a list of
    frozen dataclasses carrying an enum. None of that is JSON-native, so
    this asserts the whole shape survives a save/load rather than trusting
    `json.dumps` to have handled it.
    """
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)

    audit = InvoiceAudit(
        verified=True,
        reason=None,
        extracted=ExtractedInvoice(
            readable=True,
            currency="GBP",
            line_items=[
                ExtractedLine(
                    description="Liposomal Tripeptide Collagen",
                    sku="COLLAGEN1",
                    quantity=1,
                    unit_price=55.95,
                    line_total=55.95,
                )
            ],
            order_discount_total=14.99,
            order_reference="#140744",
        ),
        discrepancies=[
            Discrepancy(code="CURRENCY_MISMATCH", detail="GBP vs USD", severity=Severity.ESCALATE),
            Discrepancy(code="RETAIL_PRICE_UNREADABLE", detail="no price", severity=Severity.WARN),
        ],
    )
    save_gate_results(CASE_ID, invoice_audit=audit, db_path=db_path)

    loaded = load_gate_results(CASE_ID, db_path=db_path).invoice_audit
    assert loaded == audit
    # and the enum really came back as an enum, not the bare string it was
    # stored as -- `should_escalate` depends on identity comparison
    assert loaded.discrepancies[0].severity is Severity.ESCALATE
    assert loaded.should_escalate is True
    assert loaded.extracted.currency == "GBP"
    assert loaded.extracted.line_items[0].sku == "COLLAGEN1"


def test_unverified_invoice_audit_round_trips_with_no_extraction(tmp_path: Path):
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)

    audit = InvoiceAudit(verified=False, reason="the retail invoice could not be read")
    save_gate_results(CASE_ID, invoice_audit=audit, db_path=db_path)

    loaded = load_gate_results(CASE_ID, db_path=db_path).invoice_audit
    assert loaded == audit
    assert loaded.extracted is None
    assert loaded.should_escalate is False


def test_save_gate_results_missing_case_raises(tmp_path: Path):
    db_path = _db(tmp_path)
    with pytest.raises(CaseNotFoundError):
        save_gate_results("ghost-case", eligibility=EligibilityResult(eligible=True, reason=None, route="process"), db_path=db_path)


def test_load_gate_results_missing_case_raises(tmp_path: Path):
    db_path = _db(tmp_path)
    with pytest.raises(CaseNotFoundError):
        load_gate_results("ghost-case", db_path=db_path)


def test_save_gate_results_overwrites_previous_save(tmp_path: Path):
    """A case only ever reaches one terminal pipeline exit, so this is a
    full overwrite, not a merge -- confirmed directly here.
    """
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)

    save_gate_results(
        CASE_ID, eligibility=EligibilityResult(eligible=True, reason=None, route="process"), db_path=db_path
    )
    save_gate_results(
        CASE_ID,
        eligibility=EligibilityResult(eligible=True, reason=None, route="process"),
        evidence_gaps=[Gap(item=EvidenceItem.ORDER_PROOF, reason="MISSING", detail=None)],
        db_path=db_path,
    )

    loaded = load_gate_results(CASE_ID, db_path=db_path)
    assert len(loaded.evidence_gaps) == 1
    assert loaded.validation is None


def test_actions_table_has_unique_constraint_at_sqlite_level(tmp_path: Path):
    """Belt-and-suspenders: confirm the UNIQUE constraint really exists in
    the schema itself, not just that our Python-level catch happens to work.
    """
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)
    record_action(CASE_ID, "email", {"x": 1}, db_path=db_path)

    conn = get_connection(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO actions (case_id, action, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (CASE_ID, "email", "{}", "now"),
            )
    finally:
        conn.close()


# --- stats() ---------------------------------------------------------------------


def _advance_to_pending_review(case_id: str, db_path: Path, entry_event: str = "gate:calc_complete") -> None:
    """Drive a freshly-created case through the legal-transition chain up to
    PENDING_REVIEW, writing the real `entry_event` string `stats()` looks
    for (one of the 4 confirmed-by-grep events `pipeline._exit()` writes).
    """
    transition(case_id, CaseState.ELIGIBILITY, actor="system", event="transition:intake->eligibility", db_path=db_path)
    transition(case_id, CaseState.EVIDENCE, actor="system", event="transition:eligibility->evidence", db_path=db_path)
    transition(case_id, CaseState.VALIDATION, actor="system", event="transition:evidence->validation", db_path=db_path)
    transition(case_id, CaseState.CALC, actor="system", event="transition:validation->calc", db_path=db_path)
    transition(case_id, CaseState.PENDING_REVIEW, actor="system", event=entry_event, db_path=db_path)


def _approve_and_send(case_id: str, db_path: Path) -> None:
    """Drive a PENDING_REVIEW case through the real approve-and-send event
    sequence: `transition:pending_review->approved` then the SENT
    transition carrying the actual `event="approved"` string `stats()`
    counts.
    """
    transition(
        case_id,
        CaseState.APPROVED,
        actor="rep",
        event="transition:pending_review->approved",
        db_path=db_path,
    )
    transition(case_id, CaseState.SENT, actor="rep", event="approved", db_path=db_path)


def _insert_llm_call(db_path: Path, *, case_id: str, cost_usd, latency_ms: int) -> None:
    conn = get_connection(db_path)
    try:
        ensure_llm_calls_table(conn)
        conn.execute(
            """
            INSERT INTO llm_calls (
                case_id, prompt_name, prompt_hash, model, latency_ms,
                input_tokens, output_tokens, cost_usd, raw_response, created_at
            ) VALUES (?, 'p', 'h', 'm', ?, 1, 1, ?, '{}', 'now')
            """,
            (case_id, latency_ms, str(cost_usd)),
        )
        conn.commit()
    finally:
        conn.close()


def test_stats_empty_database_returns_none_rates_and_zero_counts(tmp_path: Path):
    db_path = _db(tmp_path)
    result = stats(db_path=db_path)

    assert result.approve_as_is_rate is None
    assert result.edit_rate is None
    assert result.pushback_rate is None
    assert result.escalation_rate is None
    assert result.mean_llm_cost_per_claim is None
    assert result.mean_llm_latency_ms_per_claim is None
    assert result.approved_count == 0
    assert result.reviewed_case_count == 0
    assert result.total_case_count == 0


def test_stats_approve_as_is_and_edit_rates(tmp_path: Path):
    """Case A is approved without any edit; case B is approved after the rep
    edits the draft (a `correction_recorded` event lands before the send).
    Both should count toward the approved denominator; only B should count
    as an edit.
    """
    db_path = _db(tmp_path)

    create_case("CASE-A", db_path=db_path)
    _advance_to_pending_review("CASE-A", db_path)
    _approve_and_send("CASE-A", db_path)

    create_case("CASE-B", db_path=db_path)
    _advance_to_pending_review("CASE-B", db_path)
    log_event(
        "CASE-B",
        actor="rep",
        event="correction_recorded",
        payload={"original_email_draft": "orig", "edited_email_draft": "final"},
        db_path=db_path,
    )
    _approve_and_send("CASE-B", db_path)

    result = stats(db_path=db_path)
    assert result.approved_count == 2
    assert result.approve_as_is_rate == pytest.approx(0.5)
    assert result.edit_rate == pytest.approx(0.5)
    # The two rates must sum to exactly 1.0 over the same denominator.
    assert result.approve_as_is_rate + result.edit_rate == pytest.approx(1.0)


def test_stats_pushback_rate_counted_over_reviewed_cases_only(tmp_path: Path):
    """Case A reaches PENDING_REVIEW and gets pushed back on (no approval
    yet). Case B reaches PENDING_REVIEW and is approved cleanly (no
    pushback). Case C never reaches PENDING_REVIEW at all (still in
    INTAKE) -- it must NOT count in the pushback-rate denominator.
    """
    db_path = _db(tmp_path)

    create_case("CASE-A", db_path=db_path)
    _advance_to_pending_review("CASE-A", db_path)
    log_event("CASE-A", actor="rep", event="pushback", payload={"feedback": "wrong amount"}, db_path=db_path)

    create_case("CASE-B", db_path=db_path)
    _advance_to_pending_review("CASE-B", db_path)
    _approve_and_send("CASE-B", db_path)

    create_case("CASE-C", db_path=db_path)  # sits in INTAKE, never reviewed

    result = stats(db_path=db_path)
    assert result.reviewed_case_count == 2
    assert result.pushback_rate == pytest.approx(0.5)


def test_stats_escalation_rate_uses_current_status(tmp_path: Path):
    """Case A is currently ESCALATED (reached directly from ELIGIBILITY,
    never via PENDING_REVIEW -- confirming escalation_rate doesn't depend
    on the pending-review entry events at all). Cases B and C are not
    escalated. escalation_rate = 1/3.
    """
    db_path = _db(tmp_path)

    create_case("CASE-A", db_path=db_path)
    transition("CASE-A", CaseState.ELIGIBILITY, actor="system", event="transition:intake->eligibility", db_path=db_path)
    transition("CASE-A", CaseState.ESCALATED, actor="system", event="gate:eligibility_insured", db_path=db_path)

    create_case("CASE-B", db_path=db_path)
    create_case("CASE-C", db_path=db_path)

    result = stats(db_path=db_path)
    assert result.total_case_count == 3
    assert result.escalation_rate == pytest.approx(1 / 3)
    # CASE-A never hit a pending-review entry event, so it must not be
    # double-counted as "reviewed".
    assert result.reviewed_case_count == 0


def test_stats_llm_cost_and_latency_averaged_per_case_not_per_call(tmp_path: Path):
    """Case A makes 2 LLM calls ($0.01 + $0.01 = $0.02 total, 100ms + 200ms
    = 300ms total). Case B makes 1 call ($0.10, 900ms). The per-case-then-
    across-cases mean is ($0.02 + $0.10) / 2 = $0.06 and (300 + 900) / 2 =
    600ms -- deliberately different from the naive per-call mean of
    ($0.01+$0.01+$0.10)/3 = $0.04 and (100+200+900)/3 = 400ms, so a wrong
    per-call implementation fails this assertion loudly.
    """
    db_path = _db(tmp_path)
    create_case("CASE-A", db_path=db_path)
    create_case("CASE-B", db_path=db_path)

    _insert_llm_call(db_path, case_id="CASE-A", cost_usd=Decimal("0.01"), latency_ms=100)
    _insert_llm_call(db_path, case_id="CASE-A", cost_usd=Decimal("0.01"), latency_ms=200)
    _insert_llm_call(db_path, case_id="CASE-B", cost_usd=Decimal("0.10"), latency_ms=900)

    result = stats(db_path=db_path)
    assert result.mean_llm_cost_per_claim == pytest.approx(0.06)
    assert result.mean_llm_latency_ms_per_claim == pytest.approx(600.0)
    # Sanity: confirm these do NOT match the naive per-call average.
    assert result.mean_llm_cost_per_claim != pytest.approx(0.04)
    assert result.mean_llm_latency_ms_per_claim != pytest.approx(400.0)


def test_stats_llm_cost_survives_scientific_notation_string(tmp_path: Path):
    """`_log_call` persists `cost_usd` as `str(Decimal(...))`, which for
    very small amounts renders in scientific notation (e.g. "1.2E-7").
    `CAST(cost_usd AS REAL)` must still parse it correctly.
    """
    db_path = _db(tmp_path)
    create_case(CASE_ID, db_path=db_path)
    _insert_llm_call(db_path, case_id=CASE_ID, cost_usd=Decimal("1.2E-7"), latency_ms=50)

    result = stats(db_path=db_path)
    assert result.mean_llm_cost_per_claim == pytest.approx(1.2e-7)


def test_stats_cases_with_no_llm_calls_do_not_crash_or_skew_average(tmp_path: Path):
    """A case that exists but never made an LLM call (e.g. it was denied at
    eligibility before any evidence classification ran) must simply be
    excluded from the per-claim average, not counted as a $0 case.
    """
    db_path = _db(tmp_path)
    create_case("CASE-NO-LLM", db_path=db_path)
    create_case("CASE-WITH-LLM", db_path=db_path)
    _insert_llm_call(db_path, case_id="CASE-WITH-LLM", cost_usd=Decimal("2.00"), latency_ms=1000)

    result = stats(db_path=db_path)
    assert result.mean_llm_cost_per_claim == pytest.approx(2.00)
    assert result.mean_llm_latency_ms_per_claim == pytest.approx(1000.0)
