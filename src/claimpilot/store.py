"""Business-level store API (case lifecycle + audit trail).

`claimpilot.db` owns raw connection + schema. This module owns *policy*
built on top of it: creating cases, validating and recording state
transitions against an explicit legal-transition map, saving/reading
recommendations, and recording idempotent outbound actions.

Every public function here is self-contained: it opens its own connection
via `claimpilot.db.get_connection(db_path)`, calls `ensure_schema()` so the
tables always exist, does its work, and closes the connection -- mirroring
the pattern already established by `claimpilot.llm._log_call`. `db_path`
defaults to `settings.db_path`; tests pass a `tmp_path` file so they
never touch the real on-disk database (same convention as
`claimpilot.llm.structured_call`'s `db_path` parameter).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Literal

from claimpilot.calc import CalcResult
from claimpilot.db import ensure_schema, get_connection
from claimpilot.gates.eligibility import EligibilityResult
from claimpilot.gates.evidence import Gap
from claimpilot.gates.invoice_audit import Discrepancy, ExtractedInvoice, InvoiceAudit, Severity
from claimpilot.gates.validation import ValidationResult
from claimpilot.models import CaseState, EvidenceItem, Recommendation, RecommendationLineItem

Actor = Literal["system", "rep"]
ActionType = Literal["email", "reimbursement"]


# --- Exceptions --------------------------------------------------------------


class IllegalTransitionError(Exception):
    """Raised when a requested state transition isn't legal from the case's
    current state, per `LEGAL_TRANSITIONS`.

    The plan spells this `ILLEGAL_TRANSITION`; we use the Pythonic exception
    naming convention (`IllegalTransitionError`) instead since it's a class,
    not a constant/error-code -- same semantics, same guarantee (raised,
    never silently ignored).
    """

    def __init__(self, case_id: str, from_state: CaseState, to_state: CaseState) -> None:
        self.case_id = case_id
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(f"Illegal transition for case {case_id}: {from_state.value} -> {to_state.value}")


class DuplicateActionError(Exception):
    """Raised when `record_action` is called a second time for the same
    `(case_id, action)` pair -- the idempotency backstop the plan calls out
    explicitly ("a double-clicked approve button or a retried request can
    never double-pay"). Wraps the underlying `sqlite3.IntegrityError` from
    the `UNIQUE(case_id, action)` constraint so callers never have to know
    this is backed by SQLite.
    """

    def __init__(self, case_id: str, action: str) -> None:
        self.case_id = case_id
        self.action = action
        super().__init__(f"Action {action!r} already recorded for case {case_id}")


class CaseNotFoundError(Exception):
    """Raised when an operation references a `case_id` with no `cases` row."""

    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        super().__init__(f"No case found with case_id {case_id!r}")


class CaseAlreadyExistsError(Exception):
    """Raised by `create_case` when `case_id` already has a row."""

    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        super().__init__(f"Case {case_id!r} already exists")


# --- Legal transition map -----------------------------------------------------
#
# Encodes the plan's described flow:
#   intake -> eligibility -> evidence -> validation -> calc -> pending_review
#     -> approved | needs_info | denied -> sent -> closed (+ escalated)
#
# Design notes (judgment calls made when this state machine was designed):
#
# - Every gate stage (ELIGIBILITY, EVIDENCE, VALIDATION) can short-circuit
#   straight to PENDING_REVIEW instead of only advancing to the next gate.
#   By design, every case always ends in pending_review (or escalated) --
#   the pipeline never sends anything itself -- so a failed/ineligible/
#   gap-found gate routes to PENDING_REVIEW with the appropriate decision
#   (deny/request_info) already baked into the draft, rather than jumping
#   to a terminal state directly.
# - ELIGIBILITY and VALIDATION can also go straight to ESCALATED: the
#   insured-shipment route (ELIGIBILITY) and the low-confidence-but-passed
#   route (VALIDATION) are both edge cases that need a human before even
#   entering the normal PENDING_REVIEW queue.
# - PENDING_REVIEW is the hub state: a rep can approve, deny, request more
#   info, or escalate from here.
# - APPROVED/DENIED both only lead to SENT (the outbound-guard-checked send
#   step) -- deliberately *not* directly to CLOSED, so the
#   "approved|denied -> sent -> closed" ordering is preserved and the
#   outbound guard always sits between a decision and delivery.
# - NEEDS_INFO can go to SENT (the info-request email itself is an outbound
#   send) or loop back to EVIDENCE once the customer replies with more
#   material, so a needs-info case can loop back to re-collect evidence.
# - ESCALATED can resolve back into PENDING_REVIEW (rep clears the escalation
#   into the normal review flow) or go straight to SENT (rep resolves and
#   sends directly without re-queuing).
# - SENT -> CLOSED is the final administrative close-out.
# - CLOSED is terminal: no legal outgoing transitions.
# - No `PENDING_REVIEW -> PENDING_REVIEW` self-loop (a deliberate judgment call):
#   a rep's "push back" action (feedback -> re-run the drafter -> new
#   Recommendation) doesn't advance the case anywhere -- it's still sitting
#   in the exact same review queue, waiting on the exact same rep decision,
#   just with a fresher draft. Modeling that as a state transition (even a
#   self-loop) would misrepresent it as *progress* through the state
#   machine, and would force every pushback to invent a fake `to_state`
#   just to get an audit row. Instead, `log_event()` (below) writes the
#   pushback's audit row directly, and `save_recommendation()` updates the
#   stored `Recommendation` -- no call to `transition()` at all. Same
#   reasoning applies to rep corrections recorded when the rep edits the
#   email body before sending.
#
# This is a best-effort superset for the pipeline to work within, not gospel
# -- a missing edge would block real pipeline code later, while an extra one
# is low-risk (illegal transitions are still whatever this map says at the
# time they're attempted).
LEGAL_TRANSITIONS: dict[CaseState, set[CaseState]] = {
    CaseState.INTAKE: {CaseState.ELIGIBILITY},
    CaseState.ELIGIBILITY: {CaseState.EVIDENCE, CaseState.PENDING_REVIEW, CaseState.ESCALATED},
    CaseState.EVIDENCE: {CaseState.VALIDATION, CaseState.PENDING_REVIEW},
    CaseState.VALIDATION: {CaseState.CALC, CaseState.PENDING_REVIEW, CaseState.ESCALATED},
    CaseState.CALC: {CaseState.PENDING_REVIEW},
    CaseState.PENDING_REVIEW: {
        CaseState.APPROVED,
        CaseState.NEEDS_INFO,
        CaseState.DENIED,
        CaseState.ESCALATED,
    },
    CaseState.APPROVED: {CaseState.SENT},
    CaseState.NEEDS_INFO: {CaseState.SENT, CaseState.EVIDENCE},
    CaseState.DENIED: {CaseState.SENT},
    CaseState.ESCALATED: {CaseState.PENDING_REVIEW, CaseState.SENT},
    # SENT -> EVIDENCE (not just CLOSED) matters for the NEEDS_INFO loop to
    # actually be reachable: the real sequence is NEEDS_INFO -> SENT (the
    # info-request email itself is an outbound send) -> [customer replies]
    # -> EVIDENCE (re-check the new material) -> ... From SENT alone there's
    # no way back to EVIDENCE otherwise, since the pipeline marks the
    # info-request send as SENT like any other outbound email. The caller
    # must apply this edge only when the SENT case was an info-request (not
    # a final approve/deny send) -- the map can't express that distinction,
    # only that the edge is legal when needed.
    CaseState.SENT: {CaseState.CLOSED, CaseState.EVIDENCE},
    CaseState.CLOSED: set(),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Case creation / reads ----------------------------------------------------


def create_case(case_id: str, *, merchant_id: str | None = None, db_path: Path | str | None = None) -> None:
    """Insert a new `cases` row with initial status `CaseState.INTAKE`.

    `merchant_id` is the ShipBob account identifier this case
    belongs to -- see `claimpilot.db.ensure_cases_table`'s docstring for why
    `Case.user_id` is the canonical value callers should pass, not
    `Case.account_name`. Defaults to `None` (kept optional, not required, so
    callers/tests that don't care about merchant history -- e.g. most of
    this module's own test suite -- don't need to invent one); `None` simply
    means `claimpilot.memory.merchant_context` can't attribute this case to
    any merchant's claim-frequency count.

    Raises:
        CaseAlreadyExistsError: `case_id` already has a row.
    """
    conn = get_connection(db_path)
    try:
        ensure_schema(conn)
        now = _now()
        try:
            conn.execute(
                "INSERT INTO cases (case_id, status, recommendation_json, merchant_id, created_at, updated_at) "
                "VALUES (?, ?, NULL, ?, ?, ?)",
                (case_id, CaseState.INTAKE.value, merchant_id, now, now),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            raise CaseAlreadyExistsError(case_id) from exc
    finally:
        conn.close()


def get_case(case_id: str, *, db_path: Path | str | None = None) -> sqlite3.Row | None:
    """Return the full `cases` row for `case_id`, or `None` if it doesn't exist."""
    conn = get_connection(db_path)
    try:
        ensure_schema(conn)
        return conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
    finally:
        conn.close()


def get_status(case_id: str, *, db_path: Path | str | None = None) -> CaseState:
    """Return the current `CaseState` for `case_id`.

    Raises:
        CaseNotFoundError: no `cases` row for `case_id`.
    """
    row = get_case(case_id, db_path=db_path)
    if row is None:
        raise CaseNotFoundError(case_id)
    return CaseState(row["status"])


# --- State transitions ---------------------------------------------------------


def transition(
    case_id: str,
    to_state: CaseState,
    *,
    actor: Actor,
    event: str,
    payload: dict | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Validate and record a state transition for `case_id`.

    Atomically (single SQLite transaction on one connection, opened with
    `BEGIN IMMEDIATE` so the write lock is taken before the read): validates
    `to_state` against `LEGAL_TRANSITIONS[current_state]`, and if legal,
    updates `cases.status`/`updated_at` and inserts one `audit_log` row --
    both writes commit together or neither does. `BEGIN IMMEDIATE` (rather
    than relying on SQLite's default deferred-transaction behavior, which
    only opens the transaction at the first write) makes the check-then-act
    read-validate-write sequence itself serialized against concurrent
    callers, not just the two writes -- otherwise two concurrent callers
    could both read the same pre-transition status, both pass validation,
    and both write (a real risk if these calls ever move off a single event
    loop, e.g. via `asyncio.to_thread`). This is what backs the "every
    transition writes an audit row" guarantee -- the accountability story
    for the review panel, and the enforcement point the outbound guard
    relies on ("state machine enforced in store.py").

    Args:
        case_id: which case to transition.
        to_state: the target `CaseState`.
        actor: `"system"` for pipeline-driven transitions, `"rep"` for
            human review actions (approve/deny/request-info/escalate/etc).
        event: short human-readable event label, e.g.
            `"transition:eligibility->evidence"` or `"rep_approved"`.
        payload: optional JSON-serializable event detail, stored alongside
            the audit row.
        db_path: overrides `settings.db_path` (tests only).

    Raises:
        CaseNotFoundError: no `cases` row for `case_id`.
        IllegalTransitionError: `to_state` is not a legal target from the
            case's current status. No `cases` or `audit_log` row is
            modified/inserted in this case.
    """
    conn = get_connection(db_path)
    try:
        ensure_schema(conn)
        # BEGIN IMMEDIATE takes the write lock up front, so the read below
        # and the writes further down are all serialized against any other
        # writer touching this connection's database file -- see the
        # docstring above for why this matters beyond just the two writes.
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if row is None:
            raise CaseNotFoundError(case_id)

        from_state = CaseState(row["status"])
        if to_state not in LEGAL_TRANSITIONS.get(from_state, set()):
            raise IllegalTransitionError(case_id, from_state, to_state)

        now = _now()
        conn.execute(
            "UPDATE cases SET status = ?, updated_at = ? WHERE case_id = ?",
            (to_state.value, now, case_id),
        )
        # payload_json is built (JSON serialization) *after* the UPDATE, so
        # if it fails (e.g. a payload value that isn't JSON-serializable
        # even via `default=str`, which handles the common case of a
        # caller passing a `Decimal` -- this codebase's money type --
        # straight through) the exception propagates up through the
        # `except Exception: conn.rollback()` below and the UPDATE that
        # already ran gets undone too. This is exactly the atomicity
        # property under test in `test_store.py`'s
        # `test_transition_rolls_back_status_when_audit_payload_fails`.
        payload_json = json.dumps(payload, default=str) if payload is not None else None
        conn.execute(
            "INSERT INTO audit_log (case_id, actor, event, payload_json, ts) VALUES (?, ?, ?, ?, ?)",
            (case_id, actor, event, payload_json, now),
        )
        conn.commit()
    except Exception:
        # No partial writes: either both the status update and the audit
        # row commit together, or (on any error, including a rejected
        # transition raised before either statement ran) nothing does.
        conn.rollback()
        raise
    finally:
        conn.close()


def log_event(
    case_id: str,
    *,
    actor: Actor,
    event: str,
    payload: dict | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Write one `audit_log` row for `case_id` WITHOUT touching `cases.status`.

    Two rep actions need an audit trail but are NOT
    state transitions: pushback feedback (the case stays in
    `PENDING_REVIEW`/`ESCALATED`; see the module-level note below on why we
    didn't add a `PENDING_REVIEW -> PENDING_REVIEW` self-loop to
    `LEGAL_TRANSITIONS` instead) and rep corrections to the email draft
    before sending. `transition()` was the only thing that could write an
    `audit_log` row before this, and forcing either of these through it
    would mean inventing a transition that doesn't represent a real state
    change.

    Both of those same two rep actions are ALSO recorded as
    `kind="correction"` rows via `claimpilot.memory.record_correction`
    (called from `claimpilot.web.app`, alongside -- not instead of -- the
    `log_event` calls here). This `audit_log` row remains the general,
    always-present accountability trail for the case; the `memory` table
    row is the purpose-built, structured source the feedback
    distiller actually reads from. Keep writing both: `audit_log` is
    case-centric and chronological, `memory` is merchant-centric and
    queryable by `claimpilot.memory.merchant_context`.

    Same atomic-write convention as `transition()` (`BEGIN IMMEDIATE`,
    existence check, rollback on any failure) minus the `cases.status`
    UPDATE -- kept consistent so a payload-serialization failure can't leave
    a half-written row here either, even though there's only one statement
    to roll back.

    Raises:
        CaseNotFoundError: no `cases` row for `case_id`.
    """
    conn = get_connection(db_path)
    try:
        ensure_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT case_id FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if row is None:
            raise CaseNotFoundError(case_id)

        now = _now()
        payload_json = json.dumps(payload, default=str) if payload is not None else None
        conn.execute(
            "INSERT INTO audit_log (case_id, actor, event, payload_json, ts) VALUES (?, ?, ?, ?, ?)",
            (case_id, actor, event, payload_json, now),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_audit_log(case_id: str, *, db_path: Path | str | None = None) -> list[sqlite3.Row]:
    """Return all `audit_log` rows for `case_id`, oldest first.

    Useful for the review UI (rendering the case's full event
    timeline).
    """
    conn = get_connection(db_path)
    try:
        ensure_schema(conn)
        return conn.execute(
            "SELECT * FROM audit_log WHERE case_id = ? ORDER BY id ASC",
            (case_id,),
        ).fetchall()
    finally:
        conn.close()


def list_cases_by_status(status: CaseState, *, db_path: Path | str | None = None) -> list[sqlite3.Row]:
    """Return all `cases` rows with the given `status`, oldest-updated first.

    Added for the review queue (`GET /cases`), which needs "every case
    a rep should look at" rather than a single `case_id` lookup. Ordered by
    `updated_at ASC` (oldest
    first) so the queue reads like a FIFO triage list: a case that's been
    sitting untouched the longest surfaces first, rather than newest-first
    (which could let an old case sit unseen indefinitely as fresh ones keep
    arriving).

    Deliberately takes one `status`, not a list -- the queue endpoint calls
    this once per status it cares about (`PENDING_REVIEW`, `ESCALATED`) and
    merges the results itself. Keeps this function's SQL trivial and matches
    the plan's suggested signature; a caller wanting a multi-status query can
    just call it twice.
    """
    conn = get_connection(db_path)
    try:
        ensure_schema(conn)
        return conn.execute(
            "SELECT * FROM cases WHERE status = ? ORDER BY updated_at ASC",
            (status.value,),
        ).fetchall()
    finally:
        conn.close()


# --- Recommendation ------------------------------------------------------------


def save_recommendation(
    case_id: str,
    recommendation: Recommendation,
    *,
    db_path: Path | str | None = None,
) -> None:
    """Persist `recommendation` as JSON on the case's `recommendation_json` column.

    Uses `Recommendation.model_dump_json()`, which renders `Decimal` fields
    (e.g. `amount`) as JSON strings rather than floats, so exact precision
    survives the round trip through SQLite's TEXT column (confirmed: pydantic
    v2's JSON mode serializes `Decimal` as a string, not a float).

    Raises:
        CaseNotFoundError: no `cases` row for `case_id`.
    """
    conn = get_connection(db_path)
    try:
        ensure_schema(conn)
        now = _now()
        cur = conn.execute(
            "UPDATE cases SET recommendation_json = ?, updated_at = ? WHERE case_id = ?",
            (recommendation.model_dump_json(), now, case_id),
        )
        if cur.rowcount == 0:
            raise CaseNotFoundError(case_id)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_recommendation(case_id: str, *, db_path: Path | str | None = None) -> Recommendation | None:
    """Return the case's saved `Recommendation`, or `None` if none is saved yet.

    Raises:
        CaseNotFoundError: no `cases` row for `case_id`.
    """
    row = get_case(case_id, db_path=db_path)
    if row is None:
        raise CaseNotFoundError(case_id)
    if row["recommendation_json"] is None:
        return None
    return Recommendation.model_validate_json(row["recommendation_json"])


# --- Gate results (persisted so the outbound guard can re-verify) --------------
#
# `pipeline.process_case` only ever persisted the final `Recommendation`
# (decision/amount/line_items/rationale/email_draft/confidence/risk_tier) at
# each of its 7 exit points -- the intermediate gate objects that actually
# *produced* that recommendation (`EligibilityResult`, `evidence_gaps`,
# `ValidationResult`, `CalcResult`) were computed, used to build the
# recommendation, and then discarded. That's a real gap the outbound guard
# needs closed: the guard's whole point is to re-verify an approve decision
# against what the gates *actually* found, not against the LLM-authored
# `Recommendation` (which could itself be wrong, or -- the scarier case -- a
# `recommendation_json` row that's been corrupted/tampered with between
# calc-time and approve-time). Without a separate, independently-stored
# record of the gate outcomes, there is nothing to cross-check the
# recommendation against.
#
# `GateResults` bundles the (optional) deserialized gate objects for one
# case; `save_gate_results`/`load_gate_results` persist/read them as one JSON
# blob on `cases.gate_results_json`, following the exact same
# "`db_path`-scoped connection, `ensure_schema()`, do the work, close"
# convention as every other function in this module.


@dataclass(frozen=True)
class GateResults:
    """The intermediate gate objects `pipeline.process_case` actually
    computed for a case, as saved by `save_gate_results` at whichever exit
    point the case took. Every field is optional/default-empty because not
    every pipeline path runs every gate (see `pipeline.py`'s module
    docstring point 1): a deny produced by the eligibility gate never
    reaches evidence/validation/calc, so only `eligibility` is populated for
    that case.

    `eligibility` is populated at *every* exit (every path evaluates
    eligibility before it can exit at all) -- its presence is therefore the
    signal `guard.check_outbound` uses to distinguish "gate results were
    genuinely never recorded for this case" (e.g. a case built directly via
    `create_case`/`transition` in a test, bypassing the pipeline entirely,
    or a row whose `gate_results_json` was wiped/corrupted) from "this case's
    real path through the pipeline legitimately didn't reach evidence/
    validation/calc."
    """

    eligibility: EligibilityResult | None = None
    evidence_gaps: list[Gap] = field(default_factory=list)
    validation: ValidationResult | None = None
    calc: CalcResult | None = None
    invoice_audit: InvoiceAudit | None = None


def _invoice_audit_to_json_dict(audit: InvoiceAudit) -> dict:
    """`InvoiceAudit` mixes a pydantic model (`extracted`), a list of frozen
    dataclasses, and an enum -- none of which `json.dumps` handles directly,
    so each is unwrapped explicitly here (same reason `_calc_result_to_json_
    dict` exists for `Decimal`).
    """
    return {
        "verified": audit.verified,
        "reason": audit.reason,
        "extracted": json.loads(audit.extracted.model_dump_json()) if audit.extracted is not None else None,
        "discrepancies": [
            {"code": d.code, "detail": d.detail, "severity": d.severity.value} for d in audit.discrepancies
        ],
    }


def _invoice_audit_from_json_dict(data: dict) -> InvoiceAudit:
    return InvoiceAudit(
        verified=data["verified"],
        reason=data.get("reason"),
        extracted=ExtractedInvoice.model_validate(data["extracted"]) if data.get("extracted") else None,
        discrepancies=[
            Discrepancy(code=d["code"], detail=d["detail"], severity=Severity(d["severity"]))
            for d in data.get("discrepancies") or []
        ],
    )


def _eligibility_to_json_dict(result: EligibilityResult) -> dict:
    """`EligibilityResult` has no `Decimal`/enum fields -- a plain dict of
    JSON-native values round-trips it exactly.
    """
    return {"eligible": result.eligible, "reason": result.reason, "route": result.route}


def _eligibility_from_json_dict(data: dict) -> EligibilityResult:
    return EligibilityResult(eligible=data["eligible"], reason=data["reason"], route=data["route"])


def _gap_to_json_dict(gap: Gap) -> dict:
    """`Gap.item` is an `EvidenceItem` enum member -- stored as its plain
    `.value` string (not `dataclasses.asdict()`'s default `str(enum_member)`,
    which would emit the un-parseable `"EvidenceItem.ORDER_PROOF"`).
    """
    return {"item": gap.item.value, "reason": gap.reason, "detail": gap.detail}


def _gap_from_json_dict(data: dict) -> Gap:
    return Gap(item=EvidenceItem(data["item"]), reason=data["reason"], detail=data["detail"])


def _calc_result_to_json_dict(calc: CalcResult) -> dict:
    """`CalcResult.amount` is a `Decimal` and `line_items` are pydantic
    `RecommendationLineItem`s (which themselves carry `Decimal` fields) --
    same precision-preserving approach `save_recommendation` already uses
    for `Recommendation` (pydantic's `.model_dump_json()` renders `Decimal`
    as a JSON string, not a float), applied per-line-item here since
    `CalcResult` itself is a plain dataclass, not a pydantic model.
    """
    return {
        "amount": str(calc.amount),
        "capped": calc.capped,
        "line_items": [json.loads(li.model_dump_json()) for li in calc.line_items],
    }


def _calc_result_from_json_dict(data: dict) -> CalcResult:
    return CalcResult(
        amount=Decimal(data["amount"]),
        capped=data["capped"],
        line_items=[RecommendationLineItem.model_validate(li) for li in data["line_items"]],
    )


def save_gate_results(
    case_id: str,
    *,
    eligibility: EligibilityResult | None = None,
    evidence_gaps: list[Gap] | None = None,
    validation: ValidationResult | None = None,
    calc: CalcResult | None = None,
    invoice_audit: InvoiceAudit | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Persist whichever gate objects were actually computed for `case_id` as
    one JSON blob on `cases.gate_results_json`.

    Each pipeline exit calls this exactly once (via `pipeline._exit()`, its
    single funnel point -- see that module's docstring point 1), passing
    only the gate objects that ran on the path this case took (e.g. the
    deny-on-ineligible path only ever has `eligibility`). This is therefore
    a full overwrite of the column, not a merge with whatever was previously
    stored -- there is nothing to merge, since a case only ever reaches a
    terminal exit once per pipeline run, and a pushback redraft
    never calls this again (it doesn't recompute any gates, so the
    originally-saved gate results remain the correct, current ones to
    re-verify against).

    `evidence_gaps` defaults to `None` -> stored as `[]`, matching
    `GateResults.evidence_gaps`'s own non-optional-list convention.

    Raises:
        CaseNotFoundError: no `cases` row for `case_id`.
    """
    blob = {
        "eligibility": _eligibility_to_json_dict(eligibility) if eligibility is not None else None,
        "evidence_gaps": [_gap_to_json_dict(g) for g in (evidence_gaps or [])],
        "validation": json.loads(validation.model_dump_json()) if validation is not None else None,
        "calc": _calc_result_to_json_dict(calc) if calc is not None else None,
        "invoice_audit": _invoice_audit_to_json_dict(invoice_audit) if invoice_audit is not None else None,
    }
    conn = get_connection(db_path)
    try:
        ensure_schema(conn)
        now = _now()
        cur = conn.execute(
            "UPDATE cases SET gate_results_json = ?, updated_at = ? WHERE case_id = ?",
            (json.dumps(blob), now, case_id),
        )
        if cur.rowcount == 0:
            raise CaseNotFoundError(case_id)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def load_gate_results(case_id: str, *, db_path: Path | str | None = None) -> GateResults:
    """Return the case's saved `GateResults`, or an all-`None`/empty
    `GateResults()` if none has been saved yet (e.g. a case built directly
    via `create_case`/`transition` in a test, bypassing the pipeline).

    This all-empty default is exactly the fail-closed signal
    `guard.check_outbound` relies on: `GateResults().eligibility is None`
    looks identical whether gate results were never saved at all, or
    (impossible in practice, per `pipeline.py`, but not ruled out by this
    function) saved with `eligibility=None` explicitly -- either way, an
    approve decision can't be verified against it, and the guard must treat
    that as a violation rather than silently passing.

    Raises:
        CaseNotFoundError: no `cases` row for `case_id`.
    """
    row = get_case(case_id, db_path=db_path)
    if row is None:
        raise CaseNotFoundError(case_id)
    raw = row["gate_results_json"]
    if raw is None:
        return GateResults()

    blob = json.loads(raw)
    return GateResults(
        eligibility=_eligibility_from_json_dict(blob["eligibility"]) if blob.get("eligibility") else None,
        evidence_gaps=[_gap_from_json_dict(g) for g in blob.get("evidence_gaps") or []],
        validation=ValidationResult.model_validate(blob["validation"]) if blob.get("validation") else None,
        calc=_calc_result_from_json_dict(blob["calc"]) if blob.get("calc") else None,
        invoice_audit=(
            _invoice_audit_from_json_dict(blob["invoice_audit"]) if blob.get("invoice_audit") else None
        ),
    )


# --- Actions (idempotency backstop) --------------------------------------------


def record_action(
    case_id: str,
    action: ActionType,
    payload: dict,
    *,
    db_path: Path | str | None = None,
) -> None:
    """Record an outbound action (`"email"` or `"reimbursement"`) for `case_id`.

    `actions.UNIQUE(case_id, action)` guarantees at most one row per
    `(case_id, action)` pair ever exists. This is the idempotency backstop
    the plan calls out: a double-clicked approve button or a retried
    request hits the unique constraint and gets a clear
    `DuplicateActionError` instead of silently double-sending or
    double-paying.

    Raises:
        DuplicateActionError: an action of this type was already recorded
            for this case.
    """
    conn = get_connection(db_path)
    try:
        ensure_schema(conn)
        now = _now()
        try:
            conn.execute(
                "INSERT INTO actions (case_id, action, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (case_id, action, json.dumps(payload, default=str), now),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            raise DuplicateActionError(case_id, action) from exc
    finally:
        conn.close()


# --- Stats ------------------------------------------------------------------
#
# The queue's stats bar: learning is measured -- if the approve-as-is rate
# drops after a policy note lands, the loop is hurting and you can see it.
# Every number here is computed from `audit_log` + `llm_calls` -- no new
# state, no new table.
#
# The exact `event` strings written elsewhere in the codebase (confirmed by
# grepping `src/claimpilot/web/app.py` -- do NOT change these without
# updating the writers too):
#   - "approved"              -- `transition(..., event="approved")` when a
#                                 rep clicks Approve & Send (PENDING_REVIEW/
#                                 ESCALATED -> APPROVED, `web/app.py`).
#   - "correction_recorded"   -- `log_event(..., event="correction_recorded")`
#                                 when the edit-detection branch of the
#                                 approve endpoint notices the rep changed
#                                 the drafted email body before sending.
#   - "pushback"              -- `log_event(..., event="pushback")` when a
#                                 rep sends feedback and gets a redraft.
#   - "outbound_guard_blocked" -- `transition(..., event="outbound_guard_blocked")`
#                                 when the outbound guard vetoes a send and
#                                 escalates the case instead.


@dataclass(frozen=True)
class Stats:
    """The queue-header metrics. Every field is a plain `float` (a
    0.0-1.0 rate, a dollar amount, or a millisecond count) or `None` when the
    denominator for that metric is zero -- see `stats()`'s docstring for why
    `None` (not `0.0`) is used for the "no data yet" case, and how the
    template renders each state.
    """

    approve_as_is_rate: float | None
    edit_rate: float | None
    pushback_rate: float | None
    escalation_rate: float | None
    mean_llm_cost_per_claim: float | None
    mean_llm_latency_ms_per_claim: float | None
    approved_count: int
    reviewed_case_count: int
    total_case_count: int


# Every table `ensure_schema` creates, ordered so a future FK constraint
# wouldn't be violated mid-wipe (children before parents). There are no FKs
# today, but the ordering costs nothing and stops this becoming a landmine.
_ALL_TABLES: tuple[str, ...] = ("audit_log", "actions", "llm_calls", "memory", "cases")


def reset_all(*, db_path: Path | str | None = None) -> dict[str, int]:
    """Delete every row from every table, returning `{table: rows_deleted}`.

    Deletes rows rather than unlinking the database file the way
    `scripts/seed.py`'s `reset_db()` does. That script runs as a standalone
    process before the server starts, so it can safely swap the file out
    from under nobody; this runs *inside* the live web app, where removing
    the file that open connections are pointed at is a good way to produce
    confusing `SQLITE_READONLY`/stale-handle failures mid-demo. Same visible
    outcome (empty queue), no file-handle games.

    Wrapped in one transaction so a reset either fully happens or doesn't --
    a half-wiped database (cases gone, audit log retained) would be worse
    than either end state.
    """
    conn = get_connection(db_path)
    try:
        ensure_schema(conn)
        deleted: dict[str, int] = {}
        for table in _ALL_TABLES:
            cur = conn.execute(f"DELETE FROM {table}")  # noqa: S608 -- fixed literal names, no user input
            deleted[table] = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        conn.commit()
        return deleted
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def stats(*, db_path: Path | str | None = None) -> Stats:
    """Compute the queue's learning-loop metrics from `audit_log` + `llm_calls`.

    Six numbers, each a judgment call the plan deliberately left to the
    implementer (it names the metrics, not the formulas). Documented here so
    a future reader can see exactly what's being measured and why, without
    reverse-engineering the SQL:

    - **approve_as_is_rate**: of all cases with at least one `"approved"`
      audit event, what fraction have NO `"correction_recorded"` event
      anywhere in their audit log? Note `"approved"` (confirmed by grepping
      `web/app.py`'s single `/cases/{case_id}/approve` handler) is written
      on EVERY rep-initiated send through that endpoint -- approve, deny,
      AND needs-info decisions alike, not only `decision == "approve"`
      cases -- so the denominator here is really "all cases a rep sent
      through the review UI", and the metric measures "did the rep edit the
      drafted email body before sending", regardless of which decision was
      being sent. `None` if that denominator is zero (nothing sent yet).
      `edit_rate` is the exact complement (`1 - approve_as_is_rate`) over
      the same denominator, computed directly (not derived by subtraction)
      so a future change to one formula can't silently desync the other.

    - **pushback_rate**: fraction of cases that ever reached
      `PENDING_REVIEW` (i.e. cases a rep could have looked at) that have at
      least one `"pushback"` event. We use "cases that ever entered
      pending_review" rather than "all cases in the `cases` table" as the
      denominator: a case still stuck in an early gate (eligibility/
      evidence/validation) or freshly created was never in front of a rep
      at all, so it couldn't have been pushed back on -- including it would
      dilute the rate with cases that were never eligible to contribute a
      pushback in the first place. "Ever reached pending_review" is read
      from `audit_log` by matching the exact 4 event strings
      `pipeline._exit()`'s call sites actually write on the transition INTO
      `PENDING_REVIEW` -- confirmed by grepping `pipeline.py`:
      `"gate:eligibility_denied"`, `"gate:evidence_gap"`,
      `"gate:validation_request_info"`, `"gate:calc_complete"` -- rather
      than a generic `LIKE 'transition:%->pending_review'` pattern (no
      transition into `PENDING_REVIEW` is actually logged with a
      `"transition:...->pending_review"`-shaped event string anywhere in
      the codebase today; that pattern would silently match zero rows).
      This is deliberately NOT `cases.status = 'pending_review'` (current
      status): a case that was pushed back on and has since been approved
      and sent should still count in both numerator and denominator --
      using *current* status would drop it from the denominator entirely
      once it moves on. (`ESCALATED -> PENDING_REVIEW`, the rep-clears-
      escalation hop, is legal per `LEGAL_TRANSITIONS` but no `web/app.py`
      route exercises it today, so there is no event string for it yet to
      include here.)

    - **escalation_rate**: fraction of ALL cases (every row in `cases`)
      whose CURRENT status is `ESCALATED`. We chose "currently escalated"
      over "ever reached escalated" deliberately: a case that was escalated
      and has since been resolved (rep cleared it back into
      `pending_review` and sent it) is no longer a rep's open problem, and
      a stats bar a rep glances at should answer "how much is stuck in the
      escalation bucket right now", not "how much has ever touched it
      historically" -- that's the more actionable, less alarming number for
      a live dashboard. (The "ever escalated" reading is arguably more
      honest for a *learning-loop* narrative -- it never forgets a case
      that needed a human -- but this module already exposes `audit_log`
      via `get_audit_log()` for anyone who wants to compute that version
      separately.) `None` if there are no cases at all yet.

    - **mean_llm_cost_per_claim**: `AVG` of `SUM(cost_usd)` grouped by
      `case_id`, i.e. total LLM spend summed WITHIN each case (a case makes
      several calls -- evidence classification per attachment, validation,
      drafting) and then averaged ACROSS cases. Deliberately not a flat
      average over individual `llm_calls` rows, which would understate
      "cost per claim" by treating each attachment classification call as
      its own claim. `None` if no `llm_calls` rows exist yet.

    - **mean_llm_latency_ms_per_claim**: same per-case-then-across-cases
      aggregation as cost, applied to `latency_ms` -- i.e. total LLM
      wall-clock time spent processing one claim (summed across all of that
      case's calls), averaged across cases. This is the "how long does the
      LLM pipeline spend per claim" story a stats bar cares about; the
      alternative reading (mean latency of a single call, ungrouped) is a
      different, also-reasonable metric but answers "how slow is one LLM
      call", not "how slow is a claim" -- less useful next to a per-claim
      cost figure. `None` if no `llm_calls` rows exist yet.

    Zero-denominator handling: every rate/mean is `None` (not `0.0`) when
    its denominator is zero, since `0.0` would misleadingly read as "we
    measured this and it's zero" rather than "there's nothing to measure
    yet" -- e.g. an empty database's `approve_as_is_rate` should not look
    identical to a database where every single approval involved an edit.
    Callers (the queue route / template) must handle `None` explicitly;
    `queue.html` renders `None` as an em dash.
    """
    conn = get_connection(db_path)
    try:
        ensure_schema(conn)

        total_case_count = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]

        approved_case_ids = {
            row["case_id"]
            for row in conn.execute(
                "SELECT DISTINCT case_id FROM audit_log WHERE event = 'approved'"
            ).fetchall()
        }
        corrected_case_ids = {
            row["case_id"]
            for row in conn.execute(
                "SELECT DISTINCT case_id FROM audit_log WHERE event = 'correction_recorded'"
            ).fetchall()
        }
        approved_count = len(approved_case_ids)
        if approved_count:
            edited_and_approved = len(approved_case_ids & corrected_case_ids)
            approve_as_is_rate = (approved_count - edited_and_approved) / approved_count
            edit_rate = edited_and_approved / approved_count
        else:
            approve_as_is_rate = None
            edit_rate = None

        # The exact 4 event strings `pipeline._exit()` writes on the
        # transition into PENDING_REVIEW -- see this function's docstring
        # for why these literal strings (not a `LIKE 'transition:%'`
        # pattern) are used.
        pending_review_entry_events = (
            "gate:eligibility_denied",
            "gate:evidence_gap",
            "gate:validation_request_info",
            "gate:calc_complete",
        )
        reviewed_case_ids = {
            row["case_id"]
            for row in conn.execute(
                "SELECT DISTINCT case_id FROM audit_log WHERE event IN (?, ?, ?, ?)",
                pending_review_entry_events,
            ).fetchall()
        }
        pushback_case_ids = {
            row["case_id"]
            for row in conn.execute(
                "SELECT DISTINCT case_id FROM audit_log WHERE event = 'pushback'"
            ).fetchall()
        }
        reviewed_case_count = len(reviewed_case_ids)
        if reviewed_case_count:
            pushback_rate = len(reviewed_case_ids & pushback_case_ids) / reviewed_case_count
        else:
            pushback_rate = None

        if total_case_count:
            escalated_count = conn.execute(
                "SELECT COUNT(*) FROM cases WHERE status = ?", (CaseState.ESCALATED.value,)
            ).fetchone()[0]
            escalation_rate = escalated_count / total_case_count
        else:
            escalation_rate = None

        per_case_llm = conn.execute(
            "SELECT SUM(CAST(cost_usd AS REAL)) AS total_cost, SUM(latency_ms) AS total_latency_ms "
            "FROM llm_calls GROUP BY case_id"
        ).fetchall()
        if per_case_llm:
            mean_llm_cost_per_claim = sum(r["total_cost"] for r in per_case_llm) / len(per_case_llm)
            mean_llm_latency_ms_per_claim = sum(r["total_latency_ms"] for r in per_case_llm) / len(
                per_case_llm
            )
        else:
            mean_llm_cost_per_claim = None
            mean_llm_latency_ms_per_claim = None

        return Stats(
            approve_as_is_rate=approve_as_is_rate,
            edit_rate=edit_rate,
            pushback_rate=pushback_rate,
            escalation_rate=escalation_rate,
            mean_llm_cost_per_claim=mean_llm_cost_per_claim,
            mean_llm_latency_ms_per_claim=mean_llm_latency_ms_per_claim,
            approved_count=approved_count,
            reviewed_case_count=reviewed_case_count,
            total_case_count=total_case_count,
        )
    finally:
        conn.close()
