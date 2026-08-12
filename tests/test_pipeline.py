"""Tests for the pipeline orchestrator.

Integration-tests all 7 demo fixture scenarios end-to-end through
`process_case`, using the real `FixtureClient` for case/shipment/order/
invoice/attachment *metadata* (so the real eligibility/evidence-gap-count
math is actually exercised against real fixture data), plus:

- `_NoNetworkBytesClient`: a thin wrapper around `FixtureClient` that
  delegates every method except `get_attachment_bytes`, which returns
  deterministic canned bytes instead of downloading from Azure blob storage.
  There is no local cache under `fixtures/images/` in this environment, so
  going through the real `get_attachment_bytes` would require real network
  access -- exactly what `tests/test_fixture_client.py`'s own
  `test_get_attachment_bytes_downloads_and_caches_real_image` test
  documents by skipping when the network is unavailable. This project's
  house style is "no real network/LLM calls in tests" (see `test_llm.py`'s
  module docstring), so a skip-on-network-failure pattern isn't good enough
  for an orchestration test that must be reliable in CI.
- `FakeTransport` (from `tests/test_llm.py`): scripted per scenario to
  return exactly the LLM responses needed to drive that scenario's expected
  gate outcome. `FakeTransport` is one FIFO queue shared across every
  `structured_call` site `process_case` reaches for a given run, so each
  scenario's queue must list responses in the exact order they'll be
  consumed: one per `classify_attachment` call (in `list_attachments`
  order), then one for `validate_damage` (only if the evidence gate
  passed), then always exactly one for the final `draft()` call (every exit
  path -- deny/insured/request_info/escalated/approve -- calls `draft()`
  exactly once).

Two real fixture cases (CASE-1001, CASE-1003) have only 3 attachments each,
but the evidence gate (`gates/evidence.py`) always requires all 4
`EvidenceItem` categories to be present -- 3 attachments can never cover 4
distinct categories, so at least one `MISSING` gap is always forced. Exactly
how many is NOT a fixed "always one" fact, though (a real golden-eval run
corrected this assumption for CASE-1003 -- see `evals/golden.yaml`'s
matching correction): it's exactly one only if the 3 attachments happen to
land in 3 *distinct* categories; if two of them are genuinely the same kind
of evidence (as two of CASE-1003's three real attachments turned out to be
-- both are customer-email screenshots, correctly both classified
CUSTOMER_CONFIRMATION), only 2 categories get covered and 2 gaps result.
Both CASE-1001 and CASE-1003 still end in `request_info` via an evidence
gap either way, never `approve` -- no scripting of `usable`/`confidence` can
change that, since `MISSING` fires on the total *absence* of a category,
which confidence/usability scripting cannot touch.
CASE-1002 (4 attachments) and the synthetic CASE-9002-CAP (4 attachments,
added to `fixtures/synthetic.json` for this task -- it shipped with none)
cover the approve/approve-with-cap paths instead, so orchestration coverage
of every branch is still complete.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from claimpilot import memory
from claimpilot.calc import DamagedItem, reimbursement
from claimpilot.clients.fixtures import FixtureClient
from claimpilot.config import settings
from claimpilot.db import get_connection
from claimpilot.memory import NO_MERCHANT_ID_MEMORY_CONTEXT
from claimpilot.models import CaseState
from claimpilot.pipeline import GATE_DECISION_CONFIDENCE, process_case
from claimpilot.risk import RiskTier
from claimpilot.store import (
    create_case,
    get_audit_log,
    get_case,
    get_recommendation,
    get_status,
    load_gate_results,
)
from tests.test_llm import FakeTransport
from claimpilot.llm import TransportResult

NOW = datetime(2026, 3, 25, tzinfo=timezone.utc)


class _NoNetworkBytesClient:
    """Wraps a real `ShipBobClient`, delegating everything except
    `get_attachment_bytes` (see module docstring: no real network calls in
    tests). The actual byte content is irrelevant to these tests -- the
    (fake) LLM classification/validation calls that would "look at" the
    bytes are themselves scripted, not real vision calls.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def get_attachment_bytes(self, attachment) -> bytes:
        return b"\x89PNG\r\n\x1a\n" + attachment.attachment_id.encode()


@pytest.fixture
def client() -> _NoNetworkBytesClient:
    return _NoNetworkBytesClient(FixtureClient(include_synthetic=True))


def _classification(category: str, *, confidence: float = 0.95, usable: bool = True) -> TransportResult:
    return TransportResult(
        tool_input={"category": category, "confidence": confidence, "usable": usable, "quality_issue": None},
        input_tokens=10,
        output_tokens=5,
        raw_content=[],
    )


def _validation(
    *,
    matched_skus: list[str],
    all_pass: bool = True,
    confidence: float = 0.9,
    claimed_scope: str = "unclear",
    scope_note: str | None = None,
) -> TransportResult:
    judgment = {"passed": all_pass, "confidence": confidence, "note": "looks consistent with the claim"}
    return TransportResult(
        tool_input={
            "damage_visible": judgment,
            "product_identifiable": judgment,
            "product_on_invoice": judgment,
            "packaging_documented": judgment,
            "matched_skus": matched_skus,
            # defaults to "unclear" so every pre-existing scripted case keeps
            # the outcome it had before this check existed
            "customer_claimed_scope": claimed_scope,
            "customer_scope_note": scope_note,
        },
        input_tokens=20,
        output_tokens=10,
        raw_content=[],
    )


def _draft() -> TransportResult:
    return TransportResult(
        tool_input={"rationale": "Deterministic pipeline decision.", "email_draft": "Dear customer, ..."},
        input_tokens=5,
        output_tokens=5,
        raw_content=[],
    )


def _extraction(
    *,
    readable: bool = False,
    currency: str | None = None,
    line_items: list[dict] | None = None,
    order_discount_total: float = 0.0,
    note: str | None = "the uploaded document is not legible enough to read prices from",
) -> TransportResult:
    """Canned `gates.invoice_audit.extract_invoice` response.

    Defaults to `readable=False`, which `audit_invoice` turns into an
    unverified audit with zero discrepancies -- so a test that isn't
    specifically about retail-invoice reconciliation keeps exactly the
    pipeline outcome it had before that check was added, while still
    supplying the extra scripted LLM response the call now consumes.
    Tests that *do* exercise the audit pass real `line_items`/`currency`.
    """
    return TransportResult(
        tool_input={
            "readable": readable,
            "currency": currency,
            "line_items": line_items or [],
            "order_discount_total": order_discount_total,
            "subtotal": None,
            "tax_total": None,
            "grand_total": None,
            "order_reference": None,
            "note": note,
        },
        input_tokens=15,
        output_tokens=8,
        raw_content=[],
    )


def _audit_events(db_path: Path, case_id: str) -> list[str]:
    return [row["event"] for row in get_audit_log(case_id, db_path=db_path)]


class _DescriptionOverrideClient:
    """Wraps a client, overriding a single case's `Case.description` only --
    used by the affected-count mismatch test below to exercise a real
    fixture case (CASE-1002) as if its description said something other
    than what it actually says, without needing a dedicated new fixture.
    """

    def __init__(self, inner, case_id: str, description: str) -> None:
        self._inner = inner
        self._case_id = case_id
        self._description = description

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def get_case(self, case_id: str):
        case = await self._inner.get_case(case_id)
        if case_id == self._case_id:
            case = case.model_copy(update={"description": self._description})
        return case


# --- CASE-1001: 3 attachments -> one evidence gap -> request_info -----------


async def test_case_1001_evidence_gap_ends_pending_review_request_info(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    db_path = tmp_path / "t.db"
    # 3 real attachments, classified into 3 distinct categories -- the 4th
    # (CUSTOMER_CONFIRMATION) is never classified at all, so evidence_gaps
    # reports exactly one MISSING gap (see module docstring).
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-1001", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.decision == "request_info"
    assert recommendation.amount == Decimal("0")
    assert recommendation.confidence == GATE_DECISION_CONFIDENCE

    assert get_status("CASE-1001", db_path=db_path) == CaseState.PENDING_REVIEW
    assert get_recommendation("CASE-1001", db_path=db_path) == recommendation

    events = _audit_events(db_path, "CASE-1001")
    assert events == [
        "transition:intake->eligibility",
        "transition:eligibility->evidence",
        "gate:evidence_gap",
    ]
    # Not already-closed (CASE-1001's real status is "New").
    intake_row = get_audit_log("CASE-1001", db_path=db_path)[0]
    import json

    assert json.loads(intake_row["payload_json"])["already_closed"] is False


# --- CASE-1002: 4 attachments -> full happy path -> approve ------------------


async def test_case_1002_happy_path_ends_approve(client: _NoNetworkBytesClient, tmp_path: Path):
    db_path = tmp_path / "t.db"
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["A00360"], confidence=0.9),
            _extraction(),
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.decision == "approve"
    assert recommendation.amount == Decimal("24.99")  # A00360 unit_price, qty 1, under CAP
    assert recommendation.confidence == 0.9

    assert get_status("CASE-1002", db_path=db_path) == CaseState.PENDING_REVIEW
    assert get_recommendation("CASE-1002", db_path=db_path) == recommendation

    events = _audit_events(db_path, "CASE-1002")
    assert events == [
        "transition:intake->eligibility",
        "transition:eligibility->evidence",
        "transition:evidence->validation",
        "transition:validation->calc",
        "gate:calc_complete",
    ]


# --- affected-count mismatch escalates instead of auto-approving ------------
#
# Real CASE-1002 description says "Number of affected orders: 1.", which
# already matches its single confirmed SKU (A00360) -- the happy-path test
# above is itself proof the matching-count case does NOT escalate. This test
# overrides just the description to say 2, keeping everything else (attachment
# classifications, the single-SKU vision match) identical, to prove a real
# stated/confirmed mismatch escalates instead of silently paying out.


async def test_affected_count_mismatch_escalates_instead_of_approving(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    db_path = tmp_path / "t.db"
    mismatched_client = _DescriptionOverrideClient(
        client, "CASE-1002", "Customer reports damage. Number of affected orders: 2."
    )
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["A00360"], confidence=0.9),  # only 1 distinct SKU
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-1002", client=mismatched_client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.decision == "request_info"
    assert recommendation.amount == Decimal("0")
    assert get_status("CASE-1002", db_path=db_path) == CaseState.ESCALATED

    events = _audit_events(db_path, "CASE-1002")
    assert events[-1] == "gate:validation_affected_count_mismatch"

    mismatch_row = get_audit_log("CASE-1002", db_path=db_path)[-1]
    import json

    reason = json.loads(mismatch_row["payload_json"])["reason"]
    assert "states 2" in reason
    assert "confirmed 1" in reason


# --- retail-invoice audit escalates on a real price discrepancy ------------
#
# The happy-path test above scripts `_extraction()`'s unreadable default, so
# it proves the audit stays out of the way when it can't verify. These two
# prove the other half: a readable invoice that actually disagrees stops the
# payout, and a readable invoice that agrees lets it through unchanged.


async def test_retail_invoice_price_mismatch_escalates_instead_of_approving(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    """CASE-1002's real numbers: ShipBob's invoice prices A00360 at $24.99,
    the merchant's own sales order shows the customer paid $19.99. Without
    this check the pipeline approves $24.99 -- a 25% overpayment against
    what ShipBob's published policy says it owes ("the amount they paid").
    """
    db_path = tmp_path / "t.db"
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["A00360"], confidence=0.9),
            _extraction(
                readable=True,
                currency="USD",
                line_items=[
                    {
                        "description": "CleanBoss Botanical Disinfectant & Cleaner 24 Ounce 2 Pack",
                        "sku": "A00360",
                        "quantity": 1,
                        "unit_price": 19.99,
                        "line_total": 19.99,
                        "line_discount": 0.0,
                    }
                ],
                note=None,
            ),
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.decision == "request_info"
    assert recommendation.amount == Decimal("0")
    assert get_status("CASE-1002", db_path=db_path) == CaseState.ESCALATED

    events = _audit_events(db_path, "CASE-1002")
    assert events[-1] == "gate:invoice_audit_discrepancy"

    import json

    payload = json.loads(get_audit_log("CASE-1002", db_path=db_path)[-1]["payload_json"])
    assert payload["codes"] == ["PRICE_MISMATCH"]
    assert "24.99" in payload["reason"] and "19.99" in payload["reason"]

    # the audit itself is persisted for the review UI to render
    saved = load_gate_results("CASE-1002", db_path=db_path)
    assert saved.invoice_audit is not None
    assert saved.invoice_audit.verified is True
    assert [d.code for d in saved.invoice_audit.discrepancies] == ["PRICE_MISMATCH"]


async def test_retail_invoice_agreeing_with_shipbob_still_approves(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    """The check must not simply escalate everything -- when the merchant's
    invoice agrees with ShipBob's, the case approves exactly as it did
    before this gate existed, with the audit recorded as clean.
    """
    db_path = tmp_path / "t.db"
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["A00360"], confidence=0.9),
            _extraction(
                readable=True,
                currency="USD",
                line_items=[
                    {
                        "description": "CleanBoss Botanical Disinfectant & Cleaner 24oz 2 Pack",
                        "sku": "A00360",
                        "quantity": 1,
                        "unit_price": 24.99,
                        "line_total": 24.99,
                        "line_discount": 0.0,
                    }
                ],
                note=None,
            ),
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.decision == "approve"
    assert recommendation.amount == Decimal("24.99")
    assert get_status("CASE-1002", db_path=db_path) == CaseState.PENDING_REVIEW

    saved = load_gate_results("CASE-1002", db_path=db_path)
    assert saved.invoice_audit.verified is True
    assert saved.invoice_audit.discrepancies == []


async def test_unreadable_retail_invoice_does_not_block_the_payout(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    """Fail-open, verifiably: an unreadable order-proof records *why* it
    couldn't be verified and still lets the case through, rather than
    escalating every case whose invoice is a bad phone photo.
    """
    db_path = tmp_path / "t.db"
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["A00360"], confidence=0.9),
            _extraction(readable=False, note="the photo is too dark to read any prices"),
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.decision == "approve"
    assert get_status("CASE-1002", db_path=db_path) == CaseState.PENDING_REVIEW

    saved = load_gate_results("CASE-1002", db_path=db_path)
    assert saved.invoice_audit.verified is False
    assert "too dark" in saved.invoice_audit.reason


async def test_invoice_audit_can_be_disabled_by_configuration(
    client: _NoNetworkBytesClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The off switch genuinely restores the pre-audit behavior, including
    not spending the extra vision call -- the scripted transport below has
    no `_extraction()` entry at all and would raise if one were requested.
    """
    monkeypatch.setattr(settings, "invoice_audit_enabled", False)
    db_path = tmp_path / "t.db"
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["A00360"], confidence=0.9),
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.decision == "approve"
    assert recommendation.amount == Decimal("24.99")
    saved = load_gate_results("CASE-1002", db_path=db_path)
    assert saved.invoice_audit.verified is False
    assert "disabled" in saved.invoice_audit.reason


# --- CASE-1003: 3 attachments incl. invoice screenshot -> evidence gap ------


async def test_case_1003_invoice_screenshot_handled_sensibly_ends_request_info(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    """CORRECTED 2026-08-11 (see the LLM-usage audit / `evals/golden.yaml`'s
    matching correction): this test previously scripted a 3-way split
    (ORDER_PROOF/PRODUCT_PHOTO/PACKAGING_PHOTO) that does NOT match what
    these 3 real attachments actually are -- confirmed directly by both a
    real GPT-4o golden-eval run and manual inspection of the images.
    CASE-1003's real evidence is Inv.png (a genuine invoice screenshot) plus
    TWO screenshots of the same customer support-email thread (one about a
    leaking L-Carnitine bottle, one about a leaking Liquid Glycerol bottle)
    -- there is no actual product or packaging photo anywhere in this
    case's real evidence. The scripting below now matches the real,
    verified classification a careful human (and the real model) actually
    gives these 3 attachments: ORDER_PROOF, CUSTOMER_CONFIRMATION,
    CUSTOMER_CONFIRMATION -- forcing TWO gaps (PRODUCT_PHOTO AND
    PACKAGING_PHOTO), not one. Still ends in `request_info` before
    validation/calc either way -- only the exact gap count/contents change.
    """
    db_path = tmp_path / "t.db"
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),  # Inv.png
            _classification("CUSTOMER_CONFIRMATION"),  # email screenshot #1 (L-Carnitine)
            _classification("CUSTOMER_CONFIRMATION"),  # email screenshot #2 (Liquid Glycerol)
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-1003", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.decision == "request_info"
    assert recommendation.amount == Decimal("0")
    assert get_status("CASE-1003", db_path=db_path) == CaseState.PENDING_REVIEW

    events = _audit_events(db_path, "CASE-1003")
    assert events[-1] == "gate:evidence_gap"
    gap_row = get_audit_log("CASE-1003", db_path=db_path)[-1]
    import json

    gaps = json.loads(gap_row["payload_json"])["gaps"]
    assert set(gaps) == {"PRODUCT_PHOTO", "PACKAGING_PHOTO"}


# --- CASE-1004: too old (73 days) + already Closed --------------------------


async def test_case_1004_too_old_denies_and_surfaces_already_closed(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    db_path = tmp_path / "t.db"
    # Eligibility short-circuits before any evidence/validation calls --
    # only the final draft() call happens.
    transport = FakeTransport([_draft()])

    recommendation = await process_case(
        "CASE-1004", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.decision == "deny"
    assert recommendation.amount == Decimal("0")
    assert recommendation.confidence == GATE_DECISION_CONFIDENCE

    # Denied eligibility routes to PENDING_REVIEW (not ESCALATED) -- a
    # deny is a fully-resolved recommendation, not something needing a
    # human to first untangle (unlike insured/validation-escalated cases).
    assert get_status("CASE-1004", db_path=db_path) == CaseState.PENDING_REVIEW

    rows = get_audit_log("CASE-1004", db_path=db_path)
    events = [r["event"] for r in rows]
    assert events == ["transition:intake->eligibility", "gate:eligibility_denied"]

    import json

    intake_payload = json.loads(rows[0]["payload_json"])
    # The "already closed, but processed anyway" signal: CASE-1004's real
    # status is "Closed" in the fixture data, and processing it directly by
    # ID must still run to completion (never skip) -- this audit row is the
    # rep-visible evidence that it did.
    assert intake_payload["already_closed"] is True
    assert intake_payload["case_status_at_intake"] == "Closed"

    deny_payload = json.loads(rows[1]["payload_json"])
    assert deny_payload["reason"] == "TOO_OLD"


# --- CASE-1005: missing evidence (0 attachments) -----------------------------


async def test_case_1005_missing_evidence_all_four_gaps_ends_request_info(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    db_path = tmp_path / "t.db"
    # Zero attachments -> zero classify_attachment calls -- only draft().
    transport = FakeTransport([_draft()])

    recommendation = await process_case(
        "CASE-1005", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.decision == "request_info"
    assert recommendation.amount == Decimal("0")
    assert get_status("CASE-1005", db_path=db_path) == CaseState.PENDING_REVIEW

    rows = get_audit_log("CASE-1005", db_path=db_path)
    import json

    gap_payload = json.loads(rows[-1]["payload_json"])
    assert set(gap_payload["gaps"]) == {
        "ORDER_PROOF",
        "CUSTOMER_CONFIRMATION",
        "PRODUCT_PHOTO",
        "PACKAGING_PHOTO",
    }


async def test_case_1005_invoice_zero_price_line_item_does_not_crash_calc():
    """CASE-1005's evidence gate short-circuits before step 4 (order/invoice
    fetch + calc) ever runs -- zero attachments means all four gaps fire at
    the evidence gate, so `reimbursement()` is never reached for this case
    inside `process_case` itself. The task requires proving the $0.00
    "Insert Card" line item doesn't crash `reimbursement()`, so that's
    tested directly here against the real fixture invoice, independent of
    the full pipeline path.
    """
    client = FixtureClient(include_synthetic=True)
    invoice = await client.generate_invoice(shipment_id="349164073", user_id="398045")

    zero_price_sku = next(li.sku for li in invoice.line_items if li.unit_price == Decimal("0.00"))
    result = reimbursement(invoice, [DamagedItem(sku=zero_price_sku, quantity=1)])

    assert result.amount == Decimal("0.00")
    assert result.capped is False


# --- CASE-9001-INSURED: synthetic, routes to insured_process ----------------


async def test_case_9001_insured_escalates_not_auto_processed(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    db_path = tmp_path / "t.db"
    # is_insured=True short-circuits before any evidence/validation calls.
    transport = FakeTransport([_draft()])

    recommendation = await process_case(
        "CASE-9001-INSURED", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.decision == "request_info"  # least-wrong label; see module docstring
    assert recommendation.amount == Decimal("0")
    assert recommendation.confidence == GATE_DECISION_CONFIDENCE

    assert get_status("CASE-9001-INSURED", db_path=db_path) == CaseState.ESCALATED

    events = _audit_events(db_path, "CASE-9001-INSURED")
    assert events == ["transition:intake->eligibility", "gate:eligibility_insured"]


# --- CASE-9002-CAP: synthetic, single item over $100 -> capped exactly ------


async def test_case_9002_cap_caps_at_exactly_100(client: _NoNetworkBytesClient, tmp_path: Path):
    db_path = tmp_path / "t.db"
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["A00300-CASE12"], confidence=0.85),
            _extraction(),
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-9002-CAP", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.decision == "approve"
    assert recommendation.amount == Decimal("100.00")  # $150 raw, capped
    assert recommendation.confidence == 0.85

    assert get_status("CASE-9002-CAP", db_path=db_path) == CaseState.PENDING_REVIEW
    events = _audit_events(db_path, "CASE-9002-CAP")
    assert events[-1] == "gate:calc_complete"


# --- calc-exception escalation (synthetic misbehaving classification) -------


async def test_calc_exception_escalates_instead_of_crashing(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    """A misbehaving/fake classification could name a SKU that isn't on the
    invoice (`ItemNotOnInvoice`) -- `process_case` must escalate, not crash,
    per module docstring point 6. Reuses CASE-1002 (4 attachments, clean
    evidence pass) but scripts `validate_damage` to report a `matched_skus`
    entry that doesn't exist on that case's real invoice.

    The scripted response is deliberately self-contradictory: all four
    judgments (including `product_on_invoice`) report `passed=True`, while
    `matched_skus` names a SKU that isn't on the invoice at all.
    `combine_validation` never reads `matched_skus` (validation.py docstring
    point 8), so this incoherent-but-schema-valid response sails through to
    PROCEED -- exactly the "vision model said something that doesn't
    reconcile with the invoice" failure mode this test defends against, not
    a plausible real response.
    """
    db_path = tmp_path / "t.db"
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["NOT-A-REAL-SKU"], confidence=0.9),
            _extraction(),
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.decision == "request_info"
    assert recommendation.amount == Decimal("0")
    assert recommendation.confidence == GATE_DECISION_CONFIDENCE

    assert get_status("CASE-1002", db_path=db_path) == CaseState.ESCALATED

    events = _audit_events(db_path, "CASE-1002")
    assert events == [
        "transition:intake->eligibility",
        "transition:eligibility->evidence",
        "transition:evidence->validation",
        "gate:calc_exception",
    ]
    import json

    payload = json.loads(get_audit_log("CASE-1002", db_path=db_path)[-1]["payload_json"])
    assert payload["error_type"] == "ItemNotOnInvoice"


# --- validation ESCALATED (all pass, low confidence) -------------------------


async def test_validation_escalated_low_confidence_reports_weakest_judgment(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    """Reuses CASE-1002's clean evidence pass but scripts `validate_damage`
    with one low-confidence-but-passed judgment, so `combine_validation`
    returns ESCALATED. Confidence on the resulting recommendation must be
    the weakest judgment's confidence (0.62), not `GATE_DECISION_CONFIDENCE`
    -- that low number is literally why the case escalated.
    """
    db_path = tmp_path / "t.db"
    low_conf_judgment = {"passed": True, "confidence": 0.62, "note": "photo too dark"}
    high_conf_judgment = {"passed": True, "confidence": 0.95, "note": "clear"}
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            TransportResult(
                tool_input={
                    "damage_visible": low_conf_judgment,
                    "product_identifiable": high_conf_judgment,
                    "product_on_invoice": high_conf_judgment,
                    "packaging_documented": high_conf_judgment,
                    "matched_skus": ["A00360"],
                },
                input_tokens=1,
                output_tokens=1,
                raw_content=[],
            ),
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.decision == "request_info"
    assert recommendation.amount == Decimal("0")
    assert recommendation.confidence == 0.62

    assert get_status("CASE-1002", db_path=db_path) == CaseState.ESCALATED
    events = _audit_events(db_path, "CASE-1002")
    assert events[-1] == "gate:validation_escalated"


# --- validation REQUEST_INFO (a judgment failed) -----------------------------


async def test_validation_request_info_when_a_judgment_fails(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    db_path = tmp_path / "t.db"
    failing_judgment = {"passed": False, "confidence": 0.9, "note": "no damage visible in photos"}
    passing_judgment = {"passed": True, "confidence": 0.9, "note": "fine"}
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            TransportResult(
                tool_input={
                    "damage_visible": failing_judgment,
                    "product_identifiable": passing_judgment,
                    "product_on_invoice": passing_judgment,
                    "packaging_documented": passing_judgment,
                    "matched_skus": [],
                },
                input_tokens=1,
                output_tokens=1,
                raw_content=[],
            ),
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.decision == "request_info"
    assert recommendation.amount == Decimal("0")
    assert recommendation.confidence == GATE_DECISION_CONFIDENCE

    assert get_status("CASE-1002", db_path=db_path) == CaseState.PENDING_REVIEW
    events = _audit_events(db_path, "CASE-1002")
    assert events[-1] == "gate:validation_request_info"


# --- matched_skus -> DamagedItem quantity convention (Counter) --------------


async def test_matched_skus_duplicate_entries_count_toward_quantity(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    """Positive case for the documented `Counter(matched_skus)` convention
    (module docstring point 5): CASE-1002's invoice has SKU A00300 at
    quantity=2, unit_price=$12.99. Naming it twice in `matched_skus` must
    produce `DamagedItem("A00300", 2)`, not two separate quantity-1 items
    (which would be indistinguishable from this test if `Counter` were
    replaced with a naive `set()` dedup) and not a quantity-exceeds error
    (2 damaged <= 2 invoiced is exactly at the boundary).
    """
    db_path = tmp_path / "t.db"
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["A00300", "A00300"], confidence=0.9),
            _extraction(),
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.decision == "approve"
    assert recommendation.amount == Decimal("25.98")  # 2 x $12.99, under CAP
    assert get_status("CASE-1002", db_path=db_path) == CaseState.PENDING_REVIEW


async def test_matched_skus_duplicate_entries_exceeding_invoiced_quantity_escalates(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    """CASE-1002's invoice has SKU A00360 at quantity=1. Naming it twice in
    `matched_skus` produces `DamagedItem("A00360", 2)` via the `Counter`
    convention, which exceeds the invoiced quantity -- `reimbursement()`
    raises `QuantityExceedsInvoice`, and `process_case` must escalate
    (module docstring point 6), not crash. Covers the second exception type
    in the `except (ItemNotOnInvoice, QuantityExceedsInvoice)` clause --
    `test_calc_exception_escalates_instead_of_crashing` above only exercises
    `ItemNotOnInvoice`.
    """
    db_path = tmp_path / "t.db"
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["A00360", "A00360"], confidence=0.9),
            _extraction(),
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.decision == "request_info"
    assert recommendation.amount == Decimal("0")
    assert get_status("CASE-1002", db_path=db_path) == CaseState.ESCALATED

    events = _audit_events(db_path, "CASE-1002")
    assert events[-1] == "gate:calc_exception"
    import json

    payload = json.loads(get_audit_log("CASE-1002", db_path=db_path)[-1]["payload_json"])
    assert payload["error_type"] == "QuantityExceedsInvoice"


# --- merchant_id on case creation + real merchant memory ----------


class _NoUserIdClient:
    """Wraps `client`, overriding `get_case` to strip `user_id` off whatever
    case it returns -- simulates the "no merchant identifier available"
    edge case module docstring point 3 documents, which no real fixture
    case exercises (every fixture case has a `user_id`; see
    `docs/api/postman_collection.json`/`fixtures/synthetic.json`).
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def get_attachment_bytes(self, attachment) -> bytes:
        return b"\x89PNG\r\n\x1a\n" + attachment.attachment_id.encode()

    async def get_case(self, case_id: str):
        case = await self._inner.get_case(case_id)
        return case.model_copy(update={"user_id": None})


async def test_process_case_stores_merchant_id_from_case_user_id(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    db_path = tmp_path / "t.db"
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["A00360"]),
            _extraction(),
            _draft(),
        ]
    )

    await process_case("CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW)

    row = get_case("CASE-1002", db_path=db_path)
    assert row["merchant_id"] == "283959"  # CASE-1002's real Case.user_id


async def test_merchant_scoped_policy_note_elevates_risk_and_reaches_drafter_prompt(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    """A curated `kind="policy"` note for this merchant becomes a risk flag
    (via `MerchantMemory.flags`) AND reaches the drafter's prompt -- the one
    memory-derived signal `pipeline.py` treats as a real risk factor.
    """
    db_path = tmp_path / "t.db"
    memory.record_policy_note(
        "Repeat high-value claimant.", scope="merchant", merchant_id="283959", db_path=db_path
    )
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["A00360"]),
            _extraction(),
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.risk_tier == RiskTier.ELEVATED.value
    draft_prompt = transport.calls[-1]["messages"][-1]["content"]
    assert "Repeat high-value claimant." in draft_prompt


async def test_global_policy_note_reaches_prompt_but_never_elevates_risk(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    """A global policy note applies to every merchant -- it must appear in
    the drafter's prompt (informational) but must NEVER be treated as a
    risk flag for any one merchant (module docstring point 2 in
    `claimpilot.memory` / point 3 in `pipeline.py`).
    """
    db_path = tmp_path / "t.db"
    memory.record_policy_note("Always confirm SKU before approving.", scope="global", db_path=db_path)
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["A00360"]),
            _extraction(),
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.risk_tier == RiskTier.LOW.value
    draft_prompt = transport.calls[-1]["messages"][-1]["content"]
    assert "Always confirm SKU before approving." in draft_prompt


async def test_raw_correction_note_reaches_prompt_but_never_elevates_risk(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    """A raw, automatically-written `kind="correction"` row (e.g. from a
    past rep edit) is informational context for the drafter, but must NOT
    count as a risk flag -- otherwise any merchant with so much as one past
    rep edit would read as elevated risk forever after (module docstring
    point 3 in `pipeline.py`).
    """
    db_path = tmp_path / "t.db"
    case = await client.get_case("CASE-1002")
    memory.record_correction(case, "orig", "final", feedback="tone it down", db_path=db_path)
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["A00360"]),
            _extraction(),
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.risk_tier == RiskTier.LOW.value
    draft_prompt = transport.calls[-1]["messages"][-1]["content"]
    assert "Rep correction with feedback: tone it down" in draft_prompt


async def test_prior_claim_frequency_elevates_risk(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    """Three prior cases for the same merchant within the 90-day window hit
    `HIGH_CLAIM_FREQUENCY_THRESHOLD` and elevate risk.

    Note this does NOT exercise `exclude_case_id`'s self-counting guard: the
    case being processed (CASE-1002) gets a real wall-clock `created_at`
    from `store.create_case`, which -- given this test's historical injected
    `now=NOW` (2026-03-25) -- already falls outside `merchant_context`'s
    upper bound (`created_at <= now`) regardless of exclusion. The
    self-counting guard itself is covered directly by
    `tests/test_memory.py::test_merchant_context_exclude_case_id_excludes_
    itself_from_frequency`, where `now` and the seeded rows' timestamps are
    controlled precisely enough to isolate it.
    """
    db_path = tmp_path / "t.db"
    # `create_case` always stamps `created_at` with the real wall clock
    # (`store._now()`), independent of the `now=NOW` this test injects into
    # `process_case` for eligibility date math -- so these seeded "prior"
    # rows are backdated explicitly to fall inside NOW's trailing-90-day
    # window (same reasoning `tests/test_memory.py`'s boundary tests use).
    for i in range(3):
        case_id = f"CASE-PRIOR-{i}"
        create_case(case_id, merchant_id="283959", db_path=db_path)
        conn = get_connection(db_path)
        try:
            conn.execute(
                "UPDATE cases SET created_at = ? WHERE case_id = ?",
                ((NOW - timedelta(days=10)).isoformat(), case_id),
            )
            conn.commit()
        finally:
            conn.close()

    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["A00360"]),
            _extraction(),
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.risk_tier == RiskTier.ELEVATED.value


async def test_no_merchant_id_case_skips_memory_lookup_gracefully(tmp_path: Path):
    """A case with no `user_id` (module docstring point 3) must not crash
    `process_case` -- memory lookup is skipped, `cases.merchant_id` stays
    NULL, risk tiering gets an empty `MerchantMemory`, and the drafter's
    prompt gets the explicit "no merchant identifier" marker rather than a
    misleading zero-frequency claim.
    """
    db_path = tmp_path / "t.db"
    client = _NoUserIdClient(FixtureClient(include_synthetic=True))
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["A00360"]),
            _extraction(),
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.risk_tier == RiskTier.LOW.value
    row = get_case("CASE-1002", db_path=db_path)
    assert row["merchant_id"] is None
    draft_prompt = transport.calls[-1]["messages"][-1]["content"]
    assert NO_MERCHANT_ID_MEMORY_CONTEXT in draft_prompt


# --- never sends anything itself ---------------------------------------------


def test_pipeline_module_never_calls_outbound_or_action_apis():
    """Static guard for the core safety invariant: `process_case` must never
    send an email, submit a reimbursement, or record an outbound action --
    that's the approval-endpoint and outbound-guard layer's territory. Every
    path here ends in PENDING_REVIEW or ESCALATED.
    """
    pipeline_path = Path(__file__).resolve().parents[1] / "src" / "claimpilot" / "pipeline.py"
    source = pipeline_path.read_text()
    # Check for actual call sites (`.foo(`), not the docstring's prose
    # explaining that these calls must never appear.
    assert ".send_email(" not in source
    assert ".submit_reimbursement(" not in source
    assert ".record_action(" not in source


# --- claim-scope mismatch escalates ------------------------------------------


async def test_whole_order_claim_with_partial_evidence_escalates(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    """The real CASE-1002 shape: the customer's screenshot asks to be
    refunded "in its entirety", the vision gate confirms one SKU out of three
    priced invoice lines. Without this check the case pays for one item
    against a whole-order claim, at high confidence, with nothing flagged.
    """
    db_path = tmp_path / "t.db"
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(
                matched_skus=["A00360"],
                confidence=0.95,  # high -- validation itself would happily PROCEED
                claimed_scope="entire_order",
                scope_note="refund me in its entirety",
            ),
            _extraction(),  # unreadable -> invoice audit stays out of the way
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.decision == "request_info"
    assert recommendation.amount == Decimal("0")
    assert get_status("CASE-1002", db_path=db_path) == CaseState.ESCALATED

    events = _audit_events(db_path, "CASE-1002")
    assert events[-1] == "gate:claim_scope_mismatch"

    import json

    reason = json.loads(get_audit_log("CASE-1002", db_path=db_path)[-1]["payload_json"])["reason"]
    assert "entire order" in reason
    assert "3 priced line item(s)" in reason  # $0 lines excluded; 1002 has 3 priced
    assert "refund me in its entirety" in reason


async def test_consistent_scope_still_approves(client: _NoNetworkBytesClient, tmp_path: Path):
    """The check must not escalate a claim whose breadth matches the
    evidence -- otherwise every case ends up in front of a human.
    """
    db_path = tmp_path / "t.db"
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(
                matched_skus=["A00360"],
                claimed_scope="single_item",
                scope_note="the disinfectant bottle was cracked",
            ),
            _extraction(),
            _draft(),
        ]
    )

    recommendation = await process_case(
        "CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW
    )

    assert recommendation.decision == "approve"
    assert recommendation.amount == Decimal("24.99")
    assert get_status("CASE-1002", db_path=db_path) == CaseState.PENDING_REVIEW


async def test_invoice_discrepancy_outranks_scope_but_records_both(
    client: _NoNetworkBytesClient, tmp_path: Path
):
    """When both fire, the invoice discrepancy is the reported reason (it
    names two conflicting figures, which is more actionable) -- but the
    scope finding must still be attached to the audit payload rather than
    lost to short-circuiting.
    """
    db_path = tmp_path / "t.db"
    transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(
                matched_skus=["A00360"],
                claimed_scope="entire_order",
                scope_note="everything was soaked",
            ),
            _extraction(
                readable=True,
                currency="USD",
                line_items=[
                    {
                        "description": "CleanBoss Botanical Disinfectant & Cleaner 24oz 2 Pack",
                        "sku": "A00360",
                        "quantity": 1,
                        "unit_price": 19.99,
                        "line_total": 19.99,
                        "line_discount": 0.0,
                    }
                ],
                note=None,
            ),
            _draft(),
        ]
    )

    await process_case("CASE-1002", client=client, transport=transport, db_path=db_path, now=NOW)

    assert get_status("CASE-1002", db_path=db_path) == CaseState.ESCALATED
    events = _audit_events(db_path, "CASE-1002")
    assert events[-1] == "gate:invoice_audit_discrepancy"

    import json

    payload = json.loads(get_audit_log("CASE-1002", db_path=db_path)[-1]["payload_json"])
    assert payload["codes"] == ["PRICE_MISMATCH"]
    assert "claim_scope_note" in payload  # not lost
    assert "entire order" in payload["claim_scope_note"]
