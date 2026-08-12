"""Memory demo wiring -- the self-evolving memory loop, end to end.

This is the demo's "money shot": prove that a rep's pushback feedback on one
case doesn't just redraft that one email -- it survives as a durable,
decision-free policy note (via `evolve.distill_feedback` ->
`memory.record_policy_note`) that a *later* case for the *same merchant*
actually sees in its own drafting prompt (via `memory.merchant_context()` ->
`DraftInputs.memory_context`).

Sequence (all against one shared `tmp_path` SQLite database, mirroring
`tests/test_pipeline.py` and `tests/test_web.py`'s test-injection
conventions -- no real network/LLM/on-disk-DB access anywhere in this file):

1. Process CASE-1001 (real fixture case, `user_id="334430"`, account
   "Best Paw Nutrition") via `pipeline.process_case` directly. Only 3
   attachments -> one evidence gap -> `request_info`, `PENDING_REVIEW` (same
   scripted shape as `tests/test_pipeline.py`'s
   `test_case_1001_evidence_gap_ends_pending_review_request_info`).
2. `POST /cases/CASE-1001/pushback` with the rep feedback text
   "mention their account manager Dana; drop the second apology". The
   pushback endpoint's own `FakeTransport` is scripted with two responses in
   the order `web/app.py`'s pushback handler actually makes them: the
   redraft (`draft()`) first, then the feedback distiller's own
   `structured_call` (`evolve.distill_feedback`) second. The distiller is
   scripted to return one merchant-scoped, decision-free `DistilledNote`
   that plausibly reflects the feedback ("Mention the merchant's dedicated
   account manager by name.").
3. Confirm the note actually round-tripped into durable storage via
   `memory.merchant_context("334430", ...)` -- not just "the request
   returned 303".
4. `POST /cases/CASE-1001/approve` (no edit) to close out the first case's
   review -- keeps the demo's queue state clean; also proves the loop
   doesn't require an edit-on-approve to have already worked.
5. Process CASE-9003-REPEAT (synthetic, added to `fixtures/synthetic.json`
   for this task, same `user_id="334430"` as CASE-1001 -- a same-merchant
   repeat case with no independent second real fixture merchant needed) via
   `pipeline.process_case`, with its own fresh `FakeTransport`.
   **Deliberately given a distinct `account_name`** ("Best Paw Nutrition -
   Repeat Claim" vs. CASE-1001's "Best Paw Nutrition") -- the task brief
   suggested reusing both `user_id` and `account_name`, but doing so would
   let this test pass even if `memory.py`/`pipeline.py` were consistently
   (bug-for-bug) rekeyed onto `account_name` instead of `user_id`
   (`memory.py` module docstring point 1 is exactly the mistake this
   guards against). With only `user_id` shared, the money-shot assertion
   below can only pass if `user_id` is genuinely the merchant key threaded
   through `record_policy_note` -> `merchant_context` -> `DraftInputs.
   memory_context`.
6. **The assertion that would actually catch a broken wire:** the exact
   distilled note text appears in the *second* case's draft prompt (the
   `messages` sent to the fake transport's final `create()` call) -- proving
   the full round trip: pushback feedback -> `distill_feedback` -> persisted
   `kind="policy"` memory row -> `merchant_context()` -> `DraftInputs.
   memory_context` -> the prompt text actually sent for a *different*
   case's drafting call. A vacuous `len(calls) > 0` assertion would not
   catch e.g. someone forgetting to wire `memory_context` into
   `DraftInputs`, or `merchant_context`/`record_policy_note` keying on the
   wrong merchant identifier -- this one would.

Known, documented interaction (see `evolve.py` module docstring point 7 and
`pipeline.py` module docstring point 3): once a merchant-scoped policy note
exists, `risk.tier()` treats the resulting non-empty `MerchantMemory.flags`
as one triggered risk factor, so CASE-9003-REPEAT's risk tier reads
`ELEVATED`, not `LOW` -- expected/documented behavior given the memory
system's design, not a bug this test should paper over. Asserted explicitly below so
a future change to that behavior is caught rather than silently accepted.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from fastapi.testclient import TestClient

from claimpilot import memory
from claimpilot.clients.fixtures import FixtureClient
from claimpilot.llm import TransportResult
from claimpilot.models import CaseState
from claimpilot.pipeline import process_case
from claimpilot.risk import RiskTier
from claimpilot.store import get_status
from claimpilot.web.app import create_app
from tests.test_llm import FakeTransport

NOW = datetime(2026, 3, 25, tzinfo=timezone.utc)

MERCHANT_ID = "334430"  # CASE-1001's real Case.user_id (Best Paw Nutrition)
PUSHBACK_FEEDBACK = "mention their account manager Dana; drop the second apology"
DISTILLED_NOTE_TEXT = "Mention the merchant's dedicated account manager by name."


class _RecordingClient:
    """Wraps a real `ShipBobClient`, same convention as `tests/test_web.py`'s
    `_RecordingClient`: canned attachment bytes (no real network call), and
    `send_email`/`submit_reimbursement` recorded in-memory instead of
    writing to `FixtureClient`'s real `outbox/outbox.jsonl` file.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.sent_emails: list[dict] = []
        self.reimbursements: list[dict] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def get_attachment_bytes(self, attachment) -> bytes:
        return b"\x89PNG\r\n\x1a\n" + attachment.attachment_id.encode()

    async def send_email(self, case_id, to, subject, body) -> dict:
        self.sent_emails.append({"case_id": case_id, "to": to, "subject": subject, "body": body})
        return {"success": True, "case_id": case_id}

    async def submit_reimbursement(self, case_id, order_id, user_id, shipment_id, product_name, amount) -> dict:
        self.reimbursements.append(
            {
                "case_id": case_id,
                "order_id": order_id,
                "user_id": user_id,
                "shipment_id": shipment_id,
                "product_name": product_name,
                "amount": amount,
            }
        )
        return {"status": "submitted"}


def _classification(category: str, *, confidence: float = 0.95, usable: bool = True) -> TransportResult:
    return TransportResult(
        tool_input={"category": category, "confidence": confidence, "usable": usable, "quality_issue": None},
        input_tokens=1,
        output_tokens=1,
        raw_content=[],
    )


def _validation(*, matched_skus: list[str], confidence: float = 0.9) -> TransportResult:
    judgment = {"passed": True, "confidence": confidence, "note": "looks consistent with the claim"}
    return TransportResult(
        tool_input={
            "damage_visible": judgment,
            "product_identifiable": judgment,
            "product_on_invoice": judgment,
            "packaging_documented": judgment,
            "matched_skus": matched_skus,
        },
        input_tokens=1,
        output_tokens=1,
        raw_content=[],
    )


def _draft(email_draft: str = "Dear customer, ...", rationale: str = "Deterministic pipeline decision.") -> TransportResult:
    return TransportResult(
        tool_input={"rationale": rationale, "email_draft": email_draft},
        input_tokens=1,
        output_tokens=1,
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


def _distill_notes(notes: list[dict]) -> TransportResult:
    return TransportResult(tool_input={"notes": notes}, input_tokens=1, output_tokens=1, raw_content=[])


async def test_pushback_feedback_survives_into_a_later_same_merchant_case_prompt(tmp_path: Path):
    db_path = tmp_path / "t.db"
    client = _RecordingClient(FixtureClient(include_synthetic=True))

    # --- Step 1: process the first case (CASE-1001) -------------------------
    # 3 real attachments -> one evidence gap -> request_info, PENDING_REVIEW
    # (same scripted shape as tests/test_pipeline.py's CASE-1001 test).
    first_case_transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _draft(),
        ]
    )
    first_recommendation = await process_case(
        "CASE-1001", client=client, transport=first_case_transport, db_path=db_path, now=NOW
    )
    assert first_recommendation.decision == "request_info"
    assert get_status("CASE-1001", db_path=db_path) == CaseState.PENDING_REVIEW

    # --- Step 2: push back with the rep's feedback ---------------------------
    # The pushback endpoint makes exactly two structured_call()s, in this
    # order: the redraft (draft()), then evolve.distill_feedback's own call
    # -- see web/app.py module docstring point 12 and
    # tests/test_web.py::test_pushback_triggers_distiller_and_note_becomes_available
    # for the same ordering.
    pushback_transport = FakeTransport(
        [
            _draft(
                email_draft="Dear customer, thanks for your patience -- revised per your feedback.",
                rationale="Redrafted after rep pushback.",
            ),
            _distill_notes([{"content": DISTILLED_NOTE_TEXT, "scope": "merchant"}]),
        ]
    )
    with TestClient(create_app(client=client, transport=pushback_transport, db_path=db_path)) as tc:
        pushback_resp = tc.post(
            "/cases/CASE-1001/pushback",
            data={"feedback": PUSHBACK_FEEDBACK},
            follow_redirects=False,
        )
    assert pushback_resp.status_code == 303

    # --- Step 2b: confirm the note actually round-tripped into durable
    # storage -- not just "the request returned 303". This is the real
    # memory/store plumbing (memory.merchant_context), not a mock.
    context_after_pushback = memory.merchant_context(MERCHANT_ID, db_path=db_path)
    assert context_after_pushback.policy_notes == [DISTILLED_NOTE_TEXT]

    # --- Step 3: approve the (redrafted) CASE-1001 as-is ---------------------
    # No edit here -- the pushback step above is already the main
    # distillation trigger for this demo; approve just closes out the
    # review queue for a clean end state. request_info decision -> email
    # sent, no reimbursement submitted (mirrors
    # tests/test_web.py::test_approving_request_info_sends_email_but_no_reimbursement).
    with TestClient(create_app(client=client, transport=FakeTransport([]), db_path=db_path)) as tc:
        approve_resp = tc.post("/cases/CASE-1001/approve", data={"action": "approve"}, follow_redirects=False)
    assert approve_resp.status_code == 303
    assert get_status("CASE-1001", db_path=db_path) == CaseState.SENT
    assert len(client.sent_emails) == 1
    assert client.reimbursements == []  # decision was request_info, not approve

    # --- Step 4: process the SECOND case for the SAME merchant ---------------
    # CASE-9003-REPEAT: synthetic repeat case added to fixtures/synthetic.json
    # for this task, sharing CASE-1001's real user_id/account_name so
    # merchant_context correlates the two. 4 attachments (all evidence
    # categories) -> clean approve path.
    second_case_transport = FakeTransport(
        [
            _classification("ORDER_PROOF"),
            _classification("CUSTOMER_CONFIRMATION"),
            _classification("PRODUCT_PHOTO"),
            _classification("PACKAGING_PHOTO"),
            _validation(matched_skus=["HG-FRCAST-KITTEDROLL-3PK"]),
            _extraction(),
            _draft(),
        ]
    )
    second_recommendation = await process_case(
        "CASE-9003-REPEAT", client=client, transport=second_case_transport, db_path=db_path, now=NOW
    )
    assert second_recommendation.decision == "approve"
    assert second_recommendation.amount == Decimal("42.00")

    # --- THE MONEY SHOT -------------------------------------------------------
    # The distilled note, born from case 1's pushback feedback, must appear
    # in the SECOND case's draft prompt -- the literal text, not just "some
    # call happened". This is what would fail if e.g. DraftInputs.
    # memory_context were never wired up, or merchant_context()/
    # record_policy_note keyed on the wrong merchant identifier (account_name
    # instead of user_id, or no exclude/scope filtering at all).
    #
    # Deliberately asserted via the exact "  - <note>" bullet rendering
    # `MemoryContext.to_prompt_text()` produces for `policy_notes` (see
    # `memory.py`), NOT a bare substring check -- a merchant-scoped policy
    # note also independently leaks the same literal text into the prompt's
    # separate "Risk gate: ... flags: merchant memory flag: <note>." line
    # (`draft.py`'s `_risk_text`, fed from `RiskAssessment.flags`, which is
    # ALSO sourced from `MemoryContext.policy_notes` -- see `risk.py`).
    # A bare `DISTILLED_NOTE_TEXT in prompt` check was verified (by
    # temporarily blanking `pipeline.py`'s `memory_context_text` assignment)
    # to still pass via that risk-flags leak even with `memory_context`
    # completely unwired -- so it would NOT have caught that regression.
    # The "  - " bullet prefix is unique to `to_prompt_text()`'s
    # `policy_notes` rendering (the risk-gate line uses a different prefix,
    # "merchant memory flag: ", and "; "-joins multiple flags instead), so
    # this assertion actually isolates the `memory_context` channel.
    second_case_draft_prompt = second_case_transport.calls[-1]["messages"][-1]["content"]
    assert f"  - {DISTILLED_NOTE_TEXT}" in second_case_draft_prompt

    # Known, documented interaction (evolve.py module docstring point 7):
    # a purely stylistic merchant-scoped policy note still counts as one
    # triggered risk factor via MerchantMemory.flags, so the repeat case's
    # risk tier reads ELEVATED, not LOW -- asserted explicitly (not assumed
    # away) so a future change to that behavior is caught, not silently
    # accepted.
    assert second_recommendation.risk_tier == RiskTier.ELEVATED.value
