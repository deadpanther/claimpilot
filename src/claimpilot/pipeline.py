"""Pipeline orchestrator.

The integration point that wires every gate/module together
into one end-to-end pipeline: intake -> eligibility -> evidence ->
validation -> calc -> a persisted `Recommendation`. This module owns *no*
business logic of its own (eligibility rules, calc math, risk scoring,
prompt text all live in their own modules) -- it only decides *which*
module to call next, with what data, and makes sure every step (including
every early exit) is durably recorded via `claimpilot.store`.

**Non-negotiable invariant:** `process_case` never
sends anything. Every path -- deny, request more info, escalate, or approve
-- ends with the case sitting in `PENDING_REVIEW` or `ESCALATED`, waiting for
a human rep (the review UI) to actually send something (subject to
the outbound guard). Nothing in this module calls
`client.send_email`, `client.submit_reimbursement`, or
`store.record_action` -- those live exclusively in the review UI and the outbound guard.

Design decisions / judgment calls (documented here for the full
reasoning behind each):

1. **Every exit funnels through `_exit()`.** `draft()` (LLM call) ->
   `store.save_recommendation()` -> `store.transition()` happen together, in
   that order, for all seven ways `process_case` can end (deny,
   insured-escalate, evidence-gap request_info, validation request_info,
   validation-escalate, calc-exception-escalate, approve). Centralizing this
   guarantees no branch can accidentally return a `Recommendation` that was
   never persisted, or leave a case's `cases.status` one step behind its
   saved recommendation. The recommendation is saved *before* the state
   transition commits, so nothing else watching `cases.status` can ever
   observe a "ready for review" state with no recommendation attached yet.
2. **Risk tier and merchant memory context are computed once, right after
   the shipment is fetched, and reused for every exit branch** -- not only
   the final approve path the plan's step 6 describes risk tiering for.
   `DraftInputs.risk_assessment` is a required (non-optional) field, so
   even a `deny` produced by the eligibility gate needs *some* risk
   assessment to hand `draft()`; `tier()` is pure and cheap (no I/O, no
   LLM), so computing it unconditionally is free and gives a rep reviewing
   *any* recommendation type the same risk context. Likewise, every exit
   calls `draft()`, so every exit needs the same `memory_context` text.
3. **`MerchantMemory` is built from `claimpilot.memory.
   merchant_context()`.** `MerchantMemory.flags` is populated from
   `MemoryContext.policy_notes` ONLY -- durable, curated notes (written
   deliberately via `memory.record_policy_note`, e.g. by the
   feedback distiller), not raw per-case `MemoryContext.recent_notes`
   (automatic corrections/pushback records, written on essentially every
   rep edit). Using the raw, uncurated corrections as risk flags would mean
   any merchant with so much as one prior rep edit reads as ELEVATED risk
   forever after -- which, given `record_correction` runs automatically,
   would eventually be true of every active merchant and would make the
   risk tier meaningless. Global policy notes (`MemoryContext.
   global_policy_notes`) are also excluded from `flags`: they apply to
   every merchant equally, so they carry no information about *this*
   merchant being riskier than any other. See `claimpilot.memory`'s module
   docstring points 2-3 for the full rationale (same reasoning, stated from
   the memory module's side). When `case.user_id` is `None` (no merchant
   identifier available -- shouldn't happen for real ShipBob data, but
   `Case.user_id` is optional), memory lookup is skipped entirely rather
   than guessing at a merchant_id: `MerchantMemory()` stays empty and
   `memory_context` says so explicitly, rather than reporting a
   `claims_last_90_days=0` that would misleadingly read as "confirmed zero
   history" instead of "unknown."
4. **Confidence sourcing splits into two families**, per the plan's
   "confidence honesty" principle (LLM self-reported confidence never
   drives approval, only ever pushes toward more scrutiny):
   - `GATE_DECISION_CONFIDENCE` (1.0) is used wherever the pipeline's own
     deterministic control flow -- not an LLM judgment -- decided the
     outcome with certainty: eligibility close (deny), insured routing,
     evidence gaps found, validation `REQUEST_INFO` (a judgment failed
     outright), and the calc-exception escalation (a data-integrity
     problem, not a vision-confidence problem). These are all "the rule
     fired, full stop" outcomes; reporting anything less than full
     confidence in *that fact* would be misleading.
   - `_min_judgment_confidence()` (the minimum confidence across the four
     `ValidationResult` judgments) is used for validation `ESCALATED` and
     for the final `approve`. For `ESCALATED`, this is literally the reason
     for escalating (matches the plan's "escalated because: damage
     visibility 0.62" framing) -- reporting `GATE_DECISION_CONFIDENCE` there
     would hide the very signal that triggered human review. For `approve`,
     it's the weakest link among everything that had to be true for the
     claim to go through -- an honest "how sure are we" number, never used
     to gate the decision itself (the decision was already `approve` before
     this number is computed).
5. **`matched_skus` -> `DamagedItem` quantity convention: count
   occurrences.** `ValidationResult.matched_skus` is a flat `list[str]`
   with no explicit per-item quantity field. We treat each occurrence of a
   SKU in that list as one damaged unit, via `Counter(matched_skus)` ->
   one `DamagedItem(sku, quantity)` per distinct SKU. This is the simplest
   convention consistent with the model naming the same SKU once per
   damaged unit it identifies in the photos; a future task could extend
   `ValidationResult` with an explicit quantity field if this proves too
   coarse.
6. **Calc exceptions escalate rather than crash the pipeline.**
   `reimbursement()` raising `ItemNotOnInvoice`/`QuantityExceedsInvoice`
   means the vision model named a SKU that doesn't reconcile with the
   invoice -- a sign something is wrong with the classification, not
   something a deterministic formula should paper over. We route this to
   `ESCALATED` (need a human to look at the mismatch) rather than letting
   the exception propagate and crash `process_case` -- consistent with the
   "always ends in pending_review or escalated" invariant. **Implementation
   note driven by `store.LEGAL_TRANSITIONS`:** `CALC -> ESCALATED` is *not*
   a legal transition (`CALC` only leads to `PENDING_REVIEW`), so
   `reimbursement()` is called *before* transitioning out of `VALIDATION`
   (`VALIDATION -> ESCALATED` *is* legal) -- the case only advances into the
   `CALC` state once `reimbursement()` has already succeeded.
7. **CASE-1004's "already closed" signal rides on the existing
   `INTAKE -> ELIGIBILITY` transition, not a new store function.**
   `store.py` has no standalone "write an audit row without changing
   state" primitive -- `transition()` is the only thing that inserts
   `audit_log` rows, and every case makes this exact transition early
   (before eligibility is even evaluated). Rather than invent a new store
   function for one flag, `already_closed` (and the raw `case.status` at
   intake) are included in this transition's `payload` for every case
   (`False`/actual-non-Closed-status for the normal case, `True`/`"Closed"`
   for CASE-1004) -- a uniform, always-present audit field rather than a
   special-cased extra write. Processing a case directly by
   ID is always allowed even when already `Closed` -- only the
   `/cases` list-scan (the review queue) skips closed cases; this function never
   skips.
8. **Insured escalation and validation-`ESCALATED`/calc-exception
   escalation all report `decision="request_info"`.** None of
   `approve|deny|request_info` cleanly names "this needs a human to route
   it" or "this needs a human to resolve an inconsistency" -- `request_info`
   is the least-wrong fit (it already means "not yet resolved, needs more
   input before a payout can be decided"), with `amount=Decimal("0")` and a
   rationale/prompt context that makes the *real* reason (insurance
   routing, low validation confidence, or a calc data mismatch) explicit
   rather than implying we're literally asking the customer for more
   evidence. The review UI treats `ESCALATED`-state cases as
   their own queue regardless of the `decision` label on their
   recommendation, precisely because this label is a stretch.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from claimpilot import memory, store
from claimpilot.calc import CalcResult, DamagedItem, ItemNotOnInvoice, QuantityExceedsInvoice, reimbursement
from claimpilot.clients.base import ShipBobClient, get_client
from claimpilot.config import settings
from claimpilot.draft import DraftInputs, draft
from claimpilot.gates.eligibility import EligibilityResult, check_eligibility
from claimpilot.gates.evidence import AttachmentClassification, Gap, classify_attachment, evidence_gaps
from claimpilot.gates.invoice_audit import (
    ExtractedInvoice,
    InvoiceAudit,
    audit_invoice,
    extract_invoice,
)
from claimpilot.gates.validation import (
    ValidationOutcome,
    ValidationResult,
    check_affected_count_mismatch,
    check_claim_scope_mismatch,
    combine_validation,
    validate_damage,
)
from claimpilot.llm import Transport
from claimpilot.memory import MemoryContext
from claimpilot.models import Case, CaseState, EvidenceItem, Invoice, Recommendation
from claimpilot.risk import MerchantMemory, RiskAssessment, tier

# Confidence reported on recommendations produced by deterministic,
# rule-based branches of this pipeline (eligibility close, insured routing,
# evidence gaps, validation REQUEST_INFO, calc-exception escalation) -- the
# pipeline's own control flow decided these outcomes with certainty, not an
# LLM's self-reported score. Kept distinct from `_min_judgment_confidence()`,
# used for validation ESCALATED and the final approve (see module docstring
# point 4 for the full rationale).
GATE_DECISION_CONFIDENCE = 1.0


def _min_judgment_confidence(result: ValidationResult) -> float:
    """The weakest of the four `ValidationResult` judgment confidences.

    Used for validation `ESCALATED` (this number is literally why the case
    escalated) and for the final `approve` (the weakest link supporting an
    approval that has already been decided) -- never to decide whether to
    approve in the first place. See module docstring point 4.
    """
    return min(
        result.damage_visible.confidence,
        result.product_identifiable.confidence,
        result.product_on_invoice.confidence,
        result.packaging_documented.confidence,
    )


async def _exit(
    case: Case,
    draft_inputs: DraftInputs,
    to_state: CaseState,
    *,
    event: str,
    transport: Transport | None,
    db_path: Path | str | None,
    payload: dict | None = None,
    eligibility: EligibilityResult | None = None,
    evidence_gaps: list[Gap] | None = None,
    validation: ValidationResult | None = None,
    calc: CalcResult | None = None,
    invoice_audit: InvoiceAudit | None = None,
) -> Recommendation:
    """Draft, persist, and transition -- the one path every terminal branch
    of `process_case` takes (module docstring point 1).

    `eligibility`/`evidence_gaps`/`validation`/`calc` are whichever gate
    objects the calling branch actually computed on its way here (e.g. the
    deny-on-ineligible path only ever has `eligibility`) -- passed straight
    through to `store.save_gate_results` alongside the existing
    `save_recommendation` call. This exists because the outbound guard must
    re-verify an approve decision against what the gates *actually* found,
    not against the drafted `Recommendation` alone (see `store.py`'s
    `GateResults` docstring for the full rationale).

    Order matters: the recommendation and gate results are saved *before*
    the state transition commits, so a case never briefly sits in
    `PENDING_REVIEW`/`ESCALATED` with no recommendation/gate-results
    attached for a concurrent reader to observe.
    """
    recommendation = await draft(draft_inputs, transport=transport, db_path=db_path)
    store.save_recommendation(case.case_id, recommendation, db_path=db_path)
    store.save_gate_results(
        case.case_id,
        eligibility=eligibility,
        evidence_gaps=evidence_gaps,
        validation=validation,
        calc=calc,
        invoice_audit=invoice_audit,
        db_path=db_path,
    )
    store.transition(case.case_id, to_state, actor="system", event=event, payload=payload, db_path=db_path)
    return recommendation


async def _audit_retail_invoice(
    case_id: str,
    invoice: Invoice,
    matched_skus: list[str],
    classified_with_bytes: list[tuple[AttachmentClassification, bytes]],
    *,
    transport: Transport | None,
    db_path: Path | str | None,
) -> InvoiceAudit:
    """Read the merchant's `ORDER_PROOF` attachment and reconcile it against
    ShipBob's invoice for the damaged SKUs.

    Every failure mode here degrades to an unverified `InvoiceAudit` rather
    than propagating, because this check is additive to a pipeline that
    already worked without it (`gates/invoice_audit.py` module docstring
    point 1): a flaky vision call on a supplementary verification step must
    not take down a case that would otherwise have produced a perfectly good
    recommendation from the API data. The unverified reason is persisted and
    shown to the rep, so "we couldn't check" is visible rather than silently
    indistinguishable from "we checked and it was fine".

    Picks the highest-confidence usable `ORDER_PROOF` when a case has more
    than one -- multiple order-proof attachments are legal (a merchant can
    upload both an order confirmation and a receipt) and the classifier's
    own confidence is the only ranking signal available.
    """
    if not settings.invoice_audit_enabled:
        return InvoiceAudit(verified=False, reason="retail-invoice audit is disabled by configuration")

    order_proofs = [
        (classification, content)
        for classification, content in classified_with_bytes
        if classification.category == EvidenceItem.ORDER_PROOF and classification.usable
    ]
    if not order_proofs:
        return InvoiceAudit(
            verified=False, reason="no usable order-proof attachment was available to read prices from"
        )
    order_proofs.sort(key=lambda pair: pair[0].confidence, reverse=True)
    _, order_proof_bytes = order_proofs[0]

    extracted: ExtractedInvoice | None = None
    extraction_error: str | None = None
    try:
        extracted = await extract_invoice(
            case_id, order_proof_bytes, transport=transport, db_path=db_path
        )
    except Exception as exc:  # noqa: BLE001 -- see docstring: never fail the case on this
        extraction_error = f"could not read the retail invoice ({type(exc).__name__})"

    return audit_invoice(extracted, invoice, matched_skus, extraction_error=extraction_error)


async def process_case(
    case_id: str,
    *,
    client: ShipBobClient | None = None,
    transport: Transport | None = None,
    db_path: Path | str | None = None,
    now: datetime | None = None,
) -> Recommendation:
    """Run one case through intake -> eligibility -> evidence -> validation
    -> calc, short-circuiting to the right draft type on any gate failure.

    Always ends with the case in `PENDING_REVIEW` or `ESCALATED` and a
    `Recommendation` persisted via `store.save_recommendation` -- never
    sends anything itself (see module docstring).

    Args:
        case_id: which case to process.
        client: overrides `get_client()` -- tests inject a client wrapping
            `FixtureClient` (see `tests/test_pipeline.py`) so attachment
            *bytes* never require real network access.
        transport: overrides the default LLM transport -- tests always pass
            a fake transport scripted to produce the scenario's expected
            gate outcome.
        db_path: overrides the default on-disk SQLite database -- tests
            pass a `tmp_path` file so they never touch the real database.
        now: overrides "today" for the eligibility claim-window
            calculation -- tests fix this so window math is deterministic
            regardless of when the test actually runs.
    """
    client = client or get_client()
    now = now or datetime.now(timezone.utc)

    # --- 1. Intake ------------------------------------------------------
    case = await client.get_case(case_id)
    try:
        store.create_case(case_id, merchant_id=case.user_id, db_path=db_path)
    except store.CaseAlreadyExistsError:
        # The row can legitimately already exist -- the review UI's "Fetch
        # cases" action creates `INTAKE` rows up front so the queue can show
        # what's been pulled from the API before anything is processed. The
        # intent here is "ensure this case is tracked", not "insert a
        # brand-new row", so an existing one is fine to continue from.
        #
        # This is not a way to silently re-run a decided case: the very next
        # step transitions INTAKE -> ELIGIBILITY, and `store.transition`
        # rejects that from any later state via `LEGAL_TRANSITIONS`. A case a
        # rep has already acted on still fails loudly, just at the state
        # machine (where that rule belongs) rather than at the insert.
        pass

    # --- 2. Eligibility ---------------------------------------------------
    shipment = await client.get_shipment(case.shipment_id)
    # Computed once, reused by every exit branch below (module docstring
    # point 2). `exclude_case_id=case_id` keeps this case's own just-created
    # `cases` row (inserted immediately above) from counting toward its own
    # claim-frequency history (see `claimpilot.memory`'s module docstring
    # point 6 for why that matters).
    if case.user_id is not None:
        memory_ctx: MemoryContext = memory.merchant_context(
            case.user_id, exclude_case_id=case_id, now=now, db_path=db_path
        )
        memory_context_text = memory_ctx.to_prompt_text()
        merchant_memory = MerchantMemory(
            claims_last_90_days=memory_ctx.claim_frequency_90d, flags=memory_ctx.policy_notes
        )
    else:
        memory_context_text = memory.NO_MERCHANT_ID_MEMORY_CONTEXT
        merchant_memory = MerchantMemory()
    risk_assessment: RiskAssessment = tier(shipment, merchant_memory)

    already_closed = case.status == "Closed"
    store.transition(
        case_id,
        CaseState.ELIGIBILITY,
        actor="system",
        event="transition:intake->eligibility",
        payload={"already_closed": already_closed, "case_status_at_intake": case.status},
        db_path=db_path,
    )

    eligibility = check_eligibility(case, shipment, now=now)

    if eligibility.route == "close":
        draft_inputs = DraftInputs(
            case=case,
            decision="deny",
            amount=Decimal("0"),
            confidence=GATE_DECISION_CONFIDENCE,
            risk_assessment=risk_assessment,
            memory_context=memory_context_text,
            eligibility_result=eligibility,
        )
        return await _exit(
            case,
            draft_inputs,
            CaseState.PENDING_REVIEW,
            event="gate:eligibility_denied",
            payload={"reason": eligibility.reason},
            transport=transport,
            db_path=db_path,
            eligibility=eligibility,
        )

    if eligibility.route == "insured_process":
        # No clean fit in approve|deny|request_info for "route to the
        # separate insurance process" -- request_info is the least-wrong
        # label (module docstring point 8).
        draft_inputs = DraftInputs(
            case=case,
            decision="request_info",
            amount=Decimal("0"),
            confidence=GATE_DECISION_CONFIDENCE,
            risk_assessment=risk_assessment,
            memory_context=memory_context_text,
            eligibility_result=eligibility,
        )
        return await _exit(
            case,
            draft_inputs,
            CaseState.ESCALATED,
            event="gate:eligibility_insured",
            transport=transport,
            db_path=db_path,
            eligibility=eligibility,
        )

    # route == "process"
    store.transition(
        case_id,
        CaseState.EVIDENCE,
        actor="system",
        event="transition:eligibility->evidence",
        db_path=db_path,
    )

    # --- 3. Evidence ------------------------------------------------------
    attachments = await client.list_attachments(case_id)
    classified_with_bytes: list[tuple[AttachmentClassification, bytes]] = []
    for attachment in attachments:
        content = await client.get_attachment_bytes(attachment)
        classification = await classify_attachment(
            case_id, attachment, content, transport=transport, db_path=db_path
        )
        classified_with_bytes.append((classification, content))

    classified = [classification for classification, _ in classified_with_bytes]
    gaps = evidence_gaps(classified)

    if gaps:
        draft_inputs = DraftInputs(
            case=case,
            decision="request_info",
            amount=Decimal("0"),
            confidence=GATE_DECISION_CONFIDENCE,
            risk_assessment=risk_assessment,
            memory_context=memory_context_text,
            eligibility_result=eligibility,
            evidence_gaps=gaps,
        )
        return await _exit(
            case,
            draft_inputs,
            CaseState.PENDING_REVIEW,
            event="gate:evidence_gap",
            payload={"gaps": [g.item.value for g in gaps]},
            transport=transport,
            db_path=db_path,
            eligibility=eligibility,
            evidence_gaps=gaps,
        )

    store.transition(
        case_id,
        CaseState.VALIDATION,
        actor="system",
        event="transition:evidence->validation",
        db_path=db_path,
    )

    # --- 4. Validation ------------------------------------------------------
    # NOTE: deliberately does NOT call `client.get_order()` here. An earlier
    # version of this pipeline fetched it and then never used the result --
    # confirmed dead code via `ruff`'s F841 (unused-variable) during a full
    # codebase audit, not something any test exercised or depended on (see
    # `tests/test_http_client.py`/`tests/test_fixture_client.py` for
    # `get_order`'s own, still-valid, standalone client-level tests). Every
    # real decision downstream (damage validation's SKU matching, the
    # reimbursement calc) already correctly uses `invoice`, never `order`,
    # per the brief's "price at time of fulfillment" requirement -- so this
    # was a wasted real API call on every single case processed, not a
    # latent dependency. Removing it also means one less real network call
    # `clients/retry.py`'s retry-with-backoff policy has to cover.
    invoice = await client.generate_invoice(shipment_id=case.shipment_id, user_id=case.user_id)

    product_photos = [
        content
        for classification, content in classified_with_bytes
        if classification.category == EvidenceItem.PRODUCT_PHOTO and classification.usable
    ]
    packaging_photos = [
        content
        for classification, content in classified_with_bytes
        if classification.category == EvidenceItem.PACKAGING_PHOTO and classification.usable
    ]

    # The customer's own message about the damage. Passed to the validation
    # gate alongside the photos because it is frequently the clearest
    # statement of *which* items were affected -- without it, SKU matching
    # on a multi-item order of visually similar products was picking a
    # different line between runs, and `matched_skus` is what the payout is
    # computed from. Untrusted input; see `validate_damage`'s docstring.
    customer_confirmations = [
        content
        for classification, content in classified_with_bytes
        if classification.category == EvidenceItem.CUSTOMER_CONFIRMATION and classification.usable
    ]

    validation_result = await validate_damage(
        case_id,
        product_photos,
        packaging_photos,
        invoice,
        customer_confirmations=customer_confirmations,
        case_description=case.description,
        transport=transport,
        db_path=db_path,
    )
    validation_decision = combine_validation(validation_result)

    if validation_decision.outcome == ValidationOutcome.REQUEST_INFO:
        draft_inputs = DraftInputs(
            case=case,
            decision="request_info",
            amount=Decimal("0"),
            confidence=GATE_DECISION_CONFIDENCE,
            risk_assessment=risk_assessment,
            memory_context=memory_context_text,
            eligibility_result=eligibility,
            validation_decision=validation_decision,
        )
        return await _exit(
            case,
            draft_inputs,
            CaseState.PENDING_REVIEW,
            event="gate:validation_request_info",
            payload={"reason": validation_decision.reason},
            transport=transport,
            db_path=db_path,
            eligibility=eligibility,
            evidence_gaps=gaps,
            validation=validation_result,
        )

    if validation_decision.outcome == ValidationOutcome.ESCALATED:
        draft_inputs = DraftInputs(
            case=case,
            decision="request_info",
            amount=Decimal("0"),
            confidence=_min_judgment_confidence(validation_result),
            risk_assessment=risk_assessment,
            memory_context=memory_context_text,
            eligibility_result=eligibility,
            validation_decision=validation_decision,
        )
        return await _exit(
            case,
            draft_inputs,
            CaseState.ESCALATED,
            event="gate:validation_escalated",
            payload={"reason": validation_decision.reason},
            transport=transport,
            db_path=db_path,
            eligibility=eligibility,
            evidence_gaps=gaps,
            validation=validation_result,
        )

    # --- 5. Calc ------------------------------------------------------
    # `reimbursement()` is called while the case is still in VALIDATION
    # (not yet transitioned to CALC): CALC -> ESCALATED is not a legal
    # transition per `store.LEGAL_TRANSITIONS` (CALC only leads to
    # PENDING_REVIEW), so a calc failure must escalate from VALIDATION,
    # which does legally allow it (module docstring point 6).
    # Affected-count cross-check (gates/validation.py module docstring point
    # 9): a real, evidence-based audit of ShipBob's own fixture data found
    # every real case's description states how many items/orders were
    # affected. If the merchant's own words say more (or fewer) were
    # affected than the vision review actually confirmed, that discrepancy
    # goes to a human rather than being silently resolved either way --
    # never used to adjust the calc itself (see that function's docstring
    # for why: merchant text must never directly set a dollar amount).
    affected_count_mismatch = check_affected_count_mismatch(case.description, validation_result.matched_skus)
    if affected_count_mismatch is not None:
        draft_inputs = DraftInputs(
            case=case,
            decision="request_info",
            amount=Decimal("0"),
            confidence=GATE_DECISION_CONFIDENCE,
            risk_assessment=risk_assessment,
            memory_context=memory_context_text,
            eligibility_result=eligibility,
            validation_decision=validation_decision,
        )
        return await _exit(
            case,
            draft_inputs,
            CaseState.ESCALATED,
            event="gate:validation_affected_count_mismatch",
            payload={"reason": affected_count_mismatch},
            transport=transport,
            db_path=db_path,
            eligibility=eligibility,
            evidence_gaps=gaps,
            validation=validation_result,
        )

    # Retail-invoice reconciliation (see `gates/invoice_audit.py`'s module
    # docstring for why this exists and why it can't just replace the calc
    # basis). Runs here rather than earlier for two reasons: it needs
    # `matched_skus` to know which lines are even worth comparing, and
    # running it before the validation exits above would spend a vision call
    # on cases that are already heading to a human anyway.
    invoice_audit = await _audit_retail_invoice(
        case_id,
        invoice,
        validation_result.matched_skus,
        classified_with_bytes,
        transport=transport,
        db_path=db_path,
    )
    # Claim-scope cross-check (gates/validation.py's `check_claim_scope_
    # mismatch`): does the customer's own account of what was damaged agree
    # with what the evidence actually confirmed?
    #
    # Computed here, *before* the invoice-audit branch below, but reported
    # with lower priority than it. Two reasons for that ordering: the audit
    # has already run by this point so there's no call to save by checking
    # scope first, and when both fire the invoice discrepancy is the more
    # specific, more actionable finding for a rep (it names two conflicting
    # figures; this names a disagreement about breadth). Computing it up
    # here anyway means the scope finding is still attached to the audit-log
    # payload even when the invoice audit is what actually escalates, rather
    # than being silently lost to short-circuiting.
    priced_invoice_lines = sum(1 for li in invoice.line_items if li.unit_price > 0)
    claim_scope_mismatch = check_claim_scope_mismatch(
        validation_result.customer_claimed_scope,
        validation_result.matched_skus,
        priced_invoice_lines,
        scope_note=validation_result.customer_scope_note,
    )

    if invoice_audit.should_escalate:
        draft_inputs = DraftInputs(
            case=case,
            decision="request_info",
            amount=Decimal("0"),
            confidence=GATE_DECISION_CONFIDENCE,
            risk_assessment=risk_assessment,
            memory_context=memory_context_text,
            eligibility_result=eligibility,
            validation_decision=validation_decision,
            invoice_audit=invoice_audit,
        )
        return await _exit(
            case,
            draft_inputs,
            CaseState.ESCALATED,
            event="gate:invoice_audit_discrepancy",
            payload={
                "reason": invoice_audit.summary(),
                "codes": [d.code for d in invoice_audit.escalating],
                **({"claim_scope_note": claim_scope_mismatch} if claim_scope_mismatch else {}),
            },
            transport=transport,
            db_path=db_path,
            eligibility=eligibility,
            evidence_gaps=gaps,
            validation=validation_result,
            invoice_audit=invoice_audit,
        )

    if claim_scope_mismatch is not None:
        draft_inputs = DraftInputs(
            case=case,
            decision="request_info",
            amount=Decimal("0"),
            confidence=GATE_DECISION_CONFIDENCE,
            risk_assessment=risk_assessment,
            memory_context=memory_context_text,
            eligibility_result=eligibility,
            validation_decision=validation_decision,
            invoice_audit=invoice_audit,
        )
        return await _exit(
            case,
            draft_inputs,
            CaseState.ESCALATED,
            event="gate:claim_scope_mismatch",
            payload={"reason": claim_scope_mismatch},
            transport=transport,
            db_path=db_path,
            eligibility=eligibility,
            evidence_gaps=gaps,
            validation=validation_result,
            invoice_audit=invoice_audit,
        )

    damaged = [
        DamagedItem(sku=sku, quantity=quantity)
        for sku, quantity in Counter(validation_result.matched_skus).items()
    ]

    try:
        calc_result = reimbursement(invoice, damaged)
    except (ItemNotOnInvoice, QuantityExceedsInvoice) as exc:
        draft_inputs = DraftInputs(
            case=case,
            decision="request_info",
            amount=Decimal("0"),
            confidence=GATE_DECISION_CONFIDENCE,
            risk_assessment=risk_assessment,
            memory_context=memory_context_text,
            eligibility_result=eligibility,
            validation_decision=validation_decision,
            invoice_audit=invoice_audit,
        )
        return await _exit(
            case,
            draft_inputs,
            CaseState.ESCALATED,
            event="gate:calc_exception",
            payload={"error_type": type(exc).__name__, "error": str(exc)},
            transport=transport,
            db_path=db_path,
            eligibility=eligibility,
            evidence_gaps=gaps,
            validation=validation_result,
            invoice_audit=invoice_audit,
        )

    store.transition(
        case_id,
        CaseState.CALC,
        actor="system",
        event="transition:validation->calc",
        db_path=db_path,
    )

    # --- 6/7. Approve ------------------------------------------------------
    draft_inputs = DraftInputs(
        case=case,
        decision="approve",
        amount=calc_result.amount,
        confidence=_min_judgment_confidence(validation_result),
        risk_assessment=risk_assessment,
        memory_context=memory_context_text,
        eligibility_result=eligibility,
        validation_decision=validation_decision,
        calc_result=calc_result,
        invoice_audit=invoice_audit,
    )
    return await _exit(
        case,
        draft_inputs,
        CaseState.PENDING_REVIEW,
        event="gate:calc_complete",
        transport=transport,
        db_path=db_path,
        eligibility=eligibility,
        evidence_gaps=gaps,
        validation=validation_result,
        calc=calc_result,
        invoice_audit=invoice_audit,
    )
