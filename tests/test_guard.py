"""Tests for the outbound guard.

Every invariant is exercised via a crafted violation built directly against
`check_outbound` -- no app, no database, no network (the function is pure;
see `guard.py`'s module docstring). A shared "clean" baseline scenario
(`_clean_kwargs`) is defined once and each violation test mutates exactly the
one input needed to trip that specific invariant, so a passing test is real
signal that *that* invariant (and not some other coincidental input) is what
fired.
"""

from __future__ import annotations

from decimal import Decimal


from claimpilot.calc import CalcResult
from claimpilot.config import settings
from claimpilot.gates.eligibility import EligibilityResult
from claimpilot.gates.evidence import Gap
from claimpilot.gates.validation import Judgment, ValidationResult
from claimpilot.guard import EmailToSend, GuardViolation, check_outbound
from claimpilot.models import Case, CaseState, EvidenceItem, Invoice, LineItem, RecommendationLineItem
from claimpilot.store import GateResults

CONTACT_EMAIL = "customer@example.com"
SKU = "A00360"
UNIT_PRICE = Decimal("24.99")


def _case(**overrides) -> Case:
    defaults = dict(
        case_id="CASE-1002",
        case_number="CN-1002",
        status="New",
        order_id="ORDER-1",
        user_id="USER-1",
        shipment_id="SHIP-1",
        contact_email=CONTACT_EMAIL,
    )
    defaults.update(overrides)
    return Case(**defaults)


def _invoice(**overrides) -> Invoice:
    line_items = overrides.pop(
        "line_items",
        [LineItem(product_id="P1", name="Widget", sku=SKU, quantity=1, unit_price=UNIT_PRICE)],
    )
    return Invoice(invoice_id="INV-1", shipment_id="SHIP-1", line_items=line_items)


def _passing_judgment() -> Judgment:
    return Judgment(passed=True, confidence=0.9, note="looks consistent with the claim")


def _validation_result(**overrides) -> ValidationResult:
    defaults = dict(
        damage_visible=_passing_judgment(),
        product_identifiable=_passing_judgment(),
        product_on_invoice=_passing_judgment(),
        packaging_documented=_passing_judgment(),
        matched_skus=[SKU],
    )
    defaults.update(overrides)
    return ValidationResult(**defaults)


def _calc(**overrides) -> CalcResult:
    defaults = dict(
        amount=UNIT_PRICE,
        line_items=[RecommendationLineItem(sku=SKU, quantity=1, unit_price=UNIT_PRICE, subtotal=UNIT_PRICE)],
        capped=False,
    )
    defaults.update(overrides)
    return CalcResult(**defaults)


def _gate_results(**overrides) -> GateResults:
    defaults = dict(
        eligibility=EligibilityResult(eligible=True, reason=None, route="process"),
        evidence_gaps=[],
        validation=_validation_result(),
        calc=_calc(),
    )
    defaults.update(overrides)
    return GateResults(**defaults)


def _email(**overrides) -> EmailToSend:
    defaults = dict(
        to=CONTACT_EMAIL,
        subject="Update on your ShipBob claim CASE-1002",
        body=f"Dear customer, we have approved your claim for ${UNIT_PRICE:.2f}.",
    )
    defaults.update(overrides)
    return EmailToSend(**defaults)


def _clean_kwargs() -> dict:
    """A fully self-consistent 'approve' scenario: every invariant passes.
    Each violation test below takes this as a base and mutates exactly one
    input.
    """
    return dict(
        case=_case(),
        gate_results=_gate_results(),
        calc=_calc(),
        email=_email(),
        decision="approve",
        invoice=_invoice(),
        current_status=CaseState.PENDING_REVIEW,
        intended_state=CaseState.APPROVED,
    )


def _codes(violations: list[GuardViolation]) -> list[str]:
    return [v.invariant for v in violations]


# --- happy path: nothing fires -----------------------------------------------


def test_clean_case_has_no_violations():
    assert check_outbound(**_clean_kwargs()) == []


def test_clean_deny_case_has_no_violations():
    """A deny/request_info decision has no calc/invoice to reconcile --
    confirms the guard doesn't spuriously fire when those are absent.
    """
    kwargs = dict(
        case=_case(),
        gate_results=GateResults(),
        calc=CalcResult(amount=Decimal("0"), line_items=[], capped=False),
        email=_email(body="Dear customer, unfortunately this claim has been denied."),
        decision="deny",
        invoice=None,
        current_status=CaseState.PENDING_REVIEW,
        intended_state=CaseState.DENIED,
    )
    assert check_outbound(**kwargs) == []


# --- CAP_EXCEEDED -------------------------------------------------------------


def test_cap_exceeded_when_amount_over_policy_cap():
    kwargs = dict(
        case=_case(),
        gate_results=GateResults(),
        calc=CalcResult(amount=Decimal("150.00"), line_items=[], capped=False),
        email=_email(body="Dear customer, thanks for your patience."),
        decision="deny",
        invoice=None,
        current_status=None,
    )
    violations = check_outbound(**kwargs)
    assert _codes(violations) == ["CAP_EXCEEDED"]
    assert "150.00" in violations[0].detail
    assert str(settings.cap) in violations[0].detail or f"{settings.cap:.2f}" in violations[0].detail


def test_cap_exceeded_uses_live_settings_cap_not_import_time_snapshot(monkeypatch):
    """`_check_cap` must read `settings.cap` at call time -- if it (or
    `calc.reimbursement`, the thing it's re-verifying) ever cached the cap
    into a module-level name at import time instead, this guard could drift
    from the calculator it exists to cross-check, defeating the whole point
    of the guard's independent re-verification. $150 is over the real
    default cap ($100) but under this overridden one ($200), so a clean
    (no-violation) result here is only possible if `_check_cap` actually
    read the live override.
    """
    monkeypatch.setattr(settings, "cap", Decimal("200.00"))
    kwargs = dict(
        case=_case(),
        gate_results=GateResults(),
        calc=CalcResult(amount=Decimal("150.00"), line_items=[], capped=False),
        email=_email(body="Dear customer, thanks for your patience."),
        decision="deny",
        invoice=None,
        current_status=None,
    )
    violations = check_outbound(**kwargs)
    assert _codes(violations) == []


# --- AMOUNT_MISMATCH -----------------------------------------------------------


def test_amount_mismatch_when_calc_disagrees_with_stored_gate_calc():
    """The amount about to be sent (`calc`) diverges from the `CalcResult`
    `pipeline.py` actually saved at calc time (`gate_results.calc`) --
    simulates a corrupted/tampered `recommendation_json` row.
    """
    kwargs = _clean_kwargs()
    kwargs["decision"] = "deny"  # skip approve-only gate re-verification
    kwargs["invoice"] = None  # isolate: skip the fresh-recompute leg
    kwargs["gate_results"] = _gate_results(calc=_calc(amount=Decimal("50.00")))
    kwargs["calc"] = _calc(amount=UNIT_PRICE)  # matches the email, not gate_results.calc
    violations = check_outbound(**kwargs)
    assert _codes(violations) == ["AMOUNT_MISMATCH"]
    assert "50.00" in violations[0].detail
    assert f"{UNIT_PRICE:.2f}" in violations[0].detail


def test_amount_mismatch_when_fresh_rederivation_from_invoice_disagrees():
    """The amount about to be sent (`calc`) diverges from a from-scratch
    `reimbursement(invoice, damaged)` recomputation off the invoice and the
    persisted `matched_skus` -- catches corruption of the `CalcResult`/
    `matched_skus` data itself, independent of the stored `gate_results.calc`.
    """
    kwargs = _clean_kwargs()
    kwargs["decision"] = "deny"  # skip approve-only gate re-verification
    kwargs["gate_results"] = _gate_results(calc=None)  # isolate: skip the stored-calc leg
    # Real SKU (so the SKU-hallucination check stays clean), wrong subtotal
    # (still under CAP, so this doesn't also trip CAP_EXCEEDED).
    kwargs["calc"] = _calc(
        amount=Decimal("75.00"),
        line_items=[RecommendationLineItem(sku=SKU, quantity=1, unit_price=UNIT_PRICE, subtotal=Decimal("75.00"))],
    )
    kwargs["email"] = _email(body="Dear customer, thanks for your patience.")
    violations = check_outbound(**kwargs)
    assert _codes(violations) == ["AMOUNT_MISMATCH"]
    assert "75.00" in violations[0].detail
    assert f"{UNIT_PRICE:.2f}" in violations[0].detail


def test_amount_mismatch_fails_closed_when_approve_has_no_calc_or_invoice():
    """Fail-closed case: an approve decision with NO stored `gate_results.
    calc` and NO invoice to re-derive against has nothing for the amount
    check to verify against at all. Silently returning no violation here
    would let a tampered `calc` (e.g. a corrupted `recommendation_json` row
    with a fabricated amount/line items) sail through untouched -- exactly
    the failure mode this task exists to catch. Eligibility/evidence/
    validation are all left clean so only the amount check's fail-closed
    branch fires.
    """
    kwargs = _clean_kwargs()
    kwargs["gate_results"] = _gate_results(calc=None)
    kwargs["invoice"] = None
    kwargs["calc"] = _calc(amount=Decimal("99.99"), line_items=[])
    kwargs["email"] = _email(body="Dear customer, thanks for your patience.")
    violations = check_outbound(**kwargs)
    # Both the "no stored calc" leg and the "no invoice" leg are
    # independently fail-closed -- two distinct AMOUNT_MISMATCH violations,
    # not a silent pass.
    assert _codes(violations) == ["AMOUNT_MISMATCH", "AMOUNT_MISMATCH"]
    assert all("fail closed" in v.detail.lower() for v in violations)
    assert any("calc gate result" in v.detail for v in violations)
    assert any("invoice" in v.detail for v in violations)


def test_amount_mismatch_when_matched_sku_no_longer_on_invoice():
    """`gate_results.validation.matched_skus` names a SKU that doesn't
    reconcile with the (freshly fetched) invoice -- `reimbursement()` raises
    inside the guard's own re-derivation, which must be surfaced as a
    violation, not let the exception propagate out of `check_outbound`.
    """
    kwargs = _clean_kwargs()
    kwargs["decision"] = "deny"
    kwargs["gate_results"] = _gate_results(calc=None, validation=_validation_result(matched_skus=["GHOST-SKU"]))
    kwargs["email"] = _email(body="Dear customer, thanks for your patience.")
    violations = check_outbound(**kwargs)
    assert _codes(violations) == ["AMOUNT_MISMATCH"]
    assert "GHOST-SKU" in violations[0].detail


# --- ELIGIBILITY_FAILED --------------------------------------------------------


def test_eligibility_failed_when_stored_eligibility_is_ineligible():
    """Approve decision, but the stored gate result says the case was NOT
    eligible -- simulates corrupted/incomplete stored state even though the
    persisted `Recommendation.decision` says 'approve'.
    """
    kwargs = _clean_kwargs()
    kwargs["gate_results"] = _gate_results(eligibility=EligibilityResult(eligible=False, reason="TOO_OLD", route="close"))
    violations = check_outbound(**kwargs)
    assert _codes(violations) == ["ELIGIBILITY_FAILED"]
    assert "TOO_OLD" in violations[0].detail


def test_eligibility_failed_when_eligibility_never_recorded():
    """Fail-closed: no eligibility gate result at all for an approve
    decision must block, not silently pass.
    """
    kwargs = _clean_kwargs()
    kwargs["gate_results"] = _gate_results(eligibility=None)
    violations = check_outbound(**kwargs)
    assert "ELIGIBILITY_FAILED" in _codes(violations)


# --- EVIDENCE_INCOMPLETE -------------------------------------------------------


def test_evidence_incomplete_when_gap_recorded_on_approve_decision():
    kwargs = _clean_kwargs()
    kwargs["gate_results"] = _gate_results(
        evidence_gaps=[Gap(item=EvidenceItem.CUSTOMER_CONFIRMATION, reason="MISSING", detail=None)]
    )
    violations = check_outbound(**kwargs)
    assert _codes(violations) == ["EVIDENCE_INCOMPLETE"]
    assert "CUSTOMER_CONFIRMATION" in violations[0].detail


# --- VALIDATION_FAILED ----------------------------------------------------------


def test_validation_failed_when_a_stored_judgment_did_not_pass():
    """Approve decision, but the stored validation result has a failed
    judgment -- corrupted/incomplete stored state, same shape as the
    eligibility test above.
    """
    kwargs = _clean_kwargs()
    kwargs["gate_results"] = _gate_results(
        validation=_validation_result(damage_visible=Judgment(passed=False, confidence=0.9, note="not visible"))
    )
    violations = check_outbound(**kwargs)
    assert _codes(violations) == ["VALIDATION_FAILED"]
    assert "damage_visible" in violations[0].detail


def test_validation_failed_when_validation_never_recorded():
    kwargs = _clean_kwargs()
    kwargs["gate_results"] = _gate_results(validation=None)
    violations = check_outbound(**kwargs)
    assert "VALIDATION_FAILED" in _codes(violations)


# --- EMAIL_AMOUNT_MISMATCH ------------------------------------------------------


def test_email_amount_mismatch_when_body_promises_more_than_approved():
    """The plan's own example: email draft promising $150 when calc says
    $100 must be blocked. Invoice/calc/gate_results.calc are all kept
    mutually consistent at $100 so only the email-text check fires.
    """
    kwargs = _clean_kwargs()
    invoice = _invoice(line_items=[LineItem(product_id="P1", name="Widget", sku=SKU, quantity=1, unit_price=Decimal("100.00"))])
    calc = _calc(
        amount=Decimal("100.00"),
        line_items=[RecommendationLineItem(sku=SKU, quantity=1, unit_price=Decimal("100.00"), subtotal=Decimal("100.00"))],
    )
    kwargs["invoice"] = invoice
    kwargs["calc"] = calc
    kwargs["gate_results"] = _gate_results(calc=calc)
    kwargs["email"] = _email(body="Great news, we will send you $150.00 for your claim.")
    violations = check_outbound(**kwargs)
    assert _codes(violations) == ["EMAIL_AMOUNT_MISMATCH"]
    assert "150.00" in violations[0].detail
    assert "100.00" in violations[0].detail


def test_email_amount_mismatch_covers_rep_edited_draft():
    """The plan is explicit that a rep-edited draft that accidentally
    changes the promised amount must also be caught -- there's nothing
    LLM-specific about this check, it just scans whatever the final body
    text is, edited or not.
    """
    kwargs = _clean_kwargs()  # amount stays UNIT_PRICE, consistent with invoice/matched_skus
    edited_body = "Dear customer, we will reimburse you $99.99 for the damaged item."
    kwargs["email"] = _email(body=edited_body)
    violations = check_outbound(**kwargs)
    assert _codes(violations) == ["EMAIL_AMOUNT_MISMATCH"]


def test_email_amount_matching_approved_amount_is_not_a_violation():
    kwargs = _clean_kwargs()
    kwargs["email"] = _email(body=f"Dear customer, we will send you ${UNIT_PRICE:.2f}.")
    assert check_outbound(**kwargs) == []


def test_email_with_no_dollar_amounts_is_not_a_violation():
    kwargs = _clean_kwargs()
    kwargs["email"] = _email(body="Dear customer, thank you for your patience.")
    assert check_outbound(**kwargs) == []


# --- RECIPIENT_MISMATCH ----------------------------------------------------------


def test_recipient_mismatch_when_email_to_differs_from_contact_email():
    kwargs = _clean_kwargs()
    kwargs["email"] = _email(to="attacker@evil.example")
    violations = check_outbound(**kwargs)
    assert _codes(violations) == ["RECIPIENT_MISMATCH"]
    assert "attacker@evil.example" in violations[0].detail
    assert CONTACT_EMAIL in violations[0].detail


def test_recipient_mismatch_when_case_has_no_contact_email():
    kwargs = _clean_kwargs()
    kwargs["case"] = _case(contact_email=None)
    violations = check_outbound(**kwargs)
    assert "RECIPIENT_MISMATCH" in _codes(violations)


# --- HALLUCINATED_SKU -------------------------------------------------------------


def test_hallucinated_sku_when_approved_line_item_not_on_invoice():
    kwargs = _clean_kwargs()
    kwargs["decision"] = "deny"  # skip approve-only gate re-verification
    # Both legs of the amount cross-check need gate_results.calc/.validation
    # respectively -- clear both so only the SKU safety-net check runs (a
    # hallucinated SKU in calc.line_items would otherwise also legitimately
    # trip AMOUNT_MISMATCH against a from-scratch re-derivation, since that
    # re-derivation is built only from matched_skus and could never produce
    # the same bogus line item -- a real, expected multi-invariant fire in
    # practice, just not what this test isolates).
    kwargs["gate_results"] = _gate_results(calc=None, validation=None)
    kwargs["calc"] = _calc(
        line_items=[RecommendationLineItem(sku="GHOST-SKU", quantity=1, unit_price=UNIT_PRICE, subtotal=UNIT_PRICE)]
    )
    kwargs["email"] = _email(body="Dear customer, thanks for your patience.")
    violations = check_outbound(**kwargs)
    assert _codes(violations) == ["HALLUCINATED_SKU"]
    assert "GHOST-SKU" in violations[0].detail


def test_hallucinated_sku_when_email_mentions_sku_shaped_token_not_on_invoice():
    kwargs = _clean_kwargs()
    kwargs["email"] = _email(body="Dear customer, we noticed damage to item ZZ999-FAKE in your order.")
    violations = check_outbound(**kwargs)
    assert _codes(violations) == ["HALLUCINATED_SKU"]
    assert "ZZ999-FAKE" in violations[0].detail


def test_real_invoice_sku_mentioned_in_email_is_not_flagged():
    kwargs = _clean_kwargs()
    kwargs["email"] = _email(body=f"Dear customer, item {SKU} was found damaged.")
    assert check_outbound(**kwargs) == []


def test_case_id_mentioned_in_email_body_is_not_flagged_as_hallucinated_sku():
    """`case.case_id`/`case_number`/`order_id`/`shipment_id`/`user_id` are
    legitimate for a customer email to reference and must not be mistaken
    for hallucinated SKUs (module docstring point 5's denylist).
    """
    kwargs = _clean_kwargs()
    kwargs["email"] = _email(body="Dear customer, regarding your claim CASE-1002, order ORDER-1: ...")
    assert check_outbound(**kwargs) == []


def test_hallucinated_sku_check_skipped_gracefully_without_invoice():
    kwargs = _clean_kwargs()
    kwargs["decision"] = "deny"
    kwargs["gate_results"] = GateResults()
    kwargs["calc"] = CalcResult(amount=Decimal("0"), line_items=[], capped=False)
    kwargs["invoice"] = None
    kwargs["email"] = _email(body="Dear customer, we noticed item ZZ999-FAKE in your order.")
    assert check_outbound(**kwargs) == []


# --- ILLEGAL_STATE -----------------------------------------------------------------


def test_illegal_state_when_intended_transition_not_legal_from_current_status():
    kwargs = _clean_kwargs()
    kwargs["current_status"] = CaseState.CLOSED  # terminal -- no legal outgoing transitions
    kwargs["intended_state"] = CaseState.SENT
    violations = check_outbound(**kwargs)
    assert _codes(violations) == ["ILLEGAL_STATE"]
    assert "closed" in violations[0].detail
    assert "sent" in violations[0].detail


def test_legal_state_transition_is_not_a_violation():
    kwargs = _clean_kwargs()
    kwargs["current_status"] = CaseState.ESCALATED
    kwargs["intended_state"] = CaseState.SENT  # ESCALATED -> SENT is legal
    assert check_outbound(**kwargs) == []


def test_legal_state_check_skipped_when_current_status_not_provided():
    kwargs = _clean_kwargs()
    kwargs["current_status"] = None
    assert check_outbound(**kwargs) == []


# --- multiple violations can co-occur ------------------------------------------


def test_multiple_violations_all_reported_together():
    kwargs = _clean_kwargs()
    kwargs["email"] = _email(to="attacker@evil.example", body="We will send you $999.00.")
    violations = check_outbound(**kwargs)
    codes = _codes(violations)
    assert "RECIPIENT_MISMATCH" in codes
    assert "EMAIL_AMOUNT_MISMATCH" in codes


# --- purity: no I/O anywhere in this module -------------------------------------


def test_guard_module_never_touches_the_database_network_or_client():
    """Static guard for the "genuinely pure" self-review requirement:
    `check_outbound` must never open a database connection, await anything,
    or call the ShipBob client -- every dependency is a plain argument the
    caller (web/app.py) supplies after doing its own I/O.
    """
    guard_path = __import__("pathlib").Path(__file__).resolve().parents[1] / "src" / "claimpilot" / "guard.py"
    source = guard_path.read_text()
    assert "get_connection(" not in source
    assert "await " not in source
    assert "async def" not in source
    # Real call-site patterns, not just the substring "client." (which the
    # module's own docstring legitimately uses in prose describing what the
    # *caller*, web/app.py, does).
    assert ".send_email(" not in source
    assert ".submit_reimbursement(" not in source
    assert "sqlite3" not in source
    assert "httpx" not in source


# --- unfilled placeholders -----------------------------------------------------
#
# Observed live: a redrafted CASE-9003-REPEAT email left the pipeline addressed
# to "[Merchant's Name]". The drafting prompt forbids placeholders, but the
# drafter had no merchant name available and filled the gap itself. Since a rep
# can approve a draft as-is, that is one click from reaching a real merchant.


def test_unfilled_placeholder_in_the_body_blocks_the_send():
    kwargs = _clean_kwargs()
    kwargs["email"] = _email(body="Dear [Merchant's Name],\n\nYour claim has been approved.")

    violations = check_outbound(**kwargs)

    assert "UNFILLED_PLACEHOLDER" in _codes(violations)
    detail = next(v.detail for v in violations if v.invariant == "UNFILLED_PLACEHOLDER")
    assert "Merchant's Name" in detail  # names exactly what the rep must fix


def test_signature_placeholder_also_blocks():
    kwargs = _clean_kwargs()
    kwargs["email"] = _email(body="Thanks for your patience.\n\nBest regards,\n[Your Name]")

    assert "UNFILLED_PLACEHOLDER" in _codes(check_outbound(**kwargs))


def test_multiple_placeholders_are_reported_together():
    kwargs = _clean_kwargs()
    kwargs["email"] = _email(body="Dear [Customer Name],\n\nOn [Date] we shipped.\n\n[Your Name]")

    detail = next(
        v.detail for v in check_outbound(**kwargs) if v.invariant == "UNFILLED_PLACEHOLDER"
    )
    for expected in ("Customer Name", "Date", "Your Name"):
        assert expected in detail


def test_legitimate_bracketed_prose_is_not_flagged():
    """Must not fire on ordinary text that happens to use brackets -- a check
    that blocks clean drafts is a check reps learn to click past.
    """
    for body in (
        "Dear Best Paw Nutrition,\n\nYour claim has been approved.",
        "See the attached note [1] for detail.",
        "Reference [A2] is enclosed.",
    ):
        kwargs = _clean_kwargs()
        kwargs["email"] = _email(body=body)
        assert "UNFILLED_PLACEHOLDER" not in _codes(check_outbound(**kwargs)), body
