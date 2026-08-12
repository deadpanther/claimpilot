"""Outbound guard.

The last line of defense before anything leaves the system. Every invariant
in the plan's "Guardrails & Middleware #1" section is implemented here as one
pure function, `check_outbound`, called by `web/app.py`'s approve endpoint
after the rep clicks approve (and after any edit to the draft has been
applied) but *before* `store.record_action`/`client.send_email`/
`client.submit_reimbursement`. A non-empty return value means: don't send
anything, escalate the case, and show the rep exactly which invariant(s)
fired.

**`check_outbound` is genuinely pure -- no I/O.** It never opens a database
connection, never awaits anything, never calls the ShipBob client. Every
piece of data it needs (the case, the persisted gate results, the calc
result backing the recommendation about to be sent, the invoice, the final
email, the case's current stored status) is passed in by the caller, which
does own the I/O (`web/app.py` loads gate results via `store.
load_gate_results`, fetches the invoice via `client.generate_invoice`, reads
the case's status from the `cases` row it already has). This is what makes
"would a corrupted gate_results record actually get caught" a testable
question rather than a hopeful assertion -- every test in `tests/
test_guard.py` calls this function directly with hand-built inputs, no app,
no database, no network.

Design decisions (the plan's signature/invariant list were intentionally
abbreviated -- these are the calls made to extend them):

1. **Three independently-sourced "what's about to be paid" values, cross-
   checked against each other.** The whole point of re-verifying
   "from stored gate results, not from the draft" is defeated if the
   thing being checked *is* the only copy of the truth -- comparing a value
   against itself always passes. So three values are compared, each sourced
   differently:
     - `calc` (a parameter): the amount/line-items actually about to be
       emailed and paid, reconstructed by the caller from the persisted
       `Recommendation` (`recommendation.amount` /`.line_items`) -- this is
       the *authoritative* value in the sense that `web/app.py`'s
       `submit_reimbursement` loop and the email body are both driven by
       exactly this data, so it's what the guard must ultimately trust or
       block.
     - `gate_results.calc`: the `CalcResult` saved by `pipeline.py` at
       calc-time, completely independent of whatever the `Recommendation`
       row says now. If `recommendation_json` were corrupted/tampered
       between calc-time and approve-time, `calc` (derived from it) would
       drift from this without either side "knowing" -- comparing the two
       catches exactly that.
     - A **fresh recomputation**, `reimbursement(invoice, damaged_from(
       gate_results.validation.matched_skus))`, run right here from the
       invoice and the persisted `matched_skus` -- independent of *both*
       stored `CalcResult`s. If `gate_results_json` itself were corrupted
       (e.g. `calc` patched directly in the JSON blob, or `matched_skus`
       edited), this third, from-scratch value would drift from the other
       two and get caught. This is the check the plan calls "genuinely
       re-derives the number rather than just checking amount <= CAP again."
   Any pairwise mismatch among these three is an `AMOUNT_MISMATCH`
   violation. The gate_results.calc comparison needs `gate_results.calc is
   not None`; the fresh-recompute comparison needs both `invoice is not
   None` and `gate_results.validation is not None`. For `decision !=
   "approve"` (no calc/invoice ever exists on a deny/request_info path),
   missing data is expected and each leg is skipped gracefully. For
   `decision == "approve"`, missing data is instead **fail-closed**: no
   `gate_results.calc` and/or no `invoice` means there is nothing to verify
   the approved amount against, so each is its own `AMOUNT_MISMATCH`
   violation rather than a silent pass -- otherwise a tampered `calc` with a
   fabricated amount and no corroborating gate data would sail through
   untouched, which is exactly the failure mode this whole task exists to
   prevent. See `_check_amount`'s docstring.
2. **`invoice` is optional and the caller decides when to fetch it.**
   `web/app.py` only fetches+passes an invoice when `decision == "approve"`
   (there's nothing to reconcile a deny/request_info's $0 payout against).
   A failed fetch (e.g. `NotFoundError`) is passed through as `invoice=None`
   rather than swallowed into a silent skip -- point 1's fail-closed
   handling turns "couldn't fetch the invoice" into an `AMOUNT_MISMATCH`
   violation for an approve decision, not a quietly-disabled check.
3. **Approve-decision gate re-verification reports each gate under its own
   invariant code** (`ELIGIBILITY_FAILED`/`EVIDENCE_INCOMPLETE`/
   `VALIDATION_FAILED`), each independently fail-closed on `None`/missing
   data, rather than one umbrella "gates missing" code -- "shows the rep
   exactly which invariant fired" is more useful per-gate than as one blob.
   In practice, per `pipeline.py`'s `_exit()` (every gate object a given
   exit passes is set together, never partially), `eligibility` and
   `validation` being `None` at the same time is the "gate results were
   never recorded for this case at all" signal (see `store.GateResults`
   docstring) -- both `ELIGIBILITY_FAILED` and `VALIDATION_FAILED` fire
   together in that case, which is a strong enough fail-closed signal even
   though `EVIDENCE_INCOMPLETE` (whose "true" value is indistinguishable
   between "never recorded" and "recorded as zero gaps", both `[]`) stays
   silent in that specific scenario. Documented, not hidden -- see the
   `_check_gate_reverification` docstring.
4. **`EmailToSend` is a minimal plain dataclass** (`to`/`subject`/`body`),
   not a pydantic model -- there's no external payload to validate here (the
   caller already built these strings), matching `calc.py`'s
   `DamagedItem`/`CalcResult` precedent for internal, already-trusted
   intermediate values.
5. **SKU-hallucination scope, deliberately narrow (documented, not
   over-engineered NLP).** Two checks, not one:
     - `calc.line_items` (the SKUs about to be paid) vs `invoice.line_items`
       -- every paid SKU must be real. Redundant with `calc.py`'s own
       `ItemNotOnInvoice` check at construction time, but re-verified here
       as a safety net against state corruption between calc-time and now
       (same rationale as the amount cross-check above).
     - A free-text scan of the email **body only** (never the subject,
       which is templated by `web/app.py` and contains the case ID, not a
       SKU) for tokens that look SKU-shaped (`_SKU_TOKEN_RE`: uppercase
       alphanumeric, optionally hyphen-segmented, length >= 4, containing at
       least one digit -- matches this project's real SKUs like `A00360`/
       `A00300-CASE12`) that do NOT match any real invoice SKU. Any such token
       that also matches one of the case's own identifiers (`case_id`,
       `case_number`, `order_id`, `shipment_id`, `user_id` -- values a
       legitimate customer email might reasonably reference, e.g. "Regarding
       your claim CASE-1002") is excluded from this scan via
       `_reference_id_denylist`, so the guard doesn't fire on the system's
       own identifiers being mentioned in prose. This is a best-effort
       pattern match, not language understanding -- a hallucinated SKU that
       doesn't happen to look like `_SKU_TOKEN_RE` (e.g. a plain English
       product name) will not be caught by this specific check. Only runs
       when `invoice is not None` (nothing to compare against otherwise).
6. **Legal-state check takes the caller-computed `intended_state`, not a
   hardcoded `SENT`.** `store.LEGAL_TRANSITIONS[PENDING_REVIEW]` does not
   include `SENT` directly (only via an intermediate `APPROVED`/`DENIED`/
   `NEEDS_INFO` hop, per `web/app.py`'s own two-hop approve logic) -- so
   the caller must pass whichever state it's about to transition to *next*
   (the intermediate state for a `PENDING_REVIEW` case, or `SENT` directly
   for an `ESCALATED` case, mirroring `web/app.py`'s existing
   `_DECISION_TO_INTERMEDIATE_STATE` branching) rather than this function
   guessing at the two-hop shape itself. `current_status`/`intended_state`
   default to `None`/`CaseState.SENT respectively; when `current_status is
   None` the check is skipped (nothing to validate against) -- documented,
   not silently wrong, and `web/app.py` always supplies it in practice.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from claimpilot.calc import (
    CalcResult,
    DamagedItem,
    ItemNotOnInvoice,
    QuantityExceedsInvoice,
    reimbursement,
)
from claimpilot.config import settings
from claimpilot.models import Case, CaseState, Invoice
from claimpilot.store import LEGAL_TRANSITIONS, GateResults

Decision = Literal["approve", "deny", "request_info"]

# Matches a dollar amount with exactly two decimal places, optionally with
# thousands separators (e.g. "$100.00", "$1,234.56"). Deliberately does not
# match bare numbers without a leading "$" -- the invariant is about amounts
# *presented as money* in the email, not any two-decimal number that happens
# to appear (e.g. a shipment weight of "12.34 lbs").
_DOLLAR_AMOUNT_RE = re.compile(r"\$([\d,]+\.\d{2})")

# SKU-shaped token: uppercase alphanumeric, optionally hyphen-segmented (e.g.
# "A00360", "A00300-CASE12"). Filtered further below (length/digit requirement)
# before being treated as a SKU candidate -- see module docstring point 5.
_SKU_TOKEN_RE = re.compile(r"\b[A-Z0-9]+(?:-[A-Z0-9]+)*\b")


@dataclass(frozen=True)
class GuardViolation:
    """One outbound-guard invariant that failed.

    `invariant` is a short, stable machine code (e.g. `"CAP_EXCEEDED"`) a
    caller/UI can branch or filter on; `detail` is a specific,
    human-readable explanation -- "shows the rep exactly which invariant
    fired" per the plan, so this is written to be genuinely informative
    (the actual numbers/values involved), never a generic "check failed".
    """

    invariant: str
    detail: str


@dataclass(frozen=True)
class EmailToSend:
    """The exact final email a send would deliver -- after any rep edit has
    already been applied by the caller. Plain dataclass, not pydantic:
    there's no external payload to validate, `web/app.py` already has these
    as plain strings by the time it calls `check_outbound` (matching
    `calc.py`'s `DamagedItem`-style precedent for internal, already-trusted
    intermediate values).
    """

    to: str
    subject: str
    body: str


def _damaged_items_from_matched_skus(matched_skus: list[str]) -> list[DamagedItem]:
    """Same `Counter`-based quantity convention `pipeline.py` uses when
    turning `ValidationResult.matched_skus` into `DamagedItem`s (module
    docstring point 5 there) -- reused verbatim here so the fresh
    recomputation in `_check_amount` is directly comparable to what
    `pipeline.py` itself would have computed from the same `matched_skus`.
    """
    return [DamagedItem(sku=sku, quantity=quantity) for sku, quantity in Counter(matched_skus).items()]


def _line_items_match(a: list, b: list) -> bool:
    """Compare two `RecommendationLineItem` lists for equality regardless of
    order (sorted by `sku`) -- `Decimal` fields compare by value (`Decimal
    ("100") == Decimal("100.00")`), never by string, per the project's
    existing `Decimal`-precision conventions.
    """
    if len(a) != len(b):
        return False
    a_sorted = sorted(a, key=lambda li: li.sku)
    b_sorted = sorted(b, key=lambda li: li.sku)
    return all(
        x.sku == y.sku and x.quantity == y.quantity and x.unit_price == y.unit_price and x.subtotal == y.subtotal
        for x, y in zip(a_sorted, b_sorted)
    )


def _check_cap(calc: CalcResult) -> list[GuardViolation]:
    """`amount <= settings.cap` (currently $100.00, `config.py`, env-
    overridable via `CAP`). Checked unconditionally -- even a deny/
    request_info's `calc.amount` should be `Decimal("0")`, so this only ever
    fires when something has actually gone wrong.

    Reads `settings.cap` at call time (not a module-level snapshot) so this
    guard and `calc.reimbursement` (the thing it's re-verifying) can never
    drift apart by reading the cap from two different points in time.
    """
    cap = settings.cap
    if calc.amount > cap:
        return [
            GuardViolation(
                "CAP_EXCEEDED",
                f"Approved amount ${calc.amount:.2f} exceeds the ${cap:.2f} policy cap.",
            )
        ]
    return []


def _check_amount(
    decision: Decision,
    calc: CalcResult,
    gate_results: GateResults,
    invoice: Invoice | None,
) -> list[GuardViolation]:
    """Cross-check `calc` (what's about to be paid) against two
    independently-sourced values -- see module docstring point 1 for why
    three sources, not one, are compared.

    Fail-closed for `decision == "approve"`: an approve with no stored
    `gate_results.calc` and/or no `invoice` to re-derive against has nothing
    for this check to verify the amount with at all -- silently returning
    `[]` in that case would let a tampered `calc` (e.g. a corrupted
    `recommendation_json` row with an inflated amount and fabricated line
    items) sail through untouched. Missing data on an approve is itself a
    violation here, same fail-closed principle
    `_check_gate_reverification` applies to eligibility/evidence/validation.
    For `decision != "approve"` (no calc/invoice ever exists on those
    paths), missing data is expected and not a violation.
    """
    violations: list[GuardViolation] = []

    # Leg 1: `calc` vs the `CalcResult` `pipeline.py` saved at calc-time.
    stored_calc = gate_results.calc
    if stored_calc is not None:
        if calc.amount != stored_calc.amount or not _line_items_match(calc.line_items, stored_calc.line_items):
            violations.append(
                GuardViolation(
                    "AMOUNT_MISMATCH",
                    f"Amount about to be sent (${calc.amount:.2f}) does not match the stored "
                    f"calc-gate result (${stored_calc.amount:.2f}) recorded at calc time.",
                )
            )
    elif decision == "approve":
        violations.append(
            GuardViolation(
                "AMOUNT_MISMATCH",
                "No calc gate result recorded for this case -- cannot verify the approved "
                "amount against it (fail closed).",
            )
        )

    # Leg 2: `calc` vs a from-scratch recomputation off the invoice +
    # persisted `matched_skus` -- only possible when both are available.
    if invoice is not None and gate_results.validation is not None:
        damaged = _damaged_items_from_matched_skus(gate_results.validation.matched_skus)
        try:
            fresh = reimbursement(invoice, damaged)
        except (ItemNotOnInvoice, QuantityExceedsInvoice) as exc:
            violations.append(
                GuardViolation(
                    "AMOUNT_MISMATCH",
                    f"Could not re-derive the reimbursement from the invoice and stored "
                    f"matched_skus: {exc}",
                )
            )
        else:
            if calc.amount != fresh.amount or not _line_items_match(calc.line_items, fresh.line_items):
                violations.append(
                    GuardViolation(
                        "AMOUNT_MISMATCH",
                        f"Amount about to be sent (${calc.amount:.2f}) does not match a fresh "
                        f"re-derivation from the invoice (${fresh.amount:.2f}).",
                    )
                )
    elif decision == "approve":
        violations.append(
            GuardViolation(
                "AMOUNT_MISMATCH",
                "No invoice available to re-derive the approved amount against (fail closed).",
            )
        )

    return violations


def _check_gate_reverification(decision: Decision, gate_results: GateResults) -> list[GuardViolation]:
    """Approve decision requires eligibility passed, all evidence present,
    and all 4 validation judgments passed -- **re-verified from
    `gate_results`, not from the draft/decision label**. Only applies to
    `decision == "approve"` (module docstring point 3); each of the three
    gates is independently fail-closed on `None`/missing data.
    """
    if decision != "approve":
        return []

    violations: list[GuardViolation] = []

    if gate_results.eligibility is None:
        violations.append(
            GuardViolation(
                "ELIGIBILITY_FAILED",
                "No eligibility gate result recorded for this case -- cannot verify an "
                "approve decision against it (fail closed).",
            )
        )
    elif not gate_results.eligibility.eligible:
        violations.append(
            GuardViolation(
                "ELIGIBILITY_FAILED",
                f"Stored eligibility gate result is eligible=False (reason="
                f"{gate_results.eligibility.reason!r}), but the decision is 'approve'.",
            )
        )

    if gate_results.evidence_gaps:
        gap_summary = ", ".join(f"{g.item.value} ({g.reason})" for g in gate_results.evidence_gaps)
        violations.append(
            GuardViolation(
                "EVIDENCE_INCOMPLETE",
                f"{len(gate_results.evidence_gaps)} evidence gap(s) still outstanding per "
                f"the stored gate result: {gap_summary}.",
            )
        )

    if gate_results.validation is None:
        violations.append(
            GuardViolation(
                "VALIDATION_FAILED",
                "No validation gate result recorded for this case -- cannot verify an "
                "approve decision against it (fail closed).",
            )
        )
    else:
        judgments = {
            "damage_visible": gate_results.validation.damage_visible,
            "product_identifiable": gate_results.validation.product_identifiable,
            "product_on_invoice": gate_results.validation.product_on_invoice,
            "packaging_documented": gate_results.validation.packaging_documented,
        }
        failed = [name for name, judgment in judgments.items() if not judgment.passed]
        if failed:
            violations.append(
                GuardViolation(
                    "VALIDATION_FAILED",
                    f"Stored validation gate result has failed judgment(s): {', '.join(failed)}.",
                )
            )

    return violations


def _check_email_amounts(email: EmailToSend, calc: CalcResult) -> list[GuardViolation]:
    """Every `$`-amount mentioned in the email body must equal `calc.amount`
    exactly. Zero mentions is fine -- this only fires when a mentioned
    amount actually disagrees, catching e.g. an LLM promising money the calc
    didn't grant, or a rep edit that accidentally changes the figure.
    """
    violations: list[GuardViolation] = []
    for match in _DOLLAR_AMOUNT_RE.finditer(email.body):
        mentioned = Decimal(match.group(1).replace(",", ""))
        if mentioned != calc.amount:
            violations.append(
                GuardViolation(
                    "EMAIL_AMOUNT_MISMATCH",
                    f"Email mentions ${mentioned:.2f} but the approved amount is ${calc.amount:.2f}.",
                )
            )
    return violations


# An unfilled template placeholder: a bracketed run of text that reads like
# an instruction to a human rather than content, e.g. `[Your Name]`,
# `[Merchant\'s Name]`, `[Date]`. Requires a letter first and at least three
# characters so it can\'t fire on legitimate bracketed prose a draft might
# contain (an aside, or a bracketed reference like `[1]`).
_PLACEHOLDER_RE = re.compile(r"\[[A-Za-z][^\]\n]{2,40}\]")


def _check_placeholders(email: EmailToSend) -> list[GuardViolation]:
    """Block a send whose body still contains an unfilled placeholder.

    The drafting prompt forbids these, and `draft.py` now supplies the
    merchant name so the model has no reason to invent one -- but prompt
    instructions reduce a rate, they don\'t guarantee a shape. This was
    observed live: a redrafted email went out of the pipeline addressed to
    `[Merchant\'s Name]`. Because a rep can approve a draft as-is, that is
    one click away from reaching a real merchant.

    Deliberately a hard block rather than a warning, and deliberately not
    auto-repaired: there is no safe automatic substitution (whose name?),
    and silently deleting text from an outbound email is worse than
    refusing to send it. The rep edits the draft and re-approves.
    """
    found = _PLACEHOLDER_RE.findall(email.body)
    if not found:
        return []
    shown = ", ".join(repr(f) for f in dict.fromkeys(found))
    return [
        GuardViolation(
            "UNFILLED_PLACEHOLDER",
            f"Email body still contains unfilled placeholder text ({shown}). Edit the draft "
            f"to remove it before sending -- this would reach the merchant verbatim.",
        )
    ]


def _check_recipient(case: Case, email: EmailToSend) -> list[GuardViolation]:
    """Recipient must equal `case.contact_email` exactly -- never anything
    parsed from the merchant-written description or the (possibly
    rep-edited) email body. Injection defense: a hostile/careless edit that
    tries to redirect the send has no path through this function to change
    who the guard considers valid.
    """
    if case.contact_email is None or email.to != case.contact_email:
        return [
            GuardViolation(
                "RECIPIENT_MISMATCH",
                f"Email recipient {email.to!r} does not match case.contact_email "
                f"{case.contact_email!r}.",
            )
        ]
    return []


def _reference_id_denylist(case: Case) -> set[str]:
    """The case's own identifiers -- legitimate for a customer email to
    mention (e.g. "Regarding your claim CASE-1002") and not SKU
    hallucinations. Excluded from the free-text SKU scan (module docstring
    point 5).
    """
    return {
        value.upper()
        for value in (case.case_id, case.case_number, case.order_id, case.shipment_id, case.user_id)
        if value
    }


def _is_sku_shaped(token: str) -> bool:
    """A candidate token counts as "SKU-shaped" only if, with hyphens
    stripped, it's at least 4 characters and contains at least one digit --
    matches this project's real SKU formats (`A00360`, `A00300-CASE12`) while
    excluding short/all-letter tokens that are obviously just words (e.g.
    "PLEASE", "DEAR").
    """
    stripped = token.replace("-", "")
    return len(stripped) >= 4 and any(ch.isdigit() for ch in stripped)


def _check_sku_hallucination(
    calc: CalcResult,
    email: EmailToSend,
    invoice: Invoice | None,
    case: Case,
) -> list[GuardViolation]:
    """Two checks -- see module docstring point 5 for the full scope/
    limitations discussion:
      1. Every SKU in `calc.line_items` (about to be paid) must be a real
         invoice SKU -- a safety net re-verifying what `calc.py` already
         enforces at construction time, catching corruption since.
      2. A best-effort free-text scan of the email **body** for SKU-shaped
         tokens absent from the invoice.
    Both require `invoice is not None`; skipped gracefully otherwise (no
    ground truth to compare against for a deny/request_info case with no
    invoice fetched).
    """
    if invoice is None:
        return []

    violations: list[GuardViolation] = []
    invoice_skus = {line.sku for line in invoice.line_items}

    for line_item in calc.line_items:
        if line_item.sku not in invoice_skus:
            violations.append(
                GuardViolation(
                    "HALLUCINATED_SKU",
                    f"Approved line item SKU {line_item.sku!r} is not on the invoice.",
                )
            )

    denylist = _reference_id_denylist(case)
    seen: set[str] = set()
    for match in _SKU_TOKEN_RE.finditer(email.body):
        token = match.group(0)
        if token in seen or token in denylist or not _is_sku_shaped(token):
            continue
        seen.add(token)
        if token not in invoice_skus:
            violations.append(
                GuardViolation(
                    "HALLUCINATED_SKU",
                    f"Email body mentions {token!r}, which looks like a SKU but is not on the invoice.",
                )
            )

    return violations


def _check_legal_state(current_status: CaseState | None, intended_state: CaseState) -> list[GuardViolation]:
    """The intended transition target must be legal from the case's current
    stored status per `store.LEGAL_TRANSITIONS` -- a pure lookup (never
    performs the transition itself), consistent with `check_outbound` being
    a side-effect-free function. Skipped when `current_status is None`
    (module docstring point 6).
    """
    if current_status is None:
        return []
    if intended_state not in LEGAL_TRANSITIONS.get(current_status, set()):
        return [
            GuardViolation(
                "ILLEGAL_STATE",
                f"Transition {current_status.value} -> {intended_state.value} is not legal "
                f"per the state machine.",
            )
        ]
    return []


def check_outbound(
    case: Case,
    gate_results: GateResults,
    calc: CalcResult,
    email: EmailToSend,
    *,
    decision: Decision,
    invoice: Invoice | None = None,
    current_status: CaseState | None = None,
    intended_state: CaseState = CaseState.SENT,
) -> list[GuardViolation]:
    """Re-verify every outbound invariant before a send is allowed to happen.

    Pure function: no I/O, no database access, no client calls -- every
    dependency is a plain argument (module docstring, top). Called by
    `web/app.py`'s approve endpoint after the final email (post-edit, if
    any) has been assembled but before `store.record_action`/
    `client.send_email`/`client.submit_reimbursement`; a non-empty return
    value must block all three.

    Args:
        case: the case being approved. `case.contact_email` is the only
            valid recipient (see `_check_recipient`).
        gate_results: the persisted `store.GateResults` for this case (via
            `store.load_gate_results`) -- the ground truth `pipeline.py`
            actually computed, re-verified here independently of the
            drafted `Recommendation`.
        calc: the `CalcResult` backing the amount/line-items about to
            actually be emailed and paid (reconstructed by the caller from
            the persisted `Recommendation` -- see module docstring point 1
            for why this must NOT simply be `gate_results.calc`).
        email: the exact final `to`/`subject`/`body` a send would deliver,
            after any rep edit has already been applied.
        decision: the recommendation's `decision` ("approve"/"deny"/
            "request_info") -- gates which invariants apply (the gate
            re-verification invariant only fires for "approve").
        invoice: the case's invoice, fetched fresh by the caller. `None` is
            handled two different ways depending on `decision`: for
            deny/request_info (where `web/app.py` doesn't fetch one at all
            -- nothing to reconcile against), the amount re-derivation and
            SKU-hallucination checks are both skipped gracefully. For
            `decision == "approve"`, a missing invoice is instead
            fail-closed -- an `AMOUNT_MISMATCH` violation, since there would
            be nothing left to verify an approved amount against (see
            `_check_amount`).
        current_status: the case's current stored `CaseState`, for the
            legal-state check. `None` skips that check.
        intended_state: the state the caller is about to transition to next
            (module docstring point 6) -- defaults to `SENT`, but a
            `PENDING_REVIEW` case must pass its actual next hop (the
            decision-mapped intermediate state), not `SENT` directly, since
            `PENDING_REVIEW -> SENT` alone is not a legal transition.

    Returns:
        A list of `GuardViolation`s, in a fixed, deterministic check order
        (cap, amount cross-checks, gate re-verification, email amounts,
        unfilled placeholders, recipient, SKU hallucination, legal state).
        Empty means every invariant passed.
    """
    violations: list[GuardViolation] = []
    violations.extend(_check_cap(calc))
    violations.extend(_check_amount(decision, calc, gate_results, invoice))
    violations.extend(_check_gate_reverification(decision, gate_results))
    violations.extend(_check_email_amounts(email, calc))
    violations.extend(_check_placeholders(email))
    violations.extend(_check_recipient(case, email))
    violations.extend(_check_sku_hallucination(calc, email, invoice, case))
    violations.extend(_check_legal_state(current_status, intended_state))
    return violations
