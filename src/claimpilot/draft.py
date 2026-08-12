"""Drafter (recommendation + email).

Final step of the claim-decision pipeline. Consumes every prior gate's output (eligibility,
evidence, validation, calc, risk) plus a case and produces the final
`Recommendation` a human rep reviews in the review UI.

**Non-negotiable safety invariant** (the whole point of this task, restated
because it must never regress): `decision`, `amount`, and `confidence` are
computed by deterministic code *before* `draft()` is called, and are passed
in via `DraftInputs`. The LLM's only job is writing prose (`rationale`
bullets, `email_draft`) that explains an outcome that has already been
decided. `DraftOutput` -- the LLM's forced tool schema -- has exactly two
`str` fields (`rationale`, `email_draft`); it is structurally impossible for
the model's response to carry a decision or amount, because there is no
field to put one in. `draft()` builds the final `Recommendation` by copying
`inputs.decision` / `inputs.amount` / `inputs.confidence` /
`inputs.risk_assessment.tier` directly -- never by parsing, inferring, or
cross-checking against the LLM's prose. See `tests/test_draft.py`'s
`test_draft_*_ignores_contradictory_llm_output` tests, which feed the fake
LLM a response that actively contradicts the passed-in decision/amount and
assert the final `Recommendation` still matches `DraftInputs` exactly.

Design decisions:

1. **`DraftInputs` is a grab-bag, not a 10-parameter function.** Every field
   except `case`/`decision`/`amount`/`confidence`/`risk_assessment` is
   optional, because a real case will not always have run every gate --
   e.g. a `request_info` decision produced by `evidence_gaps` alone may
   never reach the validation or calc stage. `draft()` degrades gracefully
   when a given gate's result is `None` (summarizes it as "not run for this
   case" rather than crashing), so the orchestrator can call this
   with whatever subset of gate results the pipeline actually produced for
   a given case.
2. **`memory_context` is populated by callers, not by `draft()` itself.**
   `draft()` only ever renders whatever string it's given -- it has no
   knowledge of `claimpilot.memory`. Today, `pipeline.process_case`
   passes `claimpilot.memory.MemoryContext.to_prompt_text()` (merchant
   notes/corrections + merchant and global policy notes + trailing-90-day
   claim frequency); `claimpilot.web.app`'s pushback endpoint passes that
   same merchant context PLUS the rep's immediate feedback text as its own
   labeled section (see that module's docstring for the composition and why
   the two are kept visually separate: one is durable merchant history, the
   other is a one-off "fix this now" instruction). The default `""` still
   exists for callers with no case/merchant context at all (e.g. some unit
   tests) -- `draft()` renders that as `"(none available yet)"`.
3. **Trusted vs. untrusted content, same convention as `gates/validation.py`
   and `gates/evidence.py`.** Gate results (eligibility reason, evidence
   gap details, validation decision/judgments, calc breakdown, risk tier/
   flags) are this system's own analysis -- plain trusted text, not wrapped.
   `case.description` is the only genuinely free-text, customer/merchant-
   authored field on `Case` (`status`/`sub_category`/`origin`/`case_number`
   are API vocabulary/IDs, not prose) and is wrapped in
   `<untrusted_data>` tags per the untrusted-data wrapping convention
   described in `llm.py`. `None` description
   is rendered as a plain `(no description provided)` marker rather than
   the literal string `"None"` inside the tags.
   Known limitation (same as `evidence.py`'s file-name wrapping): a
   description containing a literal `</untrusted_data>` could prematurely
   close the tag. Not solved here, consistent with existing precedent.
4. **`inputs.amount` is authoritative; `inputs.calc_result.amount` is
   context only, and the two may legitimately disagree** -- e.g. a `deny`
   decision passes `amount=Decimal("0")` while `calc_result` (if the calc
   step still ran before a later gate denied the claim) reflects what the
   raw calculation would have paid. The prompt labels `inputs.amount` as
   the "AUTHORITATIVE AMOUNT (fixed)" and any `calc_result.amount` as
   underlying-calculation context, and `draft()` does not validate that
   they match -- that would be the wrong kind of coupling. `calc_result`
   also carries `capped`, surfaced explicitly so the email can say "capped
   at the policy maximum" when true.
5. **`Recommendation.line_items` comes from `inputs.calc_result.line_items`
   verbatim (empty list if `calc_result` is `None`)** -- deterministic data,
   never LLM-authored. If a caller wants a `deny` recommendation to carry no
   line items even though a calc step ran, it should pass
   `calc_result=None`, not rely on `draft()` to infer that from `decision`.
6. **Residual risk: `email_draft` is free prose.** The structural
   guarantee above protects `Recommendation.decision`/`.amount` themselves,
   but nothing stops the model from writing a wrong dollar figure into the
   *prose* of the email body (the schema has no way to stop free text from
   containing digits). The prompt explicitly instructs the model to state
   the given amount exactly and never recompute/estimate one, but this is a
   prompting mitigation, not a structural one. The outbound guard (see
   `guard.py`) should verify the email text's stated amount (if any) against
   `Recommendation.amount` before anything is actually sent to a customer.
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from claimpilot.gates.eligibility import EligibilityResult
from claimpilot.gates.evidence import Gap
from claimpilot.gates.invoice_audit import InvoiceAudit
from claimpilot.gates.validation import ValidationDecision
from claimpilot.calc import CalcResult
from claimpilot.llm import Transport, structured_call
from claimpilot.models import Case, Recommendation
from claimpilot.risk import RiskAssessment

PROMPT_NAME = "draft_email"

_NO_DESCRIPTION_MARKER = "(no description provided)"


@dataclass(frozen=True)
class DraftInputs:
    """Consolidated inputs to `draft()`, pulled together from every gate
    that ran for a case. See module docstring point 1 for why most fields
    are optional.
    """

    case: Case
    # Already-decided outcome + amount + confidence -- computed by
    # deterministic pipeline code before `draft()` is ever called. `draft()`
    # copies these straight into the returned `Recommendation`; the LLM
    # never sees them framed as a question.
    decision: Literal["approve", "deny", "request_info"]
    amount: Decimal
    confidence: float
    risk_assessment: RiskAssessment
    eligibility_result: EligibilityResult | None = None
    evidence_gaps: list[Gap] = field(default_factory=list)
    validation_decision: ValidationDecision | None = None
    calc_result: CalcResult | None = None
    # The retail-invoice reconciliation, when it ran. Needed here because a
    # case can exit as `request_info` with ZERO evidence gaps -- the invoice
    # audit found a price/currency conflict, which is an internal records
    # problem, not missing customer evidence. Without this the drafter sees
    # only "request_info" plus a clean evidence gate and invents a plausible
    # but wrong ask (observed live: it requested product and packaging photos
    # the customer had already supplied).
    invoice_audit: InvoiceAudit | None = None
    # Populated by callers with merchant/policy memory context (module
    # docstring point 2); "" when no case/merchant context applies.
    memory_context: str = ""


class DraftOutput(BaseModel):
    """Forced tool-call output of the drafting LLM call.

    Deliberately exactly these two `str` fields -- there is no field the
    model could use to express a decision or an amount, so it is
    structurally impossible for its response to override
    `DraftInputs.decision` / `DraftInputs.amount`.
    """

    rationale: str
    email_draft: str


def _eligibility_text(result: EligibilityResult | None) -> str:
    if result is None:
        return "Eligibility gate: not run for this case."
    return (
        f"Eligibility gate: eligible={result.eligible}, "
        f"reason={result.reason or 'N/A'}, route={result.route}."
    )


def _evidence_text(gaps: list[Gap]) -> str:
    if not gaps:
        return "Evidence gate: no unresolved evidence gaps."
    lines = ["Evidence gate: the following evidence gaps were found:"]
    for gap in gaps:
        detail = gap.detail or "no detail available"
        lines.append(f"  - {gap.item.value}: {gap.reason} -- {detail}")
    return "\n".join(lines)


def _validation_text(decision: ValidationDecision | None) -> str:
    if decision is None:
        return "Validation gate: not run for this case."
    reason = decision.reason or "N/A"
    return f"Validation gate: outcome={decision.outcome.value}, reason={reason}."


# A leading `Subject: ...` line the model sometimes emits despite the prompt
# forbidding it. Anchored to the very start and to a single line so it can
# only ever match a header, never the word "subject" occurring in real prose.
_LEADING_SUBJECT_RE = re.compile(r"\A\s*subject\s*:[^\n]*\n+", re.IGNORECASE)


def _strip_leading_subject_line(body: str) -> str:
    """Drop a `Subject:` header the model put at the top of the email body.

    `web/app.py` computes the real subject itself (`_email_subject_for`) and
    sends it as the actual header, so one embedded in the body is duplicated
    -- it renders as literal "Subject: ..." text in the first line of what
    the merchant reads. Observed on a real run despite explicit prompt
    instruction, which is exactly why this deterministic backstop exists:
    prompt guidance reduces the rate, it doesn't guarantee the shape.

    Deliberately the *only* thing sanitized here. Placeholders like
    `[Your Name]` are also forbidden by the prompt but are NOT auto-stripped
    -- there's no safe rewrite (what replaces it?), and silently deleting
    text from a message a rep is about to send is worse than leaving
    something visibly wrong for them to catch.
    """
    return _LEADING_SUBJECT_RE.sub("", body, count=1)


def _invoice_audit_text(audit: InvoiceAudit | None) -> str:
    """Render the retail-invoice reconciliation for the drafter.

    Only says anything substantive when there are findings -- a clean or
    unverified audit is deliberately reported as a single flat line so it
    can't become filler the model feels obliged to mention to a customer.
    """
    if audit is None:
        return "Invoice reconciliation: not run for this case."
    if not audit.verified:
        return (
            "Invoice reconciliation: could not be verified "
            f"({audit.reason or 'no reason recorded'}). This is an internal note only -- "
            "do NOT mention it to the customer."
        )
    if not audit.discrepancies:
        return "Invoice reconciliation: ShipBob's invoice matches the merchant's retail invoice."
    lines = [
        "Invoice reconciliation: ShipBob's own invoice data DISAGREES with the "
        "retail invoice the merchant submitted. This is a discrepancy between "
        "two internal/merchant records -- it is NOT missing customer evidence, "
        "and the customer cannot fix it by sending more photos:",
    ]
    for d in audit.discrepancies:
        lines.append(f"  - [{d.severity.value}] {d.code}: {d.detail}")
    return "\n".join(lines)


def _calc_text(calc_result: CalcResult | None) -> str:
    if calc_result is None:
        return "Calc gate: not run for this case."
    lines = [
        f"Calc gate (underlying calculation, for context only -- see AUTHORITATIVE "
        f"AMOUNT above for the amount actually used): raw calculated amount="
        f"${calc_result.amount:.2f}, capped_at_policy_maximum={calc_result.capped}."
    ]
    if calc_result.line_items:
        lines.append("  Line items:")
        for li in calc_result.line_items:
            lines.append(
                f"    - sku={li.sku}, quantity={li.quantity}, "
                f"unit_price=${li.unit_price:.2f}, subtotal=${li.subtotal:.2f}"
            )
    else:
        lines.append("  No line items.")
    return "\n".join(lines)


def _risk_text(risk_assessment: RiskAssessment) -> str:
    tier = risk_assessment.tier.value
    if not risk_assessment.flags:
        return f"Risk gate: tier={tier}, no flags raised."
    flags_text = "; ".join(risk_assessment.flags)
    return f"Risk gate: tier={tier}, flags: {flags_text}."


def _recipient_block(case: Case) -> str:
    """Who the email is addressed to, so the model doesn't invent a
    placeholder for it.

    Without this the drafter has no name available and fills the gap with
    `[Merchant\'s Name]` -- observed live. A rep can approve a draft as-is,
    so a bracketed fill-in is a real risk of reaching a merchant verbatim.
    `account_name` is ShipBob\'s own record of the merchant, so it is stated
    as trusted text rather than wrapped as untrusted.
    """
    if case.account_name:
        return (
            f"Addressed to (ShipBob\'s own merchant record, trusted): {case.account_name}. "
            "Use this name in the greeting."
        )
    return (
        "Addressed to: no merchant name is on file for this case. Use a "
        "generic greeting such as \"Hello,\" -- never a placeholder."
    )


def _case_description_block(case: Case) -> str:
    description = case.description if case.description is not None else _NO_DESCRIPTION_MARKER
    return f"Case description (customer/merchant-authored, untrusted):\n<untrusted_data>{description}</untrusted_data>"


def _build_prompt_text(inputs: DraftInputs) -> str:
    """Assemble the user-message text for the drafting LLM call.

    Gate-result facts are plain trusted text (this system's own analysis,
    module docstring point 3); `case.description` is the only free-text
    field wrapped in `<untrusted_data>` tags. `inputs.decision`/`.amount`
    are stated explicitly and labeled FIXED so the model never treats them
    as open questions.
    """
    sections = [
        "The decision and payout amount for this claim have ALREADY BEEN "
        "DECIDED by deterministic ShipBob business logic. They are FIXED. "
        "Do not restate them differently, second-guess them, propose an "
        "alternative, or mention any other amount as if it might be paid. "
        "Your only job is to write the rationale bullets and the customer "
        "email explaining/reflecting this already-decided outcome.",
        f"DECISION (fixed): {inputs.decision}",
        f"AUTHORITATIVE AMOUNT (fixed): ${inputs.amount:.2f}",
        "",
        "System-generated gate facts (trusted, produced by our own pipeline "
        "-- not customer/merchant content):",
        _eligibility_text(inputs.eligibility_result),
        _evidence_text(inputs.evidence_gaps),
        _validation_text(inputs.validation_decision),
        _invoice_audit_text(inputs.invoice_audit),
        _calc_text(inputs.calc_result),
        _risk_text(inputs.risk_assessment),
        "",
        _recipient_block(inputs.case),
        "",
        _case_description_block(inputs.case),
        "",
        "Merchant/policy memory context (this system's own records, "
        "trusted -- not customer/merchant content; may include a labeled "
        "section of immediate rep feedback on a pushed-back draft):",
        inputs.memory_context if inputs.memory_context else "(none available yet)",
    ]
    return "\n".join(sections)


async def draft(
    inputs: DraftInputs,
    *,
    transport: Transport | None = None,
    db_path: Path | str | None = None,
) -> Recommendation:
    """Produce the final `Recommendation` for a case via one LLM drafting call.

    `inputs.decision` / `inputs.amount` / `inputs.confidence` /
    `inputs.risk_assessment.tier` are copied directly into the returned
    `Recommendation` -- see module docstring for the non-negotiable
    guarantee that nothing derived from the LLM's response can change them.
    `inputs.calc_result.line_items` (or `[]` if `calc_result` is `None`)
    become `Recommendation.line_items`, likewise deterministic, non-LLM
    data.

    `transport`/`db_path` are pass-through overrides to `structured_call`,
    mirroring `gates/evidence.py` / `gates/validation.py`'s test-injection
    story.
    """
    prompt_text = _build_prompt_text(inputs)
    messages = [{"role": "user", "content": prompt_text}]

    output = await structured_call(
        case_id=inputs.case.case_id,
        prompt_name=PROMPT_NAME,
        messages=messages,
        schema=DraftOutput,
        transport=transport,
        db_path=db_path,
    )

    line_items = list(inputs.calc_result.line_items) if inputs.calc_result is not None else []

    return Recommendation(
        decision=inputs.decision,
        amount=inputs.amount,
        line_items=line_items,
        rationale=output.rationale,
        email_draft=_strip_leading_subject_line(output.email_draft),
        confidence=inputs.confidence,
        risk_tier=inputs.risk_assessment.tier.value,
    )
