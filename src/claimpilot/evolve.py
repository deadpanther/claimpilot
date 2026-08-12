r"""Feedback distiller (the self-evolving loop).

On every rep edit (approve-endpoint edit-detection branch) or pushback,
`web/app.py` calls `distill_feedback()` here with the original AI draft, the
rep's final draft, and any explicit feedback text. This module makes one LLM
call to extract 0-2 short, reusable, *style/context-only* policy notes from
that edit, runs every returned note through a validator, and persists only
the notes that pass via `claimpilot.memory.record_policy_note`, so they
become available to future drafting prompts through
`claimpilot.memory.merchant_context()` / `global_policies()`.

**The whole point of this module is a security control, not a feature**:
memory must never be able to override the deterministic decision/amount
gates (`gates/eligibility.py`, `calc.py`, `gates/validation.py`). A note
that could plausibly steer a future decision (a decision word, a dollar
amount) is dropped, not passed through with a warning suffix or otherwise
softened -- see `is_decision_free()`.

Design decisions (judgment calls, documented per this project's house
style):

1. **Validator pattern: word-boundary regex, not substring matching.** The
   task write-up itself flags the substring-matching failure mode:
   `"approve"` is a substring of `"disapproves"`, and naive substring
   matching would flag a sentence like "the customer disapproves of the
   shipping delay" even though it contains no decision word. `_DECISION_WORD_RE`
   uses `\b...\b` word-boundary anchors, so it matches whole words only
   (`"approve"`, `"approved"`, ...) and does not fire on `"disapproves"`,
   `"reapproved"`, or any other word that merely *contains* one of the
   banned words as a substring. Case-insensitive, so `"Approved"`/`"APPROVED"`
   are caught too. The word list is deliberately broader than the task's
   literal example list (adds plural/gerund/noun forms: `approves`,
   `rejects`, `rejecting`, `rejection`) since a validator meant to be a real
   security boundary should not have an obvious gap a slightly different
   inflection could slip through. **This is a fixed lexical denylist, not
   semantic decision-detection** -- a paraphrase or synonym of a decision
   word (`"declined"`, `"refused"`, `"turned down"`, ...) is NOT caught by
   this regex, since it isn't in the list. This validator is *defense in
   depth*, not the load-bearing safety control: the actual structural
   guarantee that memory can never change a decision/amount is `draft.py`'s
   `DraftOutput` schema having exactly two `str` fields (`rationale`,
   `email_draft`) with no field a decision or amount could be smuggled
   into (see that module's docstring) -- a persisted note, however it's
   worded, is just more prompt text fed into that same structurally-safe
   call. Stopping at lexical coverage of the task's named vocabulary
   (rather than attempting semantic detection, which is a much larger
   undertaking than this task calls for) is a deliberate, documented
   choice, not an oversight.
2. **Dollar-amount pattern: `\$\s*\d`.** Any dollar sign, optionally
   followed by whitespace, then a digit -- `"$10"`, `"$1,234.56"`, `"$0"`,
   `"$ 10"` (a space after the sign is ordinary LLM/human output, not an
   adversarial construction, and would otherwise slip a real payout figure
   straight into a future drafting prompt), a dollar amount embedded
   mid-sentence ("...offer $10 back..."). The task's own guidance names
   `\$\d`; this is a deliberate one-character tightening (`\s*` between the
   sign and the digit) to close the "$ 10" gap, not a deviation from the
   intent. This deliberately does NOT try to catch amounts spelled out in
   words ("ten dollars") or bare numbers with no currency marker --
   reliably distinguishing "a note that mentions a quantity" (e.g. "always
   offer a discount" -- fine) from "a note that mentions a specific price"
   without a marker would need much more machinery than this task calls
   for. Documented gap, not silently assumed away.
3. **Drop-and-log, never fail the whole call.** A note that fails validation
   is dropped individually (via `logging.warning`) -- the other 0-1 note(s)
   in the same response, if any, are still processed normally. A single bad
   note must never discard an otherwise-good one, and must never raise (this
   is normal, expected model behavior to guard against, not an exceptional
   condition). An empty/whitespace-only `content` is dropped the same way
   (not a decision-word/dollar-amount violation, but equally not worth
   persisting) -- besides being useless prompt text, per point 7 below a
   persisted `scope="merchant"` note of any content, including an empty
   one, makes that merchant's `MerchantMemory.flags` non-empty and lifts
   their next claim's risk tier, so a blank note must not slip through.
4. **No `<untrusted_data>` wrapping for the diff/feedback text.** `structured_call`'s
   standing rule only guarantees the model is told what those tags mean;
   callers decide what to wrap. The diff (computed from `original_draft`/
   `final_draft`, both of which are prior LLM output plus a rep's own edit)
   and the feedback text (typed directly by a rep in this system's own UI)
   are both this system's own records, not customer/merchant-authored
   content -- the same category as the gate-result facts `draft.py`'s
   `_build_prompt_text` sends as plain trusted text. This is a judgment
   call worth being explicit about: a rep *could* paste customer text into
   the feedback box, but that's true of gate results and rationale text
   too, and this project's existing convention (`draft.py` point 3) already
   draws the untrusted-data line at "customer/merchant-authored free text
   fields on the domain model," not at "any text a human typed into a
   form." Consistent with that precedent, rep-typed feedback is not wrapped.
5. **`distill_feedback()` does not itself catch `structured_call`
   errors.** It's a straightforward async function: a transport failure or
   `StructuredCallError` (e.g. the model returns >2 notes and the schema's
   `max_length=2` constraint fails validation on both the initial attempt
   and the one retry `structured_call` allows) propagates to the caller
   normally. The "distiller failure must never block an approve/pushback
   request" requirement is handled at the `web/app.py` call sites instead
   (each wraps its `distill_feedback()` call in a narrow `try/except
   Exception`, logs a warning, and continues) -- see that module for why.
   Keeping this function itself exception-transparent means direct unit
   tests here can assert on failures precisely, rather than testing through
   a layer that always swallows them.
6. **A `scope="merchant"` note is dropped (not persisted, not raised) when
   `case.user_id is None`**, mirroring `memory.record_correction`'s guard at
   its call sites: there is no merchant to attribute a merchant-scoped note
   to. `record_policy_note` itself raises `ValueError` for this
   combination; this module checks first and logs+drops instead, since a
   case with no `user_id` is expected real-world data (per `Case.user_id`
   being optional), not a caller bug worth raising over. A `scope="global"`
   note from the same response is still persisted normally.
7. **Known interaction with `claimpilot.risk.tier()`.**
   `memory.merchant_context()` feeds `policy_notes` (merchant-scoped
   `kind="policy"` rows -- exactly what a `scope="merchant"`
   distilled note becomes once persisted) into `risk.MerchantMemory.flags`,
   and `risk.tier()` treats *any* non-empty `flags` as one triggered risk
   factor (see `risk.py` point 2/3) -- enough on its own to lift a claim
   from `LOW` to `ELEVATED`. So a purely stylistic distilled note (e.g.
   "mention the account manager by name") will, once persisted, make that
   merchant's *next* claim read as `ELEVATED` risk, even though nothing
   about the note is actually risk-relevant. This is inherited from
   `memory.py`'s existing design (see its module docstring points 2-3), not
   introduced here, but it directly affects how the demo will read once
   the distiller is wired end-to-end: a case that clearly should stay `LOW`
   risk may show `ELEVATED` purely because a past rep edit produced a style
   note. Worth a conscious call at that point (e.g. whether `risk.tier()`
   should treat `flags` sourced from `kind="policy"` rows differently than
   a hypothetical future "curated risk flag" concept) rather than treated
   as a surprise.
"""

from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from claimpilot import memory
from claimpilot.llm import Transport, structured_call
from claimpilot.models import Case

PROMPT_NAME = "distill_feedback"

logger = logging.getLogger(__name__)

# --- Validator -----------------------------------------------------------

# Word list deliberately broader than the task's literal examples (adds
# plural/gerund/noun forms) -- see module docstring point 1. `\b...\b`
# anchors mean these match whole words only: "disapproves" is NOT flagged
# (no word boundary between "dis" and "approves"), but "Approved",
# "APPROVING", "deny." (punctuation-adjacent) all are.
_DECISION_WORDS = (
    "approve",
    "approves",
    "approved",
    "approving",
    "deny",
    "denies",
    "denied",
    "denying",
    "denial",
    "reject",
    "rejects",
    "rejected",
    "rejecting",
    "rejection",
)
_DECISION_WORD_RE = re.compile(r"\b(?:" + "|".join(_DECISION_WORDS) + r")\b", re.IGNORECASE)

# A dollar sign, optional whitespace, then a digit -- module docstring
# point 2. The `\s*` is a deliberate one-character tightening of the task's
# literal `\$\d` to also catch "$ 10" (space after the sign).
_DOLLAR_AMOUNT_RE = re.compile(r"\$\s*\d")


def _violations(content: str) -> list[str]:
    """Return a list of human-readable reasons `content` fails validation
    (empty list if it passes). Used by both `is_decision_free()` and
    `distill_feedback()`'s per-note logging, so the log message and the
    boolean check can never disagree about *why* a note was rejected.
    """
    reasons = []
    if _DECISION_WORD_RE.search(content):
        reasons.append("contains a decision word")
    if _DOLLAR_AMOUNT_RE.search(content):
        reasons.append("contains a dollar amount")
    return reasons


def is_decision_free(content: str) -> bool:
    """`True` iff `content` contains no decision word and no dollar amount.

    This is the security boundary the module docstring describes: a note
    that fails this check is never persisted, regardless of how useful it
    might otherwise be. See `_DECISION_WORD_RE`/`_DOLLAR_AMOUNT_RE` for the
    exact patterns and the reasoning behind each.
    """
    return not _violations(content)


# --- Schemas ---------------------------------------------------------------


class DistilledNote(BaseModel):
    """One extracted policy note -- part of `DistillOutput`'s forced tool
    schema. `content` is validated post-hoc by `is_decision_free()` (not a
    pydantic validator on this class itself), since a validation failure
    here should silently drop the one bad note rather than fail the whole
    `structured_call` and lose 0-1 other, valid notes in the same response.
    """

    content: str
    scope: Literal["global", "merchant"]


class DistillOutput(BaseModel):
    """Forced tool-call output of the distilling LLM call.

    `max_length=2` on `notes` enforces the plan's "0-2 short policy notes"
    at the schema level: `structured_call` folds this into the tool's JSON
    Schema (`maxItems: 2`), so a model response with 3+ notes fails
    `schema.model_validate(...)`, triggers `structured_call`'s one retry
    with a corrective message, and raises `StructuredCallError` if the
    retry also returns too many -- exactly the same "retry once, then
    raise" behavior any other schema-validation failure gets. See
    `tests/test_evolve.py`'s over-limit test.
    """

    notes: list[DistilledNote] = Field(default_factory=list, max_length=2)


# --- Diff + prompt assembly -------------------------------------------------


def _compute_diff(original_draft: str, final_draft: str) -> str:
    """Unified diff between `original_draft` and `final_draft`, so the model
    sees a concrete, concise view of what actually changed rather than two
    full drafts it has to eyeball for a delta. `difflib.unified_diff` emits
    `-`/`+`-prefixed lines (no space after the sign) for removed/added
    lines; a no-op edit (identical drafts) still gets a clear marker rather
    than an empty diff block, since an empty string in the prompt reads
    ambiguously ("was the diff omitted, or genuinely empty?").
    """
    original_lines = original_draft.splitlines()
    final_lines = final_draft.splitlines()
    diff_lines = list(
        difflib.unified_diff(original_lines, final_lines, fromfile="original_draft", tofile="final_draft", lineterm="")
    )
    if not diff_lines:
        return "(no textual differences between the original and final draft)"
    return "\n".join(diff_lines)


def _build_prompt_text(diff_text: str, feedback: str | None) -> str:
    """Assemble the user-message text for the distilling LLM call.

    Neither section is wrapped in `<untrusted_data>` tags -- both the diff
    (derived from a prior LLM draft plus a rep's own edit) and the feedback
    text (typed directly by a rep in this system's own UI) are this
    system's own records, not customer/merchant-authored content. See
    module docstring point 4 for the full reasoning.
    """
    sections = [
        "Unified diff between the AI-drafted email ('original_draft') and "
        "the rep's final sent/redrafted version ('final_draft'). Lines "
        "starting with '-' were removed, lines starting with '+' were added:",
        diff_text,
        "",
        "Rep-provided feedback text explaining this edit or pushback, if any:",
        feedback if feedback else "(no explicit feedback text given -- this was a silent edit, no separate feedback typed)",
    ]
    return "\n".join(sections)


# --- Public entry point ------------------------------------------------------


async def distill_feedback(
    case: Case,
    original_draft: str,
    final_draft: str,
    feedback: str | None = None,
    *,
    transport: Transport | None = None,
    db_path: Path | str | None = None,
) -> list[DistilledNote]:
    """Distill one rep edit/pushback into 0-2 durable policy notes, persist
    the ones that pass validation, and return exactly those.

    Makes one `structured_call` (schema `DistillOutput`) with a unified diff
    of `original_draft` vs. `final_draft` plus `feedback` as context. Every
    returned note is run through `is_decision_free()`; a note that fails is
    dropped (logged via `logging.warning`, not raised -- module docstring
    point 3) and never reaches `memory.record_policy_note`. A `scope=
    "merchant"` note is additionally dropped if `case.user_id is None`
    (module docstring point 6) -- there is no merchant to attribute it to.

    Does NOT catch `structured_call` failures (e.g. `StructuredCallError`
    if the model persistently returns more than 2 notes, or a transport
    exception) -- those propagate to the caller. `web/app.py`'s two call
    sites are responsible for treating a distiller failure as non-blocking
    (module docstring point 5); this function stays exception-transparent
    so it can be tested and reasoned about directly.

    Returns the list of notes that were actually persisted (empty list if
    the model returned none, or if every returned note failed validation).
    """
    diff_text = _compute_diff(original_draft, final_draft)
    prompt_text = _build_prompt_text(diff_text, feedback)
    messages = [{"role": "user", "content": prompt_text}]

    output = await structured_call(
        case_id=case.case_id,
        prompt_name=PROMPT_NAME,
        messages=messages,
        schema=DistillOutput,
        transport=transport,
        db_path=db_path,
    )

    accepted: list[DistilledNote] = []
    for note in output.notes:
        if not note.content.strip():
            # Not a decision-word/dollar-amount violation, but equally not
            # worth persisting (module docstring point 3) -- and per point
            # 7, ANY persisted `scope="merchant"` note, blank content
            # included, makes that merchant's risk tier climb, so this must
            # be checked before the note ever reaches `record_policy_note`.
            logger.warning("distill_feedback: dropping empty/blank note for case %s", case.case_id)
            continue
        reasons = _violations(note.content)
        if reasons:
            logger.warning(
                "distill_feedback: dropping note for case %s (%s): %r",
                case.case_id,
                "; ".join(reasons),
                note.content,
            )
            continue
        if note.scope == "merchant" and case.user_id is None:
            logger.warning(
                "distill_feedback: dropping merchant-scoped note for case %s -- no user_id to "
                "attribute it to: %r",
                case.case_id,
                note.content,
            )
            continue

        memory.record_policy_note(
            content=note.content,
            scope=note.scope,
            merchant_id=case.user_id if note.scope == "merchant" else None,
            source_case_id=case.case_id,
            db_path=db_path,
        )
        accepted.append(note)

    return accepted
