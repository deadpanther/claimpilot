"""Tests for the feedback distiller (the self-evolving loop).

`distill_feedback()` is exercised with `tests/test_llm.py`'s own
`FakeTransport` pattern, injected via `structured_call`'s `transport=`
override -- same approach as `tests/test_draft.py` -- so the real
`distill_feedback.md` prompt file and the real prompt-assembly logic are
actually exercised. No real network/LLM/Anthropic SDK call happens anywhere
in this file.

The single most important test group here is the validator
(`is_decision_free` / the module's internal `_violations`): this is a real
security control (the plan is explicit that memory must never be able to
override the deterministic decision/amount gates), so several tests are
deliberately adversarial -- mixed case, punctuation-adjacent words, a
dollar amount embedded mid-sentence, and the word-boundary false-positive
case ("disapproves") the task write-up itself calls out.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from claimpilot.evolve import (
    DistilledNote,
    DistillOutput,
    distill_feedback,
    is_decision_free,
)
from claimpilot.llm import StructuredCallError, TransportResult
from claimpilot.memory import global_policies, merchant_context
from claimpilot.models import Case
from tests.test_llm import FakeTransport


def _case(case_id: str = "CASE-1", user_id: str | None = "M1") -> Case:
    return Case(case_id=case_id, status="New", user_id=user_id)


def _db(tmp_path: Path) -> Path:
    return tmp_path / "t.db"


def _notes_result(notes: list[dict]) -> TransportResult:
    return TransportResult(
        tool_input={"notes": notes},
        input_tokens=10,
        output_tokens=10,
        raw_content=[],
    )


def _transport(*note_lists: list[dict]) -> FakeTransport:
    return FakeTransport([_notes_result(notes) for notes in note_lists])


ORIGINAL_DRAFT = "Dear customer,\nWe reviewed your claim.\nBest,\nShipBob"
FINAL_DRAFT = "Dear customer,\nWe reviewed your claim carefully.\nBest,\nShipBob"


# --- is_decision_free / validator ------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        "We should approve this faster next time.",
        "Approved notes should mention the account manager.",  # capitalized
        "APPROVING claims quickly matters.",
        "Consider denying claims more gently.",
        "This was a denial that upset the customer.",
        "Please don't reject future claims like this.",
        "The rep rejected the tone, not the claim.",
        "deny.",  # punctuation immediately after the word
        "Deny,",
        "(approved)",  # punctuation-adjacent, parens
    ],
)
def test_is_decision_free_rejects_decision_words_in_various_forms(content: str) -> None:
    assert is_decision_free(content) is False


@pytest.mark.parametrize(
    "content",
    [
        "The customer disapproves of the shipping delay.",  # contains "approve" as substring
        "This packaging was later reapproved by the vendor.",  # contains "approve"
        "Mention the merchant's dedicated account manager by name.",
        "Use shorter paragraphs and avoid corporate-sounding phrases.",
        "This merchant prefers formal, no-contractions language.",
    ],
)
def test_is_decision_free_does_not_false_positive_on_substrings(content: str) -> None:
    """Word-boundary regex, not substring matching -- `"approve"` is a
    substring of `"disapproves"`/`"reapproved"`, but neither word should be
    flagged. This is the exact false-positive risk the task write-up calls
    out explicitly.
    """
    assert is_decision_free(content) is True


@pytest.mark.parametrize(
    "content",
    [
        "Always offer $10 back for this kind of issue.",
        "The customer was quoted $1,234.56 in a prior email.",
        "A one-time credit of $0 was mentioned last time.",
        "Cost is $5 more than usual.",  # dollar amount embedded mid-sentence
        "Always offer $ 10 back for delays.",  # space between sign and digit
    ],
)
def test_is_decision_free_rejects_dollar_amounts(content: str) -> None:
    assert is_decision_free(content) is False


def test_is_decision_free_allows_amount_free_generosity_language() -> None:
    """A note that talks about generosity/discounts in the abstract (no `$`
    digit) is fine -- only a note anchored to a specific figure is banned.
    """
    assert is_decision_free("Always offer a small discount for repeat issues.") is True


def test_task_example_denial_phrase_is_rejected_by_design() -> None:
    """The task write-up's own example note -- "avoid a second apology in
    denial emails" -- contains the banned word "denial" and IS rejected by
    this validator. This is intentional, not a bug: the validator is the
    security control (per the task's explicit "so memory can never override
    the deterministic gates"), so it must reject this phrasing even though
    the task also offered it as an illustrative example. `distill_feedback.md`
    is written to avoid teaching the model this vocabulary in the first
    place (e.g. "keep the closing to a single expression of regret" instead
    of "avoid a second apology in denial emails") -- this test is the
    guardrail against someone "fixing" the validator to let this phrase
    through instead of fixing the prompt.
    """
    assert is_decision_free("Avoid a second apology in denial emails.") is False


# --- distill_feedback: persistence of valid notes ---------------------------


async def test_two_valid_notes_both_persisted(tmp_path: Path) -> None:
    db_path = _db(tmp_path)
    case = _case(user_id="M1")
    transport = _transport(
        [
            {"content": "Mention the account manager by name.", "scope": "merchant"},
            {"content": "Keep paragraphs short in every email.", "scope": "global"},
        ]
    )

    result = await distill_feedback(
        case, ORIGINAL_DRAFT, FINAL_DRAFT, feedback="Please shorten this.", transport=transport, db_path=db_path
    )

    assert [n.content for n in result] == [
        "Mention the account manager by name.",
        "Keep paragraphs short in every email.",
    ]

    ctx = merchant_context("M1", db_path=db_path)
    assert ctx.policy_notes == ["Mention the account manager by name."]
    assert global_policies(db_path=db_path) == ["Keep paragraphs short in every email."]


async def test_decision_word_note_dropped_not_persisted(tmp_path: Path) -> None:
    db_path = _db(tmp_path)
    case = _case(user_id="M1")
    transport = _transport(
        [{"content": "We should approve claims like this faster.", "scope": "global"}]
    )

    result = await distill_feedback(case, ORIGINAL_DRAFT, FINAL_DRAFT, transport=transport, db_path=db_path)

    assert result == []
    assert global_policies(db_path=db_path) == []
    assert merchant_context("M1", db_path=db_path).policy_notes == []


async def test_dollar_amount_note_dropped_not_persisted(tmp_path: Path) -> None:
    db_path = _db(tmp_path)
    case = _case(user_id="M1")
    transport = _transport([{"content": "Always offer $10 back for delays.", "scope": "merchant"}])

    result = await distill_feedback(case, ORIGINAL_DRAFT, FINAL_DRAFT, transport=transport, db_path=db_path)

    assert result == []
    assert merchant_context("M1", db_path=db_path).policy_notes == []


async def test_empty_notes_response_is_a_noop(tmp_path: Path) -> None:
    db_path = _db(tmp_path)
    case = _case(user_id="M1")
    transport = _transport([])

    result = await distill_feedback(case, ORIGINAL_DRAFT, ORIGINAL_DRAFT, transport=transport, db_path=db_path)

    assert result == []
    assert global_policies(db_path=db_path) == []
    assert merchant_context("M1", db_path=db_path).policy_notes == []


async def test_one_valid_one_invalid_only_valid_persists(tmp_path: Path) -> None:
    db_path = _db(tmp_path)
    case = _case(user_id="M1")
    transport = _transport(
        [
            {"content": "Mention the account manager by name.", "scope": "merchant"},
            {"content": "We denied this claim last time too.", "scope": "merchant"},
        ]
    )

    result = await distill_feedback(case, ORIGINAL_DRAFT, FINAL_DRAFT, transport=transport, db_path=db_path)

    assert [n.content for n in result] == ["Mention the account manager by name."]
    assert merchant_context("M1", db_path=db_path).policy_notes == ["Mention the account manager by name."]


async def test_blank_content_note_dropped_only_valid_persists(tmp_path: Path) -> None:
    """An empty/whitespace-only `content` is not a decision-word or
    dollar-amount violation, but must still be dropped -- see module
    docstring point 3/7: a persisted `scope="merchant"` note of ANY
    content, blank included, makes `MerchantMemory.flags` non-empty and
    lifts that merchant's next claim's risk tier for no real reason.
    """
    db_path = _db(tmp_path)
    case = _case(user_id="M1")
    transport = _transport(
        [
            {"content": "   ", "scope": "merchant"},
            {"content": "Keep paragraphs short in every email.", "scope": "global"},
        ]
    )

    result = await distill_feedback(case, ORIGINAL_DRAFT, FINAL_DRAFT, transport=transport, db_path=db_path)

    assert [n.content for n in result] == ["Keep paragraphs short in every email."]
    assert merchant_context("M1", db_path=db_path).policy_notes == []
    assert global_policies(db_path=db_path) == ["Keep paragraphs short in every email."]


async def test_feedback_none_does_not_crash(tmp_path: Path) -> None:
    db_path = _db(tmp_path)
    case = _case(user_id="M1")
    transport = _transport([])

    result = await distill_feedback(case, ORIGINAL_DRAFT, FINAL_DRAFT, feedback=None, transport=transport, db_path=db_path)

    assert result == []
    prompt = transport.calls[0]["messages"][-1]["content"]
    assert "no explicit feedback text given" in prompt


async def test_merchant_scope_note_dropped_when_case_has_no_user_id(tmp_path: Path) -> None:
    db_path = _db(tmp_path)
    case = _case(user_id=None)
    transport = _transport([{"content": "Mention the account manager by name.", "scope": "merchant"}])

    result = await distill_feedback(case, ORIGINAL_DRAFT, FINAL_DRAFT, transport=transport, db_path=db_path)

    assert result == []
    assert global_policies(db_path=db_path) == []


async def test_global_scope_note_still_persists_when_case_has_no_user_id(tmp_path: Path) -> None:
    db_path = _db(tmp_path)
    case = _case(user_id=None)
    transport = _transport([{"content": "Keep paragraphs short in every email.", "scope": "global"}])

    result = await distill_feedback(case, ORIGINAL_DRAFT, FINAL_DRAFT, transport=transport, db_path=db_path)

    assert [n.content for n in result] == ["Keep paragraphs short in every email."]
    assert global_policies(db_path=db_path) == ["Keep paragraphs short in every email."]


# --- diff computation --------------------------------------------------------


async def test_diff_reflects_the_actual_changed_line(tmp_path: Path) -> None:
    db_path = _db(tmp_path)
    case = _case(user_id="M1")
    transport = _transport([])

    await distill_feedback(case, ORIGINAL_DRAFT, FINAL_DRAFT, transport=transport, db_path=db_path)

    prompt = transport.calls[0]["messages"][-1]["content"]
    # unified_diff emits "-"/"+"-prefixed lines (no space after the sign)
    # for the removed/added lines specifically -- not just any mention of
    # the drafts' text.
    assert "-We reviewed your claim." in prompt
    assert "+We reviewed your claim carefully." in prompt
    # Unchanged lines are not what's being asserted on here, but the diff
    # should not, e.g., just be the two full drafts pasted with no diff
    # markers at all.
    assert "@@" in prompt  # unified diff hunk header


async def test_identical_drafts_produce_a_clear_no_diff_marker(tmp_path: Path) -> None:
    db_path = _db(tmp_path)
    case = _case(user_id="M1")
    transport = _transport([])

    await distill_feedback(case, ORIGINAL_DRAFT, ORIGINAL_DRAFT, transport=transport, db_path=db_path)

    prompt = transport.calls[0]["messages"][-1]["content"]
    assert "no textual differences" in prompt


# --- schema-level max 2 notes enforcement ------------------------------------


async def test_more_than_two_notes_retries_then_raises_and_persists_nothing(tmp_path: Path) -> None:
    """`DistillOutput.notes` has `max_length=2` -- a model response with 3
    notes fails `schema.model_validate(...)`, triggering `structured_call`'s
    one retry with a corrective message; if the retry also returns too many,
    `StructuredCallError` propagates out of `distill_feedback` uncaught (this
    function is deliberately exception-transparent -- see its docstring),
    and nothing from either attempt is ever persisted.
    """
    db_path = _db(tmp_path)
    case = _case(user_id="M1")
    over_limit = [
        {"content": "Note one.", "scope": "global"},
        {"content": "Note two.", "scope": "global"},
        {"content": "Note three.", "scope": "global"},
    ]
    transport = _transport(over_limit, over_limit)

    with pytest.raises(StructuredCallError):
        await distill_feedback(case, ORIGINAL_DRAFT, FINAL_DRAFT, transport=transport, db_path=db_path)

    assert len(transport.calls) == 2  # initial attempt + exactly one retry
    assert global_policies(db_path=db_path) == []
    assert merchant_context("M1", db_path=db_path).policy_notes == []


# --- no <untrusted_data> wrapping (documented judgment call) -----------------


async def test_diff_and_feedback_are_not_wrapped_in_untrusted_data_tags(tmp_path: Path) -> None:
    """Module docstring point 4: rep-authored/system-generated content (the
    diff, the feedback text) is not customer/merchant-authored, so it is
    deliberately NOT wrapped in `<untrusted_data>` tags, unlike
    `case.description` in `draft.py`.
    """
    db_path = _db(tmp_path)
    case = _case(user_id="M1")
    transport = _transport([])

    await distill_feedback(
        case, ORIGINAL_DRAFT, FINAL_DRAFT, feedback="Please shorten this.", transport=transport, db_path=db_path
    )

    prompt = transport.calls[0]["messages"][-1]["content"]
    assert "<untrusted_data>" not in prompt


# --- schema shape sanity ------------------------------------------------------


def test_distill_output_schema_rejects_more_than_two_notes_directly() -> None:
    with pytest.raises(ValidationError):
        DistillOutput.model_validate(
            {
                "notes": [
                    {"content": "one", "scope": "global"},
                    {"content": "two", "scope": "global"},
                    {"content": "three", "scope": "global"},
                ]
            }
        )


def test_distilled_note_schema_rejects_unknown_scope() -> None:
    with pytest.raises(ValidationError):
        DistilledNote.model_validate({"content": "some note", "scope": "not_a_real_scope"})
