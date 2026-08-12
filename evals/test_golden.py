"""Golden-set eval harness.

Two very different kinds of tests live in this file:

1. A plain, always-run structural validation test
   (`test_golden_yaml_is_well_formed`) that checks `evals/golden.yaml`
   itself: all 7 known fixture cases are present, required keys exist, and
   values are shaped sensibly (booleans are booleans, amounts parse as
   `Decimal`, gap ranges are internally consistent, etc.). This test makes
   NO network/LLM calls and is part of the default suite -- it gives real
   regression value (e.g. it fails if a case is accidentally deleted from
   `golden.yaml`, or an amount field gets corrupted into something that
   doesn't parse) even in an environment where the heavy real-LLM eval
   tests below can never run.

2. `@pytest.mark.eval` tests -- one per golden case -- that run the full
   pipeline (`pipeline.process_case`) against the REAL LLM API (no
   `FakeTransport`, no injected client wrapper around attachment bytes:
   this is the entire point of a golden-set eval, per the plan's "no
   prompt change ships without green evals" rule) and assert the actual
   outcome against `golden.yaml`'s expectations -- exact match for
   deterministic/pinned fields (eligibility, and calc amount where pinned),
   membership/range checks for genuine vision-judgment fields (evidence gap
   count, validation outcome). Which real API that is depends on
   `settings.llm_provider` (post-demo-v1): `get_transport()` returns
   `AnthropicTransport` or `OpenAITransport` accordingly, unchanged from
   how every other real call in this system picks its provider.

   These are excluded from the default `pytest` run via
   `pyproject.toml`'s `addopts = "-m 'not eval'"` (marker-based deselection,
   not skip-based -- they are never even collected into the default run's
   session, not merely skipped after being selected). Running `pytest -m
   eval` explicitly selects them. Each one additionally guards on
   `_llm_provider_has_api_key()` (checks `openai_api_key` or
   `anthropic_api_key` depending on `settings.llm_provider`) via
   `pytest.mark.skipif`, so `pytest -m eval` in a credential-less
   environment (this one, right now) reports "skipped", never "failed" or
   "hung" -- there is no real key configured here, so these tests cannot
   actually be exercised in this environment; that is expected.

   Operational precondition for whoever *does* run `pytest -m eval` with a
   real key: `FixtureClient.get_attachment_bytes` downloads the real
   fixture photos from Azure blob storage on demand (there is no local
   image cache checked into this repo) -- so a real run needs network
   reachability to that blob storage host, not just a valid API key. If the
   key is present but the network is unavailable, these tests will error
   partway through a case rather than skip cleanly (unlike the
   no-API-key path, which is guarded up front).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest
import yaml

from claimpilot.config import configured_api_key, settings
from claimpilot.gates.validation import ValidationOutcome, combine_validation

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.yaml"

# Gates that sit AFTER damage validation and before the reimbursement calc,
# and can escalate a case that validated cleanly. Their existence means
# "validation returned PROCEED" no longer implies "calc ran" -- see
# `_check_calc`. Kept as a named set rather than inlined so adding a fifth
# cross-check is a one-line change here instead of a mystery eval failure.
POST_VALIDATION_ESCALATIONS = {
    "gate:validation_affected_count_mismatch",
    "gate:invoice_audit_discrepancy",
    "gate:claim_scope_mismatch",
}

REQUIRED_CASE_IDS = {
    "CASE-1001",
    "CASE-1002",
    "CASE-1003",
    "CASE-1004",
    "CASE-1005",
    "CASE-9001-INSURED",
    "CASE-9002-CAP",
}

VALID_ELIGIBILITY_REASONS = {"TOO_OLD", "WRONG_TYPE", "INSURED", "MISSING_INFO", None}
VALID_VALIDATION_OUTCOMES = {"proceed", "request_info", "escalated"}

REQUIRED_CASE_KEYS = {
    "case_id",
    "expected_eligibility",
    "expected_evidence",
    "expected_validation",
    "expected_calc",
    "notes",
}


def _load_golden() -> list[dict]:
    with GOLDEN_PATH.open() as f:
        doc = yaml.safe_load(f)
    return doc["cases"]


# --- 1. Always-run structural validation (no network, no LLM) --------------


def test_golden_yaml_is_well_formed():
    """Regression net for `golden.yaml`'s own shape -- catches a case being
    accidentally removed, a required key being dropped, or a value being
    corrupted into something that doesn't parse, independent of whether the
    real-LLM eval tests below can ever run in this environment.
    """
    cases = _load_golden()
    assert isinstance(cases, list) and len(cases) == 7

    seen_ids = set()
    for case in cases:
        missing = REQUIRED_CASE_KEYS - case.keys()
        assert not missing, f"{case.get('case_id')} missing keys: {missing}"

        case_id = case["case_id"]
        assert isinstance(case_id, str) and case_id
        seen_ids.add(case_id)

        _assert_eligibility_shape(case_id, case["expected_eligibility"])
        _assert_evidence_shape(case_id, case["expected_evidence"])
        _assert_validation_shape(case_id, case["expected_validation"])
        _assert_calc_shape(case_id, case["expected_calc"])
        assert isinstance(case["notes"], str) and case["notes"].strip()
        _assert_cross_field_consistency(case_id, case)

    assert seen_ids == REQUIRED_CASE_IDS, (
        f"golden.yaml case IDs don't match the 7 known fixture cases: "
        f"missing={REQUIRED_CASE_IDS - seen_ids}, extra={seen_ids - REQUIRED_CASE_IDS}"
    )


def _assert_eligibility_shape(case_id: str, block: dict) -> None:
    assert {"eligible", "reason"} <= block.keys(), case_id
    assert isinstance(block["eligible"], bool), case_id
    assert block["reason"] in VALID_ELIGIBILITY_REASONS, (case_id, block["reason"])
    # Structural sanity from gates/eligibility.py: an ineligible outcome
    # always carries a machine reason code (never None); INSURED is the one
    # reason code that is still `eligible=True` (it's a routing decision,
    # not a denial).
    if not block["eligible"]:
        assert block["reason"] is not None, case_id


def _assert_evidence_shape(case_id: str, block) -> None:
    if block == "not_reached":
        return
    assert isinstance(block, dict), case_id
    assert {"min_gaps", "max_gaps"} <= block.keys(), case_id
    min_gaps, max_gaps = block["min_gaps"], block["max_gaps"]
    assert isinstance(min_gaps, int) and isinstance(max_gaps, int), case_id
    assert 0 <= min_gaps <= max_gaps <= 4, (case_id, min_gaps, max_gaps)


def _assert_validation_shape(case_id: str, block) -> None:
    if block == "not_reached" or block is None:
        return
    assert isinstance(block, list) and block, case_id
    assert set(block) <= VALID_VALIDATION_OUTCOMES, (case_id, block)


def _assert_calc_shape(case_id: str, block) -> None:
    if block in ("not_reached", "unpinned") or block is None:
        return
    assert isinstance(block, str), case_id
    try:
        amount = Decimal(block)
    except InvalidOperation:
        pytest.fail(f"{case_id}: expected_calc {block!r} is not Decimal-parseable")
    assert amount >= Decimal("0"), case_id
    # settings.cap is $100.00 by default (env-overridable via CAP) -- no
    # real case should ever exceed the configured cap.
    assert amount <= settings.cap, case_id


def _assert_cross_field_consistency(case_id: str, case: dict) -> None:
    """Catches an incoherent hand-edit that the per-field shape checks above
    can't see on their own -- e.g. a case marked ineligible that still
    claims a pinned evidence/validation/calc expectation, which would be
    self-contradictory (an ineligible case's pipeline path never reaches
    evidence/validation/calc at all).
    """
    eligibility = case["expected_eligibility"]
    evidence = case["expected_evidence"]
    validation = case["expected_validation"]
    calc = case["expected_calc"]

    if not eligibility["eligible"]:
        assert evidence == "not_reached", f"{case_id}: ineligible but evidence isn't not_reached"
        assert validation in ("not_reached", None), f"{case_id}: ineligible but validation isn't not_reached"
        assert calc == "not_reached", f"{case_id}: ineligible but calc isn't not_reached"
        return

    # INSURED routes away before evidence ever runs, even though eligible=True.
    if eligibility["reason"] == "INSURED":
        assert evidence == "not_reached", f"{case_id}: insured but evidence isn't not_reached"

    if evidence == "not_reached":
        assert validation in ("not_reached", None), f"{case_id}: evidence not_reached but validation isn't"
        assert calc == "not_reached", f"{case_id}: evidence not_reached but calc isn't"
        return

    # A gap range whose minimum is >= 1 means every gap count in range is
    # >= 1 -- request_info is forced across the whole range, so validation
    # (and therefore calc) can never run.
    if evidence["min_gaps"] >= 1:
        assert validation in ("not_reached", None), (
            f"{case_id}: min_gaps>=1 forces request_info, but validation isn't not_reached"
        )
        assert calc == "not_reached", f"{case_id}: min_gaps>=1 forces request_info, but calc isn't not_reached"


# --- 2. Real-LLM golden evals (@pytest.mark.eval) ---------------------------

# Post-demo-v1: `settings.llm_provider` selects which real API `get_transport()`
# calls (see `claimpilot.llm`), so the "do we have a usable key" guard below
# must check the key for whichever provider is actually configured, not
# unconditionally `anthropic_api_key` -- a provider-blind guard would either
# skip a real `llm_provider="openai"` run that has a perfectly good
# `OPENAI_API_KEY` set, or (worse) let a run through with `anthropic_api_key`
# set but `llm_provider="openai"` and no `openai_api_key`, which would then
# fail with an `OpenAIError` at `OpenAITransport()` construction instead of
# skipping cleanly.
_NO_API_KEY_REASON = "no API key configured for settings.llm_provider, skipping real-LLM eval"


def _llm_provider_has_api_key() -> bool:
    return configured_api_key()[0]


@dataclass(frozen=True)
class _GoldenCase:
    case_id: str
    raw: dict


def _golden_cases() -> list[_GoldenCase]:
    return [_GoldenCase(case_id=c["case_id"], raw=c) for c in _load_golden()]


@pytest.mark.eval
@pytest.mark.skipif(not _llm_provider_has_api_key(), reason=_NO_API_KEY_REASON)
@pytest.mark.parametrize("golden", _golden_cases(), ids=lambda gc: gc.case_id)
async def test_golden_case_matches_real_pipeline_run(golden: _GoldenCase, tmp_path: Path):
    """Run one golden case through the real pipeline (real `FixtureClient`
    for case/shipment/order/invoice metadata, real `get_transport()` -> the
    real Anthropic or OpenAI API, per `settings.llm_provider`, for every
    vision call) and check the outcome against `golden.yaml`'s
    pinned/ranged expectations.

    Deliberately does NOT inject a `FakeTransport` or a client wrapper
    around attachment bytes -- unlike `tests/test_pipeline.py`'s scripted
    integration tests, the whole point here is exercising real vision
    judgment end to end. This import is intentionally local/lazy: importing
    `claimpilot.pipeline`/`claimpilot.store` at module scope would be fine
    functionally, but keeping the real-call imports scoped to the
    eval-marked test makes it obvious at a glance which parts of this file
    are "structural only" vs. "hits the real API".
    """
    from claimpilot import store
    from claimpilot.pipeline import process_case

    from datetime import datetime, timezone

    db_path = tmp_path / "golden.db"
    # Pinned, NOT `datetime.now(timezone.utc)`: all 7 fixture cases have
    # delivery dates in March 2026 (see fixtures/synthetic.json / the real
    # ShipBob fixture data). Using the real wall clock would make every
    # case read as TOO_OLD once enough real time has passed (it already has
    # -- "today" per this session is well past March 2026), which would
    # fail every eligibility pin in golden.yaml for the wrong reason. Same
    # value tests/test_pipeline.py uses for the same reason.
    now = datetime(2026, 3, 25, tzinfo=timezone.utc)

    await process_case(golden.case_id, db_path=db_path, now=now)

    gates = store.load_gate_results(golden.case_id, db_path=db_path)
    raw = golden.raw

    _check_eligibility(golden.case_id, gates.eligibility, raw["expected_eligibility"])
    evidence_passed = _check_evidence(golden.case_id, gates.evidence_gaps, raw["expected_evidence"])
    outcome = _check_validation(
        golden.case_id, gates.validation, raw["expected_validation"], evidence_passed
    )
    events = [row["event"] for row in store.get_audit_log(golden.case_id, db_path=db_path)]
    _check_calc(golden.case_id, gates.calc, raw["expected_calc"], outcome, events)


def _check_eligibility(case_id: str, eligibility, expected: dict) -> None:
    assert eligibility is not None, f"{case_id}: no eligibility gate result recorded"
    assert eligibility.eligible == expected["eligible"], case_id
    assert eligibility.reason == expected["reason"], case_id


def _check_evidence(case_id: str, gaps: list, expected) -> bool:
    """Checks the gap count against golden.yaml's expectation and returns
    whether evidence structurally passed cleanly (0 gaps) this run -- the
    one condition under which the pipeline's real control flow proceeds on
    to validation. `expected == "not_reached"` means the evidence gate
    itself never ran (an earlier gate short-circuited first), which must
    show up as no gaps recorded at all, not merely "gaps happen to be
    empty".
    """
    if expected == "not_reached":
        assert gaps == [], f"{case_id}: evidence gate expected not to run, but gaps were recorded: {gaps}"
        return False
    assert expected["min_gaps"] <= len(gaps) <= expected["max_gaps"], (
        case_id,
        len(gaps),
        expected,
    )
    return len(gaps) == 0


def _check_validation(case_id: str, validation, expected, evidence_passed: bool) -> ValidationOutcome | None:
    if not evidence_passed:
        assert validation is None, f"{case_id}: evidence gate didn't pass cleanly, but validation ran anyway"
        return None
    if expected in ("not_reached", None):
        pytest.fail(
            f"{case_id}: golden.yaml inconsistency -- evidence passed cleanly this run "
            "but expected_validation is not_reached (see structural cross-field check)"
        )
    assert validation is not None, f"{case_id}: evidence passed cleanly but validation didn't run"
    decision = combine_validation(validation)
    assert decision.outcome.value in expected, (case_id, decision.outcome.value, expected)
    return decision.outcome


def _check_calc(
    case_id: str, calc, expected, outcome: ValidationOutcome | None, events: list[str]
) -> None:
    reached_calc = outcome == ValidationOutcome.PROCEED

    if expected == "not_reached":
        assert calc is None, f"{case_id}: expected calc not to run, but it did"
        return

    if not reached_calc:
        assert calc is None, f"{case_id}: validation didn't PROCEED, but calc ran anyway"
        return

    # A clean validation no longer guarantees the calc ran. Three cross-checks
    # sit between them (affected-count, retail-invoice reconciliation, claim
    # scope) and any of them can escalate a case whose damage evidence was
    # perfectly fine -- e.g. ShipBob's invoice disagreeing with the merchant's
    # own about price or quantity. That is the system working, not a
    # regression, so it is accepted here rather than failed.
    #
    # Deliberately verified against the case's real audit log rather than by
    # recomputing the checks: this asserts what the pipeline actually did, so
    # a genuine "validation proceeded and calc silently didn't run" bug still
    # fails, because no escalation event would have been written.
    escalated_after_validation = POST_VALIDATION_ESCALATIONS.intersection(events)
    if calc is None and escalated_after_validation:
        return

    assert calc is not None, (
        f"{case_id}: validation proceeded but no calc result recorded, and no "
        f"post-validation cross-check escalated it either. Events: {events}"
    )
    if expected == "unpinned":
        # Amount depends on which SKU(s) the real model matched -- we only
        # assert that calc genuinely ran, not what it produced.
        return
    assert calc.amount == Decimal(expected), (case_id, calc.amount, expected)
