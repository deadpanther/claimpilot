"""Eligibility gate.

Pure, deterministic decision of whether a claim should proceed through the
normal pipeline, be closed outright, or be routed to the insured-claims path.
No I/O, no LLM calls -- everything the function needs is passed in as plain
arguments (including `now`, so the function never reaches for the wall clock
itself and stays trivially testable/deterministic).

Design decisions (see the task write-up for the full ambiguity discussion):

1. Insured takes priority over every other check. An insured shipment is
   routed to `"insured_process"` regardless of claim type or window, because
   insured claims go through a completely different (carrier-insurance)
   process and are not being *denied* -- they're being redirected. We treat
   this as `eligible=True` (it is not a "close" outcome) with
   `reason="INSURED"` documenting *why* it took the alternate route, matching
   the reason vocabulary in the plan (`TOO_OLD | WRONG_TYPE | INSURED |
   MISSING_INFO`). `Shipment.is_insured` is `bool | None`; `None` (unknown)
   is treated as falsy/uninsured (falls through to the normal checks) rather
   than assumed insured -- this is the more conservative default (avoids
   silently routing an unconfirmed-insurance case away from the standard
   review path) but is an explicit assumption worth flagging to a reviewer.
2. Missing `delivered_date` is checked next. Without it we cannot evaluate
   the claim window at all, so we surface `MISSING_INFO` rather than guessing
   `TOO_OLD`. `EligibilityResult.route` is constrained to
   `"process" | "close" | "insured_process"` (per the dataclass contract), so
   there is no dedicated "needs more info" route yet -- we map this to
   `route="close"` (not "process") since the case cannot proceed as-is. If a
   future task wants a distinct "needs_info" routing lane, that's a
   `route` type change, not something to smuggle in here.
3. Wrong claim type (`Case.sub_category != settings.eligible_sub_category`,
   including `None`) is checked next, and denies with `WRONG_TYPE` /
   `route="close"`. `eligible_sub_category` lives on `claimpilot.config.
   Settings` alongside `claim_window_days`, since both are policy values the
   gate depends on rather than implementation detail local to this module.
4. The claim window is checked last: `Case.created_date - delivered_date`
   (in days) must be `<= claim_window_days`. The boundary is inclusive --
   exactly `claim_window_days` days is still eligible; one day more is
   `TOO_OLD`.
5. `delivered_date` is preferred from `Case` (what the real case-detail API
   returns per the plan's Mock API section) and falls back to `Shipment` if
   the case's own field is absent, since both models carry the field.
6. Date fields are plain ISO-ish strings on the models (`str | None`), not
   `datetime`, so this module parses them with `datetime.fromisoformat`
   (accepting a trailing `Z` as UTC). A malformed date string is treated the
   same as a missing one (`MISSING_INFO`, since it makes the window
   uncomputable) rather than raising, since this gate must never crash the
   pipeline on dirty upstream data.
7. All parsed dates are normalized to naive (tz info stripped) before
   comparison, and the window is computed over calendar dates (`.date()`),
   not raw datetimes. This matters because the real API/fixtures return
   full timestamps with a UTC offset (e.g. `"2026-03-01T00:00:00.000+0000"`)
   while hand-built cases in tests may use bare dates (naive) -- mixing the
   two in a raw subtraction raises `TypeError`, and even same-tz timestamps
   with nonzero time-of-day would make raw `timedelta.days` floor instead of
   counting calendar days.
8. This ordering means a case that is both the wrong claim type *and*
   missing `delivered_date` reports `MISSING_INFO`, not `WRONG_TYPE` -- the
   task's prescribed check order (window-related check before type check)
   is kept as given rather than re-ordered by judgment call.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from claimpilot.config import settings
from claimpilot.models import Case, Shipment

EligibilityReason = Literal["TOO_OLD", "WRONG_TYPE", "INSURED", "MISSING_INFO"]
EligibilityRoute = Literal["process", "close", "insured_process"]


@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    reason: EligibilityReason | None  # machine code, or None when eligible outright
    route: EligibilityRoute


def _parse_date(value: str | None) -> datetime | None:
    """Parse an ISO-ish date/datetime string; return None if absent/invalid.

    The Postman mock and fixtures return full timestamps with a UTC offset
    (e.g. `"2026-03-01T00:00:00.000+0000"`), but tests and hand-built cases
    may pass bare dates (`"2026-01-01"`, naive) or `Z`-suffixed UTC. To avoid
    ever mixing naive and aware datetimes in the same subtraction (which
    raises `TypeError`), any tz info is stripped here -- the gate only ever
    compares *calendar* dates, so wall-clock offset doesn't matter.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=None)


def check_eligibility(
    case: Case,
    shipment: Shipment,
    *,
    now: datetime,
    claim_window_days: int | None = None,
) -> EligibilityResult:
    """Decide whether `case` should proceed, close, or route to insured claims.

    `now` is accepted for signature/testability symmetry and as a fallback
    reference point if `Case.created_date` is itself missing; it is not used
    when `created_date` is present, since the plan defines the claim window
    as `created_date - delivered_date`, not `now - delivered_date`.

    `claim_window_days` defaults to `settings.claim_window_days`, resolved
    here (inside the function body) rather than as the parameter's default
    value -- a `= settings.claim_window_days` default would be evaluated
    exactly once, at import time, and would then silently ignore both a
    `monkeypatch.setattr(settings, "claim_window_days", ...)` test override
    and a real env var change in production. Explicit callers (tests) may
    still pass a value directly to pin it independently of `settings`.
    """
    resolved_claim_window_days = claim_window_days if claim_window_days is not None else settings.claim_window_days

    if shipment.is_insured:
        return EligibilityResult(eligible=True, reason="INSURED", route="insured_process")

    delivered_date = _parse_date(case.delivered_date) or _parse_date(shipment.delivered_date)
    if delivered_date is None:
        return EligibilityResult(eligible=False, reason="MISSING_INFO", route="close")

    if case.sub_category != settings.eligible_sub_category:
        return EligibilityResult(eligible=False, reason="WRONG_TYPE", route="close")

    created_date = _parse_date(case.created_date) or now.replace(tzinfo=None)
    # Compare calendar dates, not raw datetimes: timestamps in the real API
    # carry time-of-day (e.g. "T00:00:00.000+0000"), and a naive
    # `(created - delivered).days` would floor sub-24h differences, letting
    # a claim that is 30 *calendar* days old read as 29 (or vice versa at
    # the boundary). Calendar-day subtraction avoids that off-by-one.
    window_days = (created_date.date() - delivered_date.date()).days
    if window_days > resolved_claim_window_days:
        return EligibilityResult(eligible=False, reason="TOO_OLD", route="close")

    return EligibilityResult(eligible=True, reason=None, route="process")
