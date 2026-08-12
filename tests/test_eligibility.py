from datetime import datetime

from claimpilot.gates.eligibility import EligibilityResult, check_eligibility
from claimpilot.models import Case, Shipment

NOW = datetime(2026, 2, 1)


def _case(**overrides) -> Case:
    defaults = dict(
        case_id="c1",
        status="New",
        sub_category="Claim | Damaged in Transit",
        created_date="2026-01-10",
        delivered_date="2026-01-01",
    )
    defaults.update(overrides)
    return Case(**defaults)


def _shipment(**overrides) -> Shipment:
    defaults = dict(shipment_id="s1", delivered_date="2026-01-01", is_insured=False)
    defaults.update(overrides)
    return Shipment(**defaults)


def test_within_window_is_eligible():
    case = _case(created_date="2026-01-15", delivered_date="2026-01-01")
    shipment = _shipment(delivered_date="2026-01-01")

    result = check_eligibility(case, shipment, now=NOW)

    assert result == EligibilityResult(eligible=True, reason=None, route="process")


def test_exact_boundary_is_eligible():
    # created_date - delivered_date == 30 days exactly (inclusive boundary).
    case = _case(created_date="2026-01-31", delivered_date="2026-01-01")
    shipment = _shipment(delivered_date="2026-01-01")

    result = check_eligibility(case, shipment, now=NOW, claim_window_days=30)

    assert result.eligible is True
    assert result.route == "process"
    assert result.reason is None


def test_one_day_past_boundary_is_too_old():
    # created_date - delivered_date == 31 days.
    case = _case(created_date="2026-02-01", delivered_date="2026-01-01")
    shipment = _shipment(delivered_date="2026-01-01")

    result = check_eligibility(case, shipment, now=NOW, claim_window_days=30)

    assert result == EligibilityResult(eligible=False, reason="TOO_OLD", route="close")


def test_wrong_claim_type_is_denied():
    case = _case(sub_category="Claim | Wrong Item")
    shipment = _shipment()

    result = check_eligibility(case, shipment, now=NOW)

    assert result == EligibilityResult(eligible=False, reason="WRONG_TYPE", route="close")


def test_missing_claim_type_is_wrong_type():
    case = _case(sub_category=None)
    shipment = _shipment()

    result = check_eligibility(case, shipment, now=NOW)

    assert result == EligibilityResult(eligible=False, reason="WRONG_TYPE", route="close")


def test_insured_routes_to_insured_process_when_otherwise_eligible():
    case = _case()
    shipment = _shipment(is_insured=True)

    result = check_eligibility(case, shipment, now=NOW)

    assert result == EligibilityResult(eligible=True, reason="INSURED", route="insured_process")


def test_insured_routes_to_insured_process_never_close_when_wrong_type():
    case = _case(sub_category="Claim | Wrong Item")
    shipment = _shipment(is_insured=True)

    result = check_eligibility(case, shipment, now=NOW)

    assert result == EligibilityResult(eligible=True, reason="INSURED", route="insured_process")


def test_insured_routes_to_insured_process_never_close_when_too_old():
    case = _case(created_date="2026-03-01", delivered_date="2026-01-01")
    shipment = _shipment(is_insured=True, delivered_date="2026-01-01")

    result = check_eligibility(case, shipment, now=NOW, claim_window_days=30)

    assert result == EligibilityResult(eligible=True, reason="INSURED", route="insured_process")


def test_missing_delivered_date_is_missing_info():
    case = _case(delivered_date=None)
    shipment = _shipment(delivered_date=None)

    result = check_eligibility(case, shipment, now=NOW)

    assert result == EligibilityResult(eligible=False, reason="MISSING_INFO", route="close")


def test_shipment_delivered_date_used_as_fallback_when_case_missing_it():
    case = _case(delivered_date=None, created_date="2026-01-10")
    shipment = _shipment(delivered_date="2026-01-01")

    result = check_eligibility(case, shipment, now=NOW)

    assert result.eligible is True
    assert result.route == "process"


def test_case_delivered_date_preferred_over_shipment_when_both_present():
    # Case delivered_date puts this outside the window; shipment's would not.
    case = _case(created_date="2026-02-01", delivered_date="2026-01-01")
    shipment = _shipment(delivered_date="2026-01-25")

    result = check_eligibility(case, shipment, now=NOW, claim_window_days=30)

    assert result.reason == "TOO_OLD"


def test_mixed_naive_and_aware_timestamps_do_not_raise():
    # Real API/fixture timestamps carry a UTC offset; hand-built dates in
    # tests are naive. The gate must not blow up comparing the two.
    case = _case(created_date="2026-01-31", delivered_date="2026-01-01T00:00:00.000+0000")
    shipment = _shipment(delivered_date="2026-01-01T00:00:00.000+0000")

    result = check_eligibility(case, shipment, now=NOW, claim_window_days=30)

    assert result.eligible is True
    assert result.route == "process"


def test_full_timestamp_boundary_uses_calendar_days_not_raw_timedelta():
    # Both timestamps carry a nonzero time-of-day; raw timedelta.days would
    # floor 30 days 23 hours down to 30, which happens to still pass here,
    # but the inverse (delivered later in the day than created) must not
    # spuriously read as fewer than 30 calendar days.
    case = _case(
        created_date="2026-01-31T00:00:00.000+0000",
        delivered_date="2026-01-01T23:00:00.000+0000",
    )
    shipment = _shipment(delivered_date="2026-01-01T23:00:00.000+0000")

    result = check_eligibility(case, shipment, now=NOW, claim_window_days=30)

    assert result.eligible is True
    assert result.route == "process"
    assert result.reason is None


def test_full_timestamp_one_day_past_boundary_is_too_old():
    case = _case(
        created_date="2026-02-01T00:00:00.000+0000",
        delivered_date="2026-01-01T23:00:00.000+0000",
    )
    shipment = _shipment(delivered_date="2026-01-01T23:00:00.000+0000")

    result = check_eligibility(case, shipment, now=NOW, claim_window_days=30)

    assert result.reason == "TOO_OLD"


def test_unknown_insurance_is_treated_as_uninsured():
    case = _case()
    shipment = _shipment(is_insured=None)

    result = check_eligibility(case, shipment, now=NOW)

    assert result.route == "process"
    assert result.reason is None


def test_missing_created_date_falls_back_to_now_and_is_within_window():
    # No created_date on the case, so the window must be computed against
    # `now`, not case creation. delivered_date is 5 days before `now`
    # here, which is within window -- if the fallback were broken (e.g.
    # ignoring `now` and defaulting to something else), this would either
    # raise or report the wrong result.
    case = _case(created_date=None, delivered_date="2026-01-15")
    shipment = _shipment(delivered_date="2026-01-15")
    now = datetime(2026, 1, 20)

    result = check_eligibility(case, shipment, now=now, claim_window_days=30)

    assert result == EligibilityResult(eligible=True, reason=None, route="process")


def test_missing_created_date_falls_back_to_now_and_is_too_old():
    # Same delivered_date as the "within window" case above, but a later
    # `now` pushes the computed window past 30 days -- proving the window is
    # actually being computed relative to `now`, not a fixed/ignored value.
    case = _case(created_date=None, delivered_date="2026-01-01")
    shipment = _shipment(delivered_date="2026-01-01")
    now = datetime(2026, 2, 5)

    result = check_eligibility(case, shipment, now=now, claim_window_days=30)

    assert result == EligibilityResult(eligible=False, reason="TOO_OLD", route="close")


def test_malformed_delivered_date_is_missing_info_not_a_crash():
    case = _case(delivered_date="not-a-date")
    shipment = _shipment(delivered_date=None)

    result = check_eligibility(case, shipment, now=NOW)

    assert result == EligibilityResult(eligible=False, reason="MISSING_INFO", route="close")


def test_malformed_case_delivered_date_falls_through_to_valid_shipment_date():
    # `_parse_date(case.delivered_date) or _parse_date(shipment.delivered_date)`
    # must fall through to the shipment's valid date when the case's own
    # delivered_date string fails to parse (not just when it's None/absent).
    case = _case(created_date="2026-01-10", delivered_date="not-a-date")
    shipment = _shipment(delivered_date="2026-01-01")

    result = check_eligibility(case, shipment, now=NOW, claim_window_days=30)

    assert result == EligibilityResult(eligible=True, reason=None, route="process")
