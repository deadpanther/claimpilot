from decimal import Decimal

from claimpilot.config import settings
from claimpilot.models import Shipment
from claimpilot.risk import MerchantMemory, RiskAssessment, RiskTier, tier


def _shipment(declared_value: str | None = None) -> Shipment:
    return Shipment(
        shipment_id="s1",
        declared_value=Decimal(declared_value) if declared_value is not None else None,
    )


def test_no_factors_triggered_is_low_with_no_flags():
    result = tier(_shipment(), MerchantMemory())

    assert isinstance(result, RiskAssessment)
    assert result.tier == RiskTier.LOW
    assert result.flags == []


def test_missing_declared_value_does_not_trigger_value_factor():
    result = tier(_shipment(declared_value=None), MerchantMemory())

    assert result.tier == RiskTier.LOW
    assert result.flags == []


def test_declared_value_below_threshold_does_not_trigger():
    below = settings.high_value_threshold - Decimal("0.01")
    result = tier(_shipment(declared_value=str(below)), MerchantMemory())

    assert result.tier == RiskTier.LOW
    assert result.flags == []


def test_declared_value_at_threshold_triggers():
    result = tier(_shipment(declared_value=str(settings.high_value_threshold)), MerchantMemory())

    assert result.tier == RiskTier.ELEVATED
    assert len(result.flags) == 1
    assert "High declared value" in result.flags[0]
    assert f"{settings.high_value_threshold:.2f}" in result.flags[0]


def test_declared_value_above_threshold_triggers():
    above = settings.high_value_threshold + Decimal("1.00")
    result = tier(_shipment(declared_value=str(above)), MerchantMemory())

    assert result.tier == RiskTier.ELEVATED
    assert len(result.flags) == 1
    assert "High declared value" in result.flags[0]


def test_claim_frequency_below_threshold_does_not_trigger():
    memory = MerchantMemory(claims_last_90_days=settings.high_claim_frequency_threshold - 1)
    result = tier(_shipment(), memory)

    assert result.tier == RiskTier.LOW
    assert result.flags == []


def test_claim_frequency_at_threshold_triggers():
    memory = MerchantMemory(claims_last_90_days=settings.high_claim_frequency_threshold)
    result = tier(_shipment(), memory)

    assert result.tier == RiskTier.ELEVATED
    assert len(result.flags) == 1
    assert "claims in the last 90 days" in result.flags[0]


def test_claim_frequency_above_threshold_triggers():
    memory = MerchantMemory(claims_last_90_days=settings.high_claim_frequency_threshold + 5)
    result = tier(_shipment(), memory)

    assert result.tier == RiskTier.ELEVATED
    assert len(result.flags) == 1


def test_empty_memory_flags_do_not_trigger():
    memory = MerchantMemory(flags=[])
    result = tier(_shipment(), memory)

    assert result.tier == RiskTier.LOW
    assert result.flags == []


def test_nonempty_memory_flags_trigger_and_surface_content():
    memory = MerchantMemory(flags=["repeat high-value claimant"])
    result = tier(_shipment(), memory)

    assert result.tier == RiskTier.ELEVATED
    assert len(result.flags) == 1
    assert "repeat high-value claimant" in result.flags[0]


def test_multiple_memory_notes_all_surface_as_one_factor():
    memory = MerchantMemory(flags=["note one", "note two"])
    result = tier(_shipment(), memory)

    assert result.tier == RiskTier.ELEVATED
    assert len(result.flags) == 2
    assert any("note one" in f for f in result.flags)
    assert any("note two" in f for f in result.flags)


def test_two_factors_triggered_is_high():
    memory = MerchantMemory(claims_last_90_days=settings.high_claim_frequency_threshold)
    result = tier(_shipment(declared_value=str(settings.high_value_threshold)), memory)

    assert result.tier == RiskTier.HIGH
    assert len(result.flags) == 2


def test_frequency_and_memory_flags_is_high():
    memory = MerchantMemory(
        claims_last_90_days=settings.high_claim_frequency_threshold,
        flags=["repeat high-value claimant"],
    )
    result = tier(_shipment(), memory)

    assert result.tier == RiskTier.HIGH
    assert len(result.flags) == 2


def test_one_factor_plus_multiple_memory_notes_is_high_not_three_factors():
    # Regression pin: flags list length is NOT the same as triggered-factor
    # count -- multiple memory notes are one factor but multiple flag
    # strings. Two triggered factors (frequency + memory-flags-nonempty)
    # should yield HIGH even though three flag strings are produced.
    memory = MerchantMemory(
        claims_last_90_days=settings.high_claim_frequency_threshold,
        flags=["note one", "note two"],
    )
    result = tier(_shipment(), memory)

    assert result.tier == RiskTier.HIGH  # 2 factors, not 3
    assert len(result.flags) == 3  # flags list length != factor count


def test_three_factors_triggered_is_high():
    memory = MerchantMemory(
        claims_last_90_days=settings.high_claim_frequency_threshold,
        flags=["repeat high-value claimant"],
    )
    result = tier(_shipment(declared_value=str(settings.high_value_threshold)), memory)

    assert result.tier == RiskTier.HIGH
    assert len(result.flags) == 3


def test_tier_never_exposes_eligibility_or_denial_semantics():
    result = tier(_shipment(), MerchantMemory())

    assert not hasattr(result, "eligible")
    assert not hasattr(result, "deny")
    assert not hasattr(result, "route")


def test_tier_reads_settings_thresholds_live_not_at_import_time(monkeypatch):
    """`tier()` must read `settings.high_value_threshold` at call time, not
    cache it into a module-level name at import time -- a stale import-time
    snapshot would silently ignore both a
    `monkeypatch.setattr(settings, ...)` test override and a real env var
    change in production. $600 is below the real default threshold ($500)
    only after it's overridden upward to $1000 -- a LOW result here is only
    possible if `tier()` actually read the live override.
    """
    monkeypatch.setattr(settings, "high_value_threshold", Decimal("1000.00"))

    result = tier(_shipment(declared_value="600.00"), MerchantMemory())

    assert result.tier == RiskTier.LOW
    assert result.flags == []
