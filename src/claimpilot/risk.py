"""Risk tiering.

Pure, deterministic assessment of how much extra scrutiny a rep should give
a claim draft before sending it. No I/O, no LLM calls, no eligibility or
denial semantics -- `tier()` only informs the reviewer; it never blocks or
approves anything. A HIGH tier just means "show a louder banner," per the
task spec.

Design decisions (see the task write-up for the full ambiguity discussion):

1. `MerchantMemory` is a **deliberately narrow** shape -- written before
   the `claimpilot.memory` store existed, as just enough of what a future
   merchant-history lookup would produce (a trailing-90-day claim count and
   a list of pre-existing human-readable notes about the merchant) for
   `tier()` to have something reasonable to consume. The memory store now
   populates it for real: `pipeline.process_case` builds `MerchantMemory`
   from `claimpilot.memory.merchant_context()`'s output (see that module
   and `pipeline.py` for exactly which memory fields become
   `claims_last_90_days`/`flags`, and why). `tier()` itself is unchanged --
   it still does no date math or memory-store I/O of its own;
   `claims_last_90_days` is assumed to already be the caller's computed
   trailing-90-day count, and tests still construct `MerchantMemory`
   directly/from fixtures.
2. Three independent risk factors, each contributing one human-readable
   flag when triggered:
     - declared value >= `settings.high_value_threshold` (config.py). Missing
       `declared_value` (`None`) does NOT trigger this factor -- per the
       task, real API data may omit it, and absence is not itself risky.
     - `claims_last_90_days` >= `settings.high_claim_frequency_threshold` (config.py).
     - `flags` is non-empty. Each pre-existing flag string is surfaced
       verbatim in the output (prefixed for context) rather than collapsed
       into one generic "merchant has flags" line, since the whole point of
       those flags is to tell the rep *what* a past rep already noted.
3. **Tier aggregation**: count how many of the three factors triggered.
       0 triggered -> LOW
       1 triggered -> ELEVATED
   2 or 3 triggered -> HIGH
   This is a simple, explicit "more independent red flags = more scrutiny"
   rule -- no factor is weighted more than another (e.g. a lone high
   declared value gets the same ELEVATED treatment as a lone claim-frequency
   hit), since the task doesn't call for differential weighting and a flat
   count is the simplest rule that satisfies "HIGH gets a louder banner,
   not automated denial."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from claimpilot.config import settings
from claimpilot.models import Shipment


class RiskTier(str, Enum):
    """`str` subclass for the same reason as `CaseState`/`EvidenceItem` in
    models.py: it serializes as a plain string (e.g. to JSON or SQLite)
    without a custom encoder, while still giving callers an enum to match on.
    """

    LOW = "LOW"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"


@dataclass(frozen=True)
class MerchantMemory:
    """Narrow shape for merchant history that `tier()` needs -- deliberately
    not a full model of everything the `claimpilot.memory` module
    tracks (e.g. it has no notion of which memory rows are policy notes vs.
    raw corrections). `pipeline.process_case` builds this from
    `claimpilot.memory.merchant_context()`'s richer `MemoryContext`; tests
    still construct it directly/from fixtures.
    """

    claims_last_90_days: int = 0
    # Pre-existing human-readable notes about this merchant from past rep
    # corrections, e.g. "repeat high-value claimant".
    flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RiskAssessment:
    tier: RiskTier
    flags: list[str]


def tier(shipment: Shipment, merchant_memory: MerchantMemory) -> RiskAssessment:
    """Assess claim risk for rep review. Informational only -- never denies
    or blocks; a HIGH tier just means the draft gets a louder banner.
    """
    # Read at call time (not snapshotted into module-level names), so a
    # `monkeypatch.setattr(settings, ...)` override in a test -- or a real
    # env var change -- is honored immediately.
    high_value_threshold = settings.high_value_threshold
    high_claim_frequency_threshold = settings.high_claim_frequency_threshold

    flags: list[str] = []

    if shipment.declared_value is not None and shipment.declared_value >= high_value_threshold:
        flags.append(
            f"High declared value: ${shipment.declared_value:.2f} "
            f"(threshold ${high_value_threshold:.2f})"
        )

    if merchant_memory.claims_last_90_days >= high_claim_frequency_threshold:
        flags.append(
            f"High claim frequency: {merchant_memory.claims_last_90_days} claims in the last "
            f"90 days (threshold {high_claim_frequency_threshold})"
        )

    triggered_factor_count = len(flags)
    if merchant_memory.flags:
        flags.extend(f"merchant memory flag: {note}" for note in merchant_memory.flags)
        triggered_factor_count += 1

    if triggered_factor_count == 0:
        risk_tier = RiskTier.LOW
    elif triggered_factor_count == 1:
        risk_tier = RiskTier.ELEVATED
    else:
        risk_tier = RiskTier.HIGH

    return RiskAssessment(tier=risk_tier, flags=flags)
