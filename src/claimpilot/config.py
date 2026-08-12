"""Environment-backed settings.

Every hardcoded, deployment-tunable value in this project -- business-policy
thresholds (`cap`, `claim_window_days`, confidence thresholds, etc.) and
infrastructure knobs (`db_path`, `allowed_attachment_host`, etc.) -- is
env-overridable through this module, so a policy or deployment change never
needs a code change. See each field's own docstring/comment for its specific
rationale and caveats. `LLM_PRICING` (below the `Settings` class) is the one
deliberate exception: a per-model pricing table, kept as a plain dict rather
than forced into individual env vars (see its own comment for why).
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT / ".env",
        extra="ignore",
        # `.env.example` ships with blank values (e.g. `SHIPBOB_API_BASE=`) as
        # a template. Without this, pydantic-settings treats a present-but-empty
        # env var as an explicit "", overriding the default below instead of
        # falling back to it -- so a user who copies the example and only
        # fills in the key would get a blank base_url. Empty env values are
        # treated as unset instead.
        env_ignore_empty=True,
    )

    # Consumed by the LLM wrapper (`claimpilot.llm`).
    anthropic_api_key: str = ""

    # Default model for `structured_call()`. Env-overridable (ANTHROPIC_MODEL)
    # so a deployment can pin a different model without a code change.
    anthropic_model: str = "claude-sonnet-5"

    # Post-demo-v1 feature: swappable LLM provider. `structured_call`'s
    # `get_transport()` factory branches on this to decide which `Transport`
    # implementation (`AnthropicTransport` vs `OpenAITransport`) backs every
    # LLM call in the system -- see `claimpilot.llm` module docstring point 3.
    # Env-overridable (LLM_PROVIDER) so a deployment can switch providers
    # without a code change; defaults to the original (and so far only
    # demo-v1-exercised) provider.
    llm_provider: Literal["anthropic", "openai"] = "anthropic"

    # Consumed by `OpenAITransport` when `llm_provider="openai"`.
    openai_api_key: str = ""

    # Default model for `structured_call()` when `llm_provider="openai"`.
    # `gpt-4o` is OpenAI's flagship multimodal model as of this writing --
    # vision-capable (needed for damage-photo validation) and
    # supports strict, schema-constrained function/tool calling (needed for
    # the same forced-structured-output guarantee `AnthropicTransport` gets
    # from Anthropic's forced `tool_choice`). Env-overridable (OPENAI_MODEL),
    # mirroring `anthropic_model` above. NOTE: OpenAI ships new flagship
    # models fairly often -- revisit this default periodically, same caveat
    # as `anthropic_model`.
    openai_model: str = "gpt-4o"

    shipbob_api_base: str = "https://e41238c7-aefe-4d20-8866-747c74eac48f.mock.pstmn.io"
    shipbob_api_key: str = ""
    use_fixtures: bool = True

    # Per-request timeout for `HttpShipBobClient`'s httpx client. httpx's own
    # default (5s for connect/read/write/pool alike) turned out to be part
    # of why the free-tier Postman mock's real, directly-observed
    # `ReadTimeout`s happened as often as they did -- a slow-but-eventually-
    # responding mock doesn't need a retry at all if it's just given a more
    # generous window first. Works together with, not instead of,
    # `clients/retry.py`'s retry-on-timeout policy below: a genuinely dead
    # connection still times out and retries; a merely slow one now often
    # just succeeds on the first attempt.
    shipbob_http_timeout_seconds: float = 15.0

    # --- Business-policy thresholds ------------------------------------------
    # Originally plain module-level constants (not `Settings` fields), on the
    # theory that they're business policy, not secrets/URLs. Promoted to
    # `Settings` fields per an explicit later request to make every
    # hardcoded, deployment-tunable value env-overridable -- a different
    # environment, a policy update, or a cost-model change can now override
    # any of these without a code change. Defaults are UNCHANGED from the
    # values these previously had as bare constants; this is a
    # configurability refactor, not a policy change. All call sites read
    # `settings.<field>` at call time (never snapshotted into a module-level
    # name or a mutable-default parameter), so a `monkeypatch.setattr(settings,
    # ...)` override in a test -- or a real env var change -- is honored
    # immediately, matching this codebase's existing test-override
    # convention (see e.g. `tests/test_openai_transport.py`).

    # The real window value comes from ShipBob's claims policy -- 30 days is
    # a placeholder until that's confirmed. Consumed by
    # `gates/eligibility.check_eligibility`.
    claim_window_days: int = 30

    # The only Case.sub_category value the eligibility gate treats
    # as an in-scope damaged-in-transit claim.
    eligible_sub_category: str = "Claim | Damaged in Transit"

    # Maximum total reimbursement amount payable for a single claim
    # (`calc.reimbursement`), independently re-verified by `guard.
    # check_outbound`'s CAP_EXCEEDED invariant before anything is sent.
    # PRODUCTION POLICY LEVER, NOT A TOY SETTING: overriding this via the
    # `CAP` env var changes the real dollar ceiling the outbound guard
    # enforces, with no additional audit trail beyond whatever change-control
    # process governs the deployment's environment/.env file -- treat an
    # override the same as a code change requiring review, not a casual
    # runtime tweak.
    cap: Decimal = Decimal("100.00")

    # Risk-tiering thresholds, consumed by `risk.tier`.

    # A shipment's declared_value at or above this amount is flagged as a
    # high-value risk factor. $500 is a placeholder pending real ShipBob
    # policy guidance, chosen as "clearly above" the $100 reimbursement `cap`
    # so it's flagging shipment/insurance value, not reimbursement amount.
    high_value_threshold: Decimal = Decimal("500.00")

    # A merchant with this many or more claims in the trailing 90 days is
    # flagged as a high-claim-frequency risk factor. Placeholder pending real
    # policy guidance.
    high_claim_frequency_threshold: int = 3

    # --- LLM wrapper ----------------------------------------------------------

    # Hard wall-clock timeout for a single `structured_call()` transport
    # attempt (passed straight through to the transport SDK's per-request
    # `timeout`). 30s is a reasonable ceiling for a single tool-forced
    # structured-output call; tune based on production latency observations
    # once there's real traffic to measure.
    llm_timeout_seconds: float = 30.0

    # Max output tokens per `structured_call()` transport attempt. Structured
    # extraction responses are small (a single tool call's JSON input), so
    # this is generous headroom rather than a tuned value -- revisit if a
    # future schema needs a much larger response.
    llm_max_tokens: int = 4096

    # --- Evidence classifier --------------------------------------------------

    # Minimum confidence `classify_attachment` must report for a
    # classification to count toward satisfying one of the 4 required
    # `EvidenceItem` categories. Below this (or `usable=False` regardless of
    # confidence), `evidence_gaps` treats the category as missing -- "ask,
    # don't guess" mirrors how a human rep would handle a photo they can't
    # actually rely on. 0.7 is a placeholder pending real calibration against
    # labeled attachments, same caveat as the other policy constants above.
    evidence_min_conf: float = 0.7

    # --- Damage validation ------------------------------------------------------

    # Minimum confidence `combine_validation` requires, across ALL FOUR
    # `ValidationResult` judgments, before an all-passed result is allowed to
    # `PROCEED` rather than being routed to `ESCALATED`. LLM self-reported
    # confidence is not a calibrated probability -- it's a coarse ordering
    # signal -- so this threshold only ever moves a case toward MORE human
    # attention (escalate/request-info), never toward auto-approval; a
    # below-threshold confidence never gets "rounded up" to PROCEED. 0.75 is
    # a placeholder pending real calibration against labeled outcomes, same
    # caveat as `evidence_min_conf` and the other policy constants above.
    validation_min_conf: float = 0.75

    # --- Retail-invoice audit --------------------------------------------------

    # Whether to read the merchant's submitted retail invoice
    # (`EvidenceItem.ORDER_PROOF`) and reconcile it against ShipBob's own
    # invoice data -- see `gates/invoice_audit.py`'s module docstring for the
    # full rationale. Costs one extra vision call per case that reaches
    # validation. Off switch exists because it is a net-new check on an
    # already-shipped pipeline: turning it off restores exactly the previous
    # behavior (API price, no reconciliation), which is the right escape
    # hatch if extraction quality turns out to be poor on a real corpus.
    invoice_audit_enabled: bool = True

    # How far ShipBob's unit price may differ from the price printed on the
    # merchant's retail invoice before it is reported as a discrepancy, in
    # currency units. Defaults to a single cent -- i.e. any real difference
    # is surfaced -- because in the sample data the gaps are dollars, not
    # rounding: every priced case disagrees, by up to 57% on one line. Raise
    # it if a real corpus turns out to carry routine sub-dollar noise.
    invoice_price_tolerance: Decimal = Decimal("0.01")

    # The currency `cap` and every stored `unit_price` are assumed to be
    # denominated in. The mock API exposes no currency field anywhere, so
    # this is an assumption the system cannot verify on its own -- the
    # retail-invoice audit exists partly to catch when it is violated (one
    # fixture case is plainly a GBP order).
    expected_currency: str = "USD"

    # --- Demo controls -----------------------------------------------------

    # Whether the review UI exposes its live demo actions: "Reset demo data"
    # (wipes every table) and "Fetch cases" / per-case "Process" (pulls from
    # the ShipBob API and runs the pipeline on demand).
    #
    # Exists so the destructive one has an off switch. A button that empties
    # the claims database is entirely reasonable in a walkthrough and
    # entirely unreasonable in a real deployment, and "nobody will click it"
    # is not an access-control model -- set this `false` anywhere that isn't
    # a demo. The endpoints themselves 404 when it's off, so hiding the
    # buttons isn't the only thing standing between a stray POST and an
    # empty database.
    demo_controls_enabled: bool = True

    # --- Infrastructure / deployment knobs -----------------------------------

    # On-disk SQLite database path. `Path`-typed (not `str`) so the default
    # keeps resolving to the same *absolute* path every module in this
    # codebase (and `scripts/seed.py`'s `DB_PATH.exists()`/`.unlink()`) has
    # always relied on -- a bare `"claimpilot.db"` string default would
    # silently become CWD-relative instead, a real behavior change this
    # refactor must not introduce.
    db_path: Path = _REPO_ROOT / "claimpilot.db"

    # SSRF allowlist: the only host `clients/attachment_guard.
    # validate_attachment_url` will fetch attachment bytes from. A different
    # environment (e.g. a non-demo Azure blob storage account) would need a
    # different host here.
    allowed_attachment_host: str = "sa032101pubdevuc.blob.core.windows.net"

    # Hard cap on a single downloaded attachment's size, in bytes (10 MiB).
    max_attachment_bytes: int = 10 * 1024 * 1024

    # Write-time cap: max `kind="policy"` memory rows retained per
    # `(scope, merchant_id)` partition -- see `claimpilot.memory` module
    # docstring point 8 for why the cap is partitioned this way rather than
    # applied globally.
    policy_note_cap_per_partition: int = 10

    # Trailing window `memory.merchant_context()` uses for both
    # `MemoryContext.claim_frequency_90d` and `risk.MerchantMemory.
    # claims_last_90_days` (also rendered into the drafter's prompt text as
    # "Claim frequency (last N days)"). Moved here from a bare
    # `memory.CLAIM_FREQUENCY_WINDOW_DAYS` module constant so a deployment
    # can tune it without a code change, matching every other business-
    # policy value in this section. NOTE: `claim_frequency_90d`/
    # `claims_last_90_days`'s Python *field names* still literally say "90"
    # -- overriding this away from 90 doesn't change those identifiers, only
    # the actual math and rendered prompt text (both correctly follow this
    # setting's live value; only the name is a naming artifact of the
    # original default). A field rename was deliberately not bundled into
    # this change, to keep it a pure configurability refactor rather than a
    # wider API change.
    claim_frequency_window_days: int = 90

    # Read-time cap on how many recent `kind IN ("note", "correction")` rows
    # `memory.merchant_context()` surfaces to the drafter's prompt --
    # distinct from `policy_note_cap_per_partition` above (a write-time cap
    # on a different, `kind="policy"`, row set). Moved here from a bare
    # `memory.MAX_RECENT_NOTES` module constant for the same reason.
    max_recent_notes: int = 5

    # --- Live-mock retry policy (clients/retry.py) ---------------------------
    # Added after directly observing intermittent `httpx.ReadTimeout`s
    # against the real (free-tier, Postman-hosted) ShipBob mock server
    # during manual testing -- these three knobs govern every retry
    # `clients/http.py` and `clients/attachment_guard.py` perform on a
    # transient network failure (timeout/connection error) or 5xx response.
    # Never applies to a 404/422 -- those are legitimate answers from the
    # server, not transient failures, and are never retried regardless of
    # these settings. See `clients/retry.py` module docstring for the full
    # policy.

    # Max retry attempts after the first (so the default of 3 means up to 4
    # total attempts per call).
    http_max_retries: int = 3

    # Seconds to wait before the first retry; doubles each subsequent
    # attempt (exponential backoff), before the jitter `clients/retry.py`
    # adds on top.
    http_retry_base_delay_seconds: float = 0.5

    # Hard ceiling on the backoff delay (before jitter) -- keeps a
    # high-attempt-count retry from waiting an unreasonably long time.
    http_retry_max_delay_seconds: float = 8.0

    # --- Review UI access control ------------------------------------------
    # Found during a full-codebase security audit: the review UI (`web/app.py`)
    # previously had ZERO authentication -- anyone who could reach the URL
    # could approve reimbursements, send emails, or delete policy notes, with
    # no access control of any kind. Opt-in HTTP Basic Auth, gated on BOTH
    # of these being non-empty (so a bare `.env` with neither set preserves
    # today's no-auth local-dev/demo behavior exactly -- this is additive,
    # not a breaking default change). Set both to turn it on for any
    # deployment reachable by more than just its own operator.
    review_ui_username: str = ""
    review_ui_password: str = ""


settings = Settings()

# Approximate per-token USD pricing, keyed by model ID, for the `llm_calls`
# audit table's `cost_usd` column. DEMO ESTIMATE ONLY -- reconcile against
# real Anthropic billing before relying on this for actual cost accounting,
# same caveat as `settings.claim_window_days`/`settings.high_value_threshold`
# above. Derived from the public *list* per-million-token price for Claude
# Sonnet 5 at time of writing ($3.00 input / $15.00 output per 1M tokens),
# converted to per-token Decimal so `_compute_cost()` can do plain
# multiplication. NOTE: Anthropic is also running introductory pricing for
# Claude Sonnet 5 ($2.00 input / $10.00 output per 1M tokens) through
# 2026-08-31 -- this table intentionally uses the standing list price, not
# the introductory rate, so the estimate doesn't silently jump when the promo
# ends; swap in real billing data (or the promo rate, dated, if that's
# preferred) when reconciling.
#
# Deliberately NOT env-configurable, unlike the `Settings` fields above: this
# is a lookup table keyed by model name, not a single scalar knob, and a
# handful of pricing entries doesn't warrant one env var per model x
# input/output rate -- that would be over-engineering a demo-estimate cost
# figure. What *is* env-configurable is `settings.anthropic_model` /
# `settings.openai_model` (the dict keys this table is looked up by) --
# `_compute_cost` already handles a model ID with no matching entry here
# gracefully (prices it at $0 rather than raising; see its docstring), so
# pointing `ANTHROPIC_MODEL`/`OPENAI_MODEL` at a model not listed below is
# safe, just silently free -- add a matching entry here (or accept the $0
# estimate) when overriding either of those env vars to a model not already
# priced.
LLM_PRICING: dict[str, dict[str, Decimal]] = {
    "claude-sonnet-5": {
        "input": Decimal("3.00") / Decimal("1000000"),
        "output": Decimal("15.00") / Decimal("1000000"),
    },
    # DEMO ESTIMATE ONLY, UNVERIFIED -- same caveat as the Claude Sonnet 5
    # entry above, but with an extra flag: these figures were not looked up
    # against OpenAI's live pricing page
    # (https://openai.com/api/pricing/) at the time this entry was added,
    # they're a widely-cited approximation for `gpt-4o` ($2.50 input /
    # $10.00 output per 1M tokens). Reconcile against OpenAI's actual
    # billing/pricing page before relying on this for real cost accounting
    # -- do not treat this number as confirmed. Key must stay byte-identical
    # to `Settings.openai_model`'s value, or `_compute_cost` silently prices
    # every OpenAI call at $0 (it treats an unknown model as free rather
    # than raising -- see `_compute_cost`'s docstring).
    "gpt-4o": {
        "input": Decimal("2.50") / Decimal("1000000"),
        "output": Decimal("10.00") / Decimal("1000000"),
    },
}


def configured_api_key() -> tuple[bool, str, str]:
    """Whether the API key for the currently-selected LLM provider is set.

    Returns `(is_set, env_var_name, human_provider_name)` -- e.g.
    `(False, "ANTHROPIC_API_KEY", "Anthropic")` -- so callers can build their
    own message for their own medium (a CLI line, an HTTP error, a banner in
    the review UI) without each re-deriving which provider is active and
    which variable to name.

    This check existed in two places before this helper (`scripts/seed.py`'s
    `check_api_key` and `evals/test_golden.py`'s
    `_llm_provider_has_api_key`), and the review UI needed a third. Three
    copies of "which key does this deployment actually need" is how one of
    them ends up silently wrong after a provider is added, so it lives here
    once, next to the settings it reads.
    """
    if settings.llm_provider == "openai":
        return bool(settings.openai_api_key), "OPENAI_API_KEY", "OpenAI"
    return bool(settings.anthropic_api_key), "ANTHROPIC_API_KEY", "Anthropic"
