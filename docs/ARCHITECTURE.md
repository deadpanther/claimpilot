# claimpilot — Architecture

`claimpilot` automates ShipBob's damaged-in-transit claim workflow: intake → eligibility → evidence
→ AI judgment → reimbursement calc → **human-approved** send, with a persistent memory system that
learns from rep corrections over time.

This document explains how the pieces fit together and what each file is responsible for. For the
known limitations and trade-offs, see [`LIMITATIONS.md`](./LIMITATIONS.md). For the original design
doc, see [`plans/2026-08-10-claim-automation.md`](./plans/2026-08-10-claim-automation.md).

## 0. 60-second tour

If you read nothing else, read this:

1. A **`Case`** comes in from ShipBob (real API or fixture data — same code either way).
2. `pipeline.process_case()` runs it through 4 gates in a fixed order — **eligibility → evidence →
   validation → calc** — each one either passes it to the next gate, or short-circuits straight to
   `PENDING_REVIEW`/`ESCALATED` with an explanation of why.
3. Three of those gates use an LLM (evidence classification and damage validation are vision calls;
   drafting is text-only) — but the LLM **only ever produces a schema-validated judgment or prose**.
   It never decides the dollar amount or the approve/deny outcome; that's plain Python.
4. Every case, no matter what happened, lands in front of a human on the **review queue**
   (`GET /cases`). A rep clicks **Approve & Send**, **Save Edits & Approve**, or **Push Back**.
5. Right before anything is actually sent, `guard.py` independently **re-derives** the answer from
   stored data (not the draft) and blocks the send if anything doesn't check out.
6. Every rep edit/pushback is optionally distilled into a short reusable note ("mention the account
   manager by name") that shows up in that merchant's *next* case — the self-evolving memory loop.

Jump straight to §7 below for a real, concrete trace of one case going through all of this with
actual data.

## 1. Data model cheat sheet

The types that show up everywhere. All are `frozen`/immutable once produced (a gate's output is a
recorded fact, not something later code mutates).

| Type | Where | Fields |
|---|---|---|
| `Case` | `models.py` | `case_id, status, sub_category, description, order_id, user_id, shipment_id, delivered_date, contact_email, account_name, created_date` |
| `Shipment` | `models.py` | `shipment_id, carrier, tracking_number, status, delivered_date, is_insured, declared_value` |
| `Invoice` / `Order` | `models.py` | `line_items: list[LineItem]` where `LineItem = {product_id, name, sku, quantity, unit_price}` |
| `EligibilityResult` | `gates/eligibility.py` | `eligible: bool, reason: TOO_OLD\|WRONG_TYPE\|INSURED\|MISSING_INFO\|None, route: process\|close\|insured_process` |
| `AttachmentClassification` | `gates/evidence.py` | `category: EvidenceItem, confidence: float, usable: bool, quality_issue: str\|None` (one per attachment) |
| `Gap` | `gates/evidence.py` | `item: EvidenceItem, reason: MISSING\|UNUSABLE\|LOW_CONFIDENCE, detail: str\|None` (one per missing/bad required item) |
| `Judgment` | `gates/validation.py` | `passed: bool, confidence: float, note: str` (×4: damage_visible, product_identifiable, product_on_invoice, packaging_documented) |
| `ValidationResult` | `gates/validation.py` | the 4 `Judgment`s + `matched_skus: list[str]` |
| `ValidationDecision` | `gates/validation.py` | `outcome: PROCEED\|REQUEST_INFO\|ESCALATED, reason: str\|None` |
| `CalcResult` | `calc.py` | `amount: Decimal, line_items: list[RecommendationLineItem], capped: bool` |
| `RiskAssessment` | `risk.py` | `tier: LOW\|ELEVATED\|HIGH, flags: list[str]` |
| `Recommendation` | `models.py` | `decision: approve\|deny\|request_info, amount, line_items, rationale, email_draft, confidence, risk_tier` — **the one thing everything else feeds into** |
| `GuardViolation` | `guard.py` | `invariant: str (e.g. "CAP_EXCEEDED"), detail: str` (empty list = send is allowed) |

## 2. State machine at a glance

```
INTAKE ──▶ ELIGIBILITY ──▶ EVIDENCE ──▶ VALIDATION ──▶ CALC ──▶ PENDING_REVIEW
              │                │              │                     │
              ▼                ▼              ▼                     ├──▶ APPROVED ──▶ SENT ──▶ CLOSED
          ESCALATED ◀──────────┴──────────────┘                     ├──▶ DENIED    ──▶ SENT
       (insured / low-       (any gate failure routes                ├──▶ NEEDS_INFO ──▶ SENT ──▶ EVIDENCE
        confidence pass)      here with an explanation)               │              (customer replies, loop back)
                                                                       └──▶ ESCALATED ──▶ PENDING_REVIEW | SENT
```

Full edge list: `store.LEGAL_TRANSITIONS`. Every arrow above is enforced there — anything not listed
raises `IllegalTransitionError` and is refused. Every transition also writes an `audit_log` row in
the same DB transaction, so the full history of any case is always reconstructable.

## 3. The core idea

Two kinds of code, kept strictly separate:

- **Deterministic code** (gates, calculator, risk tiering, the state machine, the outbound guard) —
  plain Python, fully unit-tested, makes every decision that has money or eligibility consequences.
- **LLM judgment steps** (evidence classification, damage validation, email drafting) — always
  produce *structured, schema-validated* output, always carry a confidence score, and are **never
  the final actor**. An LLM never decides `approve`/`deny`/an amount — it only classifies evidence,
  judges photos, or writes prose explaining a decision that deterministic code already made.

A human reviews every case before anything reaches the merchant or the reimbursement API. Even
after human approval, a final **outbound guard** re-derives and re-checks every safety invariant
from stored data — not from the LLM's draft — immediately before send.

## 4. Request flow (happy path)

```
process_case(case_id)
  │
  ├─ INTAKE        fetch Case + Shipment via ShipBobClient
  │
  ├─ ELIGIBILITY   check_eligibility() — pure code
  │                 ├─ ineligible (too old / wrong type / missing info) → deny draft → PENDING_REVIEW
  │                 └─ insured → ESCALATED (routed to a separate process, never auto-handled)
  │
  ├─ EVIDENCE      classify_attachment() per photo (LLM, vision) → evidence_gaps() — pure code
  │                 └─ any of the 4 required items missing/unusable → request_info draft → PENDING_REVIEW
  │
  ├─ VALIDATION    validate_damage() (LLM, vision) → combine_validation() — pure code
  │                 ├─ any judgment failed        → request_info draft → PENDING_REVIEW
  │                 └─ passed but low-confidence  → ESCALATED (needs a human's eyes, not auto-approved)
  │
  ├─ CALC          reimbursement() — pure code, Decimal math, capped at $100
  │
  ├─ (risk tier, memory context — informational, don't gate anything)
  │
  ├─ DRAFT         draft() (LLM, text-only) writes rationale + email prose;
  │                 decision & amount are passed IN, never decided by the LLM
  │
  └─ PENDING_REVIEW  (always — a human must act; the pipeline never sends anything itself)
```

A rep on the review UI then does one of three things:

- **Approve & Send** — outbound guard re-verifies everything → `send_email` (+ `submit_reimbursement`
  per line item if `decision == "approve"`) → case moves to `SENT`.
- **Save Edits & Approve** — same as above, but with a rep-edited email body; the guard checks the
  *edited* text (so a rep accidentally changing the dollar amount gets caught).
- **Push Back** — rep feedback is stored, the drafter re-runs with that feedback in context, case
  stays in `PENDING_REVIEW`. The feedback is also distilled (async, non-blocking) into a short
  reusable policy note for future merchants' drafts — the self-evolving loop.

## 5. Directory layout

```
src/claimpilot/
├── models.py              Pydantic domain models (Case, Shipment, Order, Invoice, Recommendation…)
├── config.py              Settings (env-driven) + business-policy constants
├── db.py                  SQLite connection + schema (all 5 tables)
│
├── clients/               ShipBobClient abstraction — swap fixture ↔ real API with one env var
│   ├── base.py              Protocol + NotFoundError + get_client() factory
│   ├── fixtures.py          Parses docs/api/postman_collection.json + fixtures/synthetic.json live
│   ├── http.py              Real HTTP implementation (httpx) for the Postman mock server
│   └── attachment_guard.py  SSRF-safe image downloader (host allowlist, size cap, on-disk cache)
│
├── gates/                 Deterministic + LLM-judgment gates, one per pipeline stage
│   ├── eligibility.py       Pure — window/type/insurance check
│   ├── evidence.py          LLM (vision/text) — classify attachments; pure — evidence_gaps()
│   └── validation.py        LLM (vision) — judge damage; pure — combine_validation()
│
├── calc.py                Pure — reimbursement math, Decimal-only, $100 cap
├── risk.py                Pure — LOW/ELEVATED/HIGH risk tiering
│
├── llm.py                 The ONE place every LLM call goes through — forced structured output,
│                           retry-once, timeout, cost/latency logging, untrusted-data guard
├── openai_schema.py        Adapts a Pydantic JSON schema to OpenAI's strict function-calling format
├── prompts/                Reviewable, content-hashed .md prompt files (one per LLM call site)
│
├── draft.py                LLM (text) — writes rationale + email; decision/amount passed in, not decided
│
├── store.py                Business-logic persistence: state machine, audit trail, actions, stats
├── pipeline.py              process_case() — orchestrates every gate above into one pipeline run
├── guard.py                 Outbound guard — last-line safety re-verification before send
│
├── memory.py                Merchant/global notes + corrections; feeds risk tiering & drafting
├── evolve.py                Feedback distiller — turns a rep edit/pushback into a policy note
│
└── web/                    FastAPI review UI
    ├── app.py                 Routes: queue, case detail, history, approve, pushback, memory panel
    └── templates/             Jinja2 templates (queue.html, case.html, history.html)

gates/invoice_audit.py      Reconciles ShipBob's invoice API against the merchant's retail invoice

scripts/seed.py             Resets DB + outbox, runs process_case() for all 7 demo fixture cases
evals/golden.yaml            Expected outcomes per fixture case; evals/test_golden.py asserts them
                              against the REAL LLM (pytest -m eval, excluded from the default run)
docs/api/postman_collection.json   Source of truth for the 5 real fixture cases + API shapes
fixtures/synthetic.json      3 hand-built cases covering scenarios the real data doesn't (insured,
                              over-cap, same-merchant repeat for the memory demo)
```

## 6. File-by-file

### Domain & config

- **`models.py`** — every Pydantic model in the system. `Case`/`Recommendation` are `frozen=True`
  (immutable, since they represent a decided/recorded fact). `CaseState` is the 12-value pipeline
  lifecycle enum (`intake → eligibility → evidence → validation → calc → pending_review →
  approved|needs_info|denied → sent → closed`, plus `escalated`). `EvidenceItem` is the 4-category
  enum (`ORDER_PROOF`, `CUSTOMER_CONFIRMATION`, `PRODUCT_PHOTO`, `PACKAGING_PHOTO`).
- **`config.py`** — `Settings` (a `pydantic-settings` `BaseSettings`, reads `.env`): API keys,
  `llm_provider` (`"anthropic"` default, `"openai"` alternative), model names, and every
  business-policy constant (`cap`, `claim_window_days`, `evidence_min_conf`,
  `validation_min_conf`, `high_value_threshold`, timeouts, the attachment host allowlist, etc.) —
  all overridable via env var so a policy change never needs a code change. See `.env.example` for
  the full list.

### ShipBob API access

- **`clients/base.py`** — `ShipBobClient` is a `Protocol` (structural typing) with 9 async methods
  (list/get cases, attachments, shipment, order, invoice, send email, submit reimbursement).
  `get_client()` is a cached factory that returns either `FixtureClient` or `HttpShipBobClient`
  based on `settings.use_fixtures` — nothing else in the codebase needs to know which one is active.
- **`clients/fixtures.py`** — parses `docs/api/postman_collection.json`'s example responses *live*
  (no generated/hand-copied fixture files, so it can never drift from the real API shape) for the 5
  real demo cases, plus `fixtures/synthetic.json` for 3 hand-built ones. Writes `send_email`/
  `submit_reimbursement` calls to `outbox/*.jsonl` so a demo can show "what actually got sent."
- **`clients/http.py`** — the real implementation, for once ShipBob's mock API credentials exist.
  Normalizes 404s (and the invoice endpoint's `422 invoice_unavailable`) into one `NotFoundError`.
- **`clients/attachment_guard.py`** — downloads attachment images with SSRF defenses: host
  allowlist (`settings.allowed_attachment_host`), `image/*`-only, size cap
  (`settings.max_attachment_bytes`), on-disk cache keyed by a path-traversal-sanitized attachment id.

### Deterministic gates

- **`gates/eligibility.py`** — `check_eligibility(case, shipment, now, claim_window_days) →
  EligibilityResult(eligible, reason, route)`. Insurance check has top priority (routes to
  `insured_process`, never `close`, even if the claim would otherwise fail); window is inclusive at
  the boundary; missing delivery date is `MISSING_INFO`, not `TOO_OLD`.
- **`calc.py`** — `reimbursement(invoice, damaged: list[DamagedItem]) → CalcResult`. Sums
  `unit_price × quantity` for damaged lines only, caps at `settings.cap`, and — the interesting
  part — when the raw total exceeds the cap, **proportionally scales every line item's payout** so
  the per-item amounts still sum to *exactly* the cap (with an exact-cent rounding strategy, not
  independently-rounded amounts that could be a penny off).
- **`risk.py`** — `tier(shipment, merchant_memory) → RiskAssessment(tier, flags)`. Three factors
  (high declared value, high claim frequency, merchant has memory flags); 0 factors = `LOW`, 1 =
  `ELEVATED`, 2+ = `HIGH`. Purely informational — a `HIGH` tier gets a louder UI banner, it never
  auto-denies.

### LLM layer

- **`llm.py`** — the single chokepoint for every model call. `structured_call(case_id, prompt_name,
  schema, images=…)` loads and content-hashes a prompt file, forces the model to return exactly one
  tool call matching `schema` (so output always validates), retries once on a schema-validation
  failure, enforces a hard timeout, and logs every attempt (cost, latency, token counts, raw
  response) to the `llm_calls` table for a per-claim cost metric and full audit replayability. It
  **always** prepends a standing system rule that any `<untrusted_data>`-delimited content (merchant
  descriptions, attachment text) is data to analyze, never instructions to follow — the project's
  prompt-injection defense.
- **`openai_schema.py`** — adapts a Pydantic-generated JSON schema into the shape OpenAI's strict
  function-calling mode requires (`additionalProperties: false`, every field forced `required`).
- **Provider swap**: `Transport` is a `Protocol`; `AnthropicTransport` and `OpenAITransport`
  implement it identically. `get_transport()` picks one based on `settings.llm_provider` — nothing
  in `gates/`, `draft.py`, or `evolve.py` needs to know or care which provider is active.
- **`gates/evidence.py`** — `classify_attachment()` (vision for images, text for anything else) →
  `{category, confidence, usable, quality_issue}` per attachment; `evidence_gaps()` (pure) checks
  all 4 required categories are present with `confidence ≥ evidence_min_conf` and `usable=True` —
  otherwise it's treated as missing ("ask, don't guess").
- **`gates/validation.py`** — one vision call judges 4 things (damage visible, product
  identifiable, product on invoice, packaging documented) plus which invoice SKUs matched.
  `combine_validation()` (pure): all passed + confident → proceed; anything failed → ask for more
  info, naming the specific failed check; passed but low-confidence → escalate, naming the weakest
  judgment (e.g. *"damage visibility 0.62 — photo too dark"*). Low LLM confidence only ever pushes
  toward **more** human attention, never toward auto-approval.
- **`draft.py`** — `draft(inputs: DraftInputs) → Recommendation`. The LLM's output schema has
  exactly two fields (`rationale`, `email_draft`) — structurally, there is nowhere for the model to
  put a decision or an amount. Those come from `DraftInputs.decision`/`.amount`, set by
  `pipeline.py` from the gate results above, and are copied into the final `Recommendation`
  unchanged.

### Orchestration, persistence, and safety

- **`db.py`** — the SQLite connection + schema for all 5 tables: `cases` (status, recommendation
  JSON, gate-results JSON, merchant_id), `audit_log` (every state transition and rep action),
  `llm_calls` (the cost/latency ledger), `actions` (idempotency backstop — `UNIQUE(case_id,
  action)` so a double-click can never double-send or double-pay), `memory` (policy notes +
  corrections).
- **`store.py`** — the business-logic layer on top of `db.py`. `transition()` validates every state
  change against an explicit `LEGAL_TRANSITIONS` map (raises `IllegalTransitionError` otherwise) and
  atomically writes an audit row — this is the accountability story: every state change is
  reconstructable from `audit_log`. Also: `record_action()` (idempotency), `save_gate_results()` /
  `load_gate_results()` (so the outbound guard can re-verify against what actually happened, not
  the draft), `stats()` (approve-as-is/edit/pushback/escalation rates + mean LLM cost/latency).
- **`pipeline.py`** — `process_case(case_id) → Recommendation`. Runs every gate above in order,
  short-circuits to the right draft type on any failure, persists state + gate results at every
  step, and **always** ends in `pending_review` or `escalated` — it never calls `send_email` or
  `submit_reimbursement` itself.
- **`guard.py`** — `check_outbound(case, gate_results, calc, email, decision) →
  list[GuardViolation]`. Runs after a rep clicks approve, immediately before send. Re-verifies,
  from **stored data**, not the draft: amount ≤ cap and ≤ what the invoice actually supports (by
  independently re-running `reimbursement()`); an `approve` decision's underlying gates actually
  passed; every dollar figure written in the email body matches the approved amount; the recipient
  is `case.contact_email` and nothing else; every SKU mentioned exists on the invoice; the case is
  in a legal state for the transition. Any violation blocks the send and escalates the case, with
  the specific invariant that fired shown to the rep.

### Review UI

- **`web/app.py`** — FastAPI app. `GET /cases` (queue: pending/escalated cases, risk badges, stats
  bar), `GET /cases/history` (closed/sent cases, same layout as the queue minus the action buttons),
  `GET /cases/{id}` (full case detail — evidence thumbnails, gate rationale, calc breakdown,
  editable draft, audit timeline), `POST /cases/{id}/approve` (guard check → send → `sent`, records
  a correction if edited), `POST /cases/{id}/pushback` (stores feedback, re-drafts, stays in
  review), plus the memory-notes list/delete routes.
- **`gates/invoice_audit.py`** — the financial reconciliation gate. The brief says reimbursement is
  "based on the invoice — the price at time of fulfillment, after discounts," but two documents
  could mean: ShipBob's `POST /invoices/generate`, and the retail invoice the merchant submits as
  evidence. They disagree on every priced case in the fixture set. The API is the weaker candidate
  for what that rule describes — it carries no discount concept at all (`generate_invoice` returns
  byte-identical line items to `get_order`), every generated invoice is stamped with the same
  claim-time timestamp that postdates delivery, and it has no currency field even though one
  fixture case is plainly a GBP order. But a price read off a customer-supplied image by a vision
  model is untrusted data, and this codebase never lets untrusted data set a dollar amount.
  Resolution: `calc.reimbursement` keeps computing from the API, and this module independently
  re-reads the merchant's invoice and routes disagreements to a human. Three checks —
  `PRICE_MISMATCH`, `CURRENCY_MISMATCH`, `LINE_NOT_ON_RETAIL_INVOICE`, plus `QUANTITY_MISMATCH`
  when a bundle/kit line makes per-unit prices incomparable. Fails **open**: an unreadable invoice
  is recorded as unverified and surfaced in the UI, never escalated, since this is an additive
  check on a pipeline that already worked without it.
- **`web/templates/`** — plain Jinja2 + HTML forms (no JS framework needed for a demo).

### Memory & self-evolution

- **`memory.py`** — `record_correction()`, `merchant_context(merchant_id)` (recent notes + 90-day
  claim frequency + policy notes, feeding both `risk.tier()` and `draft()`), `global_policies()`,
  `record_policy_note()` (capped at 10 notes per merchant/global partition, oldest evicted).
- **`evolve.py`** — on every rep edit/pushback, diffs the original vs. final draft and distills it
  (LLM call) into 0–2 short, reusable policy notes. A validator rejects any note containing
  decision words (`approve`/`deny`/dollar amounts) so memory can never quietly override the
  deterministic gates — it's for tone/context only ("mention the account manager by name"), never
  for outcomes.

### Evals, metrics, ops

- **`evals/golden.yaml` + `evals/test_golden.py`** — expected eligibility/evidence/validation/calc
  outcomes for all 7 original fixture cases, asserted against the **real** LLM
  (`pytest -m eval`, deselected from the default test run — see `pyproject.toml`).
- **`scripts/seed.py`** — resets the DB + outbox and runs `process_case()` for all 7 demo cases
  (real LLM calls — needs a real API key). Supports `--case ID` to seed/re-seed one case without
  wiping the rest (used for the live memory-demo repeat case).

## 7. Real trace: one case end to end

This is `CASE-1002` (CleanBoss) after a real seeded run through `pipeline.process_case()` — actual
output, not a hypothetical, captured from a live run of this app.

**1. Fetch.** `client.get_case("CASE-1002")` and `client.get_shipment(...)` return a `Case`
(`sub_category="Claim | Damaged in Transit"`, `delivered_date`/`created_date` 4 days apart,
`contact_email="mtaparia@shipbob.com"`) and a `Shipment` (`is_insured=False`).

**2. `ELIGIBILITY`.** `check_eligibility()` — pure code, no LLM — returns
`EligibilityResult(eligible=True, reason=None, route="process")`: within the 30-day window, right
claim type, not insured. `store.transition(INTAKE → ELIGIBILITY)` writes an audit row.

**3. `EVIDENCE`.** 4 real attachments are fetched and each classified individually via
`classify_attachment()` — one vision `structured_call` per image. All 4 land as usable,
confident matches for the 4 required `EvidenceItem` categories, so `evidence_gaps()` (pure) returns
`[]`. `store.transition(ELIGIBILITY → EVIDENCE)`.

**4. `VALIDATION`.** One vision call (`validate_damage()`) is made with the product + packaging
photos and the invoice's line items as text context. Real output: all 4 `Judgment`s `passed=True`
with confidence ≥ `0.75`, `matched_skus=["A00300"]`. `combine_validation()` (pure) →
`ValidationDecision(outcome=PROCEED)`. `store.transition(EVIDENCE → VALIDATION)`.

**5. `CALC`.** `reimbursement(invoice, damaged=[DamagedItem(sku="A00300", quantity=1)])` — pure,
`Decimal`-only — looks up `A00300` on the invoice and returns `CalcResult(amount=Decimal("12.99"),
capped=False)`. `store.transition(VALIDATION → CALC)`.

**6. Risk + draft.** `risk.tier(shipment, merchant_memory)` → `RiskAssessment(tier=LOW, flags=[])`
(no prior history for this merchant yet). `draft()` is called with `decision="approve"` (chosen by
`pipeline.py`, not the LLM) and `amount=Decimal("12.99")` already fixed; the LLM only writes:

> - **Eligibility**: The claim is within the 30-day window and eligible for processing.
> - **Evidence**: No unresolved evidence gaps, all required evidence categories were satisfied.
> - **Validation**: The validation process confirmed the damage and matched it to an invoice line item.
> - **Calc**: Reimbursement totaled $12.99 across 1 line item.
> - **Risk**: Tier is LOW with no flags raised, indicating a low risk assessment.

...plus an email body starting "Dear Customer, We have completed the review of your claim...". The
final `Recommendation(decision="approve", amount=Decimal("12.99"), confidence=0.90, risk_tier="LOW",
rationale=<above>, email_draft=<above>)` is saved. `store.transition(CALC → PENDING_REVIEW)`.

**7. Human review.** The case now sits on `GET /cases` with a `LOW` risk badge and `approve`
decision. A rep opens `GET /cases/CASE-1002`, sees the 4 evidence thumbnails, the rationale above,
a 1-row calc breakdown table, the editable email textarea, and the audit timeline. They click
**Approve & Send**.

**8. Guard, then send.** `POST /cases/CASE-1002/approve` loads the case, the saved
`Recommendation`, and the persisted `GateResults` (not the draft), builds the final `EmailToSend`,
and calls `check_outbound(...)`. It re-runs `reimbursement()` from the stored invoice/matched-SKUs
and confirms it still equals `$12.99`; confirms eligibility/evidence/validation all genuinely
passed; regex-scans the email body and confirms every `$` figure equals `$12.99`; confirms `to ==
case.contact_email`; confirms `A00300` is a real invoice SKU. Zero violations →
`record_action("email")` → `send_email(...)` → `record_action("reimbursement")` →
`submit_reimbursement(...)` (since `decision == "approve"`) → `store.transition(PENDING_REVIEW →
APPROVED → SENT)`. The outbox now has a real JSONL line showing exactly what went out.

If the rep had instead edited the email to say "$25.00" before clicking approve, step 8's guard
would have caught the mismatch (`EMAIL_AMOUNT_MISMATCH`), refused to call `record_action`/`send_email`
at all, and escalated the case instead — that exact scenario is one of the four demo beats in
[`LIMITATIONS.md`](./LIMITATIONS.md).

## 8. Design invariants worth remembering

1. **LLM output is always structured and never final.** Every model call is schema-forced; the
   three things with real consequences — decision, amount, recipient — are always deterministic
   inputs, never parsed from a model's prose.
2. **Low confidence only ever escalates.** Nothing in this system uses LLM confidence to *approve*
   faster or more leniently — only to ask for more evidence or hand a case to a human.
3. **The guard re-derives, it doesn't trust.** `guard.py` re-runs the reimbursement math and
   re-checks the gate results from storage — a corrupted or tampered `Recommendation` can't sail
   through on its own say-so.
4. **Everything is swappable behind a `Protocol`.** `ShipBobClient` (fixture ↔ real API) and
   `Transport` (Anthropic ↔ OpenAI) are both structural-typing interfaces with zero call-site
   awareness of which implementation is active — flip one env var.
5. **Every transition is audited.** `store.transition()` can't change `cases.status` without also
   writing an `audit_log` row, atomically, in the same DB transaction.
6. **Memory can inform tone, never override policy.** `evolve.py`'s validator is a hard block on
   any distilled note that looks like it's trying to influence a decision or amount.

See [`LIMITATIONS.md`](./LIMITATIONS.md) for the known rough
edges surfaced while building it.
