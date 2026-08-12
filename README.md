# claimpilot

Automates ShipBob's damaged-in-transit claim workflow end-to-end with human-in-the-loop review.

A claim arrives, and the system runs it through eligibility, evidence completeness, damage
validation, and reimbursement calculation, then drafts a recommendation and a customer email for a
claims rep to approve, edit, or push back on. Nothing reaches a merchant without a human pressing
a button.

## How it works

**Deterministic code decides; the LLM only judges.** Eligibility rules, reimbursement maths, the
$100 cap, risk tiering and every state transition are plain Python. The model is used at exactly
four points where judgement is genuinely required — classifying an attachment, assessing damage
from photos, reading a merchant's invoice, and writing prose. It is structurally unable to decide
an outcome or an amount: the drafting call's output schema has no field for either.

**Untrusted data can ask for a human; it can never move money.** Merchant descriptions, customer
messages and figures read off uploaded invoices are all treated as untrusted. They're used to
cross-check what the system computed, and a disagreement escalates the case — but no value
extracted from them ever feeds the payout.

**Four independent cross-checks** run against every claim that reaches a payout, each capable of
routing to a human but never of resolving the conflict itself:

| Check | Catches |
|---|---|
| Affected-count | The merchant's stated item count disagrees with the SKUs evidence confirms |
| Retail-invoice reconciliation | ShipBob's invoice price/quantity/currency disagrees with the merchant's own invoice |
| Claim scope | The customer claims more (or less) damage than the evidence supports |
| Outbound guard | The final email or payout disagrees with what the gates actually found |

**Everything is auditable.** Every state transition, gate result, LLM call (with token cost and
latency) and rep action is written to SQLite. The review UI renders that trail per case.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the pipeline flow, directory layout and
file-by-file responsibilities, and [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) for an honest
account of where it breaks and which trade-offs were deliberate.

## Running the app

**Prerequisites:** Python 3.12 (or Docker), and an **OpenAI API key**.
(`.env.example` is pre-set to OpenAI. Anthropic is equally supported: set `LLM_PROVIDER=anthropic`
and `ANTHROPIC_API_KEY` instead.)

The key matters, so to be explicit about what needs one: browsing the review UI, fetching cases
from the ShipBob API, and the entire 524-test suite all run fine **without** a key. *Processing* a
case is what makes real LLM calls -- four per case, roughly 10s and ~$0.02 -- and that needs one.
The queue page shows a banner if no key is configured, rather than letting you find out by
watching cases fail.

No ShipBob credentials are needed: the app reads the case data from the Postman collection
committed under `docs/api/` by default (`USE_FIXTURES=true`), and can be pointed at the live mock
server instead by setting `USE_FIXTURES=false`.

Two ways to run this: natively (a local Python venv) or containerized (Docker). Both end up
running the exact same code -- pick whichever fits your workflow.

### Option A: natively (venv)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # editable install -- see the Docker section below for why
                                # this matters, not just a style preference
cp .env.example .env            # then add your OPENAI_API_KEY -- see Prerequisites above
pytest                          # full test suite -- 524 tests, no API key or network needed (~30s)
```

Start the review UI:

```bash
uvicorn claimpilot.web.app:app --reload
# -> http://127.0.0.1:8000/cases
```

Populate the queue by running every fixture case through the real pipeline (makes real, billed
LLM calls -- takes roughly 10s per case):

```bash
python scripts/seed.py                  # full reset + seed all demo cases
python scripts/seed.py --case CASE-1001 # seed one case only, without resetting the rest
```

Other common commands:

```bash
pytest -m eval                  # golden-set evals against the REAL LLM API (see below)
sqlite3 claimpilot.db ".tables" # inspect the on-disk SQLite DB directly
```

### Option B: Docker

```bash
cp .env.example .env             # add your OPENAI_API_KEY first
docker compose up -d --build
# -> http://127.0.0.1:8000/cases
```

That's the whole setup -- `docker-compose.yml` builds the image from the `Dockerfile`, reads
`.env` for API keys/config, and mounts three named volumes (`claimpilot-data`,
`claimpilot-outbox`, `claimpilot-attachment-cache`) so the review queue, sent-email outbox, and
downloaded evidence photos all survive a container restart/rebuild -- they're runtime state, not
image content, and are deliberately excluded from the image itself (see `.dockerignore`).

Common Docker commands:

```bash
docker compose logs -f web                       # tail the running app's logs
docker compose exec web python scripts/seed.py   # seed demo data into the running container
docker compose exec web pytest -q                # run the full test suite inside the container
docker compose exec web sh                        # shell into the running container
docker compose restart web                        # restart after an .env change
docker compose down                               # stop (add -v to also delete the 3 volumes)
```

**Why the Dockerfile installs with `pip install -e .` (editable), not a normal install:**
`config.py`/`clients/fixtures.py`/`clients/attachment_guard.py` all locate `docs/`, `fixtures/`,
and `outbox/` via `Path(__file__).resolve().parents[N]` -- a fixed traversal count that assumes
the on-disk depth of the `src/claimpilot/...` source layout. A normal (non-editable) install
copies the package into `site-packages` *without* that `src/` layer, silently shifting every one
of those `parents[N]` calls one directory too far up. Confirmed directly while building this
image: a non-editable install resolved `_REPO_ROOT` to `/usr/local/lib/python3.12` instead of
`/app`, breaking every fixture-loading test with `FileNotFoundError` the moment the test suite ran
inside the container. `-e` keeps `__file__` pointing at the real copied-in source tree, so those
paths resolve exactly like they do in every native/CI run -- this is a real, environment-dependent
fragility in the existing path-resolution approach, not a Docker-specific quirk, and would bite
equally hard behind any other non-editable packaging (a built wheel, a different container base
image, etc.).

**Health check:** the image's `HEALTHCHECK` hits the dedicated `GET /health` endpoint (always
unauthenticated, even when `REVIEW_UI_USERNAME`/`PASSWORD` are set -- see Access control below)
every 30s; `docker compose ps` / `docker inspect` will show `healthy` once the app is actually
serving real responses, not just "the process launched".

**One thing to watch for:** don't run the native venv server and the Docker container on the same
port at the same time -- both default to `8000`, and having two processes bound to it at once
(even if the OS lets it happen without erroring) makes it genuinely ambiguous which one is actually
answering a given request. Stop one before starting the other, or remap the container's port in
`docker-compose.yml` (`ports: ["8010:8000"]`, for example).

## Walking through it

The review queue has demo controls at the top, so the whole flow can be driven from the browser:
**Fetch cases** (pulls the case list from the ShipBob API — instant), **Process all** (runs the
pipeline; backgrounded with a progress bar, ~10s per case), **+ Add demo scenarios** (see below),
and **Reset all data**. Set `DEMO_CONTROLS_ENABLED=false` to remove them entirely.

The five API cases each exercise a different path:

| Case | What it shows |
|---|---|
| `CASE-1002` CleanBoss | Full path to a payout — and the retail-invoice reconciliation catching that ShipBob's invoice and the merchant's own invoice disagree on price and quantity. Escalates instead of paying. |
| `CASE-1001` Best Paw | Missing packaging photo → asks the customer for exactly that, nothing more |
| `CASE-1003` Huge Supplements | Invoice plus two customer emails, no product or packaging photos → two evidence gaps |
| `CASE-1004` Catalyze-X | Delivered 73 days before filing → denied at the eligibility gate, before any LLM call |
| `CASE-1005` Loam Science | Zero attachments → all four evidence items requested |

To exercise a rep correction carrying into a later claim, push back on a draft with feedback
(e.g. *"mention their account manager Dana"*), which is distilled into a durable policy note, then
process another case from the same merchant and see the note reflected in its draft without anyone
re-typing it.

### Demo scenarios

Three of the business rules can't be exercised by the API data: every real shipment is uninsured, all
five belong to different merchants, and the largest line item is $59.99 — so insured routing, the
repeat-merchant memory loop and the $100 cap have nothing to fire on. **+ Add demo scenarios**
adds three synthetic cases covering exactly those gaps. They're opt-in and clearly labelled, and
they work against the live API too (`clients/synthetic.py` overlays them on whichever client is
configured), so a normal fetch never mixes invented cases into real data.

## CI

`.github/workflows/ci.yml` runs the full default test suite (no real network/LLM calls, no
secrets needed) on every push/PR. The real-LLM golden evals (`pytest -m eval`) are deliberately
not wired into CI -- they cost real money per run and are a human-run gate before merging a prompt
change (see "Golden-set evals" below), not something to run unattended on every push.

## Configuration

Everything deployment-tunable lives in `.env` (copy from `.env.example` to start) and is read via
`claimpilot.config.settings` -- no code change is ever needed to adjust a policy value, a timeout,
or a retry setting. `.env` itself already ships with every value below set explicitly to its
default, so the file doubles as living documentation of what's tunable; edit a line and restart
(`uvicorn --reload` picks it up automatically; Docker needs `docker compose restart web`).

### Secrets & provider selection

| Var | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Real LLM API key. Only the one matching `LLM_PROVIDER` is actually used. |
| `LLM_PROVIDER` | `openai` (pre-set in `.env.example`) or `anthropic` -- selects which `Transport` backs every LLM call. Both are fully supported. |
| `ANTHROPIC_MODEL` / `OPENAI_MODEL` | Model ID per provider. Defaults: `claude-sonnet-5` / `gpt-4o`. |
| `SHIPBOB_API_BASE` / `SHIPBOB_API_KEY` | Real ShipBob mock server URL + API key (only used when `USE_FIXTURES=false`). |
| `USE_FIXTURES` | `true` (default) reads case/shipment/order/invoice data from local fixtures (`docs/api/postman_collection.json` + `fixtures/synthetic.json`) -- zero network dependency. `false` hits the live mock via `SHIPBOB_API_BASE`. |

### Business-policy thresholds

These are the values a real policy change (not a code change) would touch -- all currently
placeholders pending real ShipBob guidance, per `config.py`'s own comments.

| Var | Default | Meaning |
|---|---|---|
| `CLAIM_WINDOW_DAYS` | `30` | How many days after delivery a claim can still be filed. Older -> `TOO_OLD`, denied. |
| `ELIGIBLE_SUB_CATEGORY` | `Claim \| Damaged in Transit` | The only `Case.sub_category` value treated as in-scope; anything else -> `WRONG_TYPE`, denied. |
| `CAP` | `100.00` | Max total reimbursement per claim, in USD. The real dollar ceiling `guard.py` independently re-enforces before any send. |
| `HIGH_VALUE_THRESHOLD` | `500.00` | A shipment's declared value at/above this flags a high-value risk factor. |
| `HIGH_CLAIM_FREQUENCY_THRESHOLD` | `3` | A merchant with this many+ claims in the trailing window flags a high-frequency risk factor. |
| `EVIDENCE_MIN_CONF` | `0.7` | Minimum confidence for a classified attachment to count toward satisfying a required evidence category. |
| `VALIDATION_MIN_CONF` | `0.75` | Minimum confidence, across all 4 damage-validation judgments, to `PROCEED` rather than `ESCALATED`. |
| `CLAIM_FREQUENCY_WINDOW_DAYS` | `90` | Trailing window for merchant claim-frequency counting (risk tiering + the drafter's prompt context). |
| `MAX_RECENT_NOTES` | `5` | How many recent merchant notes/corrections the drafter's prompt sees. |
| `POLICY_NOTE_CAP_PER_PARTITION` | `10` | Max stored policy notes per (scope, merchant) before the oldest are evicted. |

### Retail-invoice audit

Reconciles ShipBob's invoice API against the merchant's own submitted retail invoice (the
order-proof attachment) and escalates material disagreements to a human. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and `gates/invoice_audit.py`'s module docstring for
why both documents exist and why neither alone can be the payout basis.

| Var | Default | Meaning |
|---|---|---|
| `INVOICE_AUDIT_ENABLED` | `true` | Read and reconcile the merchant's retail invoice. Costs one extra vision call per case that reaches validation. `false` restores the exact pre-audit behavior (API price, no reconciliation). |
| `INVOICE_PRICE_TOLERANCE` | `0.01` | How far ShipBob's unit price may differ from the retail invoice's before it's reported. Defaults to one cent, i.e. any real difference is surfaced. |
| `EXPECTED_CURRENCY` | `USD` | The currency `CAP` and every stored `unit_price` are assumed to be in. The mock API has no currency field at all, so this is an assumption the system can't verify itself — the audit exists partly to catch when a retail invoice contradicts it. |

### Demo controls

Three buttons on the review queue, for driving a live walkthrough without dropping to a terminal:
**Fetch cases** (pull the case list from the ShipBob API into the queue — one API call, instant),
**Process all** (run every fetched case through the pipeline; runs in the background with a progress
bar so the browser isn't blocked for the ~10s-per-case the real vision calls take), and
**Reset all data** (wipe every table plus the outbox).

| Var | Default | Meaning |
|---|---|---|
| `DEMO_CONTROLS_ENABLED` | `true` | Exposes the three demo buttons and their endpoints. **Set `false` outside a demo** — the reset button empties every table. The endpoints 404 when disabled, so this closes the door rather than only hiding the buttons. |

Progress state is in-memory and per-process, so "Process all" assumes a single worker (the default
`docker compose` setup). It's a walkthrough affordance, not a job queue.

### LLM wrapper

| Var | Default | Meaning |
|---|---|---|
| `LLM_TIMEOUT_SECONDS` | `30.0` | Hard wall-clock timeout for a single `structured_call()` attempt. |
| `LLM_MAX_TOKENS` | `4096` | Max output tokens per attempt. |

### Live-mock resilience (retries + HTTP timeout)

Governs every real network call to the ShipBob mock (`clients/http.py`, `clients/attachment_guard.py`)
-- added after directly observing intermittent `httpx.ReadTimeout`s against the free-tier mock. A
404/422 (a real, deterministic answer) is never retried regardless of these settings; only
timeouts/connection errors and 5xx responses are.

| Var | Default | Meaning |
|---|---|---|
| `SHIPBOB_HTTP_TIMEOUT_SECONDS` | `15.0` | Per-request timeout (httpx's own default is 5s, which was part of the problem). |
| `HTTP_MAX_RETRIES` | `3` | Max retry attempts after the first (so up to 4 total attempts). |
| `HTTP_RETRY_BASE_DELAY_SECONDS` | `0.5` | Delay before the first retry; doubles each subsequent attempt (exponential backoff). |
| `HTTP_RETRY_MAX_DELAY_SECONDS` | `8.0` | Hard ceiling on the backoff delay, before jitter. |

### Infrastructure

| Var | Default | Meaning |
|---|---|---|
| `DB_PATH` | repo-root `claimpilot.db` | On-disk SQLite path. In Docker this is overridden to `/app/data/claimpilot.db` (see `docker-compose.yml`) so it lands on a volume, not ephemeral container storage. |
| `ALLOWED_ATTACHMENT_HOST` | `sa032101pubdevuc.blob.core.windows.net` | SSRF allowlist -- the only host attachment photos are ever fetched from. |
| `MAX_ATTACHMENT_BYTES` | `10485760` (10 MiB) | Hard cap on one downloaded attachment's size. |

### Access control

**Found in a full-codebase security audit: this app has no authentication by default.** Anyone who
can reach the URL can approve reimbursements, send emails, or delete policy notes. Fine for local
demo use; not fine for anything reachable beyond your own machine.

| Var | Default | Meaning |
|---|---|---|
| `REVIEW_UI_USERNAME` / `REVIEW_UI_PASSWORD` | blank (no auth) | Opt-in HTTP Basic Auth for the whole review UI. Active only when **both** are set. `/health` is always reachable unauthenticated (Docker's `HEALTHCHECK` depends on this). |

```bash
curl -u rep:yourpassword http://127.0.0.1:8000/cases   # once REVIEW_UI_USERNAME/PASSWORD are set
```

### Verifying a change actually took effect

Since `settings` is a module-level singleton read once at process start, confirm a `.env` edit
really landed with a fresh process, not just by re-reading the file:

```bash
python3 -c "from claimpilot.config import settings; print(settings.cap, settings.claim_window_days)"
```

## Golden-set evals

`evals/golden.yaml` encodes the expected eligibility outcome, evidence-gap
expectations, validation-verdict expectations, and calc amount for each of
the 7 known fixture cases. `evals/test_golden.py` runs the full pipeline
against those cases and the **real** LLM API -- whichever provider `LLM_PROVIDER` is set to
(`pytest -m eval` -- these tests are excluded from the default `pytest` run and require a real
`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`, matching whatever `LLM_PROVIDER` is configured).

**Rule: no prompt change ships without green evals.** Before merging any
change to a prompt (evidence classification, damage validation, drafting),
run `pytest -m eval` and confirm every golden case still lands within its
expected outcome. This is the regression net that makes it safe to keep
iterating on prompts.
