# syntax=docker/dockerfile:1
#
# Single-stage image: this project has no compiled/C-extension dependencies
# (see pyproject.toml -- fastapi/httpx/pydantic/anthropic/openai are all
# pure-Python-wheel installs), so there's no build-vs-runtime size tradeoff
# a multi-stage build would meaningfully help with. Installs `.[dev]` (not
# just the runtime deps) so the same image can run the app, the test suite,
# and `scripts/seed.py` without a separate dev image -- this is a small
# take-home-scale project, not a case where trimming pytest/respx off a
# production image is worth the added Dockerfile complexity.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# curl: only for the HEALTHCHECK below. Nothing else here needs a compiler
# or system library.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer first, source layer second -- editing application code
# should never bust the (slow) dependency-install cache layer.
COPY pyproject.toml ./
COPY src ./src
# `-e` (editable), not a normal install: `config.py`/`fixtures.py`/
# `attachment_guard.py` all locate `docs/`, `fixtures/`, `outbox/` etc. via
# `Path(__file__).resolve().parents[N]` -- a fixed traversal count that
# assumes the on-disk depth of the `src/claimpilot/...` source layout.
# A normal (non-editable) install copies the package into site-packages
# WITHOUT that `src/` layer, silently shifting every one of those `parents[N]`
# calls one directory too far up (confirmed directly: it resolved to
# `/usr/local/lib/python3.12` instead of `/app`, breaking every fixture-
# loading test with `FileNotFoundError`). An editable install keeps
# `__file__` pointing at the real `/app/src/claimpilot/...` copied in above,
# so those `parents[N]` calls resolve exactly like they do in every
# native/CI run today.
RUN pip install --no-cache-dir -e ".[dev]"

# Everything else the app/scripts/tests actually read at runtime.
# `docs/api/postman_collection.json` (fixtures.py's source of truth),
# `fixtures/synthetic.json`, `scripts/seed.py`, `evals/`, and `tests/`
# (so `docker compose exec web pytest` works without a rebuild).
COPY docs ./docs
COPY fixtures ./fixtures
COPY scripts ./scripts
COPY evals ./evals
COPY tests ./tests

# `claimpilot.db` (SQLite) and `outbox/`/`fixtures/images/` (the on-disk
# attachment cache) are RUNTIME STATE, never image content -- deliberately
# not copied in, and expected to be bind/volume-mounted (see
# docker-compose.yml and README.md's "Running with Docker" section) so a
# container rebuild/recreate never silently wipes the review queue.
RUN mkdir -p outbox fixtures/images data

EXPOSE 8000

# Hits the dedicated, always-unauthenticated /health route (added in a
# security audit alongside opt-in Basic Auth -- see config.py's
# review_ui_username/password) rather than /cases -- a health probe must
# never depend on credentials, and pointing this at /cases would start
# reporting false-unhealthy the moment REVIEW_UI_USERNAME/PASSWORD are set.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "claimpilot.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
