"""Shared test fixtures.

Exists for one reason: **the test suite must not depend on what happens to be
in the developer's `.env`.** `claimpilot.config.settings` is a module-level
singleton populated from the environment at import time, so without this a
run's outcome varies with whoever's machine it's on.

That wasn't hypothetical. The review UI's process endpoints refuse to start
work when no LLM API key is configured (see `web/app._require_api_key` -- it
turns an unhelpful provider-SDK auth error into an actionable message). Nine
tests exercising those endpoints passed on a machine with a real key in
`.env` and failed on a freshly-cloned checkout without one. Locally green,
broken for anyone else -- and the tests themselves weren't wrong, the suite
just wasn't hermetic. Caught by cloning the repo clean and running it the way
a reader would.
"""

from __future__ import annotations

import pytest

from claimpilot.config import settings


@pytest.fixture(autouse=True)
def _hermetic_llm_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin credentials to fixed dummy values for every test.

    Applied everywhere rather than only to the tests that need it, so this
    can't silently stop covering a future endpoint that grows its own
    credential check.

    Safe because no test in the default suite makes a real API call: every
    one injects a `FakeTransport`, and the real-API golden evals are marked
    `eval` and deselected by default (`pyproject.toml`'s `addopts`). These
    values only ever satisfy "is a key configured?" checks -- nothing
    authenticates with them.

    A test that specifically covers *missing*-credential behaviour just
    monkeypatches them back to `""` itself; the later `monkeypatch.setattr`
    wins, and `monkeypatch` unwinds both in reverse order at teardown.
    """
    monkeypatch.setattr(settings, "llm_provider", "anthropic", raising=False)
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key-not-real", raising=False)
    monkeypatch.setattr(settings, "openai_api_key", "test-key-not-real", raising=False)
