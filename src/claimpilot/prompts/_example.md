You are a placeholder prompt for `claimpilot.llm.structured_call`.

This file exists so the LLM wrapper's prompt-loading and content-hashing
code has a real `.md` file to load, hash, and send as (part of) the system
prompt in generic tests of that mechanism, independent of any specific
real prompt. Leading underscore (`_example`) marks it as internal/
non-production, the way a leading underscore marks a private module -- it
is not meant to be referenced by real pipeline code.
