# Synthetic fixtures

Real fixtures are bodies the live API returned and are listed in `specs/PROBE-RESULTS.md`.
The files below were written by hand because the API will not produce them on demand.

| File | Shape | Why it is hand-written |
|---|---|---|
| `chat_truncated.json` | JSON cut mid-object | Reproduces the gemini `max_tokens: 64` truncation observed in PROBE-RESULTS P2, which is no longer reachable now that `max_tokens` is 512. |
| `chat_no_label.json` | valid JSON, no `label` key | A model answering in prose inside JSON mode. |
| `chat_wrong_case.json` | `{"label": "Billing"}` | Case variant; spec 03 requires `invalid_label`, not silent normalisation. |
| `chat_unknown_label.json` | `{"label": "refunds"}` | A label outside the five. |
| `chat_no_usage.json` | a 200 with no `usage` block | Spec 06: must warn, not crash. |
| `http_500.json` | a provider error body | Asserts no retry happens. |
| `empty.bin` | zero bytes | A 200 with nothing in it. |

`scripts/probe.py` writes a fresh body per arm into `tests/fixtures/captured/`, which is
gitignored. That directory is a refresh workspace for diffing against a provider's current
envelope; the files the tests assert against are the ones checked in beside this note.
