# Probe results — verified evidence

Every claim in the other spec files that says "verified" points here. Run date **2026-09-20**,
against the live OpenRouter API with the key in `.env`. Raw bodies are checked in under
`tests/fixtures/` and are what the offline tests assert against.

## P1 · Jev is served on a separate endpoint and it works

`POST https://openrouter.ai/api/alpha/decisions` → **200 in 1010 ms** (cold connection).

Request body sent:

```json
{"model": "typesafe/jev-1.13",
 "state": {"ticket": "My card was charged twice this month and the invoice only shows one line."},
 "questions": {"department": {"type": "choice", "instructions": "Which team should handle this ticket.",
   "criteria": {"billing": "...", "technical": "...", "account": "...", "sales": "...", "other": "..."}}}}
```

Response body, verbatim:

```json
{"model": "typesafe/jev-1.13-20260917",
 "answers": {"department": {"type": "choice", "choice": "billing",
   "probabilities": {"sales": 0, "billing": 1, "other": 0, "technical": 0, "account": 0},
   "confidence": 1}},
 "usage": {"input_tokens": 395, "output_tokens": 52, "cost": 1.659e-05},
 "id": "gen-dec-1789847048-FaapfM5vPUHA2kclMlvY",
 "provider": "TypeSafe"}
```

**This closes the two open questions the previous spec round could not answer.**

- The envelope key is `answers`, not `choices`. Each answer carries `choice`, a full
  `probabilities` map over the criteria, and `confidence`.
- Jev **does** return a `usage` block, with the same three fields for every call. The
  `/api/v1/generation` fallback described in the earlier spec is not needed.
- `output_tokens` is 52 and billed at $0. Check: 395 × $0.042/M = $1.659e-05, exactly the
  reported `cost`. **Jev's cost is a pure function of input tokens.**

## P2 · Chat arms return comparable usage

`POST /api/v1/chat/completions`, `temperature: 0`, `max_tokens: 64`, JSON mode, same ticket.
All five answered `billing` correctly.

| Model | Wall | in | out | cost | note |
|---|---|---|---|---|---|
| anthropic/claude-sonnet-5 | 2662 ms | 71 | 11 | $0.000252 | |
| z-ai/glm-5.3-flash | 1746 ms | 57 | 56 | $0.0000366 | |
| qwen/qwen3.8-flash | 1889 ms | 99 | 51 | $0.0000388 | |
| google/gemini-3.8-flash | 11545 ms | 46 | 60 | $0.0002595 | **truncated at the 64-token cap** |
| openai/gpt-5.3-codex | 2826 ms | 54 | 30 | $0.0005145 | |

Two lessons went straight into the design: **`max_tokens` must be 512, not 64**, or reasoning
models get cut off mid-answer and score as parse failures; and `usage: {"include": true}` must
be sent to get `cost` and `completion_tokens_details.reasoning_tokens` back.

## P3 · Reasoning cannot be disabled uniformly

Same ticket, `max_tokens: 512`, three settings per model.

| Model | setting | wall | out | reasoning | cost |
|---|---|---|---|---|---|
| gemini-3.8-flash | default | 3089 ms | 176 | 171 | $0.000693 |
| gemini-3.8-flash | `{"enabled": false}` | — | — | — | **HTTP 400** |
| gemini-3.8-flash | `{"effort": "minimal"}` | 1766 ms | 5 | 0 | $0.0000518 |
| gpt-5.3-codex | default | 1964 ms | 38 | 24 | $0.0006265 |
| gpt-5.3-codex | `{"enabled": false}` | 1932 ms | 12 | 0 | $0.0002625 |
| gpt-5.3-codex | `{"effort": "minimal"}` | 2041 ms | 26 | 12 | $0.0004585 |
| claude-sonnet-5 | all three | ~2.3 s | 11 | 0 | $0.000252 |

The 400 reads: `Reasoning is mandatory for this endpoint and cannot be disabled.`

**Consequences, all binding on the implementation:**

1. There is no single "reasoning off" switch. The low arm must try `{"enabled": false}`, and
   on a 400 fall back to `{"effort": "minimal"}`. Which one answered is recorded per arm and
   printed in the report, because they are not the same setting.
2. On this task reasoning costs **13× on gemini** ($0.000693 → $0.0000518) and **2.4× on
   gpt-5.3-codex**, for an answer that was already correct without it. That is the single
   most quotable number this benchmark will produce.
3. Anthropic models do not reason here by default, so they get one arm, not two. Giving them
   a "low" arm would be two identical arms with different labels.

## P4 · Model roster is real

All 20 arms resolved against `GET /api/v1/models` on 2026-09-20; every chat model reports
`response_format` and `structured_outputs` in `supported_parameters`. `meta-llama/llama-4-scout`
is the only one without `reasoning`, which is why it has a single arm.

## How to re-run

```bash
python3 scripts/probe.py --all --write-fixtures
```

Rewrites `tests/fixtures/` and this file's tables. Costs about $0.01. Run it whenever a
number here is older than the model roster.
