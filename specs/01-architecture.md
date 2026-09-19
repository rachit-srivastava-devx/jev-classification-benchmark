# 01 · Architecture

## Bottom line

One Python package, standard library plus `pytest`. Nine modules, each with one job and a
test file that can run with no API key and no network. The only module that touches the
network is `transport.py`, and it is injected everywhere else.

## Layout

```
jev-demo/
├── README.md              # findings, written last
├── specs/                 # this directory
├── data/tickets.json      # 100 tickets + gold labels
├── scripts/probe.py       # live probe; rewrites fixtures
├── jevdemo/
│   ├── labels.py          # 5 labels + definitions — SINGLE SOURCE
│   ├── arms.py            # the 20-arm registry — SINGLE SOURCE of model ids
│   ├── transport.py       # the ONLY network code. Injected as a callable.
│   ├── jev_arm.py         # build request / parse response for the decisions endpoint
│   ├── chat_arm.py        # build request / parse response for chat completions
│   ├── pricing.py         # integer micro-dollars; cost reconciliation
│   ├── metrics.py         # pure aggregation over Prediction lists
│   ├── runner.py          # two-pass orchestration
│   └── report.py          # stdout table + results.json + doctrine HTML
└── tests/
    ├── fixtures/          # real captured API bodies (see PROBE-RESULTS)
    └── test_*.py          # one file per module
```

## The seam that makes TDD possible

```python
Transport = Callable[[str, dict, dict], tuple[int, bytes, float]]
#                     url  body  headers -> (status, raw_body, wall_ms)
```

Every arm takes a `Transport`. In tests it is `FixtureTransport`, which replays a captured
body from `tests/fixtures/` and a scripted `wall_ms`. In production it is `HttpTransport`.
**No module other than `transport.py` imports `urllib`.** A test asserting that is cheap and
catches the drift that would otherwise make the suite need a key.

## The one interface every arm implements

```python
@dataclass(frozen=True)
class Prediction:
    arm: str
    ticket_id: str
    label: str                  # in LABELS, or "" on failure
    confidence: float | None    # Jev only
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int       # 0 where the provider reports none
    cost_micro_usd: int         # integer millionths of a dollar
    latency_ms: float
    pass_name: str              # "latency" | "bulk"
    error: str | None           # transport | http | parse | invalid_label
```

`runner.py`, `metrics.py` and `report.py` never learn which arm produced a `Prediction`.
Per-arm special-casing above this line is a design failure.

## Money is integers

`cost_micro_usd: int`, never a float. OpenRouter reports cost as a float like `1.659e-05`;
`pricing.py` converts once, at the boundary, by rounding to the nearest micro-dollar, and
everything downstream is integer arithmetic. Summing 1,600 float costs and printing a total
is exactly the drift this rule exists to prevent.

`pricing.py` also **reconciles**: it independently computes
`input_tokens × price_in + output_tokens × price_out` from the model roster and asserts it
matches the provider's reported cost within 2 micro-dollars. A mismatch means either the
roster is stale or the provider billed something we did not model, and the run says so
rather than quietly reporting a wrong total. Jev is the proof case: 395 × $0.042/M = $1.659e-05,
exactly the reported figure (PROBE-RESULTS, P1).

## Wire protocols

**Jev** — `POST https://openrouter.ai/api/alpha/decisions`

```json
{"model": "typesafe/jev-1.13", "state": {"ticket": "<text>"},
 "questions": {"department": {"type": "choice", "instructions": "Which team should handle this ticket.",
                              "criteria": {"<label>": "<definition>"}}}}
```

Response: `answers.department.{choice, probabilities, confidence}` and
`usage.{input_tokens, output_tokens, cost}`. Verified, PROBE-RESULTS P1.

**Chat** — `POST https://openrouter.ai/api/v1/chat/completions`

```json
{"model": "<id>", "temperature": 0, "max_tokens": 512,
 "response_format": {"type": "json_object"},
 "usage": {"include": true},
 "reasoning": {"enabled": false},     // low arms only; see below
 "messages": [{"role": "system", "content": "<rendered from labels.py>"},
              {"role": "user", "content": "<ticket text>"}]}
```

`max_tokens: 512` is not a guess — at 64, gemini truncated mid-JSON (P2). `usage.include`
is required to get `cost` and `completion_tokens_details.reasoning_tokens` back.

**Reasoning negotiation.** `{"enabled": false}` is rejected by gemini with HTTP 400
(`Reasoning is mandatory for this endpoint`) and accepted by gpt-5.3-codex (P3). A `-low` arm
therefore tries `enabled: false` once, and on a 400 whose body names reasoning, retries with
`{"effort": "minimal"}` and records which setting answered. That retry is the **only**
retry in the system, it happens once per arm rather than once per ticket, and the resolved
setting is printed in the report — because "low" meaning two different things across arms
would otherwise be invisible.

## Orchestration: two passes, for one honest reason

Latency measured under concurrency is not latency. Cost and accuracy measured sequentially
take an hour. So:

| Pass | n | Concurrency | What its numbers are for |
|---|---|---|---|
| `latency` | first 20 tickets | **1, strictly sequential** | P50/P95 latency. The only latency reported. |
| `bulk` | all 100 | 8 workers | Accuracy, tokens, cost. Its latency is recorded but **never reported**. |

Arms run in registry order within each pass. The latency pass discards each arm's first call
as connection warm-up and says so. Estimated: ~12 min for the latency pass, ~7 min for bulk,
about **$0.45 total**, derived from the per-call costs in PROBE-RESULTS P2/P3.

## Failure policy

**No retries on ticket calls.** A retry is a second billed call and a corrupted latency
sample. A failure is recorded as `Prediction(label="", error=...)`, counts as incorrect for
accuracy, and is reported in its own row so a model is never quietly credited with an answer
it did not give. Timeout 60 s. The reasoning negotiation above is the one deliberate
exception, and it is per-arm, not per-ticket.

## Configuration

Environment only, loaded from `.env` at repo root. `OPENROUTER_API_KEY` required — the runner
exits 2 with a one-line message if it is unset, before any call. `JEVDEMO_ARMS` (comma list)
and `JEVDEMO_N` narrow a run for development. No CLI flags beyond `--dry-run`, which builds
every request and prints it without sending.
