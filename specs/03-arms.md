# 03 · Arms

## Bottom line

Every arm is a row in one registry and two pure functions: build a request, parse a response.
Neither function touches the network, which is what lets all 20 arms be tested against
captured fixtures with no key and no spend.


## Rate limiting is not a failure mode

Added after the first full run stopped on it. `openai/gpt-5.3-codex` caps new accounts at
**20 requests per minute**, and the first run sent 8 concurrent bulk requests per arm, so the
fifth arm returned 429s that the four-way failure taxonomy would have scored as
`http_error` — putting the rate limiter in the accuracy column.

The policy now:

- **429 is the only retried status.** Up to 5 attempts with 2/4/8/16 s backoff
  (`jevdemo/transport.Throttled`). It says our request rate was wrong, not that the model
  was.
- **Every other status is returned on the first attempt.** Retrying a 500 buys a second
  charge for the same broken answer.
- **Latency is the successful attempt's own wall time**, never the elapsed time across the
  backoff — timing the retries would measure this script's patience.
- **Bulk concurrency is 4, down from 8**, and the retry count is printed and written to
  `results.json` so the reader can see how much pacing the run needed.

## The registry — `jevdemo/arms.py`

```python
@dataclass(frozen=True)
class Arm:
    name: str                       # "gemini-3.8-flash-low"
    model: str                      # "google/gemini-3.8-flash"
    kind: Literal["jev", "chat"]
    reasoning: dict | None          # None | {"enabled": False} | {"effort": "minimal"}
    price_in_micro_per_mtok: int    # integers; from the roster, not remembered
    price_out_micro_per_mtok: int
```

This is the only file in which a model id appears. A test asserts every `model` in the
registry resolves against a captured copy of `GET /api/v1/models`, so a retired model id
fails the suite instead of failing halfway through a paid run.

The 16 rows are listed in spec 00. Prices, in micro-dollars per million tokens, verified
2026-09-20 (PROBE-RESULTS, P4): jev 42,000/0 · sonnet-5 2,000,000/10,000,000 ·
opus-5 5,000,000/25,000,000 · gpt-5.3-codex 1,750,000/14,000,000 · gemini-3.8-flash
750,000/3,750,000 · glm-5.3-flash 90,000/300,000 · qwen3.8-flash 150,000/470,000 ·
deepseek-v4.1-flash 150,000/600,000 · kimi-k3 1,700,000/8,500,000 · grok-4.6 2,000,000/6,000,000 ·
mistral-small-2603 150,000/600,000 · llama-4-scout 100,000/300,000.

## Jev arm — `jevdemo/jev_arm.py`

**Request.** One `Choice` question named `department`. `instructions` is the single sentence
"Which team should handle this ticket." `criteria` is `labels.render_for_criteria()`,
unchanged.

**Parse.** Reads `answers.department.choice`, `.confidence`, and the full `.probabilities`
map. The probabilities are stored in the results JSON but not reported — they are the
material for a follow-up on confidence calibration, which is out of scope here.

**A choice outside the label set raises.** It is not coerced to `other`. That would hide a
protocol misunderstanding as a model error, and the two need different fixes.

**Confidence is recorded, never acted on.** No thresholding, no escalation (spec 00, out of
scope). The report shows its distribution because "Jev was 100% confident and wrong" is worth
seeing; it does not gate anything.

## Chat arms — `jevdemo/chat_arm.py`

**System prompt, rendered from `labels.py`:**

```
You route support tickets. Reply with JSON only: {"label": "<one of the labels>"}.
The labels and their definitions:
billing: Charges, refunds, invoices, payouts, subscription price changes.
technical: ...
```

**Fixed across every chat arm:** `temperature: 0`, `max_tokens: 512`,
`response_format: {"type": "json_object"}`, `usage: {"include": true}`. Only `model` and
`reasoning` vary. A test asserts that: it builds requests for all 15 chat arms and checks
every field except those two is byte-identical.

**No few-shot examples, no reasoning scaffold, no per-model prompt tuning.** Any of the three
would raise accuracy and inflate tokens — a trade a real team would make, but one that turns
a comparison into an argument about how hard you tried on each arm. The report names this as
a limitation rather than quietly optimising a favourite.

**Parse.** `choices[0].message.content` → JSON → `label`. Then
`usage.{prompt_tokens, completion_tokens, cost}` and
`usage.completion_tokens_details.reasoning_tokens`, defaulting to 0 where absent.

## Failure taxonomy — four distinct errors, never merged

| `error` | Cause | Counts as |
|---|---|---|
| `transport` | timeout, DNS, connection reset | incorrect; reported separately |
| `http` | non-2xx | incorrect; status recorded |
| `parse` | body is not JSON, or has no `label` key | incorrect; **chat arms only** |
| `invalid_label` | a label outside the five | incorrect; **chat arms only** |

The last two have **no counterpart in the Jev arm**, and that asymmetry is a headline result,
not an implementation detail: a typed endpoint cannot return a malformed answer. Merging
these into one "error" bucket would erase the finding, so the report gives them their own row.

## Known asymmetries, stated before the numbers

1. **Jev's input is longer for the same ticket.** Its `criteria` map carries the same
   definitions the chat prompt carries, and the probe measured 395 input tokens against
   44–99 for the chat arms (PROBE-RESULTS P1, P2). Jev still wins on cost because its input
   price is 2–50× lower and its output is free. The report shows both token counts so nobody
   reads the cost win as a token win.
2. **The chat arms can fail to produce a valid label. Jev cannot.** Counted, reported.
3. **Jev returns calibrated confidence; the chat arms return nothing comparable.** Asking
   them for self-reported confidence would add output tokens and would not be calibrated.
   Not asked.
4. **`-low` does not mean the same setting on every arm** — `enabled: false` on some,
   `effort: minimal` on others, forced by the 400 in PROBE-RESULTS P3. The resolved setting
   is recorded per arm and printed in the report's footnotes.
