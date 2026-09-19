# 05 · Report

## Bottom line

Three outputs from one run: a stdout table for the person who ran it, a `results.json` that
makes every number recomputable without spending again, and one self-contained HTML report
built to the DevX doctrine for everyone else.

## 1 · stdout

```
jev-demo · 100 tickets · 16 arms · 2026-09-20T14:02:11Z · $0.4412 spent

arm                        acc      bnd   P50      in   out  rsn   $/1k    fail
jev                      92.0%    76.0%  0.31s    395     0    0  0.0166      0
glm-5.3-flash-low        91.0%    72.0%  0.68s     57    12    0  0.0087      0
...
opus-5                   96.0%    88.0%  1.94s     73    11    0  0.6400      0
gemini-3.8-flash         94.0%    84.0%  2.91s     46     5  171  0.6930      0

accuracy CI +/-9pp at n=100; arms within 9pp are tied, not ranked
latency: sequential pass, n=20, first call discarded, includes ~0.2s network floor
```

Numbers above are **illustrative placeholders**. A test asserts no digit from this block
appears as a literal in any source file.

Ordered by cost ascending, not accuracy — the question is what the cheap arms give up, and
sorting by cost puts that on the diagonal.

## 2 · results.json

```json
{"run": {"at": "...", "n": 100, "arms": 16, "total_micro_usd": 441200,
         "reasoning_settings": {"gemini-3.8-flash-low": {"effort": "minimal"}}},
 "reconciliation": [{"arm": "jev", "reported_micro": 1659, "computed_micro": 1659, "delta": 0}],
 "summary": {"<arm>": {...}},
 "predictions": [{"arm": "jev", "ticket_id": "t001", "gold": "billing", "label": "billing",
                  "confidence": 1.0, "probabilities": {...}, "input_tokens": 395,
                  "output_tokens": 52, "reasoning_tokens": 0, "cost_micro_usd": 17,
                  "latency_ms": 312.4, "pass_name": "bulk", "error": null}]}
```

All 1,920 prediction records are written, failures included. The **disagreement set** — the
tickets where arms split — is the most valuable thing in the file and the reason per-ticket
records are kept rather than only aggregates.

## 3 · HTML report — `report.html`

Single self-contained file, no network fetches, no build step. Governed by the **DevX
doctrine**: load `devx-doctrine` and inline `assets/doctrine.css` before writing it.

**Structure, in doctrine order — conclusion first:**

1. **Bottom line** — three sentences and one stat row: cheapest arm that stays within the
   accuracy band, the cost multiple against Jev, the reasoning tax. Before any methodology.
2. **The roster** — one hairline-ruled table, 16 rows, no zebra striping. Tied arms share a
   band; the band is a hairline rule, not a colour fill.
3. **Three findings**, each a bolded claim sentence then its detail:
   *"Jev costs 1/40th of the median chat arm and gives up N points of accuracy."* ·
   *"Reasoning costs 13× on gemini for an answer it gets right without it."* ·
   *"The chat arms produced N malformed answers; the typed endpoint cannot."*
4. **Where the arms disagree** — the boundary tickets, with each arm's label. The only place
   the report shows raw data, because the aggregate cannot carry it.
5. **Method and limitations** — reduced weight, at the end: two passes, warm-up discard,
   network floor, cost reconciliation, synthetic dataset, n=100 CI, and the hypotheses from
   spec 00 with which ones were wrong.

**Visual rules, from the doctrine's never-do list:** no emoji · no gradients · no rounded
corners or drop-shadows · one accent (`#1E6FFF`), used 2–4 times on the whole page · hairline
rules instead of boxes · Inter Tight body, Source Serif 4 italic for emphasis, JetBrains Mono
for every numeral in a table · max width 960px · no "click here".

**One chart, and only if it earns its place.** A 16-row cost-vs-accuracy scatter is genuinely
two-dimensional and a table cannot show the frontier; that one is justified. A bar chart of
16 costs is not — it carries less than the sorted column it would duplicate. If the scatter
does not read at a glance, the report ships without it.

## Not built

No dashboard, no live-refresh page, no per-arm drill-down. The run is reproducible from one
command and the JSON; that is the interactivity budget.
