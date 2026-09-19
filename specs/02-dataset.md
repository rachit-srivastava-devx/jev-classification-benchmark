# 02 · Dataset

## Bottom line

100 synthetic tickets, 20 per label, hand-written and hand-labelled for this demo. 25 of them
sit deliberately on a class boundary, because that is the only place the arms will disagree
and therefore the only place the comparison carries information.

## The label set

`jevdemo/labels.py` is the single source of truth. The Jev arm renders it to `criteria`; the
chat arms render it into the system prompt. Neither restates a definition in its own words —
a test asserts both renderings contain the same five keys and the same definition strings.

| Label | Definition given to every arm |
|---|---|
| `billing` | Charges, refunds, invoices, payouts, subscription price changes. |
| `technical` | Bugs, errors, outages, failed integrations, anything not working. |
| `account` | Login, password, access, permissions, seats, profile and org settings. |
| `sales` | Pricing questions, plan comparisons, features on plans they do not have. |
| `other` | None of the above applies. |

`other` exists because a model cannot pick a value it was never offered. Dropping it would
add the same 20 errors to every arm and separate nothing.

## Composition

| Label | Count | Boundary cases |
|---|---|---|
| billing | 20 | 6 |
| technical | 20 | 6 |
| account | 20 | 5 |
| sales | 20 | 4 |
| other | 20 | 4 |

**Boundary cases are the point.** Examples of the ambiguity being deliberately planted:

- A failed payment caused by an expired card → `billing`, not `technical`.
- An account locked after five bad passwords → `account`, not `technical`.
- An existing customer asking what the next tier costs → `sales`, not `billing`.
- A refund request that is really a cancellation threat → `billing`; the churn signal is not
  a label in this task.

Non-boundary tickets are there to make accuracy readable, not to be hard. A roster where
every arm scores 100% on 75 tickets and splits on 25 is a roster whose differences are legible.

## Constraints on every ticket

1. 15–60 words. The realistic range, and it keeps input-token counts comparable across arms.
2. No ticket contains its own label as a word. "I have a billing question" tests string
   matching, not classification.
3. Exactly one defensible label. If two reviewers could reasonably disagree, the ticket is
   rewritten or moved to `other` — an ambiguous gold label penalises every arm identically
   and teaches nothing.
4. No names, emails, order numbers, or anything resembling real customer data.

## File format

`data/tickets.json`:

```json
[{"id": "t001", "text": "My card was charged twice this month and the invoice only shows one line.",
  "label": "billing", "boundary": false, "note": ""}]
```

`note` carries the reason a boundary case is labelled the way it is. It is never sent to any
model; it exists so a disagreement in the results can be adjudicated later without
re-deriving the intent.

## Validation, enforced by test not by discipline

`tests/test_dataset.py` asserts, with no network and no key:

1. Exactly 100 records; ids unique and `t001`…`t100`.
2. Exactly 20 per label; every label in `LABELS`.
3. Boundary counts match the table above; every `boundary: true` record has a non-empty `note`.
4. Word count in [15, 60] for every ticket.
5. No ticket text contains its own label string, case-insensitive.

A dataset that fails any of these fails the build. This is the gate that stops the file
drifting as tickets get edited.

## Honesty note, carried into the report

These tickets are written by the demo's author, not sampled from a real inbox, and the gold
labels are one person's judgement. Every accuracy figure in the report appears next to this
sentence. A production evaluation would use real tickets labelled by the team that routes
them; the numbers here measure agreement with a synthetic ground truth, which is a weaker
claim than it looks.
