"""Two passes over the dataset, one arm at a time.

Pass 1 (latency) is strictly sequential over the first LATENCY_N tickets: a
timing taken under concurrency measures the connection pool, not the model.
Pass 2 (bulk) runs the full set concurrently, and its timings are discarded.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jevdemo import chat_arm, dataset, jev_arm, negotiate
from jevdemo.arms import ARMS, Arm
from jevdemo.metrics import BULK_PASS, LATENCY_PASS, Record, aggregate
from jevdemo.transport import HttpTransport, Transport, TransportError
from jevdemo.result import failed

LATENCY_N = 20
WORKERS = 8
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_key() -> str | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENROUTER_API_KEY":
                return value.strip().strip('"').strip("'")
    return None


def headers_for(key: str) -> dict:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def module_for(arm: Arm):
    return jev_arm if arm.kind == "jev" else chat_arm


def call(arm: Arm, ticket: dict, pass_name: str, transport: Transport, headers: dict,
         reasoning: dict | None) -> Record:
    mod = module_for(arm)
    if arm.kind == "jev":
        url, body = mod.build_request(arm, ticket["text"])
    else:
        url, body = mod.build_request(arm, ticket["text"], reasoning=reasoning)
    try:
        status, raw, elapsed = transport(url, body, headers)
    except TransportError as exc:
        return Record(arm.name, ticket["id"], pass_name, ticket["label"],
                      failed("http_error", str(exc)), 0.0)
    return Record(arm.name, ticket["id"], pass_name, ticket["label"],
                  mod.parse(raw, status), elapsed)


def run_arm(arm: Arm, tickets: list[dict], transport: Transport, headers: dict,
            log=print) -> tuple[list[Record], negotiate.Outcome]:
    outcome = negotiate.resolve(arm, tickets[0]["text"], transport, headers)
    if arm.reasoning is not None and outcome.reasoning is None:
        log(f"  {arm.name}: reasoning could not be negotiated — {outcome.error}")
    reasoning = outcome.reasoning

    records = [call(arm, t, LATENCY_PASS, transport, headers, reasoning)
               for t in tickets[:LATENCY_N]]

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        records += list(pool.map(
            lambda t: call(arm, t, BULK_PASS, transport, headers, reasoning), tickets))
    return records, outcome


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the Jev vs chat-model benchmark.")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, make no calls")
    ap.add_argument("--arm", action="append", help="limit to these arms (repeatable)")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args(argv)

    tickets = dataset.load()
    arms = [a for a in ARMS if not args.arm or a.name in set(args.arm)]
    calls = len(arms) * (LATENCY_N + len(tickets))

    print(f"{len(arms)} arms x ({LATENCY_N} latency + {len(tickets)} bulk) = {calls} calls")
    if args.dry_run:
        for a in arms:
            print(f"  {a.name:24} {a.model:36} reasoning={a.reasoning}")
        return 0

    key = load_key()
    if not key:
        print("OPENROUTER_API_KEY is not set and .env does not define it", file=sys.stderr)
        return 2

    transport = HttpTransport()
    headers = headers_for(key)
    records: list[Record] = []
    negotiated: dict[str, negotiate.Outcome] = {}
    for i, arm in enumerate(arms, 1):
        print(f"[{i}/{len(arms)}] {arm.name}", flush=True)
        arm_records, outcome = run_arm(arm, tickets, transport, headers)
        records += arm_records
        negotiated[arm.name] = outcome

    from jevdemo import report

    results = aggregate(records, {a.name: a for a in arms})
    report.write_json(Path(args.out), results, negotiated, records)
    report.print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
