#!/usr/bin/env python3
"""Capture one real response body per arm into tests/fixtures/captured/.

Costs one call per arm (20 calls, roughly a cent). The offline suite asserts
against `tests/fixtures/*.json`; this script is how those files are refreshed
when a provider changes its envelope, and how a new arm gets evidence before
any test is written against it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jevdemo import chat_arm, dataset, jev_arm, runner
from jevdemo.arms import ARMS
from jevdemo.transport import HttpTransport, TransportError

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "captured"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", help="limit to these arms (repeatable)")
    args = ap.parse_args()

    key = runner.load_key()
    if not key:
        print("OPENROUTER_API_KEY is not set and .env does not define it", file=sys.stderr)
        return 2

    ticket = dataset.load()[0]["text"]
    headers = runner.headers_for(key)
    transport = HttpTransport()
    OUT.mkdir(parents=True, exist_ok=True)

    failures = 0
    for arm in ARMS:
        if args.arm and arm.name not in set(args.arm):
            continue
        mod = jev_arm if arm.kind == "jev" else chat_arm
        url, body = mod.build_request(arm, ticket)
        try:
            status, raw, wall_ms = transport(url, body, headers)
        except TransportError as exc:
            print(f"{arm.name:24} transport failed: {exc}")
            failures += 1
            continue
        (OUT / f"{arm.name}.json").write_bytes(raw)
        parsed = mod.parse(raw, status)
        failures += parsed.failure is not None
        print(f"{arm.name:24} {status} {wall_ms:7.0f} ms  "
              f"label={parsed.label} failure={parsed.failure} "
              f"in={parsed.input_tokens} out={parsed.output_tokens} "
              f"think={parsed.reasoning_tokens}")

    print(f"\nwrote {len(list(OUT.glob('*.json')))} bodies to {OUT}")
    print(f"{failures} arm(s) did not produce a usable answer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
