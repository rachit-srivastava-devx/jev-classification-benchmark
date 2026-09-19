"""The 100 tickets, plus the five assertions that make them trustworthy.

`validate` raises rather than warns. A dataset that quietly drifts out of
balance produces a benchmark whose accuracy numbers mean nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from jevdemo.labels import LABELS

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "tickets.json"

SIZE = 100
PER_LABEL = 20
BOUNDARY = 25
MIN_WORDS = 15
MAX_WORDS = 60


class DatasetError(ValueError):
    """The dataset on disk violates one of the five assertions."""


def load(path: Path | str = DEFAULT_PATH) -> list[dict]:
    rows = json.loads(Path(path).read_text())
    validate(rows)
    return rows


def validate(rows: list[dict]) -> None:
    if len(rows) != SIZE:
        raise DatasetError(f"expected {SIZE} tickets, found {len(rows)}")

    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise DatasetError("duplicate ticket ids")

    counts = {name: 0 for name in LABELS}
    for row in rows:
        label = row.get("label")
        if label not in LABELS:
            raise DatasetError(f"{row.get('id')}: label {label!r} is not one of the five")
        counts[label] += 1

        words = row["text"].split()
        if not MIN_WORDS <= len(words) <= MAX_WORDS:
            raise DatasetError(f"{row['id']}: {len(words)} words, outside {MIN_WORDS}-{MAX_WORDS}")

        # A ticket that names its own label is a giveaway, not a classification task.
        if re.search(rf"\b{re.escape(label)}\b", row["text"], re.IGNORECASE):
            raise DatasetError(f"{row['id']}: text contains its own label {label!r}")

    unbalanced = {k: v for k, v in counts.items() if v != PER_LABEL}
    if unbalanced:
        raise DatasetError(f"expected {PER_LABEL} per label, got {unbalanced}")

    boundary = sum(1 for r in rows if r.get("note"))
    if boundary != BOUNDARY:
        raise DatasetError(f"expected {BOUNDARY} annotated boundary cases, found {boundary}")
