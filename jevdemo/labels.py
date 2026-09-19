"""The five routing labels and their definitions.

This module is the single source of truth. The Jev arm renders it to a `criteria`
map; the chat arms render it into a system prompt. Neither restates a definition
in its own words, so a change here reaches every arm at once.
"""

from __future__ import annotations

#: Ordered so that every rendering — prompt, criteria, report column — agrees.
LABELS: dict[str, str] = {
    "billing": "Charges, refunds, invoices, payouts, subscription price changes.",
    "technical": "Bugs, errors, outages, failed integrations, anything not working.",
    "account": "Login, password, access, permissions, seats, profile and org settings.",
    "sales": "Pricing questions, plan comparisons, features on plans they do not have.",
    "other": "None of the above applies.",
}


def render_for_criteria() -> dict[str, str]:
    """The label map as Jev's `criteria` argument.

    Returns:
        A fresh copy, so a caller mutating it cannot corrupt `LABELS`.
    """
    return dict(LABELS)


def render_for_prompt() -> str:
    """The label map as a chat system prompt.

    Returns:
        Instructions plus one `name: definition` line per label, in `LABELS` order.
    """
    lines = "\n".join(f"{name}: {definition}" for name, definition in LABELS.items())
    return (
        'You route support tickets. Reply with JSON only: {"label": "<one of the labels>"}.\n'
        "The labels and their definitions:\n"
        f"{lines}"
    )


def is_valid(label: object) -> bool:
    """Whether `label` is exactly one of the five labels.

    Case-sensitive and type-strict by design. Accepting "Billing" would hide a real
    behavioural difference between models behind a convenience; spec 03's
    `invalid_label` failure type exists to surface it instead.

    Args:
        label: Any value, including the `None` a missing JSON key yields.

    Returns:
        True only for an exact match against a `LABELS` key.
    """
    return isinstance(label, str) and label in LABELS
