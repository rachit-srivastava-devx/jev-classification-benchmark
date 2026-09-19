"""A1 — the label set is one source of truth for both arm styles."""
import re

from jevdemo import labels


def test_five_labels_in_declared_order():
    assert list(labels.LABELS) == ["billing", "technical", "account", "sales", "other"]


def test_definitions_are_non_empty_and_distinct():
    defs = list(labels.LABELS.values())
    assert all(d.strip() for d in defs)
    assert len(set(defs)) == 5


def test_both_renderings_carry_the_same_definitions():
    """The prompt and the criteria map must not drift apart."""
    criteria = labels.render_for_criteria()
    prompt = labels.render_for_prompt()

    assert criteria == dict(labels.LABELS)
    for name, definition in labels.LABELS.items():
        assert f"{name}: {definition}" in prompt


def test_criteria_is_a_copy_not_the_live_mapping():
    """A caller mutating the criteria must not corrupt the source of truth."""
    criteria = labels.render_for_criteria()
    criteria["billing"] = "tampered"
    assert labels.LABELS["billing"] != "tampered"


def test_prompt_names_json_and_every_label():
    prompt = labels.render_for_prompt()
    assert "JSON" in prompt
    assert '"label"' in prompt
    for name in labels.LABELS:
        assert re.search(rf"\b{name}\b", prompt)


def test_is_valid_rejects_case_variants_and_unknowns():
    """Spec 03: no silent normalisation. 'Billing' is invalid, not billing."""
    assert labels.is_valid("billing")
    assert not labels.is_valid("Billing")
    assert not labels.is_valid("refunds")
    assert not labels.is_valid("")
    assert not labels.is_valid(None)
