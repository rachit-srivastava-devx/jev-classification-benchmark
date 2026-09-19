"""A3 — the registry is the only place a model id appears."""
import json
import pathlib

from jevdemo import arms

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "models.json"


def test_twenty_arms_with_unique_names():
    assert len(arms.ARMS) == 20
    assert len({a.name for a in arms.ARMS}) == 20


def test_exactly_one_jev_arm_and_nineteen_chat_arms():
    assert [a.kind for a in arms.ARMS].count("jev") == 1
    assert [a.kind for a in arms.ARMS].count("chat") == 19


def test_every_chat_model_resolves_against_the_captured_roster():
    """A retired model id fails here, not in minute 30 of a paid run."""
    roster = {m["id"] for m in json.loads(FIXTURE.read_text())["data"]}
    unresolved = sorted(
        {a.model for a in arms.ARMS if a.kind == "chat"} - roster
    )
    assert unresolved == [], f"not in the captured roster: {unresolved}"
    assert len(roster) >= 11, f"roster fixture is too small to be a real check: {len(roster)}"


def test_jev_is_absent_from_the_chat_roster_by_design():
    """Verified 2026-09-20: a decisions model is not listed in /v1/models."""
    roster = {m["id"] for m in json.loads(FIXTURE.read_text())["data"]}
    jev = next(a for a in arms.ARMS if a.kind == "jev")
    assert jev.model not in roster


def test_registry_prices_match_the_captured_roster_exactly():
    """Guards against a price remembered rather than read."""
    roster = {m["id"]: m["pricing"] for m in json.loads(FIXTURE.read_text())["data"]}
    checked = 0
    for arm in arms.ARMS:
        if arm.kind != "chat":
            continue
        priced = roster[arm.model]
        assert arm.price_in_micro_per_mtok == round(float(priced["prompt"]) * 1e12)
        assert arm.price_out_micro_per_mtok == round(float(priced["completion"]) * 1e12)
        checked += 1
    assert checked == 19, f"only {checked} arms were price-checkable"


def test_prices_are_integers_not_floats():
    for arm in arms.ARMS:
        assert type(arm.price_in_micro_per_mtok) is int
        assert type(arm.price_out_micro_per_mtok) is int


def test_jev_price_matches_the_published_rate():
    """$0.042 per million input tokens, output free. PROBE-RESULTS P1."""
    jev = next(a for a in arms.ARMS if a.kind == "jev")
    assert jev.price_in_micro_per_mtok == 42_000
    assert jev.price_out_micro_per_mtok == 0


def test_low_arms_pair_with_a_default_arm_of_the_same_model():
    """A `-low` arm with nothing to compare against measures nothing."""
    for arm in arms.ARMS:
        if arm.name.endswith("-low"):
            assert any(
                o.model == arm.model and o.reasoning is None for o in arms.ARMS
            ), f"{arm.name} has no default-reasoning counterpart"


def test_only_low_arms_carry_a_reasoning_setting():
    for arm in arms.ARMS:
        assert (arm.reasoning is not None) == arm.name.endswith("-low")


def test_by_name_lookup_and_unknown_arm_raises():
    import pytest

    assert arms.by_name("jev").kind == "jev"
    with pytest.raises(KeyError):
        arms.by_name("no-such-arm")
