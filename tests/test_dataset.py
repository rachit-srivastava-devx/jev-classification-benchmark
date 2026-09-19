import copy

import pytest

from jevdemo import dataset
from jevdemo.dataset import DatasetError

ROWS = dataset.load()


def test_the_shipped_dataset_passes_every_assertion():
    dataset.validate(ROWS)
    assert len(ROWS) == 100


def test_labels_are_balanced_twenty_each():
    counts = {}
    for row in ROWS:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    assert counts == {k: 20 for k in counts}
    assert len(counts) == 5


def test_a_quarter_of_the_set_is_annotated_as_a_boundary_case():
    assert sum(1 for r in ROWS if r.get("note")) == 25


def test_no_ticket_names_its_own_label():
    for row in ROWS:
        assert row["label"] not in row["text"].lower()


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda r: r.pop(), "wrong size"),
        (lambda r: r[0].update(label="urgent"), "unknown label"),
        (lambda r: r[0].update(text="too short"), "word count"),
        (lambda r: r[0].update(label="technical"), "unbalanced"),
        (lambda r: [row.pop("note", None) for row in r], "no boundary cases"),
        (lambda r: r[1].update(id=r[0]["id"]), "duplicate id"),
    ],
)
def test_the_validator_actually_fails_on_broken_input(mutate, reason):
    rows = copy.deepcopy(ROWS)
    mutate(rows)
    with pytest.raises(DatasetError):
        dataset.validate(rows)


def test_a_ticket_containing_its_own_label_is_rejected():
    rows = copy.deepcopy(ROWS)
    rows[0]["text"] = "This is a billing problem " + rows[0]["text"]
    rows[0]["label"] = "billing"
    with pytest.raises(DatasetError, match="own label"):
        dataset.validate(rows)
