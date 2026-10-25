"""Dataset splitting contract tests."""

from pathlib import Path

from lung_cancer.dataset import Sample, split_samples


def test_stratified_split_preserves_labels_and_is_reproducible() -> None:
    samples = [
        Sample(Path(f"class-{label}/image-{index}.jpg"), label)
        for label in range(3)
        for index in range(30)
    ]
    first = split_samples(samples, seed=17)
    second = split_samples(samples, seed=17)

    assert [[item.path for item in part] for part in first] == [
        [item.path for item in part] for part in second
    ]
    assert [len(part) for part in first] == [62, 14, 14]
    for part in first:
        assert {item.label for item in part} == {0, 1, 2}


def test_grouped_split_keeps_patient_groups_disjoint() -> None:
    samples = [
        Sample(Path(f"class-{label}/image-{index}.jpg"), label, f"patient-{label}-{index}")
        for label in range(3)
        for index in range(12)
    ]
    train, validation, test = split_samples(
        samples, test_fraction=0.25, validation_fraction=0.25, seed=7
    )
    groups = [{item.patient_id for item in part} for part in (train, validation, test)]

    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    for part in (train, validation, test):
        assert {item.label for item in part} == {0, 1, 2}
