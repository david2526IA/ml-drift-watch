import pytest

from ml_drift_watch.psi import population_stability_index


def test_identical_distribution_is_stable() -> None:
    report = population_stability_index([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], bins=3)
    assert report["psi"] == 0.0
    assert report["status"] == "stable"


def test_shifted_distribution_requires_review() -> None:
    report = population_stability_index(list(range(100)), [99] * 100, bins=10)
    assert report["psi"] >= 0.25
    assert report["status"] == "review"


def test_invalid_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="distinct"):
        population_stability_index([1, 1], [1, 1])
