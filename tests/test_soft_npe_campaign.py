"""Selection and equal-budget experiment contracts."""

from pathlib import Path
from typing import Any

import pytest

from scripts.soft_npe_benchmark import Array
from scripts.soft_npe_campaign import select_best, validate_complete


def test_selects_own_objective_not_reported_exact_score() -> None:
    trials = [{"loss": 0.1, "exact_npe": 0.2}, {"loss": 0.2, "exact_npe": 0.9}]
    assert select_best(trials) == trials[0]


def test_partial_budget_cannot_be_summarized() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        validate_complete(
            [
                {"status": "success", "evaluations": 3499, "seed": seed, "loss": 0.1, "exact_npe": 0.9}
                for seed in range(42, 52)
            ]
        )


def test_full_exact_trial_uses_existing_ga_and_fixed_mask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import numpy as np

    from scripts import soft_npe_campaign as campaign

    class BasinEvaluator:
        def __init__(self, *args: Any) -> None:
            self.observed = np.array([1.0, 2.0, np.nan, 4.0, 2.0])
            self.masks = {"training": np.array([True, True, False, True, True])}

        def discharge(self, parameters: Array) -> Array:
            return np.tile(np.array([1.0, 2.0, np.nan, 4.0, 2.0])[:, None], (1, len(parameters)))

    selection = SimpleNamespace(forcing=None, dates=None, discharge_mm_day=np.ones((5, 1)))
    monkeypatch.setattr(campaign, "Camels", lambda _: SimpleNamespace(select=lambda *args: selection))
    monkeypatch.setattr(campaign, "CandidateEvaluator", BasinEvaluator)
    result = campaign.trial((str(tmp_path), "pilot", 0, 42, "manifest"))
    assert result["status"] == "success"
    assert result["evaluations"] == 3500
    assert result["scored_days"] == 4
    assert result["exact_npe"] == pytest.approx(1.0)
    assert result["invalid_candidates"] == 0


def test_failed_trial_records_completed_budget_and_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import soft_npe_campaign as campaign

    def missing_store(path: Path) -> None:
        raise ValueError("missing observations")

    monkeypatch.setattr(campaign, "Camels", missing_store)
    result = campaign.trial((str(tmp_path), "pilot", 3, 42, "manifest"))
    assert result["status"] == "failed"
    assert result["evaluations"] == 0
    assert "missing observations" in result["error"]
