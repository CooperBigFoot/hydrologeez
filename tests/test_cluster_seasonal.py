import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hydrologeez.camels import Basin, Camels
from hydrologeez.cluster.seasonal import camels_cycles, seasonal_cycles


def test_calendar_bins_equal_year_weight_and_leap_day() -> None:
    dates = np.arange(date(1999, 1, 1).toordinal(), date(2001, 1, 1).toordinal(), dtype=np.int64)
    values = np.zeros((2, len(dates), 2))
    values[0, dates == date(2000, 2, 29).toordinal(), 0] = 16
    values[0, dates == date(1999, 12, 31).toordinal(), 0] = 16
    values[1, :, 0] = 3
    values[:, :, 1] = -4
    result = seasonal_cycles(dates, values, ("002", "001"), ("rain", "temperature"), ("mm/day", "degrees C"))
    expected = np.zeros((2, 52, 2))
    expected[0, 8, 0] = 1  # leap-year eight-day bin mean 2, averaged with 1999 mean 0
    expected[0, 51, 0] = 1  # eight-day December bin mean 2, averaged with 2000 mean 0
    expected[1, :, 0] = 3
    expected[:, :, 1] = -4
    assert_allclose(result.values, expected)
    assert result.ids == ("002", "001")
    assert (result.start, result.stop) == ("1999-01-01", "2001-01-01")


@pytest.mark.parametrize("problem", ["gap", "duplicate", "partial", "nan", "shape", "ids", "variables"])
def test_invalid_inputs_have_explicit_diagnostics(problem: str) -> None:
    dates = np.arange(date(2001, 1, 1).toordinal(), date(2002, 1, 1).toordinal(), dtype=np.int64)
    values = np.ones((1, len(dates), 1))
    ids = ("001",)
    variables = ("rain",)
    if problem == "gap":
        dates = np.delete(dates, 30)
        values = np.delete(values, 30, axis=1)
    elif problem == "duplicate":
        dates[30] = dates[29]
    elif problem == "partial":
        dates, values = dates[1:], values[:, 1:]
    elif problem == "nan":
        values[0, 10, 0] = np.nan
    elif problem == "shape":
        values = values[:, :-1]
    elif problem == "ids":
        ids = ("",)
    else:
        variables = ()
    with pytest.raises(
        ValueError,
        match={
            "gap": "daily",
            "duplicate": "daily",
            "partial": "complete calendar years",
            "nan": "001.*rain.*2001-01-11",
            "shape": "shape",
            "ids": "identities",
            "variables": "variables",
        }[problem],
    ):
        seasonal_cycles(dates, values, ids, variables, ("mm/day",))


def test_camels_reads_only_selected_forcing_and_retains_order(tmp_path: Path) -> None:
    first, last = date(2001, 1, 1).toordinal(), date(2002, 1, 1).toordinal()
    records = []
    for index, gauge in enumerate(("001", "002")):
        folder = tmp_path / gauge
        folder.mkdir()
        values = np.full((365, 3), np.nan)
        values[:, 0] = index + 1
        values[:, 1] = -index - 2
        np.save(folder / "forcing.npy", values)
        records.append(
            asdict(Basin(gauge, 40, 1e6, None, None, first, last, first, last, "forcing", "flow", 0, 365, 365, 0))
        )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "format": 1,
                "basins": records,
                "source": "test source",
                "forcing_variables": ["precipitation", "mean_temperature", "oudin_pet"],
                "forcing_units": ["mm/day", "degrees C", "mm/day"],
            }
        )
    )
    store = Camels(tmp_path)
    result = camels_cycles(store, ["002", "001"], "2001-01-01", "2002-01-01")
    assert result.ids == ("002", "001")
    assert_allclose(result.values[:, 0], [[2, -3], [1, -2]])
    assert result.units == ("mm/day", "degrees C")
    assert result.provenance["source"] == "test source"
    with pytest.raises(ValueError, match="002.*oudin_pet.*2001-01-01"):
        camels_cycles(store, ["002"], "2001-01-01", "2002-01-01", ("oudin_pet",))
    with pytest.raises(ValueError, match="coverage.*001"):
        camels_cycles(store, ["001"], "2000-01-01", "2002-01-01")
    with pytest.raises(ValueError, match="unknown"):
        camels_cycles(store, ["001"], "2001-01-01", "2002-01-01", ("unknown",))
    values = np.load(tmp_path / "001" / "forcing.npy")
    values[0, 0] = -1
    np.save(tmp_path / "001" / "forcing.npy", values)
    with pytest.raises(ValueError, match="001.*precipitation.*2001-01-01"):
        camels_cycles(store, ["001"], "2001-01-01", "2002-01-01")
    temperature = camels_cycles(store, ["001"], "2001-01-01", "2002-01-01", ("mean_temperature",))
    assert_allclose(temperature.values, -2)
