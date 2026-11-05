"""Score saved discharge against prepared CAMELS observations without model runs."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from hydrologeez.camels import Camels
from hydrologeez.hbv_light.output import OutputStore
from hydrologeez.metrics import evaluate


@dataclass(frozen=True)
class DischargeScore:
    """One complete period; counts distinguish source presence from usable observations."""

    basin_id: str
    parameter_id: str
    start: str
    stop: str
    observed_days: int
    present_days: int
    scores: dict[str, float]


def evaluate_discharge(
    output: Path,
    observations: Path,
    *,
    basin_ids: Sequence[str],
    parameter_ids: Sequence[str],
    periods: Sequence[tuple[str, str]],
) -> list[DischargeScore]:
    """Score each basin × parameter × [start, stop) period in caller order.

    Keep one simulation's complete period in memory at a time, plus one basin's
    observations. Dates outside observation coverage are missing, not truncated.
    Output reads reject unavailable pairs and dates outside recorded coverage.
    Counts impose no eligibility threshold; scores retain metrics.evaluate's NaNs
    and component diagnostics. Inputs must remain unchanged during evaluation.
    """
    if any(not ids or len(set(ids)) != len(ids) for ids in (basin_ids, parameter_ids)):
        raise ValueError("Select unique nonempty basin and parameter identities")
    if not periods or len(set(periods)) != len(periods):
        raise ValueError("Select unique nonempty scoring periods")
    store, camels = OutputStore(output), Camels(observations)
    if "discharge" not in store.variables:
        raise ValueError("Output store must contain discharge")
    if store.metadata["metadata"]["discharge"]["units"] != "mm/day":
        raise ValueError("Saved discharge units must be mm/day")
    if camels.metadata["observation_units"] != "cfs":
        raise ValueError("Prepared observation units must be cfs for conversion to mm/day")
    bounds = [(date.fromisoformat(start).toordinal(), date.fromisoformat(stop).toordinal()) for start, stop in periods]
    if any(
        not date.fromisoformat(store.start).toordinal() <= a < z <= date.fromisoformat(store.stop).toordinal()
        for a, z in bounds
    ):
        raise ValueError("Scoring period exceeds recorded coverage or is empty")
    results: list[DischargeScore] = []
    for basin in basin_ids:
        observed = camels.observations(basin)
        for parameter in parameter_ids:
            for start, stop in periods:
                selected = store.read(
                    variables=["discharge"], basin_ids=[basin], parameter_ids=[parameter], start=start, stop=stop
                )
                if selected.metadata["discharge"].units != "mm/day":
                    raise ValueError("Selected discharge units must be mm/day")
                _, destination, source = np.intersect1d(selected.dates, observed.dates, return_indices=True)
                aligned = np.full(len(selected.dates), np.nan)
                aligned[destination] = observed.discharge_mm_day[source]
                results.append(
                    DischargeScore(
                        selected.basin_ids[0],
                        selected.parameter_ids[0],
                        start,
                        stop,
                        int(np.count_nonzero(~np.isnan(aligned))),
                        int(np.count_nonzero(observed.present[source])),
                        evaluate(aligned, selected.values["discharge"][:, 0]),
                    )
                )
                del selected
    return results
