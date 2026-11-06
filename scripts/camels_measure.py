"""Bounded local CAMELS input measurement; not a runtime dependency."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    import json
    import platform
    from collections import Counter
    from time import perf_counter

    import numba
    import numpy as np

    from hydrologeez.camels import Camels
    from hydrologeez.hbv_light import Forcing, Memory, Parameters, simulate

    numba.set_num_threads(1)
    start = perf_counter()
    store = Camels(args.store)
    opened = perf_counter() - start
    records = []
    for count, start_date, stop_date in [(2, "2000-02-01", "2000-03-01"), (128, "1990-01-01", "2000-01-01")]:
        ids = list(store.ids[:count])[::-1]
        begin = perf_counter()
        selected = store.select(ids, start_date, stop_date)
        elapsed = perf_counter() - begin
        n = count * 4
        params = Parameters(
            np.tile(np.arange(count, dtype=np.int64), 4),
            np.tile([0.0, 2.0, 0.05, 0.1, 1.0, 100.0, 1.0, 0.5], (n, 1)),
            np.tile([1.0, 0.5, 0.1, 0.05], (n, 1)),
            np.full(n, 2.5),
        )
        initial = Memory(np.tile([0.0, 0.0, 50.0], (n, 1)), np.zeros((n, 2)), np.zeros(n * 2))
        reference = simulate(
            Forcing(selected.forcing.values.copy(), np.ones(count), np.arange(count + 1, dtype=np.int64)),
            params,
            initial,
        )
        times = []
        for _ in range(4):
            begin = perf_counter()
            result = simulate(selected.forcing, params, initial)
            times.append(perf_counter() - begin)
            np.testing.assert_array_equal(result.outlet, reference.outlet)
            for name in ("zone", "groundwater", "routing"):
                np.testing.assert_array_equal(getattr(result.final, name), getattr(reference.final, name))
        records.append(
            {
                "basins": count,
                "days": len(selected.dates),
                "parameter_copies": 4,
                "selection_seconds": elapsed,
                "forcing_materialized_bytes": selected.forcing.values.nbytes,
                "simulation_seconds": times,
            }
        )
    flags: Counter[str] = Counter()
    present = missing = absent = 0
    for b in store.basins:
        o = store.observations(b.gauge_id)
        flags.update(o.qc[o.present].tolist())
        present += int(o.present.sum())
        missing += int((o.present & np.isnan(o.discharge_mm_day)).sum())
        absent += b.forcing_stop - b.forcing_start - int(o.present.sum())
    assert present == 8415489 and missing == 69481 and absent == 162575
    assert sum(b.invalid_forcing_rows for b in store.basins) == 0
    # Independently audited real source values, including leap date and published missing flow.
    for day, p, t, pet, q in [
        ("1980-02-29", 0.01, -17.92, 0.0, 0.3085156210913192),
        ("1980-07-01", 3.01, 14.555, 3.3157159628066526, 0.5315128770380271),
    ]:
        from datetime import date, timedelta

        s = store.select(["01013500"], day, (date.fromisoformat(day) + timedelta(days=1)).isoformat())
        np.testing.assert_allclose(s.forcing.values[0, 0], [p, t, pet], rtol=1e-13)
        np.testing.assert_allclose(s.discharge_mm_day[0, 0], q, rtol=1e-13)
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "platform": platform.platform(),
        "threads": 1,
        "metadata_open_seconds": opened,
        "measurements": records,
        "observation_rows": present,
        "published_missing": missing,
        "absent_over_forcing": absent,
        "qc": dict(flags),
        "basins": len(store.ids),
        "forcing_gaps": sum(b.missing_forcing_dates for b in store.basins),
        "invalid_forcing": sum(b.invalid_forcing_rows for b in store.basins),
    }
    args.evidence.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
