"""Simulate 1% homogeneity-test critical values beyond the published n <= 100 tables.

The ECA&D ATBD (2013, Tables 3-6) tabulates 1% critical values for n = 20...100,
obtained by simulation (SNHT, Pettitt) or from Buishand (1982). Index series with
more than 100 reliable values (most monthly series) need values for larger n.
This script simulates iid standard normal series with the statistics exactly as
implemented in ``hydrologeez.quality_control.homogeneity`` and reports the 99th
percentile (1st percentile for the von Neumann ratio). Run with ``--check`` to
compare the simulation against the published n <= 100 values first.
"""

import argparse

import numpy as np

from hydrologeez.quality_control.homogeneity import PUBLISHED_CRITICAL_VALUES, homogeneity_statistics


def simulate(n: int, replicates: int, seed: int) -> tuple[float, float, float, float]:
    rng = np.random.Generator(np.random.PCG64(seed + n))
    statistics = np.empty((replicates, 4))
    chunk = max(1, 2_000_000 // n)
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        statistics[start:stop] = homogeneity_statistics(rng.standard_normal((stop - start, n)))
    snht, buishand = np.quantile(statistics[:, :2], 0.99, axis=0)
    # The published Pettitt "1%" values reproduce the 98th percentile of max|Xk| (a one-sided 1% level);
    # extend the table with the same quantile so it stays continuous at n = 100.
    pettitt = np.quantile(statistics[:, 2], 0.98)
    von_neumann = np.quantile(statistics[:, 3], 0.01)
    return float(snht), float(buishand), float(pettitt), float(von_neumann)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--replicates", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20180417)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="*", default=[150, 200, 300, 400, 500, 700, 1000, 1500, 2000, 3000])
    args = parser.parse_args()
    sizes = [int(n) for n in PUBLISHED_CRITICAL_VALUES[:, 0]] if args.check else args.sizes
    for n in sizes:
        snht, buishand, pettitt, von_neumann = simulate(n, args.replicates, args.seed)
        print(f"({n}, {snht:.2f}, {buishand:.3f}, {pettitt:.0f}, {von_neumann:.3f}),", flush=True)


if __name__ == "__main__":
    main()
