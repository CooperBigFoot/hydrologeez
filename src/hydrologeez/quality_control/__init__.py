"""Retrospective streamflow quality control after Gudmundsson et al. (2018), GSIM Part 2.

Three capabilities, usable separately or together through the archive:

- ``screen``: daily screening by reliable provider flags or automated checks (``daily``).
- ``compute_indices``: yearly, seasonal and monthly indices with reliability (``indices``).
- ``assess_indices``: homogeneity of index series (``homogeneity``).

Screening is retrospective and uses each full basin record: adding later observations can
change earlier statistical flags, so an archive is not a training-cutoff-specific QC. The
archive stores evidence only; experiments decide which days or basins to exclude.

Usage::

    from pathlib import Path

    from hydrologeez.camels import Camels
    from hydrologeez.quality_control import StreamflowQualityArchive, camels_records, camels_source, write_archive

    store = Camels(Path("data/camels-us/prepared"))
    write_archive(Path("data/camels-us/streamflow-qc"), camels_source(store), camels_records(store))

    archive = StreamflowQualityArchive(Path("data/camels-us/streamflow-qc"))
    daily = archive.daily(["01013500"], "1990-10-01", "2000-10-01")
    accepted = daily.observed & ~daily.suspect          # or combine selected daily.reasons
    yearly = archive.indices(["01013500"], "yearly")     # values, counts, reliable
    homogeneity = archive.homogeneity(["01013500"], "yearly")  # classification [basin, index]

Other sources provide ``DischargeRecord`` objects and a ``DischargeSource`` that declares units,
the factor to m³/s used by the outlier transform, and provider-flag meanings.
"""

from hydrologeez.quality_control.archive import (
    DailyQuality,
    HomogeneityQuality,
    IndexQuality,
    StreamflowQualityArchive,
    write_archive,
)
from hydrologeez.quality_control.camels import camels_records, camels_source
from hydrologeez.quality_control.daily import REASONS, DailyScreening, DischargeRecord, DischargeSource, screen
from hydrologeez.quality_control.homogeneity import CLASSIFICATIONS, TESTS, Homogeneity, assess, assess_indices
from hydrologeez.quality_control.indices import INDICES, RESOLUTIONS, IndexSeries, compute_indices

__all__ = [
    "CLASSIFICATIONS",
    "INDICES",
    "REASONS",
    "RESOLUTIONS",
    "TESTS",
    "DailyQuality",
    "DailyScreening",
    "DischargeRecord",
    "DischargeSource",
    "Homogeneity",
    "HomogeneityQuality",
    "IndexQuality",
    "IndexSeries",
    "StreamflowQualityArchive",
    "assess",
    "assess_indices",
    "camels_records",
    "camels_source",
    "compute_indices",
    "screen",
    "write_archive",
]
