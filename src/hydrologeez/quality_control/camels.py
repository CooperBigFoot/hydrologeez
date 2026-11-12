"""CAMELS prepared store as a streamflow QC source.

USGS approval codes follow GSIM Table 2: "A" and "A:e" are reliable, "P" and "P:e" suspect.
"M" marks a missing value (-999 in the source) and carries no QC verdict. Original cfs values
and flag strings are passed through unchanged.
"""

import json
from collections.abc import Iterator, Sequence

import numpy as np

from hydrologeez.camels import Camels
from hydrologeez.quality_control.daily import DischargeRecord, DischargeSource

CUBIC_METRES_PER_CUBIC_FOOT = 0.028316846592


def camels_source(store: Camels) -> DischargeSource:
    provenance = {
        "store": str(store.directory),
        "source": store.metadata["source"],
        "usgs_source": store.metadata["usgs_source"],
    }
    return DischargeSource(
        json.dumps(provenance),
        "ft3/s",
        CUBIC_METRES_PER_CUBIC_FOOT,
        {"A": "reliable", "A:e": "reliable", "P": "suspect", "P:e": "suspect", "M": "unflagged"},
    )


def camels_records(store: Camels, ids: Sequence[str] | None = None) -> Iterator[DischargeRecord]:
    """Full observation records in store order, or in ``ids`` order."""
    for gauge in store.ids if ids is None else ids:
        observations = store.observations(gauge)
        qc = np.asarray(observations.qc)
        yield DischargeRecord(
            gauge,
            observations.dates,
            np.asarray(observations.discharge_cfs),
            ~np.asarray(observations.present) | (qc == "M"),
            qc,
        )
