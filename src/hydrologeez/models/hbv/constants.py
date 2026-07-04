"""HBV-Light numerical constants.

Single-zone (lumped) target. Only ``maxbas`` is hard-validated ([1.0, 7.0],
tied to the fixed length-7 routing buffer); the other 13 bounds are ADVISORY
(calibration only) and are NOT range-checked anywhere.
"""

# State-size constants. Single-zone (lumped) state size:
#   n_zones*ZONE_STATE_SIZE + LUMPED_STATE_SIZE + ROUTING_BUFFER_SIZE
#   = 3*1 + 2 + 7 = 12
ROUTING_BUFFER_SIZE: int = 7  # triangular-UH convolution buffer length
ZONE_STATE_SIZE: int = 3  # [SP, LW, SM] per zone
LUMPED_STATE_SIZE: int = 2  # [SUZ, SLZ]
STATE_SIZE: int = 12  # single-zone flat layout [SP, LW, SM, SUZ, SLZ, b0..b6]

N_PARAMS: int = 14
MAXBAS_MAX: float = 7.0  # == ROUTING_BUFFER_SIZE; effective routing upper bound

# Canonical parameter order.
PARAM_NAMES: tuple[str, ...] = (
    "tt",
    "cfmax",
    "sfcf",
    "cwh",
    "cfr",
    "fc",
    "lp",
    "beta",
    "k0",
    "k1",
    "k2",
    "perc",
    "uzl",
    "maxbas",
)

# CODE bounds. Only maxbas is hard-validated; the rest are advisory (calibration only).
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "tt": (-2.5, 2.5),
    "cfmax": (0.5, 10.0),
    "sfcf": (0.4, 1.4),
    "cwh": (0.0, 0.2),
    "cfr": (0.0, 0.2),
    "fc": (50.0, 700.0),
    "lp": (0.3, 1.0),
    "beta": (1.0, 6.0),
    "k0": (0.05, 0.99),
    "k1": (0.01, 0.5),
    "k2": (0.001, 0.2),
    "perc": (0.0, 6.0),
    "uzl": (0.0, 100.0),
    "maxbas": (1.0, 7.0),
}
