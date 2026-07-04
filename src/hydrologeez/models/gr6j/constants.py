"""GR6J numerical constants.

CODE bounds are used (x6 is [1, 50]; the documented [0.01, 20] is
intentionally not used).
"""

# Routing split fractions
B: float = 0.9  # fraction of effective rainfall to UH1 (slow branch)
C: float = 0.4  # fraction of UH1 output routed to the exponential store

# Unit hydrograph
D: float = 2.5  # S-curve exponent
NH: int = 20  # UH1 length in days; UH2 length is 2 * NH
UH1_LEN: int = NH  # 20
UH2_LEN: int = 2 * NH  # 40
X4_MAX: float = 10.0  # declared upper bound on x4 (masked-kernel sizing)

# Percolation
PERC_CONSTANT: float = 25.62890625  # (9 / 4) ** 4

# Numerical safeguards
MAX_TANH_ARG: float = 13.0
MAX_EXP_ARG: float = 33.0
EXP_BRANCH_THRESHOLD: float = 7.0

# Flat state layout: [S, R, Exp, uh1[20], uh2[40]]
STATE_SIZE: int = 63

# Parameter bounds (code bounds)
PARAM_NAMES: tuple[str, ...] = ("x1", "x2", "x3", "x4", "x5", "x6")
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "x1": (1.0, 2500.0),
    "x2": (-5.0, 5.0),
    "x3": (1.0, 1000.0),
    "x4": (0.5, 10.0),
    "x5": (-4.0, 4.0),
    "x6": (1.0, 50.0),
}
