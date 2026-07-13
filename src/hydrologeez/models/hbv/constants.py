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
