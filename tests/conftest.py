"""Suite-wide x64-enablement seam (the SINGLE canonical mechanism).

pytest imports this conftest before collecting any test module. JAX reads the
``JAX_ENABLE_X64`` environment variable at ITS OWN import time, so setting it
here -- before any ``jax`` / ``hydrologeez`` import anywhere in the suite --
guarantees x64 is enabled and that every downstream test module
(test_precision, test_ssm, and all later modules) collects without tripping the
raise-on-import float64 guard.

Do NOT call ``jax.config.update`` here and do NOT import jax/hydrologeez at the
top of this file; the env-var seam must be set first. Parallel worktrees must
reuse THIS mechanism rather than inventing a colliding one.
"""

import os

os.environ["JAX_ENABLE_X64"] = "1"
