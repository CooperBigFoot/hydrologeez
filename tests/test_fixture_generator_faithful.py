"""Fixture generator faithfulness against committed artifacts."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

gen_gr6j = importlib.import_module("generate_gr6j_fixtures")
gen_hbv = importlib.import_module("generate_hbv_fixtures")

FIXTURES = Path(__file__).parent / "fixtures"
RTOL = 1e-4
ATOL = 1e-6

FIXTURE_BUILDERS = {
    **gen_gr6j.FIXTURE_BUILDERS,
    **gen_hbv.FIXTURE_BUILDERS,
}


@pytest.mark.parametrize("fixture_name", list(FIXTURE_BUILDERS))
def test_fixture_generator_rebuilds_committed_fixture(fixture_name: str) -> None:
    builder = FIXTURE_BUILDERS[fixture_name]
    with np.load(FIXTURES / fixture_name, allow_pickle=False) as data:
        rebuilt = builder(data)
        assert set(rebuilt) == set(data.files)
        for key, got in rebuilt.items():
            ref = data[key]
            err_msg = f"{fixture_name}:{key}"
            if np.issubdtype(ref.dtype, np.floating):
                np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL, err_msg=err_msg)
            else:
                np.testing.assert_array_equal(got, ref, err_msg=err_msg)
