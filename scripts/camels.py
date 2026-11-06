"""Prepare local CAMELS inputs without modifying source files."""

import argparse
from pathlib import Path

from hydrologeez.camels import prepare


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--usgs-areas", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.raw, args.output, args.usgs_areas)


if __name__ == "__main__":
    main()
