"""Prepare all available CAMELS meteorological products."""

import argparse
from pathlib import Path

from hydrologeez.camels_meteorology import prepare_meteorology


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_meteorology(args.raw, args.output)


if __name__ == "__main__":
    main()
