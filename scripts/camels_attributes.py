"""Prepare local CAMELS-US static attributes without daily records."""

import argparse
from pathlib import Path

from hydrologeez.camels import prepare_attributes


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_attributes(args.raw, args.output)


if __name__ == "__main__":
    main()
