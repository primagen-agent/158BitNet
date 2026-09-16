#!/usr/bin/env python3
"""Export the experimental typed-memory link heads for the C runtime."""

import argparse

from train_typed_pair_verifier import (
    export_typed_link_binary,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    print(export_typed_link_binary(
        args.checkpoint, args.output))


if __name__ == "__main__":
    main()
