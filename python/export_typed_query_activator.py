#!/usr/bin/env python3

import argparse

from train_typed_query_activator import (
    export_typed_query_activator_binary,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("pair_checkpoint")
    parser.add_argument("pair_binary")
    parser.add_argument("output")
    args = parser.parse_args()
    print(export_typed_query_activator_binary(
        args.checkpoint,
        args.pair_checkpoint,
        args.pair_binary,
        args.output,
    ))


if __name__ == "__main__":
    main()
