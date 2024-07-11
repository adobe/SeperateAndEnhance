# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from cleanfid import fid
import argparse

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dir1",
        type= str,
        required=True,
        help=(
            "where the generated samples are saved"
        ),
    )
    parser.add_argument(
        "--dir2",
        type= str,
        required=True,
        help=(
            "where the real samples are saved"
        ),
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    score = fid.compute_fid(args.dir1, args.dir2, num_workers=0)
    print(score)
