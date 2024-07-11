# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from model_pipeline import CustomDiffusionPipeline
from utils import filter, safe_dir
import torch
from pathlib import Path

import os
import argparse


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--prompt",
        help=(
            "prompt for evaluation"
        ),
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help=(
            "pretrianed model checkpoint directory"
        ),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help=(
            "number of samples to generate for each prompt"
        ),
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

def main(args):
    prompt = args.prompt
    
    pipe = CustomDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
    pipe.load_model(args.checkpoint_dir)

    num_of_imgs = args.num_samples
    if not os.path.exists('./samples'):
        os.mkdir('./samples')
    if not os.path.exists('./samples/{}'.format(prompt)):
        os.mkdir('./samples/{}'.format(prompt))
    for i in range(num_of_imgs):
        g = torch.Generator('cuda').manual_seed(i)
        images = pipe([prompt], num_inference_steps=50, guidance_scale=6., eta=1., generator = g, ).images
        image_filename = './samples/{}/{}.png'.format(prompt,i)
        images[0].save(image_filename)

if __name__ == '__main__':
    args = parse_args()
    main(args)
    
