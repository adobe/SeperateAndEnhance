# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import torch
import torch.nn as nn 
import numbers
import math 
import torch.nn.functional as F


import itertools

import torch.utils.checkpoint
from transformers import AutoTokenizer, PretrainedConfig
from packaging import version

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)

from diffusers.models.cross_attention import CrossAttention
from diffusers.utils.import_utils import is_xformers_available

from model import CustomDiffusionAttnProcessor, CustomDiffusionPipeline, set_use_memory_efficient_attention_xformers
from ptp_utils import AttentionStore, register_attention_control


def load_model(accelerator, args):
    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    controller = AttentionStore()
    register_attention_control(unet,controller)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet = create_model(unet)
    
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    if accelerator.mixed_precision != "fp16":
        unet.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        if args.use_norm:
            args.learning_rate = args.learning_rate * 2.

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = itertools.chain([x[1] for x in unet.named_parameters() if 'attn2.to_k' in x[0]])



    # Optimizer creation
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    return tokenizer, text_encoder_cls, noise_scheduler, text_encoder, vae, unet, controller, optimizer

def replace_prompt(source_prompt, prompt1, prompt2, prompt3):
    target_prompt = []
    for prompt in source_prompt:
        if prompt1 in prompt:
            target_prompt += [prompt.replace(prompt1, prompt3)]
        elif prompt2 in prompt:
            target_prompt += [prompt.replace(prompt2, prompt3)]
        else:
            target_prompt += [prompt]
    return target_prompt

def get_index(source_prompt, target):
    idx = []
    for p in source_prompt:
        pp = p.split()
        idx.append(pp.index(target)+1)
    return idx


def create_model(unet):
    for name, params in unet.named_parameters():
        if 'attn2.to_k' in name:
            params.requires_grad = True
        else:
            params.requires_grad = False

    # change attn class
    def change_attn(unet):
        for layer in unet.children():
            if type(layer) == CrossAttention:
                bound_method = set_use_memory_efficient_attention_xformers.__get__(layer, layer.__class__)
                setattr(layer, 'set_use_memory_efficient_attention_xformers', bound_method)
            else:
                change_attn(layer)

    change_attn(unet)
    unet.set_attn_processor(CustomDiffusionAttnProcessor())
    return unet


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def freeze_params(params):
    for param in params:
        param.requires_grad = False



def safe_dir(dir):
    if not dir.exists():
        dir.mkdir()
    return dir

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)


def compute_separate_loss(amap1, amap2):
    iou = amap1 * amap2 / (amap1 + amap2)
    return iou.max() 

def compute_enhance_loss(amap1, amap2, smoothing):
    amap1 = F.pad(amap1.unsqueeze(0), (1, 1, 1, 1), mode='reflect')
    amap1 = smoothing(amap1).squeeze(0)
    amap2 = F.pad(amap2.unsqueeze(0), (1, 1, 1, 1), mode='reflect')
    amap2 = smoothing(amap2).squeeze(0)
    aamap1 = amap1.max()
    aamap2 = amap2.max()
    threshold = 1
    if aamap1 > aamap2:
        return threshold - aamap2 
    else:
        return threshold - aamap1
