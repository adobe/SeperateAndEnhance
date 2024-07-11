# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import os
import itertools
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.utils import check_min_version
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm

from diffusers import (
    UNet2DConditionModel,
)

from diffusers.optimization import get_scheduler

from model import CustomDiffusionPipeline
from ptp_utils import aggregate_attention
from utils import GaussianSmoothing, load_model, get_index, compute_separate_loss, compute_enhance_loss

from config import parse_args


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.14.0")



def main(args):

    # accelerator config

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if accelerator.is_main_process:
        if args.experiment_name is not None:
            output_dir = os.path.join('./logs',args.experiment_name)
            os.makedirs(output_dir, exist_ok=True)

    tokenizer, text_encoder_cls, noise_scheduler, text_encoder, vae, unet, controller, optimizer = load_model(accelerator, args)
    if args.use_norm:
        unet_clone = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )
        unet_clone = unet_clone.to(accelerator.device)
        unet_clone.requires_grad_(False)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )


    unet, optimizer, lr_scheduler = accelerator.prepare(
        unet, optimizer, lr_scheduler
    )


    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    global_step = 0
    first_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    smoothing = GaussianSmoothing(channels=4, kernel_size=3, sigma=0.5, dim=2).cuda()
    batch_size = args.train_batch_size

    for step in range(first_step, args.max_train_steps):
        unet.train()
        controller.reset()

        with accelerator.accumulate(unet):
            p_full = ["a photo of {} and {}".format(args.c1,args.c2) for _ in range(batch_size)]

            # strategy 1: no norm loss                               
            latents = torch.randn(batch_size, 4, args.resolution//8, args.resolution//8).to(accelerator.device)
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(
                latents, noise, timesteps)

            input_id_full = tokenizer(p_full,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
            ).input_ids.cuda()
            hid_state_full = text_encoder(input_id_full)[0]
            # print(noisy_latents.shape, n_real_samples,)
            # import pdb 
            # pdb.set_trace()
            sample_out = unet(noisy_latents, timesteps, hid_state_full).sample
            amap_full = aggregate_attention(
                attention_store=controller,
                res=16,
                from_where=("up", "down", "mid"),
                is_cross=True,
                bs = batch_size)

            index_c1 = get_index(p_full, args.c1)
            index_c2 = get_index(p_full, args.c2)

            loss_sep = 0
            loss_en = 0
            
            for ii in range(batch_size):
                cur_c1_index = index_c1[ii]
                cur_c2_index = index_c2[ii]
                amap_c1 = amap_full[ii,:,:,:,cur_c1_index]
                amap_c2 = amap_full[ii,:,:,:,cur_c2_index]
                loss_sep += compute_separate_loss(amap_c1,amap_c2)  
                loss_en += compute_enhance_loss(amap_c1,amap_c2,smoothing)              
            
            loss = args.lambda_sep * loss_sep / batch_size + args.lambda_en * loss_en / batch_size 
            

            
            if args.use_norm:
                p_norm = ["a photo of {}".format(args.c1) for _ in range(batch_size//2)] + ["a photo of {}".format(args.c2) for _ in range(batch_size//2)]
                input_id_norm = tokenizer(p_norm,
                        truncation=True,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        return_tensors="pt",
                        ).input_ids.cuda()
                bsz = input_id_norm.shape[0]
                norm_encoder_hidden_states = text_encoder(input_id_norm)[0]
                norm_latents = torch.randn(bsz,4, args.resolution//8, args.resolution//8).to(accelerator.device)
                norm_latents = norm_latents * vae.config.scaling_factor


                norm_noise = torch.randn_like(norm_latents)
                # Sample a random timestep for each image
                norm_timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                norm_timesteps = norm_timesteps.long()
                norm_noisy_latents = noise_scheduler.add_noise(
                        norm_latents, norm_noise, norm_timesteps)
                    
                model_pred = unet(norm_noisy_latents,  norm_timesteps,
                                norm_encoder_hidden_states).sample
                target_prior = unet_clone(norm_noisy_latents,  norm_timesteps,
                                norm_encoder_hidden_states).sample.detach()
                norm_loss = F.mse_loss(
                    model_pred.float(), target_prior.float(), reduction="mean")
                loss += args.norm_weight * norm_loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = (
                    itertools.chain([x[1] for x in unet.named_parameters() if ('attn2' in x[0])])
                )
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    pipeline = CustomDiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        text_encoder=accelerator.unwrap_model(text_encoder),
                        tokenizer=tokenizer,
                        revision=args.revision,
                    )
                    save_path = os.path.join(output_dir,f"delta-{global_step}")
                    pipeline.save_pretrained(save_path)

        if global_step % args.checkpointing_steps == 0:
            validation_prompt = args.validation_prompt
            # create pipeline
            pipe = CustomDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
            pipe.load_model(os.path.join(output_dir,'delta-{}'.format(global_step)))
            # run inference
            generator = torch.Generator(device=accelerator.device).manual_seed(666)
            images = pipe([validation_prompt], num_inference_steps=25, guidance_scale=6., eta=1., generator = generator).images
            if not os.path.exists('./training_samples'):
                os.mkdir('./training_samples')
            if not os.path.exists(os.path.join('./training_samples',args.experiment_name)):
                os.mkdir(os.path.join('./training_samples',args.experiment_name))
            image_outpath = os.path.join(os.path.join('./training_samples',args.experiment_name),'{}-{}.png'.format(global_step,validation_prompt))
            images[0].save(image_outpath)

            del pipe
            torch.cuda.empty_cache()

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        pipeline = CustomDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            revision=args.revision,
        )
        save_path = os.path.join(output_dir, "delta.bin")
        pipeline.save_pretrained(save_path)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
