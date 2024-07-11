# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

CUDA_VISIBLE_DEVICES=1 python train_large.py \
--pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
--experiment_name="large" \
--resolution=512  \
--train_batch_size=4  \
--learning_rate=5e-7  \
--max_train_steps=10000 \
--scale_lr \
--enable_xformers_memory_efficient_attention \
--dataloader_num_workers 0 \
--checkpointing_steps 250 \
--use_norm \
--norm_weight 0.5 \
--validation_prompt "a photo of a bear and a book" \
