# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

CUDA_VISIBLE_DEVICES=2 python train_individual.py \
--pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
--experiment_name="bear" \
--resolution=512  \
--train_batch_size=4  \
--learning_rate=2e-6  \
--max_train_steps=200 \
--scale_lr \
--c1="bear" \
--c2="book" \
--enable_xformers_memory_efficient_attention \
--dataloader_num_workers 0 \
--checkpointing_steps 100 \
--validation_prompt "a photo of a bear and a book" \
