#!/bin/bash
command="CUDA_VISIBLE_DEVICES=2 python target_i2c.py --gpu 2 --source_model ./model_source/20240725-0923-single_gpu_imagenet1k_ce_vit_B_16.pkl --source_centers ./model_source/20240725-0923vit_B_16_imagenet1k_source_centers_mean.pkl"
echo "Running: $command"
eval $command

echo "All commands are executed."