#!/bin/bash
command="CUDA_VISIBLE_DEVICES=1 python target_c2i.py --gpu 1 --source_model ./model_source/20240717-1602-single_gpu_cal256_ce_vit_B_16_best.pkl --source_centers ./model_source/20240717-1602vit_B_16_cal256_source_centers_mean.pkl"
echo "Running: $command"
eval $command

echo "All commands are executed."
