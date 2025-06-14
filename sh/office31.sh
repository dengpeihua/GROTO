#!/bin/bash

for target in 1 2
do
    command="CUDA_VISIBLE_DEVICES=0 python target_office31.py --gpu 0 --source 0 --target $target --source_model ./model_source/20240521-2241source-free-OH_amazon_ce_singe_gpu_vit_B_16_imagenet1k_best.pkl --source_centers ./model_source/20240521-2241source_centers_mean.pkl"
    echo "Running: $command"
    eval $command
done

for target in 0 2
do
    command="CUDA_VISIBLE_DEVICES=0 python target_office31.py --gpu 0 --source 1 --target $target --source_model ./model_source/20240521-2255source-free-OH_dslr_ce_singe_gpu_vit_B_16_imagenet1k_best.pkl --source_centers ./model_source/20240521-2255source_centers_mean.pkl"
    echo "Running: $command"
    eval $command
done

for target in 0 1
do
    command="CUDA_VISIBLE_DEVICES=0 python target_office31.py --gpu 0 --source 2 --target $target --source_model ./model_source/20240521-2258source-free-OH_webcam_ce_singe_gpu_vit_B_16_imagenet1k_best.pkl --source_centers ./model_source/20240521-2258source_centers_mean.pkl"
    echo "Running: $command"
    eval $command
done

echo "All commands are executed."
