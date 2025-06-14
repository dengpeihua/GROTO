#!/bin/bash
for target in 1 2 3
do
    command="CUDA_VISIBLE_DEVICES=2 python target_officehome.py --gpu 2 --source 0 --target $target --source_model ./model_source/20240701-1714source-free-OH_Art_ce_singe_gpu_vit_B_16_imagenet1k_best.pkl --source_centers ./model_source/20240701-1714source_centers_mean.pkl"
    echo "Running: $command"
    eval $command
done


for target in 0 2 3
do
    command="CUDA_VISIBLE_DEVICES=2 python target_officehome.py --gpu 2 --source 1 --target $target --source_model ./model_source/20240701-1801source-free-OH_Clipart_ce_singe_gpu_vit_B_16_imagenet1k_best.pkl --source_centers ./model_source/20240701-1801source_centers_mean.pkl"
    echo "Running: $command"
    eval $command
done


for target in 0 1 3
do
    command="CUDA_VISIBLE_DEVICES=2 python target_officehome.py --gpu 2 --source 2 --target $target --source_model ./model_source/20240701-1918source-free-OH_Product_ce_singe_gpu_vit_B_16_imagenet1k_best.pkl --source_centers ./model_source/20240701-1918source_centers_mean.pkl"
    echo "Running: $command"
    eval $command
done


for target in 0 1 2
do
    command="CUDA_VISIBLE_DEVICES=2 python target_officehome.py --gpu 2 --source 3 --target $target --source_model ./model_source/20240701-1806source-free-OH_RealWorld_ce_singe_gpu_vit_B_16_imagenet1k_best.pkl --source_centers ./model_source/20240701-1806source_centers_mean.pkl"
    echo "Running: $command"
    eval $command
done
echo "All commands are executed."
