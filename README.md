# Multi-Granularity Class Prototype Topology Distillation for Class-Incremental  Source-Free Unsupervised Domain Adaptation
[CVPR 2025] Official implementation of paper "Multi-Granularity Class Prototype Topology Distillation for Class-Incremental  Source-Free Unsupervised Domain Adaptation"

# Data Preparation
The files of data list and their corresponding labels have been put in the directory ./data_splits, and the imagenet_list.txt can be downloaded at https://drive.google.com/drive/folders/1MGFO41tVIsG1ckQmh0t3q9kJjmnXM2i2?usp=sharing.

Please manually download the office31, office-home and ImageNet-Caltech benchmarks from the websites (https://github.com/jindongwang/transferlearning/tree/master/data, http://www.vision.caltech.edu/Image_Datasets/Caltech256, https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar) and put them into the corresponding directory (e.g., './dataset/office-home').

Put the corresponding file in your path (e.g., './dataset/office-home/Art.txt')
