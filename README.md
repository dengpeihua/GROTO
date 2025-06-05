# Multi-Granularity Class Prototype Topology Distillation for Class-Incremental  Source-Free Unsupervised Domain Adaptation
[CVPR 2025] Official implementation of paper "Multi-Granularity Class Prototype Topology Distillation for Class-Incremental  Source-Free Unsupervised Domain Adaptation"
![1748399243391](https://github.com/user-attachments/assets/6f019bcc-2f37-4826-a967-2525b2cb8f63)

## 🔧 Installation
- Clone this repository:
```bash
git clone https://github.com/dengpeihua/GROTO.git
cd GROTO
```
- Install the environment by running the following command:
```bash
conda env create -f environment.yml
```

## 📚 Data Preparation
The files of data list and their corresponding labels have been put in the directory `./data_splits`, and the `imagenet_list.txt` can be downloaded at [GoogleDrive](https://drive.google.com/drive/folders/1MGFO41tVIsG1ckQmh0t3q9kJjmnXM2i2?usp=sharing).

Please manually download the office31, office-home and ImageNet-Caltech benchmarks from the [link](https://github.com/jindongwang/transferlearning/tree/master/data), [link](http://www.vision.caltech.edu/Image_Datasets/Caltech256) and [link](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar), then put them into the corresponding directory (e.g., './dataset/office-home').

Put the corresponding file in your path (e.g., './dataset/office-home/Art.txt')

## ⚡ Source Pre-trained
To obtain the pre-trained model on the source domain:
- Pre-train on the amazon domain of Office-31-CI:
```bash
python source_train_office31.py --gpu 0 --source 0
```

- Pre-train on the Art domain of Office-Home-CI:
```bash
python source_train_office-home.py --gpu 0 --source 0
```

- from Caltech256 to ImageNet84:
```bash
python source_train_cal256.py --gpu 0
```

- from ImageNet1K to Caltech84:
```bash
python source_train_imagenet1k.py --gpu 0
```

## ➡️ Class-Incremental Source-Free Unsupervised Domain Adaptation
Run the `.sh` scripts to obtain the results of different datasets.

## 🙏 Acknowledgments
We would like to thank the authors of previous related projects for generously sharing their code and insights: [ViT](https://github.com/lukemelas/PyTorch-Pretrained-ViT), [PCT](https://github.com/korawat-tanwisuth/Proto_DA) and [ProCA](https://github.com/Hongbin98/ProCA).

## 🤝 Citation
If you find our work useful, please consider giving a citation 📝 :)
```bibtex
@article{deng2024multi,
  title={Multi-Granularity Class Prototype Topology Distillation for Class-Incremental Source-Free Unsupervised Domain Adaptation},
  author={Deng, Peihua and Zhang, Jiehua and Sheng, Xichun and Yan, Chenggang and Sun, Yaoqi and Fu, Ying and Li, Liang},
  journal={arXiv preprint arXiv:2411.16064},
  year={2024}
}
```
