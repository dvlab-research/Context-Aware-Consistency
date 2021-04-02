# Semi-supervised Semantic Segmentation with Directional Context-aware Consistency (CAC)
*Lai Xin<sup>\*</sup>, Zhuotao Tian<sup>\*</sup>, Li Jiang, Shu Liu, Hengshuang Zhao, Liwei Wang, Jiaya Jia*

This is the official PyTorch implementation of our paper [**Semi-supervised Semantic Segmentation with Directional Context-aware Consistency**](https://jiaya.me/papers/semiseg_cvpr21.pdf) that has been accepted to 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2021).

# Highlight 
Our method achives the state-of-the-art performance on semi-supervised semantic segmentation. Based on [**CCT**](https://github.com/yassouali/CCT), this Repository also supports efficient distributed training with multiple GPUs.

# Get Started
## Environment
The repository is tested on Ubuntu 18.04.3 LTS, Python 3.6.9, PyTorch 1.6.0 and CUDA 10.2
```
pip install -r requirements.txt
```

## Datasets Preparation
1. Firstly, download the [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) Dataset, and the extra annotations from [SegmentationClassAug](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0).
2. Extract the above compression files into your desired path, and make it follow the directory tree as below.

```
-VOCtrainval_11-May-2012
    -VOCdevkit
        -VOC2012
            -Annotations
            -ImageSets
            -JPEGImages
            -SegmentationClass
            -SegmentationClassAug
            -SegmentationObject
```

3. Set 'data_dir' in the config file into '[YOUR_PATH]/VOCtrainval_11-May-2012'.

## Training

Firsly, you should download the PyTorch ResNet101 or ResNet50 ImageNet-pretrained weight into the 'pretrained/' directory using the following commands

```
cd Context-Aware-Consistency
cd pretrained
wget https://download.pytorch.org/models/resnet50-19c8e357.pth # ResNet50
wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth # ResNet101
```

Run the following commands for training.

- train the model on the 1/8 labeled data (the 0-th data list) of PASCAL VOC with the segmentation network and the backbone set to DeepLabv3+ and ResNet50 respectively.
```
python3 train.py --config configs/voc_cac_deeplabv3+_resnet50_1over8_datalist0.json
```

- train the model on the 1/8 labeled data (the 0-th data list) of PASCAL VOC with the segmentation network and the backbone set to DeepLabv3+ and ResNet101 respectively.
```
python3 train.py --config configs/voc_cac_deeplabv3+_resnet101_1over8_datalist0.json
```

## Testing
For testing, run the following command.

```
python3 train.py --config [CONFIG_PATH] --resume [CHECKPOINT_PATH] --test True
```

# Related Repositories

This repository highly depends on the **CCT** repository at https://github.com/yassouali/CCT. We thank the authors of CCT for their great work and clean codes.

Besides, we also borrow some codes from **Semseg** at https://github.com/hszhao/semseg, and also **MoCo** at https://github.com/facebookresearch/moco. Thanks a lot for their great work.

# Citation
If you find this project useful, please consider citing:

```
@inproceedings{lai2021cac,
  title     = {Semi-supervised Semantic Segmentation with Directional Context-aware Consistency},
  author    = {Lai Xin, Zhuotao Tian, Li Jiang, Shu Liu, Hengshuang Zhao, Liwei Wang and Jiaya Jia},
  booktitle = {CVPR},
  year      = {2021}
}
```