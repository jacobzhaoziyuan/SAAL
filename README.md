
<div align="center">

# Self-supervised  Assisted  Active  Learning  for  Skin  Lesion  Segmentation
  
[![EMBC2022](https://img.shields.io/badge/arXiv-2205.07021-blue)](https://arxiv.org/abs/2205.07021)
[![EMBC2022](https://img.shields.io/badge/Conference-EMBC2022-green)](https://arxiv.org/abs/2205.07021)

</div>

Pytorch implementation of our method for EMBC 2022 paper: "Self-supervised  Assisted  Active  Learning  for  Skin  Lesion  Segmentation".

Contents
---
- [Abstract](#Abstract)
- [Dataset](#Dataset)
- [Installation](#Installation)
- [Pretrain](#Pretrain)
- [Training](#Training)
- [Citation](#Citation)
- [Acknowledgement](#Acknowledgement)

## Abstract

Label scarcity has been a long-standing issue for biomedical image segmentation, due to high annotation costs and professional requirements. Recently, active learning (AL) strategies strive to reduce annotation costs by querying a small portion of data for annotation, receiving much traction in the field of medical imaging. However, most of the existing AL methods have to initialize models with some randomly selected samples followed by active selection based on various criteria, such as uncertainty and diversity. Such random-start initial- ization methods inevitably introduce under-value redundant samples and unnecessary annotation costs. For the purpose of addressing the issue, we propose a novel self-supervised assisted active learning framework in the cold-start setting, in which the segmentation model is first warmed up with self- supervised learning (SSL), and then SSL features are used for sample selection via latent feature clustering without accessing labels. We assess our proposed methodology on skin lesions segmentation task. Extensive experiments demonstrate that our approach is capable of achieving promising performance with substantial improvements over existing baselines.

<p align="center">
<img src="https://github.com/jacobzhaoziyuan/SAAL/blob/main/assets/archi.png" width="700">
</p>

## Dataset

Download the ISIC 2017: International Skin Imaging Collaboration (ISIC) dataset from [ISIC 2017](https://challenge.isic-archive.com/data/) which composes of 2000 RGB dermoscopy images with binary masks of lesions.

**Preprocess**: refer to the image pre-processing method in [CEAL](https://github.com/marc-gorriz/CEAL-Medical-Image-Segmentation) .

* **Unlabeled Pool**: 1600 samples

* **Labeled Pool**: 400 samples

## Installation

```
Install PyTorch 1.10.2 + CUDA 11.2 
```

## Pretrain

#### 1. Self-Supervised Learning Model

A self-supervised learning (SSL) framework was first built to explore features from the unlabeled dataset and save SSL model weights to warm up the segmentation model.

```
cd Self-supervised
run PreTrain.py
```

#### 2. Embedded Feature Clustering

Then, based on the clustering results of the potential features extracted from the SSL, we develop a criterion to select representative samples from the unlabeled data to form the initial dataset for warm-up training.

```
run cluster.py
```

## Training

Prior to the AL procedure, the pre-trained weights from the self-supervised learning model are adopted to initialize the segmentation model. 

```
cd segmentation
bash run.sh
```


Citation
---
If you find the codebase useful for your research, please cite the paper:
```
@misc{zhao2022selfsupervised,
    title={Self-supervised Assisted Active Learning for Skin Lesion Segmentation},
    author={Ziyuan Zhao and Wenjing Lu and Zeng Zeng and Kaixin Xu and Bharadwaj Veeravalli and Cuntai Guan},
    year={2022},
    eprint={2205.07021},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```




## Acknowledgement

Part of the code is adapted from open-source codebase and original implementations of algorithms, we thank these authors for their fantastic and efficient codebase:

- ModelGenisis: https://github.com/MrGiovanni/ModelsGenesis
- CEAL: https://github.com/marc-gorriz/CEAL-Medical-Image-Segmentation
- UNet: https://github.com/zhixuhao/unet
- DSAL: https://github.com/jacobzhaoziyuan/DSAL
