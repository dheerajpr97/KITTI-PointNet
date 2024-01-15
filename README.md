# KITTI-PointNet

This repository hosts the implementation and tools necessary for training and evaluating PointNet models on the KITTI dataset for tasks such as point cloud classification and segmentation.

## Overview

The repository is structured to facilitate the processing of point cloud data, model training and evaluation, and provides utility scripts and Jupyter notebooks for a comprehensive workflow.

## Structure

- `data_3d_semantics/`: Raw semantic 3D data from KITTI.
- `preprocess_scripts/`: Scripts for preprocessing point cloud data.
- `processed/`: Contains processed data split into training and validation sets.
- `evaluation/`: Evaluation scripts for classification and segmentation tasks.
- `models/`: PointNet model definitions and utilities.
- `notebooks/`: Jupyter notebooks for data analysis, preprocessing, and model implementation.
- `training/`: Training scripts for classification and segmentation.
- `utils/`: Utilities for file handling and other functions.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed on your system and the necessary packages.

### Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/dheerajpr97/KITTI-PointNet.git
cd KITTI-PointNet
pip install -r requirements.txt
```

## Preprocessing
Run the preprocessing scripts to prepare the data:

```bash
python preprocess_scripts/data_cls.py
python preprocess_scripts/data_seg.py
```

## Training
Train the models using the scripts in the training directory:

```bash
python training/train_cls.py
python training/train_seg.py
```

## Evaluation
Evaluate the trained models with the evaluation scripts:

```bash
python evaluation/evaluate_cls.py
python evaluation/evaluate_seg.py
```

### License
This project is under the MIT License. See the LICENSE file for more information.

### Contributions
Contributions are welcome. Please open an issue to discuss your ideas or submit a pull request.

### Acknowledgements
Thanks to the KITTI-360 Dataset Providers.

    @article{Liao2022PAMI,
       title =  {{KITTI}-360: A Novel Dataset and Benchmarks for Urban Scene Understanding in 2D and 3D},
       author = {Yiyi Liao and Jun Xie and Andreas Geiger},
       journal = {Pattern Analysis and Machine Intelligence (PAMI)},
       year = {2022},
    }
Appreciation to the creators of the PointNet architecture and its PyTorch implementation.

    @article{qi2016pointnet,
      title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
      author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
      journal={arXiv preprint arXiv:1612.00593},
      year={2016}
    }
    @article{Pytorch_Pointnet_Pointnet2,
          Author = {Xu Yan},
          Title = {Pointnet/Pointnet++ Pytorch},
          Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
          Year = {2019}
    }

### To do:

* Add classification scripts 
* Add mIoU evaluation and inference scripts
* Add example visualizations
* Add PointNet++ functionality
* Compare performance of architectures
