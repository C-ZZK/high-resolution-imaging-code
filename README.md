<img src="https://github.com/C-ZZK/high-resolution-imaging-code/blob/main/Figure4.png" width="600">

## Description

DA-ID-LSM is a Python framework for Domain-Adaptive Image-Domain Least-Squares Migration using deep learning. It is designed to overcome the critical challenge of feature discrepancies between synthetic and field seismic data, a common issue that limits the field seismic data applicability of deep learning models in geophysics.The method leverages a U-Net based convolutional neural network to learn the complex, nonlinear mapping from a conventional migrated seismic image to a high-resolution subsurface reflectivity model. A key innovation is the integration of a Maximum Mean Discrepancy (MMD) loss function during training. This explicitly minimizes the statistical distribution gap between the features of synthetic training data (source domain) and field data (target domain), enabling the network to learn domain-invariant features. This ensures that the model generalizes effectively from synthetic examples to field seismic data applications. The result is a robust and efficient workflow that produces images with higher resolution, enhanced signal-to-noise ratios, and better lateral continuity compared to conventional RTM and standard ID-LSM methods. 


## Installation

pip install high-resolution-imaging-code

## Requirements

- numpy
- matplotlib
- scipy
- PIL
- numba
- scikit-image
- tqdm
- OpenCV-Python
  
Language: Python 3.8 or higher

Framework: PyTorch 1.8.0  or higher
```python
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
## Usage

The workflow of DA-ID-LSM is shown in the figure below. Migrated images from both synthetic (source) and field (target) domains are fed into the U-Net. The network's training is guided by two loss functions: a Mean Squared Error (MSE) loss to ensure accurate reconstruction of the synthetic data against its true reflectivity label, and an MMD loss to align the feature distributions of the source and target domain outputs.
<img src="https://github.com/C-ZZK/high-resolution-imaging-code/blob/main/Figure1.png" width="600">

# Project Structure

```python
├── checkpoints4train/      # To save/load trained model weights (.pth files)
├── data/
│   ├── seam/               # For target (real-world) data
│   │   ├── seam_mig.dat    # RTM image of the target area
│   │   ├── ImgPSFdeblur.dat    # ID-LSM image of the target area
│   │   ├── deconvPythonShow.py    # Plot code
│   │   ├── model_ref_1501x1751_8x8.dat    # True reflectivity of seam
│   │   └── PSF.dat         # Point Spread Function (PSF) of the target area
│   └── ref.dat             # True reflectivity model (e.g., Marmousi) for training
├── model/                  # Contains model definition scripts (e.g., Trainer.py)
│   ├──  Trainer.py
│   ├──  unetm.py           # U-Net structure
├── utils/                  # Contains utility scripts (e.g., configs, data loaders)
│   ├── config.py
│   ├── config4train2.py    # hyperparameter for  train
│   ├── dataset2.py         # dataset function
│   ├── mmd.py              # MMD function
│   └── ormsby.py           # ormsby wavelet
├── DA-ID-LSM
│   ├── train.py                # Script to train the model
│   └── test.py                 # Script to apply the model for testing
├── result/                 # Output directory for train.py (loss plot, intermediate test images)
├── testresult/             # Output directory for the final result from test.py
└── README.md               # This file
```
Note: The model/ and utils/ directories, which contain the model implementation and helper functions, must be created and populated with the necessary scripts.

## 1. Training the Model(train.py)
To run the below cells, you must first import the library:

```python
import os
import cv2
import numpy as np
import scipy.io as io
import scipy.signal
from scipy.interpolate import interp1d
import torch
from torch.utils import data as data_
from tqdm import tqdm
from torchvision import transforms as tfs
from model.Trainer import InpTrainer
from utils.config4train2 import opt
from utils.dataset2 import Dataset
from utils.ormsby import ormsby_wavelet
from scipy.io import savemat
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt
```
The model is trained using pairs of conventional migrated images and their corresponding true reflectivity models (for the synthetic source domain), alongside unlabeled migrated images from the field (target domain).
Key parameters and the training process are managed within the Python scripts. The Adam optimizer is used to update network parameters.

A reasonable set of input parameters are as follows:
```python

nz = 1500  # nz
nx = 1751  # nx
ee =1  # epoch
cropsize = 128   # cropsize
num_train_dataset = 5000  # dataset of train
num_test_dataset = 2000   # dataset of validation
batch_size = 20  # train batch_size
PSFLx = 21    # Central coordinate interval
PSFLz = 21   # Central coordinate interval
PSFSx = 11    # PSF_Scale 
PSFSz = 11    # PSF_Scale   
PSFNx = 84    # PSF_Number  
PSFNz = 72    # PSF_Number  
PSFBx = 11    # PSF_Initial  Central coordinate
PSFBz = 11    # PSF_Initial   Central coordinate

```
# Loss function 
 Loss function in the trainer.py, loss1 is MSE loss, loss2 is MMD loss.
 ```python
 loss1 = self.lossMSE(outimg, ref)
 loss2 = self.lossmmd(data_out,target1)
 loss = loss1 +  0.02*loss2
```
# Execute the training script
```python
python train.py
```
# Train process 
Once training begins, you can monitor its progress and observe the final loss function.

<img src="https://github.com/C-ZZK/high-resolution-imaging-code/blob/main/Figure5.png" width="600">
<img src="https://github.com/C-ZZK/high-resolution-imaging-code/blob/main/Figure6.png" width="600">

## 2. Testing the Model (test.py)
The testing script applies a trained model to a large target image (e.g., model_image_1501x1751.dat). It uses a sliding window to process the image patch by patch and stitches the results together.

Before running, you must update test.py to point to your trained model checkpoint.Inside test.py, set the path to your trained model

```python
modelpath = './checkpoints4train/Network_36.pth'
```

Execute the test script from your terminal:
```python
python test.py
```

The final high-resolution output will be saved as a binary .dat file (result_seam.dat) in the testresult/ directory. The image below shows an example of a low-resolution input and the corresponding high-resolution output from the model.

<img src="https://github.com/C-ZZK/high-resolution-imaging-code/blob/main/Figure2.png" width="600">
<img src="https://github.com/C-ZZK/high-resolution-imaging-code/blob/main/Figure3.png" width="600">

## 3. Visualize the results (deconvPythonShow.py)
To visualize the results, run the deconvPythonShow.py script located in the ./data/seam/ . This will generate four images: the conventional RTM result, the ID-LSM result, the DA-ID-LSM result, and the true reflectivity model.
```python
cd ./data/seam/

python deconvPythonShow.py
```
# RTM result

<img src="https://github.com/C-ZZK/high-resolution-imaging-code/blob/main/rtm.png" width="600">

# ID-LSM result

<img src="https://github.com/C-ZZK/high-resolution-imaging-code/blob/main/ID-LSM.png" width="600">

# DA-ID-LSM result

<img src="https://github.com/C-ZZK/high-resolution-imaging-code/blob/main/DA-ID-LSM.png" width="600">

# Reflectivity

<img src="https://github.com/C-ZZK/high-resolution-imaging-code/blob/main/ref.png" width="600">

## Google Colab notebook

If you don't want to deal with any local Python environments and installations, you should be able to run meanderpy in [this Google Colab notebook](https://colab.research.google.com/drive/119cnd_cEKT2Zdwv4AuFAHG-p8bIVI3A9?usp=sharing).

```python
%run setup.py install

%run /content/high-resolution-imaging-code-main/DA-ID-LSM/train.py

%run /content/high-resolution-imaging-code-main/DA-ID-LSM/test.py

%run /content/high-resolution-imaging-code-main/data/seam/deconvPythonShow.py
```


# Related Publications
If you use this project in your work, please consider citing one or more of these publications:

Gretton, A., Borgwardt, K.M., Rasch, M.J., Schölkopf, B., Smola, A.J., 2007. A kernel method for the two-sample problem. Advances in Neural Information Processing Systems 19, 513–520.
Liu, S., Ni, W., Fang, W., Fu, L., 2023. Absolute acoustic impedance inversion using convolutional neural networks with transfer learning. Geophysics 88, R163-R174.

# Acknowledgements
The study is sponsored by the National Natural Science Foundation of China (No. 42374139) and the China National Petroleum Corporation Innovation Fund (2024DQ02-0138). We are also grateful for the support of the China University of Geosciences (Wuhan) Postgraduate Joint-Training Practice Base Construction Projects.

# License
This project is licensed under the [Apache License 2.0](https://github.com/C-ZZK/high-resolution-imaging-code/blob/main/LICENSE.txt).








