<img src="https://raw.githubusercontent.com/zsylvester/meanderpy/755e95a8a87c82df6384694e0192fc8a02af4782/meanderpy_logo.svg" width="300">

## Description

DA-ID-LSM is a Python framework for Domain-Adaptive Image-Domain Least-Squares Migration using deep learning. It is designed to overcome the critical challenge of feature discrepancies between synthetic and field seismic data, a common issue that limits the field seismic data applicability of deep learning models in geophysics.The method leverages a U-Net based convolutional neural network to learn the complex, nonlinear mapping from a conventional migrated seismic image to a high-resolution subsurface reflectivity model. A key innovation is the integration of a Maximum Mean Discrepancy (MMD) loss function during training. This explicitly minimizes the statistical distribution gap between the features of synthetic training data (source domain) and field data (target domain), enabling the network to learn domain-invariant features. This ensures that the model generalizes effectively from synthetic examples to field seismic data applications. The result is a robust and efficient workflow that produces images with higher resolution, enhanced signal-to-noise ratios, and better lateral continuity compared to conventional RTM and standard ID-LSM methods. 


## Installation

git clone https://github.com/C-ZZK/high-resolution-imaging-code.git
cd high-resolution-imaging-code
pip install -r requirements.txt

## Requirements

- numpy
- os
- PyTorch
- matplotlib
- scipy
- PIL
- numba
- scikit-image
- tqdm
- OpenCV-Python
  
Language: Python 3.8 or higher
Framework: PyTorch 1.8.0  or higher

## Usage

<img src="https://raw.githubusercontent.com/zsylvester/meanderpy/master/meanderpy_sketch.png" width="600">

The workflow of DA-ID-LSM is shown in the figure below. Migrated images from both synthetic (source) and field (target) domains are fed into the U-Net. The network's training is guided by two loss functions: a Mean Squared Error (MSE) loss to ensure accurate reconstruction of the synthetic data against its true reflectivity label, and an MMD loss to align the feature distributions of the source and target domain outputs.

1. Training the Model
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

```
# Execute the training script
python train.py

The initial Channel object can be created using the 'generate_initial_channel' function. This creates a straight line, with some noise added. However, a Channel can be created (and then used as the first channel in a ChannelBelt) using any set of x,y,z,W,D variables.

```python
ch = mp.generate_initial_channel(W, depths[0], Sl, deltas, pad, n_bends) # initialize channel
chb = mp.ChannelBelt(channels=[ch], cutoffs=[], cl_times=[0.0], cutoff_times=[]) # create channel belt object
```

The core functionality of 'meanderpy' is built into the 'migrate' method of the 'ChannelBelt' class. This is the function that computes migration rates and moves the channel centerline to its new position. The last Channel of a ChannelBelt can be further migrated through applying the 'migrate' method to the ChannelBelt instance.

```python
chb.migrate(nit,saved_ts,deltas,pad,crdist,depths,Cfs,kl,kv,dt,dens,t1,t2,t3,aggr_factor) # channel migration
```

ChannelBelt objects can be visualized using the 'plot' method. This creates a map of all the channels and cutoffs in the channel belt; there are two styles of plotting: a 'stratigraphic' view and a 'morphologic' view (see below). The morphologic view tries to account for the fact that older point bars and oxbow lakes tend to be gradually covered with vegetation.

```python
# migrate an additional 1000 iterations and plot results
chb.migrate(1000,saved_ts,deltas,pad,crdist,depths,Cfs,kl,kv,dt,dens,t1,t2,t3,aggr_factor)
fig = chb.plot('strat', 20, 60, chb.cl_times[-1], len(chb.channels)) # plotting
```

<img src="https://raw.githubusercontent.com/zsylvester/meanderpy/master/meanderpy_strat_vs_morph.png" width="1000">

A series of movie frames (in PNG format) can be created using the 'create_movie' method:

```python
chb.create_movie(xmin,xmax,plot_type,filename,dirname,pb_age,ob_age,scale,end_time)
```
The frames have to be assembled into an animation outside of 'meanderpy'.

## Build 3D model

'meanderpy' includes the functionality to build 3D stratigraphic models. However, this functionality is decoupled from the centerline generation, mainly because it would be computationally expensive to generate surfaces for all centerlines, along their whole lengths. Instead, the 3D model is only created after a Channelbelt object has been generated; a model domain is defined either through specifying the xmin, xmax, ymin, ymax coordinates, or through clicking the upper left and lower right corners of the domain, using the matplotlib 'ginput' command:

<img src="https://raw.githubusercontent.com/zsylvester/meanderpy/master/define_3D_domain.png" width="600">

Important parameters for a fluvial 3D model are the following:

```python
Sl = 0.0              # initial slope (matters more for submarine channels than rivers)
t1 = 500              # time step when incision starts
t2 = 700              # time step when lateral migration starts
t3 = 1400             # time step when aggradation starts
aggr_factor = 4e-9    # aggradation rate (in m/s, it kicks in after t3)
h_mud = 0.4           # thickness of overbank deposit for each time step
dx = 10.0             # gridcell size in meters
```
The first five of these parameters have to be specified before creating the centerlines. The initial slope (Sl) in a fluvial model is best set to zero, as typical gradients in meandering rivers are very low and artifacts associated with the along-channel slope variation will be visible in the model surfaces [this is not an issue with steeper submarine channel models]. t1 is the time step when incision starts; before t1, the centerlines are given time to develop some sinuosity. At time t2, incision stops and the channel only migrates laterally until t3; this is the time when aggradation starts. The rate of incision (if Sl is set to zero) is set by the quantity 'kv x dens x 9.81 x D x dt x 0.01' (as if the slope was 0.01, but of course it is not), where kv is the vertical incision rate constant. This approach does not require a new incision rate constant. The rate of aggradation is set by 'aggr_factor x dt' (so 'aggr_factor' must be a small number, as it is measured in m/s). 'h_mud' is the maximum thickness of the overbank deposit in each time step, and 'dx' is the gridcell size in meters. 'h_mud' has to be large enough that it matches the channel aggradation rate; weird artefacts are generated otherwise.

The Jupyter notebook has two examples for building 3D models, for a fluvial and a submarine channel system. The 'plot_xsection' method can be used to create a cross section at a given x (pixel) coordinate (this is the first argument of the function). The second argument determines the colors that are used for the different facies (in this case: brown, yellow, brown RGB values). The third argument is the vertical exaggeration.

```python
fig1,fig2,fig3 = chb_3d.plot_xsection(343, [[0.5,0.25,0],[0.9,0.9,0],[0.5,0.25,0]], 4)
```
This function also plots the basal erosional surface and the final topographic surface. An example topographic surface and a zoomed-in cross section are shown below.

<img src="https://raw.githubusercontent.com/zsylvester/meanderpy/master/fluvial_meanderpy_example_map.png" width="400">

<img src="https://raw.githubusercontent.com/zsylvester/meanderpy/master/fluvial_meanderpy_example_section.png" width="900">

## Google Colab notebook

If you don't want to deal with any local Python environments and installations, you should be able to run meanderpy in [this Google Colab notebook](https://colab.research.google.com/drive/1eZgGD_eXddaAeqxmI9guGIcTjjrLXmKO?usp=sharing).

## Related publications

If you use meanderpy in your work, please consider citing one or more of these publications:

Sylvester, Z., Durkin, P., and Covault, J.A., 2019, High curvatures drive river meandering: Geology, v. 47, p. 263–266, [doi:10.1130/G45608.1](https://doi.org/10.1130/G45608.1).

Sylvester, Z., and Covault, J.A., 2016, Development of cutoff-related knickpoints during early evolution of submarine channels: Geology, v. 44, p. 835–838, [doi:10.1130/G38397.1](https://doi.org/10.1130/G38397.1).

Covault, J.A., Sylvester, Z., Hubbard, S.M., and Jobe, Z.R., 2016, The Stratigraphic Record of Submarine-Channel Evolution: The Sedimentary Record, v. 14, no. 3, p. 4-11, [doi:10.2210/sedred.2016.3](https://www.sepm.org/files/143article.hqx9r9brxux8f2se.pdf).

Sylvester, Z., Pirmez, C., and Cantelli, A., 2011, A model of submarine channel-levee evolution based on channel trajectories: Implications for stratigraphic architecture: Marine and Petroleum Geology, v. 28, p. 716–727, [doi:10.1016/j.marpetgeo.2010.05.012](https://doi.org/10.1016/j.marpetgeo.2010.05.012).

## Acknowledgements

While the code in 'meanderpy' was written relatively recently, many of the ideas implemented in it come from numerous discussions with Carlos Pirmez, Alessandro Cantelli, Matt Wolinsky, Nick Howes, and Jake Covault. Funding for this work comes from the [Quantitative Clastics Laboratory industrial consortium](http://www.beg.utexas.edu/qcl) at the Bureau of Economic Geology, The University of Texas at Austin.

## License

meanderpy is licensed under the [Apache License 2.0](https://github.com/zsylvester/meanderpy/blob/master/LICENSE.txt).
