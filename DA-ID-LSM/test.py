# -*- coding: utf-8 -*-
# --- 1. IMPORTS ---
import cv2
import numpy as np
from utils.config import opt
from model.Trainer import InpTrainer
import scipy.io as io
import torch
import os
from scipy.io import savemat
from scipy.signal import butter, filtfilt
import scipy.signal

# --- 2. FUNCTIONS ---
def test_model(data, model):
    """
    Runs inference on a single data patch using the provided model.

    Args:
        data (np.ndarray): The input data patch, expected to be pre-processed.
        model (InpTrainer): The trainer object containing the loaded neural network.

    Returns:
        np.ndarray: The output patch from the model.
    """
    model.eval()
    data = torch.tensor(data)

    outputimg = model.test_onebatch(data.cuda())
    outimg = outputimg[0][0].detach().cpu().numpy()

    return outimg


# --- 3. MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    # opt._parse()
    
    # opt._parse() # A call to parse command-line arguments, currently commented out.
    
    # --- 1. DATA LOADING ---
    # Load the target area data (e.g., a large seismic image).
    nz = 1500
    nx = 1751  

    test_RTM = np.fromfile('./data/seam/seam_mig.dat', dtype=np.float32)
    test = test_RTM.reshape((nz, nx), order='F') 
    # --- 2. SETUP AND MODEL INITIALIZATION ---
    # Define the directory to save the final result.   
    saveroot = './testresult/'
    if not os.path.exists(saveroot):
        os.makedirs(saveroot)
    
    modelpath = './checkpoints4train/Network_36.pth'

    if not os.path.exists(modelpath):
        print("error")
    else:
        print(f"modelpath is:{modelpath}")
    
    model = InpTrainer(opt)
    model.load_net(modelpath)
    h, w = np.shape(test)
    cropsize = 128
    step = 16  
    numh = (h-cropsize-1)//step+2
    numw = (w-cropsize-1)//step+2
    result = np.zeros((h, w))
    weight = np.zeros((h, w))

    for hh in range(numh):
        for ww in range(numw):
            if hh == 0:
                hstar = 0
            elif hh == numh - 1:
                hstar = h - cropsize
            else:
                hstar = step * hh

            if ww == 0:
                wstar = 0
            elif ww == numw - 1:
                wstar = w - cropsize
            else:
                wstar = step * ww

            #------------------------------
            subdata = test[hstar:hstar+cropsize, wstar:wstar+cropsize]
            scale = np.max(np.abs(subdata))
            subdata = subdata / (scale+0.0000001)
            subdata = subdata.reshape((1, 1, cropsize, cropsize)).astype(np.float32)
            out = test_model(subdata, model)

            #-------------------------------
            if hh == 0:
                haa = 0
                hbb = hstar + cropsize
                haa1 = 0
                hbb1 = cropsize
            elif hh == numh - 1:
                haa = hstar
                hbb = h
                haa1 = 0
                hbb1 = cropsize
            else:
                haa = hstar
                hbb = hstar+cropsize
                haa1 = 0
                hbb1 = cropsize


            if ww == 0:
                waa = 0
                wbb = wstar + cropsize
                waa1 = 0
                wbb1 = cropsize
            elif ww == numw - 1:
                waa = wstar
                wbb = w
                waa1 = 0
                wbb1 = cropsize
            else:
                waa = wstar 
                wbb = wstar + cropsize 
                waa1 = 0
                wbb1 = cropsize

            result[haa:hbb,waa:wbb] = result[haa:hbb,waa:wbb]+out[haa1:hbb1, waa1:wbb1]*scale
            weight[haa:hbb,waa:wbb] = weight[haa:hbb,waa:wbb]+np.ones((hbb-haa,wbb-waa))

    # --- D. FINALIZE AND SAVE THE RESULT ---
    result = result/(weight+0.0000001)
    result[test == 0] = 0
    result = result.astype(np.float32)
       
    result = result.transpose()
    result.tofile(saveroot+'result_seam.dat')
