# -*- coding: utf-8 -*-

# --- 1. IMPORTS ---
# Import necessary libraries.
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

# --- 2. GLOBAL CONFIGURATION ---
# Define the root directory to save results.
SAVEROOT = './result/'
# Create the directory if it doesn't already exist.
if not os.path.exists(SAVEROOT):
    os.makedirs(SAVEROOT)


# --- 3. FUNCTIONS ---

def test_model(dataloader, model, test_num, ifsave=True, name='test'):
    """
    Evaluates the model on the test dataset.

    Args:
        dataloader (DataLoader): The data loader for the test set.
        model (InpTrainer): The trainer object containing the model.
        test_num (int): The number of batches to test.
        ifsave (bool): If True, saves the output images.
        name (str): A prefix for the saved image filenames.
    """
    num = 0
    model.eval()
        # Iterate over the test dataloader.
    for ii, (data, ref,target) in enumerate(dataloader):  

        
        outputimg = model.test_onebatch(data.cuda())
        outputimg = outputimg.detach().cpu()
        num += 1
        if ifsave:
            for i in range(1):
                inputimg = data[i][0].numpy()  ####
                refimg = ref[i][0].numpy()  ####
                outimg = outputimg[i][0].numpy()
                tarimg = target[i][0].numpy()
                inputimg = ((inputimg + 1) * 255 / 2).astype(np.uint8)
                outimg = ((np.clip(outimg, -1, 1) + 1) * 255 / 2).astype(np.uint8)
                refimg = ((np.clip(refimg, -1, 1) + 1) * 255 / 2).astype(np.uint8)
                tarimg = ((np.clip(tarimg, -1, 1) + 1) * 255 / 2).astype(np.uint8)

                cv2.imwrite(SAVEROOT + name + 'outimg' + str(ii) + '.jpg', outimg)
                cv2.imwrite(SAVEROOT + name + 'inputimg' + str(ii) + '.jpg', inputimg)
                cv2.imwrite(SAVEROOT + name + 'labelimg' + str(ii) + '.jpg', refimg)
                cv2.imwrite(SAVEROOT + name + 'tragetimg' + str(ii) + '.jpg', tarimg)
        if ii > test_num:
            break

    return


def train(data1, ref2,Kernel,target):

    # opt._parse()
    for ee in range(1): 
        # --- Training Hyperparameters ---
        cropsize = 128   
        num_train_dataset = 5000  
        batch_size = 20   
        
        # --- Dataset and DataLoader Setup ---
        # Create the training dataset object.
        train_dataset = Dataset(data1, ref2, Kernel,target, 'train', cropsize, num_train_dataset) 
        train_dataloader = data_.DataLoader(train_dataset, batch_size, shuffle=True)

        num_test_dataset = 2000  
        test_dataset = Dataset(data1, ref2,Kernel,target,'test', cropsize, num_test_dataset)
        test_dataloader = data_.DataLoader(test_dataset, batch_size, shuffle=True)
        
        # --- Model Initialization ---
        # Initialize the trainer, which handles the model and optimization.
        trainer = InpTrainer(opt)
        if opt.load_net:
            trainer.load_net(opt.load_net)
        # print('model construct completed')

        if opt.test_only:
            eval_result1 = test_model(test_dataloader, trainer, test_num=opt.test_num, ifsave=True, name='test')
            print('eval_loss: ', eval_result1)
            return
            
        # --- Training Loop ---
        # Initialize an array to store loss values for each batch.
        # los = 0.0
        all_loss = np.zeros(int(num_train_dataset/batch_size), dtype=float, order='F')
        print("=====epoch %s start=====" % ee)

        # Loop through the training data using tqdm for a progress bar.
        for ii, (dd, rr, tt) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # ------------训练-------------
            trainer.train()
            loss = trainer.train_onebatch2(dd.cuda(), rr.cuda(),tt.cuda())
            los = loss.detach().cpu().numpy()
            all_loss[ii] = los
            save_flag = int(num_train_dataset/batch_size)/2
            if (ii + 1) % save_flag == 0:
                print('loss:', loss.detach().cpu().numpy())
                test_num = 50  
                test_model(test_dataloader, trainer, test_num, ifsave=True, name='test')

                savemodel = trainer.save_net(best_map=los)  
                print("save to %s !" % savemodel)
        print("=====epoch %s end=====" % ee)
        # all_loss_sm = savgol_filter(all_loss, 15, 2, mode='nearest')
        # all_loss_sm = gaussian_filter(all_loss, sigma=20)

        # --- Plotting ---
        # Plot the loss curve for the epoch.
        plt.plot(all_loss,color='red', linewidth=1.5)
        plt.title('Iterative loss function')
        plt.xlabel('Iteration')
        plt.ylabel('Loss Value')

if __name__ == '__main__':

    m1root = []
    
    # --- Data Loading and Preprocessing Module ---

    # 1. Load target area data (real-world data to be processed).
    nz = 1500  # sampling
    nx = 1751  # sampling
    # Load the Reverse Time Migration (RTM) image from a binary file.
    target_RTM = np.fromfile('./data/seam/seam_mig.dat', dtype=np.float32)
    target = target_RTM.reshape((nz, nx), order='F')      
    # target = io.loadmat('./data/rtm.mat')["rtm"]
    target = target /np.max(np.abs(target))
    
    # 2. Load and process Point Spread Functions (PSFs) for the target area.
    # The PSF describes the response of an imaging system to a point source.
    target_PSF1 = np.fromfile('./data/seam/PSF.dat', dtype=np.float32)
    PSF_2d1 = target_PSF1.reshape((21, 21,1), order='F')
    PSF_2d1 = PSF_2d1 /np.max(np.abs(PSF_2d1))
    Kernel1 = PSF_2d1
    # # --- Parameters for extracting PSF kernels ---
    # PSFLx = 21    # Central coordinate interval
    # PSFLz = 21   # Central coordinate interval
    # PSFSx = 11    # PSF_Scale 
    # PSFSz = 11    # PSF_Scale   
    # PSFNx = 84    # PSF_Number  
    # PSFNz = 72    # PSF_Number  
    # PSFBx = 11    # PSF_Initial  Central coordinate
    # PSFBz = 11    # PSF_Initial   Central coordinate
    # Kernel1 = np.zeros((PSFSz, PSFSx, PSFNx * PSFNz))
    # n = 0
    # for iz in range(PSFBz, PSFBz + PSFLz * PSFNz, PSFLz):
    #     for ix in range(PSFBx, PSFBx + PSFLx * PSFNx, PSFLx):
    #         n = n + 1
    #         Kernel1[:, :, n-1] = PSF_2d1[iz-5:iz+6, ix-5:ix+6]

    # 3. Load the true reflectivity model (ground truth).
    ref_1d = np.fromfile('./data/ref.dat', dtype=np.float32)   
    ref_2d = ref_1d.reshape((551, 2001), order='F')
    
    ref_2d = ref_2d /np.max(np.abs(ref_2d))

    
    # 4. Synthesize the training input data (the "label").

    f1, f2, f3, f4 = 3, 12, 60, 80   
    dt = 0.004  
    tlen = 1.0
    wavelet = ormsby_wavelet(f1, f2, f3, f4, dt, tlen)
    wavelet1 = wavelet.reshape(len(wavelet), 1)
    label_2d = scipy.signal.convolve2d(ref_2d, wavelet1, mode='same')
    
    # 5. Start the training process.
    # Note: Based on a typical deconvolution task, `label_2d` (convolved data) is the network input,
    # and `ref_2d` (true reflectivity) is the desired output/label.
    # The Dataset class likely handles mapping these inputs correctly.
    train([ref_2d], label_2d,Kernel1,target)

