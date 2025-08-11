import torch.nn
from torch.utils.data import dataset
import numpy as np
from numpy import uint8
import random
from scipy.io import savemat
import scipy.signal
from torchvision import transforms as tfs
from torchvision import transforms
from PIL import Image
import scipy.misc
import torchvision.transforms.functional as TF
import random
from utils.mmd import MMDLoss
from typing import Sequence
class Dataset(dataset.Dataset):

    def  __init__(self, data, ref2, kernel,target, mode, cropsize, num):
        self.data = data
        self.kernel = kernel
        self.target = target
        self.datanum = len(data)
        self.ref2 = ref2
        self.mode = mode
        self.cropsize = cropsize
        self.cropsize1 = cropsize
        self.kernelsize_z = kernel.shape[0]
        self.kernelsize_x = kernel.shape[1]
        random1 = np.random.RandomState(0)

        self.seed = random1.randint(100000, size=num)



    def __len__(self):
        return len(self.seed)

    def __getitem__(self, index):
        idd = index%self.datanum
        data = self.data[idd]
        ref2 = self.ref2
        Kernel =self.kernel
        target = self.target

        h, w = np.shape(data)
        h1,w1 = np.shape(target)
        kernelsize_z =self.kernelsize_z
        kernelsize_x =self.kernelsize_x
        # hpsf = random.randint(0, 8)
        # wpsf = random.randint(0, 19)
        npsf = Kernel.shape[2]
        ipsf = random.randint(50,50)
        if self.mode == 'train':

            Kernel = Kernel[:,:,ipsf]
            Kernel = Kernel / (np.max(np.abs(Kernel)) + 0.000000001)
            
            hstar = random.randint(0, h-self.cropsize)
            wstar = random.randint(0, w-self.cropsize)
            data = data[hstar:hstar+self.cropsize, wstar:wstar+self.cropsize]
            data = data / (np.max(np.abs(data))+0.000000001)
            ref2 = ref2[hstar:hstar + self.cropsize, wstar:wstar + self.cropsize]
            ref2 = ref2 / (np.max(np.abs(ref2)) + 0.000000001)

            h1star = random.randint(0, h1 - self.cropsize1)
            w1star = random.randint(0, w1 - self.cropsize1)
            target = target[h1star:h1star + self.cropsize1, w1star:w1star + self.cropsize1]
            target = target / (np.max(np.abs(target)) + 0.000000001)
            data = scipy.signal.convolve(data, Kernel, mode='same',method='fft')
            data = data / (np.max(np.abs(data)) + 0.000000001)


            
            data = np.reshape(data, (1, self.cropsize, self.cropsize)).astype(np.float32)
            ref2 = np.reshape(ref2, (1, self.cropsize, self.cropsize)).astype(np.float32)
            target = np.reshape(target, (1, self.cropsize, self.cropsize)).astype(np.float32)

            return data, ref2,target
        
        else:
           
            Kernel = Kernel[:,:,ipsf]
            Kernel = Kernel / (np.max(np.abs(Kernel)) + 0.000000001)
            random1 = np.random.RandomState(self.seed[index])
            hstar = random1.randint(0, h - self.cropsize)
            wstar = random1.randint(0, w - self.cropsize)
            data = data[hstar:hstar + self.cropsize, wstar:wstar + self.cropsize]
            data = data / (np.max(np.abs(data)) + 0.000000001)
            ref2 = ref2[hstar:hstar + self.cropsize, wstar:wstar + self.cropsize]
            ref2 = ref2 / (np.max(np.abs(ref2)) + 0.000000001)

            h1star = random.randint(0, h1 - self.cropsize)
            w1star = random.randint(0, w1 - self.cropsize)

            target = target[h1star:h1star + self.cropsize, w1star:w1star + self.cropsize]
            target = target / (np.max(np.abs(target)) + 0.000000001)

            data = scipy.signal.convolve(data, Kernel, mode='same',method='fft')
            data = data / (np.max(np.abs(data)) + 0.000000001)        
            data2 = np.zeros((self.cropsize, self.cropsize))
            for tt in range(self.cropsize):
                data2[:, tt] = data[:, tt] + 0 * max(abs(data[:, tt])) * np.random.normal(0, 1, self.cropsize)
           
            data = np.reshape(data, (1, self.cropsize, self.cropsize)).astype(np.float32)
            ref2 = np.reshape(ref2, (1, self.cropsize, self.cropsize)).astype(np.float32)
            target = np.reshape(target, (1, self.cropsize, self.cropsize)).astype(np.float32)
            return data, ref2,target





