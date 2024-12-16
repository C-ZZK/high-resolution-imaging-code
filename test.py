import cv2
import numpy as np
from utils.config import opt
from model.Trainer import InpTrainer
import scipy.io as io
import torch
import os
from scipy.io import savemat
from scipy.signal import butter, filtfilt

def test_model(data, model):
    model.eval()
    data = torch.tensor(data)

    outputimg = model.test_onebatch(data.cuda())
    outimg = outputimg[0][0].detach().cpu().numpy()

    return outimg



if __name__ == '__main__':
    opt._parse()
    
    test = io.loadmat('./data/image.mat')["image"]
    test = test / np.max(np.abs(test))
    saveroot = './testresult/'
    # saveroot = './testresult/model_mar_581x1501'
    if not os.path.exists(saveroot):
        os.makedirs(saveroot)
    modelpath = './checkpointsfor_mar/Unet__0.011069189310073852.pth'
    model = InpTrainer(opt)
    model.load_net(modelpath)
    h, w = np.shape(test)
    print(h)
    print(w)
    cropsize = 128
    step = 16
    numh = (h-cropsize-1)//step+2
    numw = (w-cropsize-1)//step+2
    print(numh)
    print(numw)
    result = np.zeros((h, w))
    weight = np.zeros((h, w))
    r = 20

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
                hbb = hstar + cropsize - r
                haa1 = 0
                hbb1 = cropsize-r
            elif hh == numh - 1:
                haa = hstar + r
                hbb = h
                haa1 = r
                hbb1 = cropsize
            else:
                haa = hstar+r
                hbb = hstar+cropsize-r
                haa1 = r
                hbb1 = cropsize-r


            if ww == 0:
                waa = 0
                wbb = wstar + cropsize - r
                waa1 = 0
                wbb1 = cropsize - r
            elif ww == numw - 1:
                waa = wstar + r
                wbb = w
                waa1 = r
                wbb1 = cropsize
            else:
                waa = wstar + r
                wbb = wstar + cropsize - r
                waa1 = r
                wbb1 = cropsize-r

            result[haa:hbb,waa:wbb] = result[haa:hbb,waa:wbb]+out[haa1:hbb1, waa1:wbb1]*scale
            weight[haa:hbb,waa:wbb] = weight[haa:hbb,waa:wbb]+np.ones((hbb-haa,wbb-waa))


    result = result/(weight+0.0000001)
         
    io.savemat(saveroot+'deblur_mar.mat',{'result':result})
    inputimg = ((test+1)*255/2).astype(np.uint8)
    outimg = result / np.max(np.abs(result))
    outimg = ((outimg+1)*255/2).astype(np.uint8)

    # cv2.imwrite(saveroot +'output.jpg', outimg)
    # cv2.imwrite(saveroot + 'input.jpg', inputimg)
    
    # ref = io.loadmat('./data/mar_ref_75hz.mat')["mar_ref"]
    # ref = ref / np.max(np.abs(ref))
    # ref_cop = ((ref+1)*255/2).astype(np.uint8)
    
    