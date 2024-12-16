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
from scipy.io import savemat
import matplotlib.pyplot as plt
# SAVEROOT = './result/model_mar_581x1501/'
SAVEROOT = './result/'
if not os.path.exists(SAVEROOT):
    os.makedirs(SAVEROOT)



def test_model(dataloader, model, ifsave=True, test_num=1000, name='test'):
    num = 0
    model.eval()
    for ii, (data, ref) in enumerate(dataloader):
        outputimg = model.test_onebatch(data.cuda())
        outputimg = outputimg.detach().cpu()
        num += 1
        if ifsave:
            for i in range(1):
                inputimg = data[i][0].numpy()  ####
                refimg = ref[i][0].numpy()  ####
                outimg = outputimg[i][0].numpy()

                inputimg = ((inputimg + 1) * 255 / 2).astype(np.uint8)
                outimg = ((np.clip(outimg, -1, 1) + 1) * 255 / 2).astype(np.uint8)
                refimg = ((np.clip(refimg, -1, 1) + 1) * 255 / 2).astype(np.uint8)
                cv2.imwrite(SAVEROOT + name + 'outimg' + str(ii) + '.jpg', outimg)
                cv2.imwrite(SAVEROOT + name + 'inputimg' + str(ii) + '.jpg', inputimg)
                cv2.imwrite(SAVEROOT + name + 'labelimg' + str(ii) + '.jpg', refimg)

        if ii > test_num:
            break

    return


def train(data1, ref2,Kernel,target):
    opt._parse()
    for ee in range(5): 
          
        train_dataset = Dataset(data1, ref2, Kernel,target, 'train', 128, 4000)
        train_dataloader = data_.DataLoader(train_dataset,
                                            batch_size=20,
                                            shuffle=True,
                                           )

        test_dataset = Dataset(data1, ref2,Kernel,target,'test', 128, 2000)
        test_dataloader = data_.DataLoader(test_dataset,
                                           batch_size=20,
                                           shuffle=True)

        trainer = InpTrainer(opt)
        if opt.load_net:
            trainer.load_net(opt.load_net)
        print('model construct completed')

        if opt.test_only:
            eval_result1 = test_model(test_dataloader, trainer, ifsave=True, test_num=opt.test_num, name='test')
            print('eval_loss: ', eval_result1)
            return

        los = 0.0
        print("=====epoch %s start=====" % ee)
        for ii, (dd, rr, tt) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # ------------训练-------------
            trainer.train()
            loss = trainer.train_onebatch2(dd.cuda(), rr.cuda(),tt.cuda())
            los = los + loss.detach().cpu().numpy()

            if (ii + 1) % 40 == 0:
                print('loss:', loss.detach().cpu().numpy())

                test_model(test_dataloader, trainer, ifsave=True, test_num=opt.test_num, name='test')

                savemodel = trainer.save_net(best_map=los / 40)
                los = 0
                print("save to %s !" % savemodel)
        print("=====epoch %s end=====" % ee)
def interpolate_row(row):
    f = interp1d(np.arange(len(row)), row, kind='linear', fill_value='extrapolate')
    return f(np.linspace(0, len(row) - 1, 1600))

if __name__ == '__main__':

    m1root = []

    target = io.loadmat('./data/image.mat')["image"]
    target = target /np.max(np.abs(target))
    data1 = io.loadmat('./data/mar_ref_75hz.mat')["mar_ref"]
    data1 = data1 / np.max(np.abs(data1))
    Kernel = io.loadmat('./data/psf_3D.mat')["psf_3D_mig"]
    ref = io.loadmat('./data/mar_ref_75hz.mat')["mar_ref"]
    ref = ref / np.max(np.abs(ref))
    
    print(np.shape(ref))
    train([data1], ref,Kernel,target)

    
