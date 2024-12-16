import  torch
import numpy as nn
from torch.autograd import Variable

input=torch.ones(1,3,5,5)
input=Variable(input)
x=torch.nn.Conv2d(in_channels=3,out_channels=2,kernel_size=3,groups=1)
out=x(input)
print('filters and bias',list(x.parameters()))
print(('out',out))
# print('input',input)
f_p=list(x.parameters())[0]
f_p=f_p.data.numpy()
# f_p1=list(x.parameters())[0][1]
# f_p1=f_p1.data.numpy()
print('channel 1',f_p[0][:].sum())
print('channel 2',f_p[1][:].sum())

# print('22',f_p1[0:2].sum())
# print("the result of first channel in image:", f_p[0].sum()+(-0.1104))