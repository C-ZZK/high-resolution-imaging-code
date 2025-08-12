#! /usr/bin/env python3
# =============================================================================
#   Program Name : pythow.py
#   Introduction :
#   Parameters   :
#   Author       : Sheng Shen
#                   WPI, Tongji University
#   Date         : 2021.11.29
#   CopyRight    : 2021 -
# =============================================================================

import scipy.io as io
import scipy.io as scio
import scipy.signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.font_manager as fm
from matplotlib.ticker import MaxNLocator
import os
from scipy.signal import savgol_filter
import numpy as np
from scipy.signal import butter, filtfilt
import scipy.signal

# =============================================================================
# Part 1 : Parameter define
# =============================================================================
nz  = 1500
nx  = 1751
npz = 160
npx = 40
dz  = 8.0 / 1000
dx  = 8.0 / 1000
dt  = 0.004


# colormap  = scio.loadmat('../petrel.mat')['petrel']
# cmapPetrel= colors.ListedColormap(colormap, 'indexed')
cmap              = 'Greys' #  'RdGy_r' 'gist_yarg'binary
cmap              = 'Greys'
maxRatio          = 30.0
minRatio          = 15.0
flagvMaxEqualvMin = False   
flagVRangeEqual   = True   
flagFKRangeEqual  = False   

# 字体设置
label_fontsize = 10
ticks_fontsize = 10
dpi            = 200
vFigsize       = (8,6)
fkFigsize      = (3,3)



font_label = {'family': 'Arial',
              'weight': 'normal',
              'size': label_fontsize,
              }
fontType   = 'Arial'


flag_show = 1
flag_save = 0
fn_save   = '../pic/'

constrained_layout = True
axisType           = 'tight'


flag_part = 1
_xmin     = 0
_xmax     = 13.6
_zmin     = 0
_zmax     = 12
_xminloc  = int(_xmin / dx)
_xmaxloc  = int(_xmax / dx)


_zminloc  = int(_zmin / dz)
_zmaxloc  = int(_zmax / dz)
print(_zminloc)
print(_zmaxloc)
print(_xminloc)
print(_xmaxloc)
xMaxNLocator   = 5 
yMaxNLocator   = 5

class SN:
    def __init__(self):
        self._fk_xmin = ''
        self._fk_xmax = ''
        self._fk_zmin = ''
        self._fk_zmax = ''
        pass
    pass


fn_part = '1_' # '1_' , '2_'

part1 = SN()
part1._fk_xmin = 0
part1._fk_xmax = 6
part1._fk_zmin = 2
part1._fk_zmax = 6
part2 = SN()
part2._fk_xmin = 10
part2._fk_xmax = 20
part2._fk_zmin = 4.8
part2._fk_zmax = 6.8
if fn_part == '1_' :
    _fk_xmin = part1._fk_xmin
    _fk_xmax = part1._fk_xmax
    _fk_zmin = part1._fk_zmin
    _fk_zmax = part1._fk_zmax
    pass
if fn_part == '2_' :
    _fk_xmin = part2._fk_xmin
    _fk_xmax = part2._fk_xmax
    _fk_zmin = part2._fk_zmin
    _fk_zmax = part2._fk_zmax
    pass


_fxminloc  = int(_fk_xmin / dx)
_fxmaxloc  = int(_fk_xmax / dx)
_fzminloc  = int(_fk_zmin / dz)
_fzmaxloc  = int(_fk_zmax / dz)


line1_xloc = int(nx / 3)
line2_xloc = int(nx / 4 * 3)
print(line1_xloc)
print(line2_xloc)

iw         = np.linspace(0, 1.0 / dt, _zmaxloc - _zminloc)
ifw        = np.linspace(0, 1.0 / dt, _fzmaxloc - _fzminloc)
iz         = np.linspace(0, (nz - 1) * dz, nz)
x1         = np.linspace(line1_xloc * dx, line1_xloc * dx, nz)
x2         = np.linspace(line2_xloc * dx, line2_xloc * dx, nz)


print('# Step 1 : Parameters defination')
print('     [n1,n2] is : [', nz, ',', nx, ']')
print('     [d1,d2] is : [', dz, ',', dx, ']')

# =============================================================================
# Part 2 : Auto load data
# =============================================================================
# load data
# PSF
# fn      = '../psf/16Hz/PSF_nz676_nx786.dat'
# _psf    = np.fromfile(fn, dtype='float32')
# _psf    = np.reshape(_psf, [676, 786], order='F')
# psf     = np.zeros((676, 786))
# psf     = _psf
# psf[0:int(2.5 /(20*1e-3)),:].fill(0)
# psf     =  io.loadmat('./data/label_150Hz.mat')["label_150Hz"]
# psf     =  io.loadmat('./data/psf.mat')["psf"]
target_PSF = np.fromfile('./data/seam/final_matrix.dat', dtype=np.float32)
psf = target_PSF.reshape((nz, nx), order='F')
#psf = psf[150:701,:]
# cimg
# target_RTM = np.fromfile('./data/mig_4_d1_OBN_depth_01_55_ex3_sm20', dtype=np.float32)
target_RTM = np.fromfile('./data/seam/seam_mig.dat', dtype=np.float32)
cimg = target_RTM.reshape((nz, nx), order='F') 
#cimg    = cimg / np.max(np.abs(cimg))
# cimg = cimg[500:1201,:]

# dimg
# dimg    = io.loadmat('./data/lsm.mat')["lsm"]
target_LSM = np.fromfile('./data/ImgPSFdeblur.dat', dtype=np.float32)
dimg = target_LSM.reshape((nz, nx), order='F') 
#dimg = dimg / np.max(abs(dimg))
# dimg = dimg[500:1201,:]

# test = io.loadmat('./data/Marmousi2Ref.mat')["ref"]
# test = test[150:701,:]
# test = test / np.max(np.abs(test))

# emdimg
# target_result = np.fromfile('./testresult/result_d1.dat', dtype=np.float32)
target_result = np.fromfile('./testresult/result_seam.dat', dtype=np.float32)
emdimg = target_result.reshape((nz, nx), order='F') 


target_result_filt = np.fromfile('./data/seam/model_ref_1501x1751_8x8.dat', dtype=np.float32)
target_result_filt = target_result_filt.reshape((1501, nx), order='F') 
emdimg1 = target_result_filt[0:1500,:]



# vel
# target_vel = np.fromfile('./data/DN3D_VEL_line2100.dat', dtype=np.float32)
# vel = target_vel.reshape((nz, nx), order='F')
# fn_vel = '../vel/vel_psdm_nz676_nx786_dz20_dx20.bin'
# vel    = np.fromfile(fn_vel, dtype='float32')
# vel    = np.reshape(vel, [676, 786], order='F')
# emdimg = data1
# io.savemat('./testresult/20241218/filter/result_1_5_30_35.mat',{'result':emdimg})
# noise_data     =  io.loadmat('./testresult/resultadd5%cornoise.mat')["result"]
# noise_data     =  noise_data / np.max(abs(noise_data))
# noise_data     =  noise_data[0:321, :]
print('# Step 2 : Auto load data')
print('     cimg  : convolution profile')
print('     dimg  : deconvolution profile')
print('     emdimg  : deconvolution profile')
print('     vel  : velocity profile')


# trick 1 : zeros some part
nzeros              = _zmin
cimg[0:int(nzeros/dz), :].fill(0)
dimg[0:int(nzeros/dz), :].fill(0)
emdimg[0:int(nzeros/dz), :].fill(0)

# trick 2 : regularization
max_dimg = np.max(np.abs(dimg[_zminloc:_zmaxloc,_xminloc:_xmaxloc]))
emdimg   = emdimg / np.max(np.abs(emdimg[_zminloc:_zmaxloc,_xminloc:_xmaxloc])) * max_dimg
cimg     = cimg / np.max(np.abs(cimg[_zminloc:_zmaxloc,_xminloc:_xmaxloc])) * max_dimg



# # =============================================================================
# # Part 3 : Plot
# # =============================================================================     

# # show : cimg
plt.figure(figsize=vFigsize, dpi=dpi, constrained_layout=constrained_layout)
# vmin = cimg.min() / minRatio
# vmax = cimg.max() / maxRatio
vmin = cimg.min() /16
vmax = cimg.max() /30
# if flagvMaxEqualvMin:
#     vmax = vmin * -1
#     pass
im = plt.imshow(cimg, extent=(0, nx * dx, nz * dz, 0),
                vmin=vmin,  vmax=vmax, cmap=cmap)
# plt.plot(x1, iz, linestyle='--', color='green', linewidth=1.5)
# plt.plot(x2, iz, linestyle='--', color='green', linewidth=1.5)
plt.axis(axisType)
cbar = plt.colorbar()
cbar.set_label('Amplitude')
ax = plt.gca()
ax.xaxis.set_ticks_position('top')
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname(fontType) for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname(fontType) for y1_label_temp in y1_label]
ax.xaxis.set_label_position('top')
plt.xlabel('Distance (km)', font_label)
plt.ylabel('Depth (km)', font_label)
plt.xticks(fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)

plt.xlim(_xmin, _xmax)
plt.ylim(_zmax, _zmin)
# if flag_save == 1:
#     plt.savefig(fn_save + 'cimg.png')
#     pass

# # show : dimg
plt.figure(figsize=vFigsize, dpi=dpi, constrained_layout=constrained_layout)
# vmin = dimg.min() / minRatio
# vmax = dimg.max() / maxRatio
vmin = dimg.min() /16
vmax = dimg.max() /30
# if flagVRangeEqual:
#     vmin = dimg.min() / minRatio
#     vmax = dimg.max() / maxRatio
#     pass
# if flagvMaxEqualvMin:
#     vmax = vmin * -1
#     pass
im = plt.imshow( dimg, extent=(0, nx * dx, nz * dz, 0),
                 vmin=vmin,  vmax=vmax, cmap=cmap)
# plt.plot(x1, iz, linestyle='--', color='blue', linewidth=1.5)
# plt.plot(x2, iz, linestyle='--', color='blue', linewidth=1.5)
plt.axis(axisType)
ax = plt.gca()

ax.xaxis.set_ticks_position('top')
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
cbar = plt.colorbar()
cbar.set_label('Amplitude')
x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname(fontType) for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname(fontType) for y1_label_temp in y1_label]
ax.xaxis.set_label_position('top')
plt.xlabel('Distance (km)', font_label)
plt.ylabel('Depth (km)', font_label)
plt.xticks(fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
plt.xlim(_xmin, _xmax)
plt.ylim(_zmax, _zmin)
if flag_save == 1:
    plt.savefig(fn_save + 'dimg.png')
    pass

# # # show : emdimg
plt.figure(figsize=vFigsize, dpi=dpi, constrained_layout=constrained_layout)
vmin = emdimg.min() /16
vmax = emdimg.max() /30
# # if flagVRangeEqual:
# #     vmin = cimg.min() / minRatio
# #     vmax = cimg.max() / maxRatio
# #     pass
# # if flagvMaxEqualvMin:
# #     vmax = vmin * -1
# #     pass
im = plt.imshow( emdimg, extent=(0, nx * dx, nz * dz, 0),
                 vmin=vmin,  vmax=vmax, cmap=cmap)
# plt.plot(x1, iz, linestyle='--', color='red', linewidth=1.5)
# plt.plot(x2, iz, linestyle='--', color='red', linewidth=1.5)
plt.axis(axisType)

cbar = plt.colorbar()
cbar.set_label('Amplitude')
ax = plt.gca()

ax.xaxis.set_ticks_position('top')
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname(fontType) for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname(fontType) for y1_label_temp in y1_label]
ax.xaxis.set_label_position('top')
plt.xlabel('Distance (km)', font_label)
plt.ylabel('Depth (km)', font_label)
plt.xticks(fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
plt.xlim(_xmin, _xmax)
plt.ylim(_zmax, _zmin)
# if flag_save == 1:
#     plt.savefig(fn_save + 'emdimg.png')
#     pass

# # # show : emdimg1
plt.figure(figsize=vFigsize, dpi=dpi, constrained_layout=constrained_layout)
vmin = emdimg1.min() /8
vmax = emdimg1.max() /8
im = plt.imshow( emdimg1, extent=(0, nx * dx, nz * dz, 0),
                 vmin=vmin,  vmax=vmax, cmap=cmap)
# # plt.plot(x1, iz, linestyle='--', color='red', linewidth=1.5)
# # plt.plot(x2, iz, linestyle='--', color='red', linewidth=1.5)
plt.axis(axisType)
# plt.colorbar()
cbar = plt.colorbar()
cbar.set_label('Amplitude')
ax = plt.gca()

ax.xaxis.set_ticks_position('top')
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname(fontType) for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname(fontType) for y1_label_temp in y1_label]
ax.xaxis.set_label_position('top')
plt.xlabel('Distance (km)', font_label)
plt.ylabel('Depth (km)', font_label)
plt.xticks(fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
plt.xlim(_xmin, _xmax)
plt.ylim(_zmax, _zmin)
if flag_save == 1:
    plt.savefig(fn_save + 'emdimg1.png')
    pass




                 
