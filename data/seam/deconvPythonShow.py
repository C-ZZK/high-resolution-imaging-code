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

# 颜色设置
# colormap  = scio.loadmat('../petrel.mat')['petrel']
# cmapPetrel= colors.ListedColormap(colormap, 'indexed')
cmap              = 'Greys' #  'RdGy_r' 'gist_yarg'binary
cmap              = 'Greys'
maxRatio          = 30.0
minRatio          = 15.0
flagvMaxEqualvMin = False   # img和FK，colorbar的最大值和最小值的负数相同,作用于maxRtio和minRatio后
flagVRangeEqual   = True    # img，不同图的colorbar范围相同
flagFKRangeEqual  = False   # FK谱，不同图的colorbar范围相同

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

# 展示&保存位置
flag_show = 1
flag_save = 0
fn_save   = '../pic/'

constrained_layout = True
axisType           = 'tight'

# 放大展示部分[_xmin,_xmax   _ymin,_ymax] 米
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
# 频谱展示区域
xMaxNLocator   = 5 # xy刻度个数
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


# 辅助向量
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

# fs = 1 / 0.004
# dt = 0.004
# emdimg1 =np.zeros((nz,nx),dtype=np.float64)
# # 设计滤波器
# fc = 1  # 截止频率为1Hz，即高通滤波器
# Wn = 2 * fc / fs
# b, a = butter(4, Wn, btype='high')
# for i in range(nx):
#     emdimg1[:, i] = filtfilt(b, a, emdimg[:, i])  # 按列进行滤波
# emdimg = emdimg1

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
# show : psf
# plt.figure(figsize=vFigsize, dpi=dpi, constrained_layout=constrained_layout)
# vmin = psf.min() / 16
# vmax = psf.max() / 30
# # _tmp = psf / np.max(abs(psf)) * 3.0  + vel / np.max(abs(vel))
# # _tmp = _tmp / np.max(_tmp) * np.max(abs(vel))
# # vmin = _tmp.min() / 1.5
# # vmax = _tmp.max() / 2.0
# im = plt.imshow(psf, extent=(0, nx * dx, nz * dz, 0),
#                  vmin=vmin,  vmax=vmax, cmap=cmap)
# plt.axis(axisType)
# ax = plt.gca()
# plt.colorbar()

# ax.xaxis.set_ticks_position('top')
# ax.tick_params(axis="x", direction="in")
# ax.tick_params(axis="y", direction="in")
# x1_label = ax.get_xticklabels()
# [x1_label_temp.set_fontname(fontType) for x1_label_temp in x1_label]
# y1_label = ax.get_yticklabels()
# [y1_label_temp.set_fontname(fontType) for y1_label_temp in y1_label]
# ax.xaxis.set_label_position('top')
# plt.xlabel('Distance (km)', font_label)
# plt.ylabel('Depth (km)', font_label)
# plt.xticks(fontsize=ticks_fontsize)
# plt.yticks(fontsize=ticks_fontsize)
# plt.xlim(_xmin, _xmax)
# plt.ylim(_zmax, _zmin)
# if flag_save == 1:
#     plt.savefig(fn_save + 'psf.png')
#     pass


# # show : vel
# plt.figure(figsize=vFigsize, dpi=dpi, constrained_layout=constrained_layout)
# _vel = vel 
# vmin = _vel.min() 
# vmax = _vel.max() 
# im = plt.imshow(_vel, extent=(0, nx * dx, nz * dz, 0),
#                 vmin=vmin,  vmax=vmax, cmap='jet')
# plt.axis(axisType)
# # plt.colorbar()
# ax = plt.gca()
# ax.xaxis.set_ticks_position('top')
# ax.tick_params(axis="x", direction="in")
# ax.tick_params(axis="y", direction="in")
# x1_label = ax.get_xticklabels()
# [x1_label_temp.set_fontname(fontType) for x1_label_temp in x1_label]
# y1_label = ax.get_yticklabels()
# [y1_label_temp.set_fontname(fontType) for y1_label_temp in y1_label]
# ax.xaxis.set_label_position('top')
# plt.xlabel('Distance (km)', font_label)
# plt.ylabel('Depth (km)', font_label)
# plt.xticks(fontsize=ticks_fontsize)
# plt.yticks(fontsize=ticks_fontsize)
# plt.xlim(_xmin, _xmax)
# plt.ylim(_zmax, _zmin)
# if flag_save == 1:
#     plt.savefig(fn_save + 'vel.png')
#     pass

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
# # 默认框的颜色是黑色，第一个参数是左上角的点坐标 第二个参数是宽，第三个参数是长
# ax.add_patch(plt.Rectangle((part1._fk_xmin, part1._fk_zmin),
#      part1._fk_xmax-part1._fk_xmin, part1._fk_zmax-part1._fk_zmin,
#      color="purple", fill=False, linewidth=2, linestyle='--'))
# # ax.add_patch(plt.Rectangle((part2._fk_xmin, part2._fk_zmin),
# #      part2._fk_xmax-part2._fk_xmin, part2._fk_zmax-part2._fk_zmin,
# #      color="yellow", fill=False, linewidth=3.5, linestyle='--'))
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
# # # 默认框的颜色是黑色，第一个参数是左上角的点坐标 第二个参数是宽，第三个参数是长
# ax.add_patch(plt.Rectangle((part1._fk_xmin, part1._fk_zmin),
#      part1._fk_xmax-part1._fk_xmin, part1._fk_zmax-part1._fk_zmin,
#      color="purple", fill=False, linewidth=2, linestyle='--'))
# # ax.add_patch(plt.Rectangle((part2._fk_xmin, part2._fk_zmin),
# #      part2._fk_xmax-part2._fk_xmin, part2._fk_zmax-part2._fk_zmin,
# #      color="yellow", fill=False, linewidth=3.5, linestyle='--'))
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
# 默认框的颜色是黑色，第一个参数是左上角的点坐标 第二个参数是宽，第三个参数是长
# ax.add_patch(plt.Rectangle((part1._fk_xmin, part1._fk_zmin),
#      part1._fk_xmax-part1._fk_xmin, part1._fk_zmax-part1._fk_zmin,
#      color="purple", fill=False, linewidth=2, linestyle='--'))
# # ax.add_patch(plt.Rectangle((part2._fk_xmin, part2._fk_zmin),
# #      part2._fk_xmax-part2._fk_xmin, part2._fk_zmax-part2._fk_zmin,
# #      color="yellow", fill=False, linewidth=3.5, linestyle='--'))
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
# # 默认框的颜色是黑色，第一个参数是左上角的点坐标 第二个参数是宽，第三个参数是长
# ax.add_patch(plt.Rectangle((part1._fk_xmin, part1._fk_zmin),
#      part1._fk_xmax-part1._fk_xmin, part1._fk_zmax-part1._fk_zmin,
#      color="purple", fill=False, linewidth=3.5, linestyle='--'))
# # ax.add_patch(plt.Rectangle((part2._fk_xmin, part2._fk_zmin),
# #      part2._fk_xmax-part2._fk_xmin, part2._fk_zmax-part2._fk_zmin,
# #      color="yellow", fill=False, linewidth=3.5, linestyle='--'))
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


# # # show : frequency spectrum 1

# plt.figure(figsize=(8, 5), dpi=dpi, constrained_layout=constrained_layout)
# ccimg   = np.linspace(0, 0, _fzmaxloc - _fzminloc)
# # cdimg   = np.linspace(0, 0, _fzmaxloc - _fzminloc)
# cemdimg = np.linspace(0, 0, _fzmaxloc - _fzminloc)
# # cemdimg1 = np.linspace(0, 0, _fzmaxloc - _fzminloc)
# for ix in range(_fxminloc, _fxmaxloc):
#     ccimg   = ccimg + np.abs(np.fft.fft(cimg[_fzminloc:_fzmaxloc, ix]))
#     # cdimg   = cdimg + np.abs(np.fft.fft(dimg[_fzminloc:_fzmaxloc, ix]))
#     cemdimg = cemdimg + np.abs(np.fft.fft(emdimg[_fzminloc:_fzmaxloc, ix]))
#     # cemdimg1 = cemdimg1 + np.abs(np.fft.fft(emdimg1[_fzminloc:_fzmaxloc, ix]))
#     pass
# ccimg   = ccimg / np.max(ccimg)
# # cdimg   = cdimg / np.max(cdimg)
# cemdimg = cemdimg / np.max(cemdimg)
# # cemdimg1 = cemdimg1 / np.max(cemdimg1)
# plt.plot(ifw, ccimg, color='black', linewidth=1.5, label='image')
# # plt.plot(ifw, cdimg, color='blue', linewidth=1.5, label='lsm')
# plt.plot(ifw, cemdimg, color='red', linewidth=1.5, label='deconvolution')
# # plt.plot(ifw, cemdimg1, color='yellow', linewidth=1.5, label='deconvolution_filt')
# ax = plt.gca()
# ax.tick_params(axis="x", direction="in")
# ax.tick_params(axis="y", direction="in")
# x1_label = ax.get_xticklabels()
# [x1_label_temp.set_fontname(fontType) for x1_label_temp in x1_label]
# y1_label = ax.get_yticklabels()
# [y1_label_temp.set_fontname(fontType) for y1_label_temp in y1_label]
# plt.legend(prop=font_label)
# plt.axis(axisType)
# plt.xlabel('Frequency (Hz)', font_label)
# plt.ylabel('Normalized Amplitude', font_label)
# # plt.ylabel('Amplitude', font_label)
# plt.xticks(fontsize=ticks_fontsize)
# plt.yticks(fontsize=ticks_fontsize)
# plt.xlim(0, 80)
# plt.grid()
# if flag_save == 1:
#     plt.savefig(fn_save + '2frequencyspectrum.png')
#     pass



# # # # show : frequency spectrum 2
# # # emdimg1=data1
# plt.figure(figsize=(8,5), dpi=dpi, constrained_layout=constrained_layout)
# ccimg   = np.linspace(0, 0, _zmaxloc - _zminloc)
# cdimg   = np.linspace(0, 0, _zmaxloc - _zminloc)
# cemdimg = np.linspace(0, 0, _zmaxloc - _zminloc)
# cemdimg1 = np.linspace(0, 0, _zmaxloc - _zminloc)
# for ix in range(_xminloc, _xmaxloc):
#     ccimg   = ccimg + np.abs(np.fft.fft(cimg[_zminloc:_zmaxloc, ix]))
#     cdimg   = cdimg + np.abs(np.fft.fft(dimg[_zminloc:_zmaxloc, ix]))
#     cemdimg = cemdimg + np.abs(np.fft.fft(emdimg[_zminloc:_zmaxloc, ix]))
#     cemdimg1 = cemdimg1 + np.abs(np.fft.fft(emdimg1[_zminloc:_zmaxloc, ix]))
#     pass
# ccimg   = ccimg / np.max(ccimg)
# cdimg   = cdimg / np.max(cdimg)
# cemdimg = cemdimg / np.max(cemdimg)
# cemdimg1 = cemdimg1 / np.max(cemdimg1)
# # ccimg = savgol_filter(ccimg, 15, 3, mode= 'nearest')
# # cdimg = savgol_filter(cdimg, 15, 3, mode= 'nearest')
# # cemdimg = savgol_filter(cemdimg, 15, 3, mode= 'nearest')
# plt.plot(iw, ccimg, color='black', linewidth=2, label='image')
# plt.plot(iw, cdimg, color='blue', linewidth=2, label='lsm')
# plt.plot(iw, cemdimg, color='red', linewidth=2, label='deconvolution')
# plt.plot(iw, cemdimg1, color='yellow', linewidth=2, label='deconv_ref')
# ax = plt.gca()
# ax.tick_params(axis="x", direction="in")
# ax.tick_params(axis="y", direction="in")
# x1_label = ax.get_xticklabels()
# [x1_label_temp.set_fontname(fontType) for x1_label_temp in x1_label]
# y1_label = ax.get_yticklabels()
# [y1_label_temp.set_fontname(fontType) for y1_label_temp in y1_label]
# plt.legend(prop=font_label)
# plt.axis(axisType)
# plt.xlabel('Frequency (Hz)', font_label)
# plt.ylabel('Normalized Amplitude', font_label)
# # plt.ylabel('Amplitude', font_label)
# plt.xticks(fontsize=ticks_fontsize)
# plt.yticks(fontsize=ticks_fontsize)
# plt.xlim(0, 80)
# plt.grid()
# # if flag_save == 1:
# #     plt.savefig(fn_save + 'frequencyspectrum.png')
# #     pass



# # # # show : frequency spectrum psf
# plt.figure(figsize=(8,5), dpi=dpi, constrained_layout=constrained_layout)
# ccimg   = np.linspace(0, 0, _zmaxloc - _zminloc)
# cpsf   = np.linspace(0, 0, _zmaxloc - _zminloc)
# for ix in range(_xminloc, _xmaxloc):
#     ccimg   = ccimg + np.abs(np.fft.fft(cimg[_zminloc:_zmaxloc, ix]))
#     cpsf   = cpsf + np.abs(np.fft.fft(psf[_zminloc:_zmaxloc, ix]))
#     pass
# ccimg   = ccimg / np.max(ccimg)
# cpsf   = cpsf / np.max(cpsf)
# # ccimg = savgol_filter(ccimg, 15, 3, mode= 'nearest')
# # cdimg = savgol_filter(cdimg, 15, 3, mode= 'nearest')
# # cemdimg = savgol_filter(cemdimg, 15, 3, mode= 'nearest')
# plt.plot(iw, ccimg, color='black', linewidth=2, label='rtm')
# plt.plot(iw, cpsf, color='blue', linewidth=2, label='psf')
# #plt.plot(iw, cemdimg, color='red', linewidth=2, label='deconv_withMMD')
# ax = plt.gca()
# ax.tick_params(axis="x", direction="in")
# ax.tick_params(axis="y", direction="in")
# x1_label = ax.get_xticklabels()
# [x1_label_temp.set_fontname(fontType) for x1_label_temp in x1_label]
# y1_label = ax.get_yticklabels()
# [y1_label_temp.set_fontname(fontType) for y1_label_temp in y1_label]
# plt.legend(prop=font_label)
# plt.axis(axisType)
# plt.xlabel('Frequency (Hz)', font_label)
# plt.ylabel('Normalized Amplitude', font_label)
# # plt.ylabel('Amplitude', font_label)
# plt.xticks(fontsize=ticks_fontsize)
# plt.yticks(fontsize=ticks_fontsize)
# plt.xlim(0, 150)
# plt.grid()
# if flag_save == 1:
#     plt.savefig(fn_save + 'frequencyspectrum.png')
#     pass





# # show : trace compare
# plt.figure(figsize=vFigsize, dpi=dpi, constrained_layout=constrained_layout)
# plt.subplot(2, 1, 1)
# line_xloc = line1_xloc

# iiz = np.linspace(5,10,501)
# # print(iiz)
# plt.plot(iiz, cimg[500:1001, line_xloc],
#          linewidth=2.0, color='black',label='image')
# # plt.plot(iiz, dimg[500:1000, line_xloc],
# #          linewidth=2.0, color='blue', alpha=0.6,label='lsm')
# plt.plot(iiz, emdimg[500:1001, line_xloc],
#          linewidth=2.0, color='red', label='deconv')
# # plt.title(title, fontsize=label_fontsize)
# plt.yticks([])
# plt.xlabel('Depth (km)', font_label)
# plt.ylabel('Amplitude', font_label)
# plt.tick_params(labelsize=ticks_fontsize)
# ax = plt.gca()
# ax.tick_params(axis="x", direction="in")
# ax.tick_params(axis="y", direction="in")
# x1_label = ax.get_xticklabels()
# [x1_label_temp.set_fontname(fontType) for x1_label_temp in x1_label]
# y1_label = ax.get_yticklabels()
# [y1_label_temp.set_fontname(fontType) for y1_label_temp in y1_label]
# ax.xaxis.set_ticks_position('top')
# ax.xaxis.set_label_position('top')
# plt.grid()
# plt.legend(fontsize=ticks_fontsize , loc=1)
# # plt.xlim(6, 9)
# plt.subplot(2, 1, 2)
# line_xloc = line2_xloc
# plt.plot(iiz, cimg[500:1001, line_xloc],
#          linewidth=2.0, color='black', label='image')
# # plt.plot(iiz, dimg[500:1000, line_xloc],
# #          linewidth=2.0, color='blue', label='lsm')
# plt.plot(iiz, emdimg[500:1001, line_xloc],
#          linewidth=2.0, color='red', label='deconv')
# plt.axis(axisType)
# # plt.title(title, fontsize=label_fontsize)
# plt.xlabel('Depth (km)', font_label)
# plt.ylabel('Amplitude', font_label)
# plt.tick_params(labelsize=ticks_fontsize)
# ax = plt.gca()
# ax.tick_params(axis="x", direction="in")
# ax.tick_params(axis="y", direction="in")
# x1_label = ax.get_xticklabels()
# [x1_label_temp.set_fontname(fontType) for x1_label_temp in x1_label]
# y1_label = ax.get_yticklabels()
# [y1_label_temp.set_fontname(fontType) for y1_label_temp in y1_label]
# plt.grid()
# plt.legend(fontsize=ticks_fontsize , loc=1)
# plt.xlim(6, 9)
# if flag_save == 1:
#     plt.savefig(fn_save + 'tracecompare.png')
#     pass


# #show : cimg_part
# plt.figure(figsize=fkFigsize, dpi=dpi, constrained_layout=constrained_layout)
# vmin = cimg.min() /16
# vmax = cimg.max() /30
# if flagvMaxEqualvMin:
#     vmax = vmin * -1
#     pass
# im = plt.imshow( cimg, extent=(0, nx * dx, nz * dz, 0),
#                  vmin=vmin,  vmax=vmax, cmap=cmap)
# #plt.colorbar(im, orientation='vertical')
# plt.axis(axisType)
# ax = plt.gca()
# ax.xaxis.set_ticks_position('top')
# ax.tick_params(axis="x", direction="in")
# ax.tick_params(axis="y", direction="in")
# x1_label = ax.get_xticklabels()
# [x1_label_temp.set_fontname(fontType) for x1_label_temp in x1_label]
# y1_label = ax.get_yticklabels()
# [y1_label_temp.set_fontname(fontType) for y1_label_temp in y1_label]
# ax.xaxis.set_label_position('top')
# plt.xlabel('Distance (km)', font_label)
# plt.ylabel('Depth (km)', font_label)
# plt.xticks(fontsize=ticks_fontsize)
# plt.yticks(fontsize=ticks_fontsize)
# plt.xlim(_fk_xmin, _fk_xmax)
# plt.ylim(_fk_zmax, _fk_zmin)
# plt.gca().xaxis.set_major_locator(MaxNLocator(xMaxNLocator))
# plt.gca().yaxis.set_major_locator(MaxNLocator(yMaxNLocator))
# if flag_save == 1:
#     plt.savefig(fn_save + fn_part + 'cimgpart.png')
#     pass

# # # show : dimg_part
# plt.figure(figsize=fkFigsize, dpi=dpi, constrained_layout=constrained_layout)
# vmin = dimg.min() /16
# vmax = dimg.max() /30
# if flagvMaxEqualvMin:
#     vmax = vmin * -1
#     pass
# im = plt.imshow( dimg, extent=(0, nx * dx, nz * dz, 0),
#                  vmin=vmin,  vmax=vmax, cmap=cmap)
# # plt.plot(x1, iz, linestyle='--', color='red', linewidth=1.5)
# # plt.plot(x2, iz, linestyle='--', color='red', linewidth=1.5)
# plt.axis(axisType)
# ax = plt.gca()
# ax.xaxis.set_ticks_position('top')
# ax.tick_params(axis="x", direction="in")
# ax.tick_params(axis="y", direction="in")
# x1_label = ax.get_xticklabels()
# [x1_label_temp.set_fontname(fontType) for x1_label_temp in x1_label]
# y1_label = ax.get_yticklabels()
# [y1_label_temp.set_fontname(fontType) for y1_label_temp in y1_label]
# ax.xaxis.set_label_position('top')
# plt.xlabel('Distance (km)', font_label)
# plt.ylabel('Depth (km)', font_label)
# plt.xticks(fontsize=ticks_fontsize)
# plt.yticks(fontsize=ticks_fontsize)
# plt.xlim(_fk_xmin, _fk_xmax)
# plt.ylim(_fk_zmax, _fk_zmin)
# plt.gca().xaxis.set_major_locator(MaxNLocator(xMaxNLocator))
# plt.gca().yaxis.set_major_locator(MaxNLocator(yMaxNLocator))
# if flag_save == 1:
#     plt.savefig(fn_save + fn_part + 'dimgpart.png')
#     pass


# show : emdimg_part
# plt.figure(figsize=fkFigsize, dpi=dpi, constrained_layout=constrained_layout)
# vmin = emdimg1.min() /10
# vmax = emdimg1.max() /10
# # if flagVRangeEqual:
# #     vmin = cimg.min() / minRatio
# #     vmax = cimg.max() / maxRatio
# #     pass
# # if flagvMaxEqualvMin:
# #     vmax = vmin * -1
# #     pass
# im = plt.imshow( emdimg1, extent=(0, nx * dx, nz * dz, 0),
#                  vmin=vmin,  vmax=vmax, cmap=cmap)
# # plt.plot(x1, iz, linestyle='--', color='red', linewidth=1.5)
# # plt.plot(x2, iz, linestyle='--', color='red', linewidth=1.5)
# plt.axis(axisType)
# ax = plt.gca()
# ax.xaxis.set_ticks_position('top')
# ax.tick_params(axis="x", direction="in")
# ax.tick_params(axis="y", direction="in")
# x1_label = ax.get_xticklabels()
# [x1_label_temp.set_fontname(fontType) for x1_label_temp in x1_label]
# y1_label = ax.get_yticklabels()
# [y1_label_temp.set_fontname(fontType) for y1_label_temp in y1_label]
# ax.xaxis.set_label_position('top')
# plt.xlabel('Distance (km)', font_label)
# plt.ylabel('Depth (km)', font_label)
# plt.xticks(fontsize=ticks_fontsize)
# plt.yticks(fontsize=ticks_fontsize)
# plt.xlim(_fk_xmin, _fk_xmax)
# plt.ylim(_fk_zmax, _fk_zmin)
# plt.gca().xaxis.set_major_locator(MaxNLocator(xMaxNLocator))
# plt.gca().yaxis.set_major_locator(MaxNLocator(yMaxNLocator))
# if flag_save == 1:
#     plt.savefig(fn_save + fn_part + 'emdimgpart.png')
#     pass


# ratio = 5.0
# cmap = 'jet'
# # show : cimg_part fk
# cnwz = int((_fk_zmax - _fk_zmin) / dz)
# cnwx = int((_fk_xmax - _fk_xmin) / dx)
# _zhub = int((_fk_zmin + _fk_zmax) / 2 / dz)
# _xhub = int((_fk_xmax + _fk_xmin) / 2 / dx)
# part = np.zeros((cnwz, cnwx))
# part = cimg[_zhub - int(cnwz / 2): _zhub + int(cnwz / 2),
#             _xhub - int(cnwx / 2 + 0.5): _xhub + int(cnwx / 2 + 0.5)]
# cpart = np.fft.fftn(part)
# cpart = np.fft.fftshift(cpart)
# abscpart = np.abs(cpart)
# abscpart = abscpart - np.min(abscpart)
# abscpart = abscpart / np.max(abscpart)

# plt.figure(figsize=fkFigsize, dpi=dpi, constrained_layout=constrained_layout)
# vmin = abscpart.min() / minRatio
# #vmax = abscpart.max() / maxRatio
# vmax = abscpart.max() / 6
# im = plt.imshow(abscpart,
#                 extent=(-1.0 / dx / 1000 / 2, 1.0 / dx / 1000 / 2,
#                         1.0 / dz / 1000 / 2, -1.0 / dz / 1000 / 2),
#                 vmin=vmin,  vmax=vmax, cmap=cmap)
# #plt.colorbar(im, orientation='vertical')
# plt.axis(axisType)
# plt.grid(color='grey', linewidth=1.0, linestyle='--')
# ax = plt.gca()
# ax.xaxis.set_ticks_position('top')
# ax.tick_params(axis="x", direction="in")
# ax.tick_params(axis="y", direction="in")
# x1_label = ax.get_xticklabels()
# [x1_label_temp.set_fontname(fontType) for x1_label_temp in x1_label]
# y1_label = ax.get_yticklabels()
# [y1_label_temp.set_fontname(fontType) for y1_label_temp in y1_label]
# ax.xaxis.set_label_position('top')
# plt.xlabel('Kx (m$^{-1}$)', font_label)
# plt.ylabel('Kz (m$^{-1}$)', font_label)
# plt.xticks(fontsize=ticks_fontsize)
# plt.yticks(fontsize=ticks_fontsize)
# plt.xlim(-1.0 / dx / 1000 / 2, 1.0 / dx / 1000 / 2)
# plt.ylim(1.0 / dz  / 1000 / 2 * 0.7, -1.0 / dz / 1000 / 2 * 0.7)
# plt.gca().xaxis.set_major_locator(MaxNLocator(xMaxNLocator))
# plt.gca().yaxis.set_major_locator(MaxNLocator(yMaxNLocator))
# ax.ticklabel_format(style='plain', scilimits=(-1,2), axis='both')
# if flag_save == 1:
#     plt.savefig(fn_save + fn_part + 'cimgpart_fk.png')
#     pass

# # show : dimg_part fk
# cnwz = int((_fk_zmax - _fk_zmin) / dz)
# cnwx = int((_fk_xmax - _fk_xmin) / dx)
# _zhub = int((_fk_zmin + _fk_zmax) / 2 / dz)
# _xhub = int((_fk_xmax + _fk_xmin) / 2 / dx)
# part = np.zeros((cnwz, cnwx))
# part = dimg[_zhub - int(cnwz / 2): _zhub + int(cnwz / 2),
#             _xhub - int(cnwx / 2 + 0.5): _xhub + int(cnwx / 2 + 0.5)]
# cpart = np.fft.fftn(part)
# cpart = np.fft.fftshift(cpart)
# abscpart = np.abs(cpart)
# abscpart = abscpart - np.min(abscpart)
# abscpart = abscpart / np.max(abscpart)
# plt.figure(figsize=fkFigsize, dpi=dpi, constrained_layout=constrained_layout)
# vmin = abscpart.min() / minRatio
# #vmax = abscpart.max() / maxRatio
# vmax = abscpart.max() / 6
# #if flagFKRangeEqual:
# #    vmin = vmin
# #    vmax = vmax
# #else:
# #    vmin = abscpart.min() / minRatio
# #    vmax = abscpart.max() / maxRatio
# #   pass
# im = plt.imshow(abscpart,
#                 extent=(-1.0 / dx / 1000 / 2, 1.0 / dx / 1000 / 2,
#                         1.0 / dz / 1000 / 2, -1.0 / dz / 1000 / 2),
#                 vmin=vmin,  vmax=vmax, cmap=cmap)
# #plt.colorbar(im, orientation='vertical')
# plt.grid(color='grey', linewidth=1.0, linestyle='--')
# plt.axis(axisType)
# ax = plt.gca()
# ax.xaxis.set_ticks_position('top')
# ax.tick_params(axis="x", direction="in")
# ax.tick_params(axis="y", direction="in")
# x1_label = ax.get_xticklabels()
# [x1_label_temp.set_fontname(fontType) for x1_label_temp in x1_label]
# y1_label = ax.get_yticklabels()
# [y1_label_temp.set_fontname(fontType) for y1_label_temp in y1_label]
# ax.xaxis.set_label_position('top')
# plt.xlabel('Kx (m$^{-1}$)', font_label)
# plt.ylabel('Kz (m$^{-1}$)', font_label)
# plt.xticks(fontsize=ticks_fontsize)
# plt.yticks(fontsize=ticks_fontsize)
# plt.xlim(-1.0 / dx / 1000 / 2, 1.0 / dx / 1000 / 2)
# plt.ylim(1.0 / dz  / 1000 / 2 * 0.7, -1.0 / dz / 1000 / 2 * 0.7)
# plt.gca().xaxis.set_major_locator(MaxNLocator(xMaxNLocator))
# plt.gca().yaxis.set_major_locator(MaxNLocator(yMaxNLocator))
# ax.ticklabel_format(style='plain', scilimits=(-1,2), axis='both')
# if flag_save == 1:
#     plt.savefig(fn_save + fn_part + 'dimgpart_fk.png')
#     pass


# # show : emdimg_part fk
# cnwz = int((_fk_zmax - _fk_zmin) / dz)
# cnwx = int((_fk_xmax - _fk_xmin) / dx)
# _zhub = int((_fk_zmin + _fk_zmax) / 2 / dz)
# _xhub = int((_fk_xmax + _fk_xmin) / 2 / dx)
# part = np.zeros((cnwz, cnwx))
# part = emdimg1[_zhub - int(cnwz / 2): _zhub + int(cnwz / 2),
#               _xhub - int(cnwx / 2 + 0.5): _xhub + int(cnwx / 2 + 0.5)]
# cpart = np.fft.fftn(part)
# cpart = np.fft.fftshift(cpart)
# abscpart = np.abs(cpart)
# abscpart = abscpart - np.min(abscpart)
# abscpart = abscpart / np.max(abscpart)
# plt.figure(figsize=fkFigsize, dpi=dpi, constrained_layout=constrained_layout)
# vmin = abscpart.min() / 20
# #vmax = abscpart.max() / maxRatio
# vmax = abscpart.max() / 30
# #vmin = vmin / 1.6
# #vmax = vmax / 1.6
# im = plt.imshow(abscpart,
#                 extent=(-1.0 / dx / 1000 / 2, 1.0 / dx / 1000 / 2,
#                         1.0 / dz / 1000 / 2, -1.0 / dz / 1000 / 2),
#                 vmin=vmin,  vmax=vmax, cmap=cmap)
# #plt.colorbar(im, orientation='vertical')
# plt.grid(color='grey', linewidth=1.0, linestyle='--')
# plt.axis(axisType)
# ax = plt.gca()
# ax.xaxis.set_ticks_position('top')
# ax.tick_params(axis="x", direction="in")
# ax.tick_params(axis="y", direction="in")
# x1_label = ax.get_xticklabels()
# [x1_label_temp.set_fontname(fontType) for x1_label_temp in x1_label]
# y1_label = ax.get_yticklabels()
# [y1_label_temp.set_fontname(fontType) for y1_label_temp in y1_label]
# ax.xaxis.set_label_position('top')
# plt.xlabel('Kx (m$^{-1}$)', font_label)
# plt.ylabel('Kz (m$^{-1}$)', font_label)
# plt.xticks(fontsize=ticks_fontsize)
# plt.yticks(fontsize=ticks_fontsize)
# plt.xlim(-1.0 / dx / 1000 / 2, 1.0 / dx / 1000 / 2)
# plt.ylim(1.0 / dz  / 1000 / 2 * 0.7, -1.0 / dz / 1000 / 2 * 0.7)
# plt.gca().xaxis.set_major_locator(MaxNLocator(xMaxNLocator))
# plt.gca().yaxis.set_major_locator(MaxNLocator(yMaxNLocator))
# ax.ticklabel_format(style='plain', scilimits=(-1,2), axis='both')
# if flag_save == 1:
#     plt.savefig(fn_save + fn_part + 'emdimgpart_fk.png')
#     pass


# if flag_show == 1:
#     plt.show()
#     pass

# io.savemat('result.mat',{'result':emdimg})

                 
