import numpy as np

def ormsby_wavelet(f1, f2, f3, f4, dt, tlen):
    
    # 参数说明：
    # f1: 低截止频率
    # f2: 低通频率
    # f3: 高通频率
    # f4: 高截止频率
    # dt: 时间采样间隔
    # tlen: 子波长度
    
    t = np.arange(-tlen / 2, tlen / 2+dt, dt)
    
    # 计算 Ormsby 子波
    term1 = (np.pi * f1**2 / (f2 - f1)) * np.sinc(f1 * t)**2
    term2 = (np.pi * f2**2 / (f2 - f1)) * np.sinc(f2 * t)**2
    term3 = (np.pi * f3**2 / (f4 - f3)) * np.sinc(f3 * t)**2
    term4 = (np.pi * f4**2 / (f4 - f3)) * np.sinc(f4 * t)**2
    
    # 组合子波
    w = term1 - term2 - term3 + term4
    
    return w
