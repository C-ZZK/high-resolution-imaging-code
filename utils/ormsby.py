import numpy as np

def ormsby_wavelet(f1, f2, f3, f4, dt, tlen):
    

    
    t = np.arange(-tlen / 2, tlen / 2+dt, dt)
    

    term1 = (np.pi * f1**2 / (f2 - f1)) * np.sinc(f1 * t)**2
    term2 = (np.pi * f2**2 / (f2 - f1)) * np.sinc(f2 * t)**2
    term3 = (np.pi * f3**2 / (f4 - f3)) * np.sinc(f3 * t)**2
    term4 = (np.pi * f4**2 / (f4 - f3)) * np.sinc(f4 * t)**2
    

    w = term1 - term2 - term3 + term4
    
    return w
