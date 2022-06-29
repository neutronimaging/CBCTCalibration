import numpy as np

def goldenSequence(x) :
    phi = 0.5* (1+np.sqrt(5))
    return np.fmod(x*phi*180.0,180.0)
    
Nproj    = 21
Nrepeat = 1

angles = np.repeat(np.linspace(0,180,Nproj),Nrepeat)

scan(sp2_ry,angles.tolist())