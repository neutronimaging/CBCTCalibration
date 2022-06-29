import numpy as np

def goldenSequence(x, arc=180) :
    phi = 0.5* (1+np.sqrt(5))
    return np.fmod(x*phi*180.0,arc)
    
Nproj    = 4000
Nrepeat = 1

x = np.repeat(range(Nproj),Nrepeat)

angles = goldenSequence(x)

scan(sp2_ry,angles.tolist())