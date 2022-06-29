import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt

def misclasification_distance(gt,pr,decimals=0) :
    '''
    Counts the number of false positves and false negatives as function of distance from the edge of the ground truth.

            Parameters:
                    gt (np array) : The ground truth image
                    pr (np array) : The prediction image

            Returns:
                    binary_sum (str): Binary string of the sum of a and b
    '''
    
    dpos = np.around(edt(gt != 0),decimals=decimals)
    dneg = np.around(edt(gt == 0),decimals=decimals)
    
    m=4*gt*pr+ 2*gt*(1-pr) + 3*(1-gt)*pr + (1-gt)*(1-pr)
    
    dneg=dneg*(m==3)
    
    duneg = np.unique(dneg)
    fpcnt = []
    
    for d in duneg :
        fpcnt.append((dneg==d).sum())

    dpos=dpos*(m==2)
    
    dupos = np.unique(dpos)
    fncnt = []
    
    for d in dupos :
        fncnt.append((dpos==d).sum())
       
    res = {'fp_dist':duneg,'fp_count':fpcnt,'fn_dist':dupos,'fn_count':fncnt,'confusion_map':m}

    return res
    
    
    
    