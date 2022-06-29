import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import skimage.io as io
import skimage.filters as filt
import skimage.morphology as morph
from skimage.draw import disk
import amglib.imageutils as amg
import pandas as pd
from skimage.feature import match_template

def get_dot_template(img, roi,se=morph.disk(5)) :
    """
    Extracts sub image from the region given by a ROI
    
    Parameters
    ----------
    img : ndarray (2D)
        The source image
    roi : array
        Region of interest to extract. Organized as (x0,y0,x1,y1)
    se  : ndarray (2D)
        Structure element for the median filter used to remove outliers from the template.
    

    Returns
    -------
    ndarray (2D)
        The template image

    """
    template=filt.median(img[roi[0]:roi[2],roi[1]:roi[3]],se)
    
    return template

def get_black_bodies(img, greythres,areas = [2000,4000],R=2) :
    """
    Gets black body information using a global threshold. This is not te ideal implementation.
    
    Arguments:
    img - the image to analyse
    greythres - threshold level to identify the bb's
    areas - size limitation for the bb regions. A list with upper and lower bound
    R - radius of the dots in the mask image
    
    Returns:
    mask - the mask image
    r,c - coordinates of the bbs
    df - data frame containing all information about the dots
    """
    
    bb=img<greythres
    lbb=label(bb)
    
    plt.imshow(bb)

    mask = np.zeros(lbb.shape)
    regions=[]

    for region in regionprops(lbb):
        if (region.area<areas[0]) or (areas[1]<region.area) :
            lbb[lbb==region.label]=0
        else:
            
            #regions.append(region)
            x,y = region.centroid
            rr, cc = disk((x,y), R)
            mask[rr, cc] = 255  
            dotdata = img[rr,cc]
            regions.append({'label': region.label,
                            'mean' : np.mean(dotdata),
                            'median' : np.median(dotdata),
                            'r' : region.centroid[0],
                            'c' : region.centroid[1]})

    mask=mask.astype('uint8')
    r,c = np.where(0<mask)
    df = pd.DataFrame.from_dict(regions)
    
    return mask,r,c,df

def get_black_bodies_mask(img,mask, areas = [2000,4000],R=2) :
    """
    Gets black body information using a global threshold. This is not te ideal implementation.
    
    Arguments:
    img - the image to analyse
    greythres - threshold level to identify the bb's
    areas - size limitation for the bb regions. A list with upper and lower bound
    R - radius of the dots in the mask image
    
    Returns:
    mask - the mask image
    r,c - coordinates of the bbs
    df - data frame containing all information about the dots
    """
    
    bb= 0<mask 
    
    lbb=label(bb)

    mask = np.zeros(lbb.shape)
    regions=[]

    for region in regionprops(lbb):
        if (region.area<areas[0]) or (areas[1]<region.area) :
            lbb[lbb==region.label]=0
        else:
            
            #regions.append(region)
            x,y = region.centroid
            rr, cc = disk((x,y), R)
            mask[rr, cc] = 255  
            dotdata = img[rr,cc]
            regions.append({'label': region.label,
                            'mean' : np.mean(dotdata),
                            'median' : np.median(dotdata),
                            'r' : region.centroid[0],
                            'c' : region.centroid[1]})

    mask=mask.astype('uint8')
    r,c = np.where(0<mask)
    df = pd.DataFrame.from_dict(regions)
    
    return mask,r,c,df


def get_blackbodies_by_templatematching(img,template,threshold,R=10,area_th=[0.4,0.6]) :
    """
    Finds the black body dots in a image using template matching.
    
    Parameters
    ----------
    img      : ndarray (2D)
    template : ndarray (2D)
    threshold : float
    R : int
    
    Returns
    -------
    ndarray (2D)
        A bilevel mask image marking out the bb dots with a standard radius given by R.
    ndarray (1D)
        row coordinates of the dot pixels
    ndarray (1D)
        column coordinates of the dot pixels
    pandas data frame
        Data frame containing the information about the detected dots 
    
    """
    tm = match_template(img, template,pad_input=True)
    lbb=label(threshold<tm)
    mask = np.zeros(lbb.shape)
    regions=[]

    area=(template<filt.threshold_otsu(template)).sum()
    areas = [area_th[0]*area, area_th[1]*area]
    
    for region in regionprops(lbb):
        if (region.area<areas[0]) or (areas[1]<region.area) :
            lbb[lbb==region.label]=0
        else:
            x,y = region.centroid
            rr, cc = disk((x,y), R)
            mask[rr, cc] = 255  
            dotdata = img[rr,cc]
            regions.append({'label': region.label,
                            'mean' : np.mean(dotdata),
                            'median' : np.median(dotdata),
                            'r' : region.centroid[0],
                            'c' : region.centroid[1]})

    mask=mask.astype('uint8')
    r,c = np.where(0<mask)
    df = pd.DataFrame.from_dict(regions)
    
    return mask,r,c,df,tm


def compute_scatter_image(img,r,c) :
    """
    Computes a scatter image using a 2D polynomial fit of the image intensity behind the provided coordinates.
    
    Arguments:
    img - image with black bodies
    r,c - coordinates of the blackbodies

    Returns:
    An image with the scatter estimate.
    """
    
    y = img[r,c]

    H=np.transpose([r,c,r**2,c**2,r*c])
    H=np.concatenate((np.ones([H.shape[0],1]),H),axis=1)

    q=np.linalg.lstsq(H, y, rcond=None)

    res=polynomial_image(img.shape,q[0])
    return res

def compute_scatter_image_from_df(df,shape,info='median') :
    """
    Computes a scatter image using a 2D polynomial fit of the image intensity behind the provided coordinates.
    
    Arguments:
    df - dataframe containing the bb information
    shape - list with two elements describing the image dimensions
    info - selects the df column containing the information to fit

    Returns:
    An image with the scatter estimate.
    """
        
    y = df[info]
    H=np.transpose([df['r'],df['c'],df['r']**2,df['c']**2,df['r']*df['c']])
    H=np.concatenate((np.ones([H.shape[0],1]),H),axis=1)
    
    q=np.linalg.lstsq(H, y, rcond=None)

    res=polynomial_image(shape,q[0])
    
    return res

def polynomial_image(size,q) :
    """
    Computes an image with intensities described by a 2D polynomial
    
    Arguments:
    size - list with two elements to set the image size
    q - list/numpy array with polynomial coefficients.
    
    Returns:
    A polynomial image
    """
    c,r = np.meshgrid(np.arange(size[1]),np.arange(size[0]))

    img = np.ones(size)*q[0]
    img = img + r*q[1]
    img = img + c*q[2]
    img = img + r**2*q[3]
    img = img + c**2*q[4]
    img = img + r*c*q[5]

    return img

def check_scatter_image(bb, 
                        est,  
                        mask, 
                        areas=[100,1000], 
                        show=True, 
                        ax=None, 
                        cmap='jet', 
                        sym_cmap=False,
                        v=None) :  
    """
    Visualization function to compute errors and show the outcome of a scatter image fit
    
    Arguments:
    bb    - 
    est   -  
    mask  -
    areas - 
    show  - 
    ax    - 
    cmap  - 
    sym_cmap -
    
    Returns:
    A dataframe with region statistics for each bb-dot
    """
    lbb=label(mask)

    diff = bb - est 
    
    regdata = []
    
    for region in regionprops(lbb):
        if (region.area<areas[0]) or (areas[1]<region.area) :
            lbb[lbb==region.label]=0
        else:
            x,y = region.centroid
            
            med   = np.median(diff[lbb==region.label])
            mean  = diff[lbb==region.label].mean()
            mse   = (diff**2)[lbb==region.label].mean()
            
            regdata.append({'label':  region.label, 
                            'Median': med, 
                            'Mean':   mean, 
                            'MSE':    mse, 
                            'RMSE':   np.sqrt(mse),
                            'area':   region.area, 
                            'centroid': region.centroid, 
                            'x':      y, 
                            'y':      x})
    
    df=pd.DataFrame.from_dict(regdata)
    
    if show :
        if ax is None :
            fig,ax =plt.subplots(1,1,figsize=(10,10))
            
        ax.imshow(bb,vmin=0,vmax=8000,cmap='gray')
        
        if sym_cmap :
            if v is None: 
                v = np.max([np.abs(df['Median'].max()),np.abs(df['Median'].min())])
            a=ax.scatter(df['x'],df['y'],c=df['Median'],vmin=-v,vmax=v,cmap=cmap)
        else :
            a=ax.scatter(df['x'],df['y'],c=df['Median'],cmap=cmap)
            
        plt.gcf().colorbar(a,ax=ax,shrink=0.5)
    
    return df

def normalize(img,ob,dc, dose_roi=None,logarithm=True) :
    """
    Open beam correction of an image
    
    Arguments:
    img - the image to normalize
    ob  - open beam image
    dc  - dark current image
    dose_roi - list of coordinates for a dose roi [r0,c0,r1,c1] (optional)
    logarithm - compute the minus logarithm
    
    Returns:
    The Normalized image
    """
    
    img=img-dc
    img[img<1]=1
    ob=ob-dc
    ob[ob<1]=1
    
    D=1
    
    if dose_roi is not None:
        dob=ob[dose_roi[0]:dose_roi[2],dose_roi[1]:dose_roi[3]]
        dimg=img[dose_roi[0]:dose_roi[2],dose_roi[1]:dose_roi[3]]
        D=np.median(dob,axis=0).mean()/np.median(dimg,axis=0).mean()
        
    n=D*img/ob
    
    if logarithm :
        n=-np.log(n)
        
    return n

def remove_dc(img,dc) :
    """
    Subtracts the dark current form the image and replaces non-positive numbers by 1
    
    Arguments:
    img - image to process
    dc  - dark current image
    
    Returns: 
    The dc corrected image
    """
    
    if type(dc) == type(img) :
        img = img - dc
        img[img<1]=1
    return img

def D(img, roi) :
    """
    Computes the dose from a region in an image
    
    Arguments:
    img - the image
    roi - region of interest a list with [r0,c0,r1,c1]
    
    Returns
    A scalar value of the measured dose.
    """
    return np.median(img[roi[0]:roi[2],roi[1]:roi[3]],axis=0).mean()

def normalization_with_BB(sample,ob,dc,bbsample,bbob,ssample,sob,roi,tau) :
    """
    Normalizes an image with scattering correction
    
    Arguments:
    sample   - sample image
    ob       - open beam image
    dc       - dark current image
    bbsample - sample image with bbs
    bbob     - open beam image with bbs
    ssample  - estimated scattering image for the sample image
    sob      - estimated scattering image for the open beam image
    roi      - region of interest a list with [r0,c0,r1,c1]
    tau      - bb area correction factor
    
    Returns:
    A scattering corrected normalized image
    
    """
    
    sample   = remove_dc(sample,dc)
    ob       = remove_dc(ob,dc)
    bbsample = remove_dc(bbsample,dc)
    bbob     = remove_dc(bbob,dc)
    
    dsample   = D(sample,roi)
    dob       = D(ob,roi)
    dbbsample = D(bbsample,roi)
    dbbob     = D(bbob,roi)
    dssample  = D(ssample,roi)
    dsob      = D(sob,roi)
    
    ds = (dsample/(tau*(dbbsample-(1-1/tau)*dssample)))
    do = (dob/(tau*(dbbob-(1-1/tau)*dsob)))
    corrected = (sample - ssample*ds)/(ob - sob*do) * (do/ds)
    
    return corrected


def normalization_with_BB_doselist(sample,ob,dc,bbsample,bbob,ssample,sob,dsample,dob,dssample,dbbob,dsob,tau) :
    sample   = remove_dc(sample,dc)
    ob       = remove_dc(ob,dc)
    bbsample = remove_dc(bbsample,dc)
    bbob     = remove_dc(bbob,dc)
    
    dbbsample = D(bbsample,roi)
    dbbob     = D(bbob,roi)
    dssample  = D(ssample,roi)
    dsob      = D(sob,roi)
    
    ds = (dsample/(tau*(dbbsample-(1-1/tau)*dssample)))
    do = (dob/(tau*(dbbob-(1-1/tau)*dsob)))
    corrected = (sample - ssample*ds)/(ob - sob*do) * (do/ds)
    
    return corrected
