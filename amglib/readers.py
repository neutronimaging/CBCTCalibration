import numpy as np
import astropy.io.fits as fits
from amglib.imageutils import *
import tifffile as tiff
from tqdm.notebook import tqdm

def read_image(fname) :
    ''' Reads a single image in fits or tiff format
        Arguments:
        fname - full filename of the image to read
    
        Returns:
        A 2D numpy array in single precision float
    '''
    ext = fname.split('.')[-1]

    img = []
    
    if ext == 'tif' :
        img = tiff.imread(fname).astype('float32')
    
    if ext == 'fits' :
        img = fits.getdata(fname,ext=0).astype('float32')
        
    return img

def read_images(fname,first,last, average = 'none', averageStack=False, stride=1, count=1, size = 5) :
    ''' Reads a series of images into either a 3D stack or as a single image representing the average
    
        Arguments:
            fname   - file name mask the indexing is formatted using the {0:0N} format where N is the number of digits in the file index.
            first   - first image to read
            last    - last image to read
            average - averaging method valid values: 'mean','weighted', and 'median' 
            averageStack - compute an average of the images using the settings stride, count, average and size.
            stride  - stride beween observations, mostly the same as count
            count   - number on images to average
            size    - size of the window used to compute local statistics in each slice
    
        Returns:
        Either a 2D or 3D numpy array in single precision float depending on the averageStack value.
    '''
    tmp = read_image(fname.format(first))
    img = np.zeros([(last-first+1) // stride,tmp.shape[0],tmp.shape[1]],dtype='float32')
    img[0]=tmp.astype('float32')
    if 1<count :
        for idx in tqdm(np.arange(first,last+1,step=stride)) : 
            img[(idx-first) // stride] =  read_images(fname,
                                                     idx,
                                                     idx+count-1,
                                                     stride=1,
                                                     count=1,
                                                     size=size, 
                                                     average=average,
                                                     averageStack=True)
    else :
        for idx in tqdm(np.arange(first,last+1,step=stride)) : 
            img[(idx-first) // stride] = read_image(fname.format(idx))

    
    if averageStack == True :
        if average == 'mean' :
            img = averageimage(img)
        if average == 'weighted' :
            img = weightedaverageimage(img,size=size)
        if average == 'median' :
            img = medianimage(img)
        
    return img

def get_sinograms(fmask,ob,dc,N,lines,stride=1, counts=1) :
    '''Prepares sinograms from the provided images 
    
    '''
    img = fits.getdata(fmask.format(1),ext=0)
    sino = np.zeros([len(lines),N,img.shape[1]])
    
    angles   = goldenSequence(N)
    angleIdx = np.argsort(angles)
    ob = ob-dc
    ob[ob<1]=1
    D0 = ob[100:200,100:200].mean()
    
    for idx in range(len(angleIdx)) :
        for count in range(counts) :
            fileIdx = 1 + angleIdx[idx]*stride + count
            img = fits.getdata(fmask.format(fileIdx),ext=0).astype(float)-dc
            img[img<1]=1
            D = img[100:200,100:200].mean()
            # Normalize (img-dc)/(ob-dc)*(D0/D)
            img = -np.log((D0/D)*(img)/(ob))
            for lineIdx,line in enumerate(lines) :
                sino[lineIdx,idx] = sino[lineIdx,idx] + img[line]
                
    sino = sino / counts
    
    angles = angles[angleIdx]
    
    return sino,angles


def save_TIFF(fname, img, startIndex=0) :
    '''Save a numpy array as tiff image
    
    Arguments:
        fname - the file name or mask. A mask is used when a 3D array is provided.
        img   - the numpy array either 2D or 3D
        startIndex - the first file index when a 3D array is saved.
        
    Returns:
        the function doesn't return anything
    '''
    if len(img.shape)<3 :
        tiff.imsave(fname,np.squeeze(img), {'photometric': 'minisblack'})
    else :
        for idx in tqdm(range(img.shape[0])) :
            tiff.imsave(fname.format(idx+startIndex),np.squeeze(img[idx]), {'photometric': 'minisblack'})

def read_fits_meta_data(fname,parlist = []) :
    '''Reads selected meta data from a fits file
    
    Arguments:
        fname   - full file name of the fits file
        parlist - a list of strings to select the meta data tags.
        
    Returns:
        A dict containing the requested values
    '''
    hdul = fits.open(fname)
    hdul.verify('silentfix')
    data = {}
    for par in parlist :
        value = hdul[0].header[par]
        if isinstance(value,str) and value[0]=='b' :
            value=value[2:-1]
        data[par] = value
        
    return data

def read_fits_meta_data2(fmask,first,last,parlist = []) :
    '''Reads selected meta data from a series of fits files
    
    Arguments:
        fmask   - file name mask of the fits file the index template is using the format {0:0N} where N is the number of digits in the index
        first   - first file index in the series
        last    - last file index in the series
        parlist - a list of strings to select the meta data tags.
        
        
    Returns:
        A dict containing lists of the requested values from each file
    '''
    data = {}
    
    for par in parlist :
        data[par]= []
        
    for idx in np.arange(first,last+1) :
        hdul = fits.open(fmask.format(idx))
        hdul.verify('silentfix')
    
        for par in parlist :
            value = hdul[0].header[par]
            if isinstance(value,str) and value[0]=='b' :
                value=value[2:-1]
            data[par].append(value)
            
        hdul.close()
        
    return data
