# +
import numpy as np

def line_pattern(segmentlength,f):
    """ Generates a 1D bilevel test pattern with increasing frequency
    
    Arguments:
    segmentlength -- number of pixels for each frequency
    f -- list of periods
    
    """
    x=np.arange(0.0,segmentlength)
    xx=[]
    for ff in f :
        xx=np.append(xx,np.mod(np.floor(x/ff),2))

    return xx


# -

def line_pattern_2d(segmentwidth,segmentheight,f) :
    """ Generates a 2D bilevel line pattern with increasing frequency

    Arguments:
    segmentwidth -- number of pixels for each frequency
    segmentheight -- number of pixels for each line 
    f -- list of periods
    
    """
    x=linepattern(segmentwidth,f)
    y=[np.ones(len(x)),x,np.ones(len(x))]
    
    return np.repeat(x,segmentheight)

def contraststeps(neutrons=100,scalex=20,scaley=50) :
    """ Generates a constrst step wedge
    """
    img=np.array([[  1,   1,   1,   1,   1,   1,   1,   1,   1,  1,    1,   1, 1.0],
         [1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.0, 1.0],
         [  1,   1,   1,   1,   1,   1,   1,   1,   1,  1,    1,   1, 1.0]])
    
    img=np.repeat(img,scalex,axis=1)
    img=np.repeat(img,scaley,axis=0)
    
    return img


# +
def gaussian_2D(sigma,th) :
    """ Makes a rotated 2D gaussian, size is given by max(sigma)
    
    Arguments :
    sigma -- list/array with two elements to describe minor and major axis
    th    -- rotation angle of the gaussian
    
    Returns:
    
    """
    theta = np.radians(th)
    c, s = np.cos(theta), np.sin(theta)
    #R = np.array(((c, -s), (s, c)))
    
    N=np.ceil(np.max(sigma)*3)
    u=np.linspace(-N,N,2*int(N)+1)
    x,y = np.meshgrid(u,u)
    rx = c*x-s*y
    ry = s*x+c*y
    
    g=np.exp(-(rx**2)/(2*sigma[0]**2)-ry**2/(2*sigma[1]**2))
    g=g/g.max()
    return g

def make_spots(N_spots, size=[2048,2048]) :
    """Makes an image with realistic spots uniformly distributed over the image
    
    Arguments: 
    N_spots - number of spots to generate
    size    - size of image
    
    Returns:
    A numpy array with spots
    """
    img=np.zeros(size);
    
    m_width = [0.4,1]
    s_width = [0.6,2.2]
    m_ampl = 1400
    
    minor = np.random.weibull(3,size=N_spots)*m_width[0]+1
    major = np.random.weibull(1.1,size=N_spots)*m_width[1]+1
    ampl  = np.random.weibull(1.2,size=N_spots)*m_ampl
    theta = np.random.uniform(low=0,high=360,size=N_spots)
    
    r=np.floor(np.random.uniform(low=0,high=size[0],size=N_spots)).astype(int)
    c=np.floor(np.random.uniform(low=0,high=size[1],size=N_spots)).astype(int)
    
    for i in range(len(minor)) :
        g=gaussian_2D(sigma=[minor[i],major[i]],th=theta[i])
        g=g*(0.6<g)*ampl[i]

        if (size[0]-r[i]<=g.shape[0]) :
            g=g[0:(size[0]-r[i])]
            
        if (size[1]-c[i]<=g.shape[1]) :
            g=g[:,0:(size[1]-c[i])]
            
        img[r[i]:(r[i]+g.shape[0]),c[i]:(c[i]+g.shape[1])]=np.maximum(g,img[r[i]:(r[i]+g.shape[0]),c[i]:(c[i]+g.shape[1])])
        
    return img

def spottify_image(img,N_spots) :
    spots = make_spots(img.shape,N_spots)
    
     # res = np.maximum(img,spots)
    res = img + spots
    
    return res, 0<spots 
