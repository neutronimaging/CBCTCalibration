import numpy as np
import matplotlib.pyplot as plt
import skimage as im
import amglib.readers as io
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.morphology import binary_erosion as erode
from skimage.morphology import binary_dilation as dilate
from skimage.morphology import disk
from skimage.morphology import h_maxima
import skimage.morphology.greyreconstruct as gr
from skimage.morphology import label
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import matplotlib.colors as colors
from skimage.color      import hsv2rgb, rgb2hsv
from matplotlib.colors import ListedColormap
from tqdm.notebook import tqdm
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
from skimage.measure import label, regionprops, regionprops_table

def normalizeData(img,ob,dc) :
    ob=ob-dc
    ob[ob<1]=1
    lcal=img.copy();
    for idx in np.arange(0, img.shape[0]):
        tmp=(img[idx,:,:]-dc)
        tmp[tmp<=0]=1
        lcal[idx,:,:]=(tmp/ob)
    lcal=-np.log(lcal)
    
    return lcal


def removeBaseline(img) :
    img=img[:,:,0:1750]
    baseline = img.mean(axis=0).mean(axis=0)
    baseline = baseline.reshape(1,baseline.shape[0])

    b2=np.matmul(np.ones([img.shape[1],1]),baseline)
    res=img;
    for idx in np.arange(0,img.shape[0]) :
        res[idx,:,:]=res[idx,:,:]-b2
    return res


def thresholdBBs(bcal,k) :
    s=bcal.std()
    m=bcal.mean()
    
    return (m+k*s)< bcal


def min_impose(dimg,markers) :
    fm=markers.copy()
    fm[markers != 0] = 0
    fm[markers == 0] = dimg.max()
    dimg2 = np.minimum(fm,dimg+1)
    res   = gr.reconstruction(fm,dimg2,method='erosion')
    
    return res

def randomCM(N, low=0.2, high=1.0,seed=42, bg=0) :
    np.random.seed(seed=seed)
    clist=np.random.uniform(low=low,high=high,size=[N,3]); 
    m = ortho_group.rvs(dim=3)
    if bg is not None : 
        clist[0,:]=bg;
        
    rmap = ListedColormap(clist)
    
    return rmap

def goldenCM(N,increment=1.0,s=0.5,v=0.7,bg=0) :
    phi= 0.5*(np.sqrt(5)-1)
    
    hsv = np.zeros([N,3]);
    hsv[:, 0] = increment*phi*np.linspace(0,N-1,N)-np.floor(increment*phi*np.linspace(0,N-1,N))
    hsv[:, 1] = s
    hsv[:, 2] = v
    rgb = hsv2rgb(hsv)
    if bg is not None : rgb[0,:]=bg    
    cm = ListedColormap(rgb) 
    return cm


def watershed_segmentation_display(img):
    cog,ws1=findBeadsWS(img)
    
    labels = np.unique(ws1).astype(int)
    
    fig,ax=plt.subplots(1,1,figsize=(10,10))
    ax.imshow(ws1,interpolation='None',cmap=goldenCM(labels.max()))
    ax.plot(cog[:,1],cog[:,0],'r+')

def findBeadsWS(img, selem= disk(3),h=2) :
    p=img.mean(axis=0)
    m=np.where((0.5*p.max())<p)
    a=np.min(m)
    b=np.max(m)
    w=b-a
    a=a-w
    b=b+w
    if a<0 : a=0
    if len(p)<=b : b=len(p)-1
    img2=img[:,a:b]
    distance = ndi.distance_transform_edt(erode(img2,selem))

    localmax = h_maxima(distance,h)
    rdmap    = distance.max()-distance
    labels   = label(localmax)
    ws1 = watershed(min_impose(rdmap,labels),labels,mask=img2)
    
    rp = regionprops_table(ws1,properties=('area','centroid'))
       
    cog=np.zeros([rp['centroid-0'].shape[0],2])
    cog[:,0]=rp['centroid-0']
    cog[:,1]=rp['centroid-1']+a
    ws = np.zeros(img.shape)
    ws[:,a:b] = ws1

    return cog,ws

def buildBeadList(img,selem=disk(12),c=1.96) :
    beadlist = []

    for idx in np.arange(0, img.shape[2]) :
        cog=findBeadsWS(img[:,:,idx],selem,c)
        beadlist.append(cog)

    return beadlist


def display_beads(tcal, idx):
    c=4
    cog=findBeadsWS(tcal[idx,:,:])

    plt.figure(figsize=[12,8])
    plt.imshow(tcal[idx,:,:])

    plt.plot(cog[:,1]-1,cog[:,0]-1,'r+')

    cog1=findBeadsWS(tcal[idx+1,:,:])
    plt.plot(cog1[:,1]-1,cog1[:,0]-1,'rx')

    cog2=findBeadsWS(tcal[idx-1,:,:])
    plt.plot(cog2[:,1]-1,cog2[:,0]-1,'ro')
    
    
def getBeads(img,selem=disk(2)) :
    N=img.shape[0]
  
    beads = []
    
    for proj in tqdm(np.arange(0,N)) :
        cog,_ = findBeadsWS(img[proj,:,:])
        beads.append(cog)
    return beads

def identifyEllipses_old(img,selem=disk(2)) :
    N=img.shape[0]
    ellipses = []
    params=[]
    cog_allbeads=[]    
    beads = []
    for proj in tqdm(np.arange(0,N)) :
        cog = findBeadsWS(img[proj,:,:])
        beads.append(cog)


    for idx in range(len(min(beads,key=len))):
        ellipse = []
        for p in range(N) :
            ellipse.append(beads[p][idx,:].tolist())
        ellipses.append(np.array(ellipse))
        ell = EllipseModel()
        a_ellipse = np.array(ellipse)
        ell.estimate(a_ellipse)
        if ell.params==None:
            continue
        cog_onebead=[]
        for p in range(N) :
            cog_onebead.append(beads[p][idx,:])
        cog_allbeads.append(cog_onebead)
        xc, yc, a, b, theta = ell.params
        if theta> 1:
            theta=theta-(np.pi/2)
        params.append([yc,xc,max(a,b),min(a,b),theta])
    return params,cog_allbeads

def identifyEllipses(img,selem=disk(2), beads=None) :
    N=img.shape[0]
    
    ellipses = []
    params=[]
    cog_allbeads=[]  
    res = []
    if beads is None :
        beads = getBeads(img,selem)
    
    for idx in range(len(min(beads,key=len))):
        ellipse = []
        for p in range(N) :
            ellipse.append(beads[p][idx,:].tolist())
        ellipses.append(np.array(ellipse))
        ell = EllipseModel()
        a_ellipse = np.array(ellipse)
        ell.estimate(a_ellipse)
        if ell.params==None:
            continue
        cog_onebead=[]
        r=ell.residuals(a_ellipse)
        res.append(4)
        
        sort=np.argsort(r)
        
        ell.estimate(a_ellipse[sort[:len(sort)//2],:])
        for p in range(N) :
            cog_onebead.append(beads[p][idx,:])
        cog_allbeads.append(cog_onebead)
        xc, yc, a, b, theta = ell.params
        if theta> 1:
            theta=theta-(np.pi/2)
        params.append([yc,xc,max(a,b),min(a,b),theta])
    return params,cog_allbeads,res

def show_ellipses(e2,cog_allbeads):
    #for idx in range(len(min(cog_allbeads,key=len))):
    for idx in range(len(e2)):
        print("ID Number = ", idx)
        print("center = ",  (e2[idx][0], e2[idx][1]))
        theta=e2[idx][4]
        print("angle of rotation = ",  theta)
        print("axes major/minor = ", e2[idx][2],e2[idx][3])
        a_ellipse=np.array(cog_allbeads[idx])
        x=a_ellipse[:,0]
        y=a_ellipse[:,1]
        fig, axs = plt.subplots(1,2, figsize=(15,4),sharex=True, sharey=True)
        axs[0].scatter(y, x)
        for (num,(yy,xx)) in enumerate(zip(y,x)) :
            axs[0].annotate("{0}".format(num),xy=(yy,xx),  xycoords='data')

        axs[1].scatter(y, x)
        axs[1].scatter(e2[idx][0], e2[idx][1], color='red', s=100)

        ell_patch = Ellipse((e2[idx][0], e2[idx][1]), 2*e2[idx][2], 2*e2[idx][3], theta, edgecolor='red', facecolor='none')

        axs[1].add_patch(ell_patch)
        plt.show()

        
def estimate_cor(e2):
    x_centres=[]
    y_centres=[]
    for i in range(len(e2)):
        x_centres.append(e2[i][0])
        y_centres.append(e2[i][1])
        
    x_centres=np.array(x_centres)
    y_centres=np.array(y_centres)
    theta=np.polyfit(y_centres, x_centres, 1)
    res= np.abs(theta[1]+theta[0]*np.array(y_centres)-x_centres)
    idx = np.where(res<(res.mean()+res.std()))
    theta=np.polyfit(y_centres[idx[0]], x_centres[idx[0]], 1)
    
    plt.scatter(x_centres,y_centres)
    plt.plot(theta[1]+theta[0]*np.array(y_centres), y_centres ,'r')
    plt.show()
    print("The parameters of the COR obtained are as follows:",(theta[1]))
    tilt = np.arctan(theta[0])*180/np.pi
    print("The tilt of the center of rotation is in degrees: ", tilt)
    result ={"COR" : theta[1], "Ctilt" : tilt}
    return result
    
def estimate_magnification(tcal,idx):
    cog,_=findBeadsWS(tcal[idx,:,:])
    pixel_pitch=0.139
    d=np.sort(np.diff(cog[:,0]))
    m=d[6:-6].mean()
    s=d[6:-6].std()
    k=1
    plt.plot(np.diff(cog[:,0]),'.')
    plt.plot(d,'.')
    plt.fill_between([0, 37],y1=[m-k*s,m-k*s],y2=[m+k*s,m+k*s],alpha=0.2)

    print(pixel_pitch*np.array([m,d.mean(),np.median(d)]))
    return pixel_pitch*m
    
def plot_allellipses(e2):
    fig, ax = plt.subplots()

    for idx in range(len(e2)):
        ellipse = Ellipse((e2[idx][0], e2[idx][1]), e2[idx][2]*2, e2[idx][3]*2, e2[idx][4])
        ax.add_artist(ellipse)
    ax.set_xlabel("x_coordinate")
    ax.set_ylabel("y_coordinate")
    ax.set_xlim(500, 1500)
    ax.set_ylim(800, 2000)
    plt.show()
    
    
def estimate_piercingpoint(e2):
    radius=[]
    height=[]
    
    for i in range(len(e2)):
        radius.append(e2[i][3])
        height.append(e2[i][1])
        
    theta =[]
    temp_r= radius.copy()
    temp_h= height.copy()
    k=0
    while k!=1:
        count=0
        theta=np.polyfit(height, radius, 1)
        radius= temp_r.copy()
        height= temp_h.copy()
        for x,y in zip(height, radius):
            if (theta[1]+theta[0]*x >= y+0.5 or theta[1]+theta[0]*x <= y-0.5):
                count=count+1
                temp_h.remove(x)
                temp_r.remove(y)
        if count ==0:
            k=1
    plt.plot(height,radius,'.')
    plt.xlabel('height')
    plt.ylabel('radius(minor axis length)')
    
    plt.plot(height,theta[0]*np.array(height)+theta[1])
    x_centres=[]
    y_centres=[]
    for i in range(len(e2)):
        x_centres.append(e2[i][0])
        y_centres.append(e2[i][1])
    theta1=np.polyfit(y_centres, x_centres, 1)
    pp_y= -theta[1]/theta[0]
    pp_x= theta1[1]+theta1[0]*(pp_y)
    print("Piercing Point is at: ",(pp_x,pp_y))
    return pp_x, pp_y


def medianDistance(cog) :
    return np.median(np.diff(cog[:,0]))

def itemList(cog) :
    d=medianDistance(cog)
    idx=np.floor((cog[:,0]-cog[0,0])/d+0.5).astype(int)
    
    idxList={}
    for (i,c) in zip(idx,cog) :
        idxList[i]=c
        
    return idxList
   
def estimate_sod_sdd(tcal, e2, vpiercing, mag, factor=1.08):    
    pixelSize = 0.139
    R = 10
    e0,_=findBeadsWS(tcal[0,:,:])
    c0=itemList(e0)
    sod = []
    sdd = [] 
    for idx in range(np.array(e2).shape[0]) :
        h=0
        if e2[idx][1] < vpiercing :
            hb=(vpiercing - (e2[idx][1]-e2[idx][3])) * pixelSize
            ha=(vpiercing - (e2[idx][1]+e2[idx][3])) * pixelSize
            if idx in c0 :
                h = (vpiercing - c0[idx][1])*pixelSize
        else :
            ha=(e2[idx][1]-e2[idx][3]-vpiercing) * pixelSize
            hb=(e2[idx][1]+e2[idx][3]-vpiercing) * pixelSize
            if idx in c0 :
                h = (c0[idx][1]-vpiercing)*pixelSize


        est_sod= (hb+ha)*R*factor/(hb-ha)
        est_sdd= (est_sod)*mag
        sod.append(np.abs(est_sod))
        sdd.append(np.abs(est_sdd))
        print("h: {0:0.3f}, SOD: {1:0.2f}, SDD: {2:0.2f}, magn: {3:0.2f}, ha: {4:0.2f}, hb: {5:0.2f}".format(h,est_sod, est_sdd,est_sdd/est_sod,ha,hb))
    sd_sod= np.std(sod)
    sd_sdd= np.std(sdd)
    sod = np.mean(sod)
    sdd = np.mean(sdd)
    result = {'sod': sod, 'sdd': sdd, 'magnification': sdd/sod, 'sd_sod':sd_sod ,'sd_sdd': sd_sdd}
    print("Mean SOD= ", sod)
    print("Mean SDD= ", sdd)
    print("Magnification= ", sdd/sod)
    print("Standard deviation in SOD= ", sd_sod)
    print("Standard deviation in SDD= ", sd_sdd)
    print("Pixel size= ", pixelSize/mag)
    return result 