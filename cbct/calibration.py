import numpy as np

from scipy import ndimage as ndi

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
import matplotlib.patches as patches

from tqdm.notebook import tqdm

import skimage as im
from skimage.filters      import threshold_otsu
from skimage.segmentation import watershed
from skimage.morphology   import binary_erosion as erode
from skimage.morphology   import binary_dilation as dilate
from skimage.morphology   import disk
from skimage.morphology   import h_maxima
from skimage.morphology   import label
from skimage.feature      import peak_local_max
from skimage.color        import hsv2rgb, rgb2hsv
from skimage.measure      import EllipseModel
from skimage.measure      import label, regionprops, regionprops_table

import cv2

from uncertainties import ufloat, umath
import uncertainties as un

from sklearn import linear_model
    # from sklearn.linear_model import RANSACRegressor  # https://developer.nvidia.com/blog/dealing-with-outliers-using-three-robust-linear-regression-models/

import amglib.readers as io


def normalizeData(img,ob,dc, flipProjection=False) : 
    """Normalize the data using the given offset and dark current images.
    Arguments:
    - img: the image to normalize
    - ob: the offset image
    - dc: the dark current image
    - flipProjection: if True, the projection is flipped vertially after normalization

    Returns: A normalized image stack
    """

    ob=ob-dc
    ob[ob<1]=1
    lcal=img.copy();
    for idx in np.arange(0, img.shape[0]):
        tmp=(img[idx,:,:]-dc)
        tmp[tmp<=0]=1
        lcal[idx,:,:]=(tmp/ob)
    lcal=-np.log(lcal)
    
    if flipProjection : 
        lcal = lcal[:,::-1,:]

    return lcal


def removeBaseline(img, usemin=False) :
    """
    Remove the baseline from the image stack.
    Arguments:
    - img: the image stack
    - usemin: if True, the baseline is the minimum of the image, otherwise, the baseline is the mean of the image.
    """
    if usemin :
        baseline = img.min(axis=0)
    else :
        img=img[:,:,0:1750]
        baseline = img.mean(axis=0).mean(axis=0)
        baseline = baseline.reshape(1,baseline.shape[0])

        baseline=np.matmul(np.ones([img.shape[1],1]),baseline)
    
    res=img.copy()
    print(baseline.shape,res.shape)
    for idx in np.arange(0,res.shape[0]) :
        res[idx,:,:]=res[idx,:,:]-baseline

    return res

def erode_dilate(thresh_img, kernel_size=3, erosion_iterations=14, dilation_iterations=1):
    """
    Perform erosion and dilation on the thresholded image to remove small noise and fill small gaps.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    eroded_img  = np.zeros(thresh_img.shape)
    dilated_img = np.zeros(thresh_img.shape)
    for i in range(len(thresh_img)):
        # Perform erosion to remove small noise
        eroded_img[i] = cv2.erode(thresh_img[i], kernel, iterations=erosion_iterations)

        # Perform dilation to fill small gaps
        dilated_img[i] = cv2.dilate(eroded_img[i], kernel, iterations=dilation_iterations)

    return eroded_img, dilated_img


def find_beads(img, projidx=None,sensitivity=0.2, silent=True, dispidx=None):
    if projidx is None:
        projidx = range(img.shape[0])

    bead_positions_all = []
    bead_intersections_all = []
    bead_ids_all = []
    for i in projidx:
        # Find the contours of the beads in the current projection
        contours, _ = cv2.findContours(img[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Compute the center of mass of each contour using its moments
        bead_positions = []
        for j, contour in enumerate(contours):
            moments = cv2.moments(contour)
            if moments['m00'] == 0:
                continue
            cx = int(moments['m10'] / moments['m00']+0.000001)
            cy = int(moments['m01'] / moments['m00']+0.000001)
            bead_positions.append([cy, cx])


        # Fit a line to the bead positions using PCA
        cov = np.cov(np.array(bead_positions).T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        line_dir = eigenvectors[:, np.argmax(eigenvalues)]
        mean_pos = np.mean(bead_positions, axis=0)
        x1 = int(mean_pos[1] - 1000 * line_dir[1])
        y1 = int(mean_pos[0] - 1000 * line_dir[0])
        x2 = int(mean_pos[1] + 1000 * line_dir[1])
        y2 = int(mean_pos[0] + 1000 * line_dir[0])

        # Compute the intersection of each bead with the line
        bead_intersections = []
        bead_ids = []
        for j, bead_pos in enumerate(bead_positions):
            # Compute the projection of the bead onto the line
            a = np.array([x1, y1])
            b = np.array([x2, y2])
            p = np.array(bead_pos)
            ap = p - a
            ab = b - a
            proj = a + np.dot(ap, ab) / np.dot(ab, ab) * ab

            # Compute the distance from the projection to the bead
            dist = np.linalg.norm(p - proj)

            # If the distance is small enough, use the projection as the intersection point
            if dist < sensitivity:
                bead_intersections.append(proj)
            else:
                # Otherwise, use the bead position as the intersection point
                bead_intersections.append(bead_pos)

            # Append the bead ID to the list
            bead_ids.append(j+1)

        bead_positions_all.append(bead_positions)
        bead_intersections_all.append(bead_intersections)
        bead_ids_all.append(bead_ids)
        
        # Assign unique IDs to each bead
        num_beads = len(bead_positions)
        bead_ids = np.arange(1, num_beads + 1)

        if not silent:
            print(f"Detected {num_beads} beads with IDs {bead_ids}")
            print(f"Bead positions: {bead_positions}")
            print(f"Bead intersections: {bead_intersections}")

        # Show the result for some specific projections, contour_img is not implemented
        # if i in [0, 9, 19, 29]:
        #     fig, ax = plt.subplots()
        #     ax.imshow(contour_img)
        #     ax.set_title(f"Projection {i+1}")
        #     ax.set_xlabel('X position')
        #     ax.set_ylabel('Y position')
        #     ax.set_aspect('equal', 'box')
        #     plt.show()


    return bead_positions_all, bead_intersections_all, bead_ids_all


def get_trajectories(img, bead_positions,silent=True):
    # Define the initial feature points as the bead positions in the first projection
    feature_points = np.array(bead_positions[0], dtype=np.float32).reshape(-1, 1, 2)

    # Define the Lucas-Kanade parameters
    lk_params = dict(winSize=(15, 15),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Loop over each subsequent projection and track the feature points
    trajectories = []
    for i in range(1, img.shape[0]):
        # Convert the images to CV_8U format
        img1 = cv2.convertScaleAbs(img[i-1])
        img2 = cv2.convertScaleAbs(img[i])

        # Calculate the optical flow of the feature points from the previous frame to the current frame
        p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, feature_points, None, **lk_params)


        # Remove any feature points that were not tracked successfully
        st = st.reshape(-1)
        feature_points = feature_points[st == 1]
        p1 = p1[st == 1]

        # Detect new feature points in the current frame
        new_features = np.array(bead_positions[i], dtype=np.float32).reshape(-1, 1, 2)
        feature_points = np.vstack((feature_points, new_features))

        # Calculate the trajectories of the feature points up to the current frame
        trajectories.append(feature_points.reshape(-1, 2))

    # Convert the trajectories to NumPy arrays
    # trajectories = np.array(trajectories)

    return trajectories

def remove_ellipse_outliers(trajectories, threshold=1.5):
    major_axis_lengths = [ellipse[1][1] for ID, (trajectory, ellipse) in trajectories.items() if ellipse is not None]

    # Calculate the median major axis length
    median_major_axis_length = np.median(major_axis_lengths)

    # Define a threshold for major axis length as 2 times the median
    major_axis_length_threshold = 1.5 * median_major_axis_length

    # Remove outlier ellipses with major axis length greater than the threshold
    for ID, (trajectory, ellipse) in list(trajectories.items()):
        if ellipse is not None and ellipse[1][1] > major_axis_length_threshold:
            trajectories.pop(ID)
    
    return trajectories

def plot_trajectories(trajectories, bead_positions=None, showEllipses=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    for bpos in bead_positions:
        tmp = np.array(bpos)
        ax.scatter(tmp[:, 1], tmp[:, 0], color='grey', marker='o', s=5)

    
    for ID in trajectories:
        if trajectories[ID][1] is not None:
            # Plot the trajectory and ellipse
            trajectory = trajectories[ID][0]
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'b')

            if showEllipses:
                ellipse = trajectories[ID][1]   
                ellipse_patch = Ellipse(xy=ellipse[0], width=ellipse[1][0], height=ellipse[1][1], angle=ellipse[2], edgecolor='r', facecolor='none')
                ax.add_patch(ellipse_patch)
    
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('Bead Positions and Trajectories')
    

    plt.show()

def get_trajectories2(img, bead_positions,bead_ids,silent=True):
    # Find the minimum number of beads in all the projections
    min_beads = np.min([len(bead_positions[i]) for i in range(img.shape[0])])

    # Remove extra beads from each projection
    for i in range(img.shape[0]):
        if len(bead_positions[i]) > min_beads:
            num_extra_beads = len(bead_positions[i]) - min_beads
            bead_positions[i] = bead_positions[i][num_extra_beads:]

    # Convert the remaining bead positions to NumPy arrays
    bead_positions_all = [np.array(bead_positions) for bead_positions in bead_positions]

    # Define the initial feature points as the bead positions in the first projection, along with their IDs
    initial_bead_positions = bead_positions[0][:min_beads]
    initial_bead_ids = bead_ids[0][:min_beads]
    feature_points = np.hstack((initial_bead_positions, np.array(initial_bead_ids).reshape(-1, 1))).astype(np.float32).reshape(-1, 1, 3)

    # Define the Lucas-Kanade parameters
    lk_params = dict(winSize=(20, 20),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Loop over each subsequent projection and track the feature points
    trajectories = {}
    for i in range(1, img.shape[0]):
        # Calculate the optical flow of the feature points from the previous frame to the current frame
        img0 = img[i-1].astype(np.uint8)
        img1 = img[i].astype(np.uint8)  
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, 
                                               img1, 
                                               feature_points[:, :, :2], 
                                               None, 
                                               **lk_params)

        # Remove any feature points that were not tracked successfully
        st = st.reshape(-1)
        feature_points = feature_points[st == 1]
        p1 = p1[st == 1]

        # Update the IDs of the feature points
        if feature_points.size > 0:
            feature_points[:, :, 2] = feature_points[:, :, 2] * st.reshape(-1, 1)

        # Detect new feature points in the current frame
        new_features = np.hstack((bead_positions_all[i][:min_beads], np.array(bead_ids[i][:min_beads]).reshape(-1, 1))).astype(np.float32).reshape(-1, 1, 3)
        feature_points = np.vstack((feature_points, new_features))

        # Calculate the trajectories of the feature points up to the current frame
        for j in range(feature_points.shape[0]):
            x, y, ID = feature_points[j, 0]
            if ID not in trajectories:
                trajectories[ID] = []
            trajectories[ID].append([y, x])

    # Convert the trajectories to NumPy arrays
    for ID in trajectories:
        trajectories[ID] = np.array(trajectories[ID])


    # Fit ellipses to each bead trajectory up to the current frame
    for ID in trajectories:
        if len(trajectories[ID]) > 10: # Only fit ellipse if at least 11 points have been tracked
            # Convert the trajectory to a NumPy array
            trajectory = np.array(trajectories[ID])

            # Fit an ellipse to the trajectory using the method of moments
            ellipse = cv2.fitEllipse(trajectory)

            # Add the ellipse parameters to the trajectory dictionary
            trajectories[ID] = (trajectory, ellipse)
        else:
            # If there are not enough points, add None to the trajectory dictionary
            trajectories[ID] = (trajectory, None)

    return trajectories


# fitting an ellipse 
# based on Fitzgibbon, Pilu, & Fisher (1996) and Halir & Flusser (1998)
def fit(x,y):
    D1   = np.vstack([x**2,x*y,y**2]).T
    D2   = np.vstack([x,y,np.ones_like(x)]).T
    S1,S2,S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    C1   = np.array([[0,0,2],[0,-1,0],[2,0,0]])
    M    = np.linalg.inv(C1) @ (S1 - S2 @ np.linalg.inv(S3) @ S2.T)
    vec  = np.linalg.eig(M)[1]
    cond = 4*(vec[0]*vec[2]) - vec[1]**2
    a1   = vec[:,np.nonzero(cond > 0)[0]]
    return np.vstack([a1,np.linalg.inv(-S3) @ S2.T @ a1]).flatten()

def errors(x,y,coeffs):
    z           = np.vstack((x**2,x*y,y**2,x,y,np.ones_like(x)))
    numerator   = np.sum(((coeffs.reshape(1,6) @ z)-1)**2)
    denominator = (len(x)-6)*np.sum(z**2,axis=1)
    unc         = np.sqrt(numerator/denominator)
    return tuple(ufloat(i,j) for i,j in zip(coeffs,unc))

def convert(coeffs):
    """ Convert the fitting parameters to ellipse parameters with uncertainties
    Args:
    - coeffs: the coefficients of the fitted ellipse

    Returns: The center, major axis, minor axis, and rotation angle of the fitted ellipse
    """
    a,b,c,d,e,f = coeffs
    b /= 2
    d /= 2
    e /= 2
    x0 = (c*d - b*e) / (b**2 - a*c)
    y0 = (a*e - b*d) / (b**2 - a*c)
    center       = (x0, y0)
    numerator    = 2 * (a*e**2 + c*d**2 + f*b**2 - 2*b*d*e - a*c*f)
    denominator1 = (b*b-a*c)*((c-a)*umath.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    denominator2 = (b*b-a*c)*((a-c)*umath.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    major = umath.sqrt(numerator/denominator1) if numerator/denominator1 > 0 else 0
    minor = umath.sqrt(numerator/denominator2) if numerator/denominator2 > 0 else 0
    phi   = .5*umath.atan((2*b)/(a-c))
    major, minor, phi = (major, minor, phi) if major > minor else (minor, major, np.pi/2+phi)
    return center, major, minor, phi

def line(coeffs,n=100):
    t = np.linspace(0,2*np.pi,n)
    center,major,minor,phi = convert(coeffs)
    x = major*np.cos(t)*np.cos(phi) - minor*np.sin(t)*np.sin(phi) + center[0]
    y = major*np.cos(t)*np.sin(phi) + minor*np.sin(t)*np.cos(phi) + center[1]
    return x,y

# alternative using matplotlib artists
def artist(coeffs,*args,**kwargs):
    center,major,minor,phi = convert(coeffs)
    return Ellipse(xy=(center[0],center[1]),width=2*major,height=2*minor,
                     angle=np.rad2deg(phi),*args,**kwargs)

# obtaining the confidence interval/area for ellipse fit
def confidence_area(x,y,coeffs,f=1): # f here is the multiple of sigma
    c = coeffs
    res = c[0]*x**2 + c[1]*x*y + c[2]*y**2 + c[3]*x + c[4]*y + c[5]
    sigma = np.std(res)
#     print('Sigma = ', sigma)
    c_up = np.array([c[0],c[1],c[2],c[3],c[4],c[5] + f*sigma])
    c_do = np.array([c[0],c[1],c[2],c[3],c[4],c[5] - f*sigma])
   
    if convert(c_do) > convert(c_up):
        c_do, c_up = c_up,c_do
    return c_up,c_do

def fit_ellipses(trajectories, silent=True, iterations = 5) :
    trajectories_x = []
    trajectories_y = []

    for ID in trajectories:
        if trajectories[ID][1] is not None:
            trajectories_x.append(np.ravel(trajectories[ID][0][:, 0]).tolist())
            trajectories_y.append(np.ravel(trajectories[ID][0][:, 1]).tolist())

    trajectories_x = np.array(trajectories_x)
    trajectories_y = np.array(trajectories_y)

    ell_all   = []
    err_all   = []
    Sigma_all = [] 
    M = len(trajectories)
    if not silent:
        print('M = ', M)

    for iii in range(M):
        
        if not silent:
            print('ID = ', iii)
        
        sigma_ell = []

        x = trajectories_x[iii]
        y = trajectories_y[iii]

        for ell_ii in range(iterations): # This should be cleaned up
            # do fitting
            params1 = fit(x,y)
            params1 = errors(x,y,params1)


            center, a, b, phi = convert(params1)
            c_up, c_do = confidence_area(x,y,[i.n for i in params1],f=2) 
            if convert(c_do) > convert(c_up):
                c_do, c_up = c_up,c_do

            # do the outlier removal
            mask1 = artist(c_up,ec='none') 
            mask2 = artist(c_do,ec='none') 

            # The ellipse1
            g_ell_center = getattr(mask1, "center")
            g_ell_width  = getattr(mask1, "width")
            g_ell_height = getattr(mask1, "height")
            angle        = getattr(mask1, "angle")

            # g_ellipse = patches.Ellipse(g_ell_center, g_ell_width, g_ell_height, angle=angle, fill=False, edgecolor='skyblue', linewidth=2)
            # ax.add_patch(g_ellipse)

            cos_angle = np.cos(np.radians(180.-angle))
            sin_angle = np.sin(np.radians(180.-angle))

            xc = x - g_ell_center[0]
            yc = y - g_ell_center[1]

            xct = xc * cos_angle - yc * sin_angle
            yct = xc * sin_angle + yc * cos_angle 

            rad_cc = (xct**2/(g_ell_width/2.)**2) + (yct**2/(g_ell_height/2.)**2)

            # The ellipse2
            # g_ell_center2 = getattr(mask2, "center")
            # g_ell_width2  = getattr(mask2, "width")
            # g_ell_height2 = getattr(mask2, "height")
            # angle2        = getattr(mask2, "angle")

            # # g_ellipse2 = patches.Ellipse(g_ell_center2, g_ell_width2, g_ell_height2, angle=angle2, fill=False, edgecolor='skyblue', linewidth=2)
            # # ax.add_patch(g_ellipse2)

            # cos_angle2 = np.cos(np.radians(180.-angle2))
            # sin_angle2 = np.sin(np.radians(180.-angle2))

            # xc = x - g_ell_center[0]
            # yc = y - g_ell_center[1]

            # xct = xc * cos_angle - yc * sin_angle
            # yct = xc * sin_angle + yc * cos_angle 

            # rad_cc = (xct**2/(g_ell_width/2.)**2) + (yct**2/(g_ell_height/2.)**2)

            # The ellipse2
            g_ell_center2 = getattr(mask2, "center")
            g_ell_width2  = getattr(mask2, "width")
            g_ell_height2 = getattr(mask2, "height")
            angle2        = getattr(mask2, "angle")

            # g_ellipse2 = patches.Ellipse(g_ell_center2, g_ell_width2, g_ell_height2, angle=angle2, fill=False, edgecolor='skyblue', linewidth=2)
            # ax.add_patch(g_ellipse2)

            cos_angle2 = np.cos(np.radians(180.-angle2))
            sin_angle2 = np.sin(np.radians(180.-angle2))

            xc2  = x - g_ell_center2[0]
            yc2  = y - g_ell_center2[1]

            xct2 = xc2 * cos_angle2 - yc2 * sin_angle2
            yct2 = xc2 * sin_angle2 + yc2 * cos_angle2 

            rad_cc2 = (xct2**2/(g_ell_width2/2.)**2) + (yct2**2/(g_ell_height2/2.)**2)

            # define new X and Y as modified
            xy_array = np.array([x,y])
            indices1 = np.where(rad_cc2 <= 1.)[0]
            if len(indices1) > 0:
                modified_array1 = np.delete(xy_array, indices1, 1)
                indices2 = np.intersect1d(np.where(rad_cc >= 1.)[0], np.arange(modified_array1.shape[1]))
                if len(indices2) > 0:
                    modified_array2 = np.delete(modified_array1, indices2, 1)
                    x = modified_array2[0]
                    y = modified_array2[1]
                else:
                    x = modified_array1[0]
                    y = modified_array1[1]
            else:
                pass
    
            sigma =  abs(((c_up-c_do)/2)[5])
            sigma_ell.append(sigma)


        ell = convert(params1)
        ell_all.append(ell)
        Sigma_all.append(sigma_ell)
        # plot of fit and confidence area
        if not silent:
            fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
            ax1.plot(x,y,'sr',label='Data')
            ax1.plot(*line([i.n for i in params1]),'--b',lw=1,label='Fit')
            ax1.add_patch(artist(c_up,ec='none',fc='r',alpha=0.15,label=r'2$\sigma$'))
            ax1.add_patch(artist(c_do,ec='none',fc='white'))
            ax1.legend()
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax2.plot(sigma_ell)
            ax2.set_title('sigma decay')

            print('last sigma value = ', sigma_ell[-1])
            plt.show()
    sigma_ell = np.array(Sigma_all)
    return ell_all, sigma_ell

def ellipses_to_geometry(ellipses) :
    geometry = {'x_centers': [],
                'y_centers': [],
                'major_axes': [],
                'minor_axes': [],
                'angles': [],
                'stddev': {
                    'x_centers': [],
                    'y_centers': [],
                    'major_axes': [],
                    'minor_axes': [],
                    'angles': []
                }}
    for ellipse in ellipses:
        geometry['x_centers'].append(un.nominal_value(ellipse[0][0]))
        geometry['y_centers'].append(un.nominal_value(ellipse[0][1]))
        geometry['major_axes'].append(un.nominal_value(ellipse[1]))
        geometry['minor_axes'].append(un.nominal_value(ellipse[2]))
        geometry['angles'].append(un.nominal_value(ellipse[3]))
        geometry['stddev']['x_centers'].append(un.std_dev(un.nominal_value(ellipse[0][0])))
        geometry['stddev']['y_centers'].append(un.std_dev(un.nominal_value(ellipse[0][1])))
        geometry['stddev']['major_axes'].append(un.std_dev(un.nominal_value(ellipse[1])))
        geometry['stddev']['minor_axes'].append(un.std_dev(un.nominal_value(ellipse[2])))
        geometry['stddev']['angles'].append(un.std_dev(un.nominal_value(ellipse[3])))

    return geometry

def plot_ellipse_parameters(geom) :
    fig,axis=plt.subplots(1,2,figsize=(15,5))
    axis[0].plot(geom['x_centers'],geom['y_centers'],'o')
    #axis[0].axis('equal')
    axis[0].set(title='Ellipse Centers',xlabel='X position', ylabel='Y position')
    axis[1].plot(geom['minor_axes'],geom['y_centers'],'or',label='Minor axes')
    axis[1].plot(geom['major_axes'],geom['y_centers'],'ob',label='Major axes')
    axis[1].legend()
    axis[1].set(title='Ellipse Radii',xlabel='Radius', ylabel='Y position');

def build_lines(geometry):
    line_points = np.zeros((len(geometry['x_centers']),4))
    N=line_points.shape[0]
    for i in range(N):
        x0 = geometry['y_centers'][i]
        y0 = geometry['x_centers'][i]
        angle = geometry['angles'][i]
        min_axis = geometry['minor_axes'][i]
        x1 = x0 + min_axis * np.cos(np.rad2deg(angle))
        y1 = y0 + min_axis * np.sin(np.rad2deg(angle))
        line_points[i]=[x0, y0, x1, y1]
    
    return line_points

def plot_lines(line_points,ax=None):
    if ax is None: 
        fig, ax = plt.subplots(figsize=(5,5))

    for i in range(line_points.shape[0]):
        x0, y0, x1, y1 = line_points[i]
        ax.plot([y0, y1], [x0, x1], 'r-')
    # ax.set_aspect('equal', 'box')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    plt.show()

def fit_piercing_point(geometry, silent=True):
    line_points = build_lines(geometry)
    lines = []
    threshold = 10  # distance threshold for RANSAC
    num_iterations = 100  # number of iterations for RANSAC

    for idx in np.arange(line_points.shape[0]):
        x0, y0, x1, y1 = line_points[idx]
        x = np.array([x0, x1]).reshape(-1, 1)
        y = np.array([y0, y1])
        ransac_model = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(),
                                                    residual_threshold=threshold,
                                                    max_trials=num_iterations)
        ransac_model.fit(x, y)
        k, b = ransac_model.estimator_.coef_, ransac_model.estimator_.intercept_
        lines.append([k[0], -1, b])  # convert line equation to the form ax + by = c

    # Find intersection point of lines using linear algebra
    A = np.array([[line[0], line[1]] for line in lines])
    b = np.array([-line[2] for line in lines])
    x_int, y_int = np.linalg.lstsq(A, b, rcond=None)[0]

    # Print results
    if not silent:
        fig, ax = plt.subplots(1,2,figsize=(15, 5))
        plot_lines(line_points,ax=ax[0])
        # for idx in range(len(geometry['x_centers'])):
        #     x=geometry['x_centers'][idx]
        #     y=geometry['y_centers'][idx]
        #     minor_axis=geometry['minor_axes'][idx]
        #     major_axis=geometry['major_axes'][idx]
        #     angle=geometry['angles'][idx]
        #     ellipse = patches.Ellipse((y, x), 2*major_axis, 2*minor_axis, angle=angle, fill=False, edgecolor='r', linewidth=1)
        #     ax[1].add_patch(ellipse)
        ax[1].plot(y_int, x_int, 'bo',label='Intersection point')
        ax[1].set_xlabel('X position')
        ax[1].set_ylabel('Y position')
        ax[1].legend()
        print(f"Intersection point: ({x_int},{y_int})")
    
    return x_int, y_int

def center_of_rotation(ellipses, silent=True, oldversion=False,ax = None):
    if ax is None and not silent:
        fig,ax = plt.subplots(1,1,figsize=(5,5))

    M = len(ellipses)
    if oldversion:
        x_centres=[]
        y_centres=[]

        for i in range(len(ellipses)):
            x_centres.append(ellipses[i][0][1])
            y_centres.append(ellipses[i][0][0])

        arr_x = np.array(x_centres)
        
        arr_x_nominal = []
        arr_x_std     = []
        for iii in range(M):
            arr_x_nominal.append(un.nominal_value(un.nominal_value(arr_x)[iii]))
            arr_x_std.append(un.std_dev(un.nominal_value(arr_x)[iii]))
        arr_x_nominal_arr = np.array(arr_x_nominal).reshape(M,1)
        arr_x_std_arr     = np.array(arr_x_std).reshape(M,1)


        arr_y = np.array(y_centres)

        arr_y_nominal = []
        arr_y_std = []
        for iii in range(len(un.nominal_value(arr_y))):
            arr_y_nominal.append(un.nominal_value(un.nominal_value(arr_y)[iii]))
            arr_y_std.append(un.std_dev(un.nominal_value(arr_y)[iii]))
        arr_y_nominal_arr = np.array(arr_y_nominal).reshape(M,1)
        arr_y_std_arr     = np.array(arr_y_std).reshape(M,1)


        # Removing the outlier in the data 
        elements_x = arr_x_nominal_arr
        elements_y = arr_y_nominal_arr
    else:
        elements_x=np.array([un.nominal_value(ellipses[idx][0][1]) for idx in range(len(ellipses))]).reshape(M,1)
        elements_y=np.array([un.nominal_value(ellipses[idx][0][0]) for idx in range(len(ellipses))]).reshape(M,1)

    median_x = np.median(elements_x, axis=0)
    sd_x     = np.std(elements_x, axis=0)

    outlier_x = [x for x in elements_x if (x > median_x + 1 * sd_x)]
    if not silent:
        print('outlier values: ', outlier_x)

    for i in range(len(outlier_x)):
        index_x        = np.where(elements_x == outlier_x[i-1])
        final_list_x_1 = np.array(np.delete(elements_x, index_x)).reshape(-1, 1)
        final_list_y_1 = np.array(np.delete(elements_y, index_x)).reshape(-1, 1)
        index_x_2      = np.where(final_list_x_1 == outlier_x[i])
        final_list_x   = np.array(np.delete(final_list_x_1, index_x_2)).reshape(-1, 1)
        final_list_y   = np.array(np.delete(final_list_y_1, index_x_2)).reshape(-1, 1)

    # Using RANSACRegressor for minimum sensitivity to the outliers      
    ransac      = linear_model.RANSACRegressor(random_state=10).fit(final_list_x, final_list_y)
    plotline_y  = np.arange(final_list_x.min(), final_list_x.max()).reshape(-1, 1)
    a = ransac.predict(plotline_y)
    ransac_coef = ransac.estimator_.coef_

    COR_new = np.median(final_list_y) 
    tilt_angle = np.degrees(np.arctan(ransac_coef))
    
    if not silent:
        print(ransac_coef)
        devaition_range = final_list_y.max()- final_list_y.min()
        print('number of pixels the center of rotation deviated from top to bottom of the tomogram: ', np.abs(devaition_range))
        print('Center of rotation vertical: ', COR_new)
        # form above (cell) estimation:
        print('Tilt slope: ', ransac_coef.ravel())
        # print('Tilt degree: ', ransac_coef.ravel())

        #### Final value of COR can be obtained by subtractin of the projection value taken 
        print("The tilt of the center of rotation in degrees: ", tilt_angle)
        ax.plot(a, plotline_y,'r', label='Tilted center line')
        ax.scatter(final_list_y, final_list_x, label='Ellipse centers')
        #ax.hlines(COR_new, xmin = final_list_x.min() , xmax =final_list_x.max() , colors='gray', linestyles='--', label='')
        ax.axvline(COR_new,color='gray', linestyle='--', label='Center') #, xmin = final_list_x.min() , xmax =final_list_x.max() , colors='gray', linestyles='--', label='')
        ax.text(COR_new+0.2,final_list_x.min(), 'Center of rotation');
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.legend()   
    return COR_new, tilt_angle[0,0]

def fit_beamline(ellipses,vpiercing,pixelSize=0.139, ringRadius=10.0,silent=True):
    if not silent:
        fig,ax = plt.subplots(1,2,figsize=(15,5))

    R = 10              # mm
    # c0=itemList(findBeadsWS(tcal[0,:,:]))
    sod = []
    sdd = []
    mag = []
    cte = pixelSize/R
    for idx in range(len(ellipses)) :
        ha=(ellipses[idx][0][1]-ellipses[idx][2]-vpiercing) * pixelSize
        hb=(ellipses[idx][0][1]+ellipses[idx][2]-vpiercing) * pixelSize

        est_sod= (hb+ha)*R/(hb-ha)
        sod.append(np.abs(est_sod))

        magnification_each_elipse = cte * (ellipses[idx][1]) 
        sdd_each_elipse = magnification_each_elipse * est_sod
        sdd.append(np.abs(sdd_each_elipse))
        mag.append(magnification_each_elipse)

    sod_mean = np.median([un.nominal_value(x) for x in sod])
    sdd_mean = np.median([un.nominal_value(x) for x in sdd])
    mag_mean = np.median([un.nominal_value(x) for x in mag])
    mag_ratio = sdd_mean/sod_mean

    if not silent:
        print("SOD median= ", un.nominal_value(sod_mean))
        print("SDD median= ", un.nominal_value(sdd_mean))
        print("Magnification median= ", mag_mean)
        print("Magnification ratio= ", mag_ratio)

    index = []
    sod_nominal = []
    for i in range(len(sod)):
        sod_nominal.append(un.nominal_value(np.mean(sod[i])))
        index.append(np.argmin(un.nominal_value(np.mean(sod[i])))+1)
    sod_nominal_array = np.array(sod_nominal)    
    sod_mean2 = np.mean(sod_nominal)
    sod_std   = np.std(sod_nominal)

    if not silent:
        print('SOD stdev= ',sod_std)
        ax[0].boxplot(sod_nominal)
        ax[0].plot(1, sod_mean2,'sr', label='Mean SOD= {0:0.2f}'.format(sod_mean2))
        ax[0].scatter(index, sod_nominal,label='SODs')
        ax[0].set_title("SOD")
        ax[0].legend()


    index2 = []
    sdd_nominal = []
    for i in range(len(sdd)):
        sdd_nominal.append(un.nominal_value(np.mean(sdd[i])))
        index2.append(np.argmin(un.nominal_value(np.mean(sdd[i])))+1)
    sdd_nominal_array = np.array(sdd_nominal)    
    sdd_mean2 = np.mean(sdd_nominal)
    sdd_std = np.std(sdd_nominal)
    
    if not silent:
        print('SDD stdev= ', sdd_std)
        ax[1].boxplot(sdd_nominal)
        ax[1].plot(1, sdd_mean2,'sr',label='Mean SDD= {0:0.2f}'.format(sdd_mean2))
        ax[1].scatter(index2, sdd_nominal,label='SDDs')
        ax[1].set_title("sdd")  
        ax[1].legend()

        print("SOD mean= ", sod_mean2)
        print("SDD mean= ",sdd_mean2)
    return sod_mean2, sdd_mean2, mag_mean, mag_ratio

def fit_cbct_geometry():
    return None


# Old functions
# def thresholdBBs(bcal,k) :
#     s=bcal.std()
#     m=bcal.mean()
    
#     return (m+k*s)< bcal


# def min_impose(dimg,markers) :
#     fm=markers.copy()
#     fm[markers != 0] = 0
#     fm[markers == 0] = dimg.max()
#     dimg2 = np.minimum(fm,dimg+1)
#     res   = gr.reconstruction(fm,dimg2,method='erosion')
    
#     return res

# def randomCM(N, low=0.2, high=1.0,seed=42, bg=0) :
#     np.random.seed(seed=seed)
#     clist=np.random.uniform(low=low,high=high,size=[N,3]); 
#     m = ortho_group.rvs(dim=3)
#     if bg is not None : 
#         clist[0,:]=bg;
        
#     rmap = ListedColormap(clist)
    
#     return rmap

# def goldenCM(N,increment=1.0,s=0.5,v=0.7,bg=0) :
#     phi= 0.5*(np.sqrt(5)-1)
    
#     hsv = np.zeros([N,3]);
#     hsv[:, 0] = increment*phi*np.linspace(0,N-1,N)-np.floor(increment*phi*np.linspace(0,N-1,N))
#     hsv[:, 1] = s
#     hsv[:, 2] = v
#     rgb = hsv2rgb(hsv)
#     if bg is not None : rgb[0,:]=bg    
#     cm = ListedColormap(rgb) 
#     return cm


# def watershed_segmentation_display(img):
#     cog,ws1=findBeadsWS(img)
    
#     labels = np.unique(ws1).astype(int)
    
#     fig,ax=plt.subplots(1,1,figsize=(10,10))
#     ax.imshow(ws1,interpolation='None',cmap=goldenCM(labels.max()))
#     ax.plot(cog[:,1],cog[:,0],'r+')

# def findBeadsWS(img, selem= disk(3),h=2) :
#     p=img.mean(axis=0)
#     m=np.where((0.5*p.max())<p)
#     a=np.min(m)
#     b=np.max(m)
#     w=b-a
#     a=a-w
#     b=b+w
#     if a<0 : a=0
#     if len(p)<=b : b=len(p)-1
#     img2=img[:,a:b]
#     distance = ndi.distance_transform_edt(erode(img2,selem))

#     localmax = h_maxima(distance,h)
#     rdmap    = distance.max()-distance
#     labels   = label(localmax)
#     ws1 = watershed(min_impose(rdmap,labels),labels,mask=img2)
    
#     rp = regionprops_table(ws1,properties=('area','centroid'))
       
#     cog=np.zeros([rp['centroid-0'].shape[0],2])
#     cog[:,0]=rp['centroid-0']
#     cog[:,1]=rp['centroid-1']+a
#     ws = np.zeros(img.shape)
#     ws[:,a:b] = ws1

#     return cog,ws

# def buildBeadList(img,selem=disk(12),c=1.96) :
#     beadlist = []

#     for idx in np.arange(0, img.shape[2]) :
#         cog=findBeadsWS(img[:,:,idx],selem,c)
#         beadlist.append(cog)

#     return beadlist


# def display_beads(tcal, idx):
#     c=4
#     cog=findBeadsWS(tcal[idx,:,:])

#     plt.figure(figsize=[12,8])
#     plt.imshow(tcal[idx,:,:])

#     plt.plot(cog[:,1]-1,cog[:,0]-1,'r+')

#     cog1=findBeadsWS(tcal[idx+1,:,:])
#     plt.plot(cog1[:,1]-1,cog1[:,0]-1,'rx')

#     cog2=findBeadsWS(tcal[idx-1,:,:])
#     plt.plot(cog2[:,1]-1,cog2[:,0]-1,'ro')
    
    
# def getBeads(img,selem=disk(2)) :
#     N=img.shape[0]
  
#     beads = []
    
#     for proj in tqdm(np.arange(0,N)) :
#         cog,_ = findBeadsWS(img[proj,:,:])
#         beads.append(cog)
#     return beads

# def identifyEllipses_old(img,selem=disk(2)) :
#     N=img.shape[0]
#     ellipses = []
#     params=[]
#     cog_allbeads=[]    
#     beads = []
#     for proj in tqdm(np.arange(0,N)) :
#         cog = findBeadsWS(img[proj,:,:])
#         beads.append(cog)


#     for idx in range(len(min(beads,key=len))):
#         ellipse = []
#         for p in range(N) :
#             ellipse.append(beads[p][idx,:].tolist())
#         ellipses.append(np.array(ellipse))
#         ell = EllipseModel()
#         a_ellipse = np.array(ellipse)
#         ell.estimate(a_ellipse)
#         if ell.params==None:
#             continue
#         cog_onebead=[]
#         for p in range(N) :
#             cog_onebead.append(beads[p][idx,:])
#         cog_allbeads.append(cog_onebead)
#         xc, yc, a, b, theta = ell.params
#         if theta> 1:
#             theta=theta-(np.pi/2)
#         params.append([yc,xc,max(a,b),min(a,b),theta])
#     return params,cog_allbeads

# def identifyEllipses(img,selem=disk(2), beads=None) :
#     N=img.shape[0]
    
#     ellipses = []
#     params=[]
#     cog_allbeads=[]  
#     res = []
#     if beads is None :
#         beads = getBeads(img,selem)
    
#     for idx in range(len(min(beads,key=len))):
#         ellipse = []
#         for p in range(N) :
#             ellipse.append(beads[p][idx,:].tolist())
#         ellipses.append(np.array(ellipse))
#         ell = EllipseModel()
#         a_ellipse = np.array(ellipse)
#         ell.estimate(a_ellipse)
#         if ell.params==None:
#             continue
#         cog_onebead=[]
#         r=ell.residuals(a_ellipse)
#         res.append(4)
        
#         sort=np.argsort(r)
        
#         ell.estimate(a_ellipse[sort[:len(sort)//2],:])
#         for p in range(N) :
#             cog_onebead.append(beads[p][idx,:])
#         cog_allbeads.append(cog_onebead)
#         xc, yc, a, b, theta = ell.params
#         if theta> 1:
#             theta=theta-(np.pi/2)
#         params.append([yc,xc,max(a,b),min(a,b),theta])
#     return params,cog_allbeads,res

# def show_ellipses(e2,cog_allbeads):
#     #for idx in range(len(min(cog_allbeads,key=len))):
#     for idx in range(len(e2)):
#         print("ID Number = ", idx)
#         print("center = ",  (e2[idx][0], e2[idx][1]))
#         theta=e2[idx][4]
#         print("angle of rotation = ",  theta)
#         print("axes major/minor = ", e2[idx][2],e2[idx][3])
#         a_ellipse=np.array(cog_allbeads[idx])
#         x=a_ellipse[:,0]
#         y=a_ellipse[:,1]
#         fig, axs = plt.subplots(1,2, figsize=(15,4),sharex=True, sharey=True)
#         axs[0].scatter(y, x)
#         for (num,(yy,xx)) in enumerate(zip(y,x)) :
#             axs[0].annotate("{0}".format(num),xy=(yy,xx),  xycoords='data')

#         axs[1].scatter(y, x)
#         axs[1].scatter(e2[idx][0], e2[idx][1], color='red', s=100)

#         ell_patch = Ellipse((e2[idx][0], e2[idx][1]), 2*e2[idx][2], 2*e2[idx][3], theta, edgecolor='red', facecolor='none')

#         axs[1].add_patch(ell_patch)
#         plt.show()

        
# def estimate_cor(e2):
#     x_centres=[]
#     y_centres=[]
#     for i in range(len(e2)):
#         x_centres.append(e2[i][0])
#         y_centres.append(e2[i][1])
        
#     x_centres=np.array(x_centres)
#     y_centres=np.array(y_centres)
#     theta=np.polyfit(y_centres, x_centres, 1)
#     res= np.abs(theta[1]+theta[0]*np.array(y_centres)-x_centres)
#     idx = np.where(res<(res.mean()+res.std()))
#     theta=np.polyfit(y_centres[idx[0]], x_centres[idx[0]], 1)
    
#     plt.scatter(x_centres,y_centres)
#     plt.plot(theta[1]+theta[0]*np.array(y_centres), y_centres ,'r')
#     plt.show()
#     print("The parameters of the COR obtained are as follows:",(theta[1]))
#     tilt = np.arctan(theta[0])*180/np.pi
#     print("The tilt of the center of rotation is in degrees: ", tilt)
#     result ={"COR" : theta[1], "Ctilt" : tilt}
#     return result
    
# def estimate_magnification(tcal,idx):
#     cog,_=findBeadsWS(tcal[idx,:,:])
#     pixel_pitch=0.139
#     d=np.sort(np.diff(cog[:,0]))
#     m=d[6:-6].mean()
#     s=d[6:-6].std()
#     k=1
#     plt.plot(np.diff(cog[:,0]),'.')
#     plt.plot(d,'.')
#     plt.fill_between([0, 37],y1=[m-k*s,m-k*s],y2=[m+k*s,m+k*s],alpha=0.2)

#     print(pixel_pitch*np.array([m,d.mean(),np.median(d)]))
#     return pixel_pitch*m
    
# def plot_allellipses(e2):
#     fig, ax = plt.subplots()

#     for idx in range(len(e2)):
#         ellipse = Ellipse((e2[idx][0], e2[idx][1]), e2[idx][2]*2, e2[idx][3]*2, e2[idx][4])
#         ax.add_artist(ellipse)
#     ax.set_xlabel("x_coordinate")
#     ax.set_ylabel("y_coordinate")
#     ax.set_xlim(500, 1500)
#     ax.set_ylim(800, 2000)
#     plt.show()
    
    
# def estimate_piercingpoint(e2):
#     radius=[]
#     height=[]
    
#     for i in range(len(e2)):
#         radius.append(e2[i][3])
#         height.append(e2[i][1])
        
#     theta =[]
#     temp_r= radius.copy()
#     temp_h= height.copy()
#     k=0
#     while k!=1:
#         count=0
#         theta=np.polyfit(height, radius, 1)
#         radius= temp_r.copy()
#         height= temp_h.copy()
#         for x,y in zip(height, radius):
#             if (theta[1]+theta[0]*x >= y+0.5 or theta[1]+theta[0]*x <= y-0.5):
#                 count=count+1
#                 temp_h.remove(x)
#                 temp_r.remove(y)
#         if count ==0:
#             k=1
#     plt.plot(height,radius,'.')
#     plt.xlabel('height')
#     plt.ylabel('radius(minor axis length)')
    
#     plt.plot(height,theta[0]*np.array(height)+theta[1])
#     x_centres=[]
#     y_centres=[]
#     for i in range(len(e2)):
#         x_centres.append(e2[i][0])
#         y_centres.append(e2[i][1])
#     theta1=np.polyfit(y_centres, x_centres, 1)
#     pp_y= -theta[1]/theta[0]
#     pp_x= theta1[1]+theta1[0]*(pp_y)
#     print("Piercing Point is at: ",(pp_x,pp_y))
#     return pp_x, pp_y


# def medianDistance(cog) :
#     return np.median(np.diff(cog[:,0]))

# def itemList(cog) :
#     d=medianDistance(cog)
#     idx=np.floor((cog[:,0]-cog[0,0])/d+0.5).astype(int)
    
#     idxList={}
#     for (i,c) in zip(idx,cog) :
#         idxList[i]=c
        
#     return idxList
   
# def estimate_sod_sdd(tcal, e2, vpiercing, mag, factor=1.08):    
#     pixelSize = 0.139
#     R = 10
#     e0,_=findBeadsWS(tcal[0,:,:])
#     c0=itemList(e0)
#     sod = []
#     sdd = [] 
#     for idx in range(np.array(e2).shape[0]) :
#         h=0
#         if e2[idx][1] < vpiercing :
#             hb=(vpiercing - (e2[idx][1]-e2[idx][3])) * pixelSize
#             ha=(vpiercing - (e2[idx][1]+e2[idx][3])) * pixelSize
#             if idx in c0 :
#                 h = (vpiercing - c0[idx][1])*pixelSize
#         else :
#             ha=(e2[idx][1]-e2[idx][3]-vpiercing) * pixelSize
#             hb=(e2[idx][1]+e2[idx][3]-vpiercing) * pixelSize
#             if idx in c0 :
#                 h = (c0[idx][1]-vpiercing)*pixelSize


#         est_sod= (hb+ha)*R*factor/(hb-ha)
#         est_sdd= (est_sod)*mag
#         sod.append(np.abs(est_sod))
#         sdd.append(np.abs(est_sdd))
#         print("h: {0:0.3f}, SOD: {1:0.2f}, SDD: {2:0.2f}, magn: {3:0.2f}, ha: {4:0.2f}, hb: {5:0.2f}".format(h,est_sod, est_sdd,est_sdd/est_sod,ha,hb))
#     sd_sod= np.std(sod)
#     sd_sdd= np.std(sdd)
#     sod = np.mean(sod)
#     sdd = np.mean(sdd)
#     result = {'sod': sod, 'sdd': sdd, 'magnification': sdd/sod, 'sd_sod':sd_sod ,'sd_sdd': sd_sdd}
#     print("Mean SOD= ", sod)
#     print("Mean SDD= ", sdd)
#     print("Magnification= ", sdd/sod)
#     print("Standard deviation in SOD= ", sd_sod)
#     print("Standard deviation in SDD= ", sd_sdd)
#     print("Pixel size= ", pixelSize/mag)
#     return result 