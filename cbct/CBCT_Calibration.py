# CBCT calibration core functions

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as pil
import skimage as im

from scipy.signal import medfilt
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.segmentation import clear_border
from skimage.morphology import dilation
from skimage.morphology import binary_opening
from skimage.morphology import disk
from skimage.morphology import h_maxima
from skimage.morphology import label
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import matplotlib.colors as colors
from skimage.color import hsv2rgb, rgb2hsv
from matplotlib.colors import ListedColormap
from tqdm.notebook import tqdm
from skimage.measure import EllipseModel
from skimage.measure import regionprops
from matplotlib.patches import Ellipse
import uncertainties as un
from uncertainties import umath
import cv2
from scipy.stats import scoreatpercentile
from scipy.optimize import curve_fit, minimize

from scipy.stats import ortho_group
from matplotlib.colors import ListedColormap
from skimage.color      import hsv2rgb, rgb2hsv



def randomCM(N, low=0.2, high=1.0,seed=42, bg=0) :
    np.random.seed(seed=seed)
    clist=np.random.uniform(low=low,high=high,size=[N,3]); 
    m = ortho_group.rvs(dim=3)
    if bg is not None : clist[0,:]=bg;
        
    rmap = ListedColormap(clist)
    
    return rmap


class CBCTCalibration:
    def __init__(self):
        self.projections         = None
        self.projections_flat    = None
        self.projections_bilevel = None
        self.trajectories        = None
        self.ellipses            = None
        self.histogram           = None
        self.beads               = None
        self.calibration         = {"COR" : None, "SDD" : None, "SOD" : None, "pp": None}

    def set_projections(self, proj,ob,dc, verticalflip=False, show=False, amplification=10, stack_window=5):
        ob = ob-dc
        ob[ob<=0] = 1

        self.projections = np.copy(proj)
        for idx in range(proj.shape[0]):
            tmp = self.projections[idx]
            tmp = tmp-dc
            tmp[tmp<=0] = 1
            self.projections[idx] = tmp/ob
        
        self.projections[1.5<self.projections] = 1.5
        if verticalflip:
            self.projections = self.projections[:,::-1,:] 
        
        if show:
            plt.imshow(self.projections[0],vmin=0,vmax=1)

        # self.remove_projection_baseline(show=show)
        self.flatten_projections(amplification=amplification, stack_window=stack_window,show=show)

    def remove_projection_baseline(self, show=False):
        

        self.projections_flat = np.copy(self.projections)
        b2 = np.median(self.projections,axis=0)
        
        for idx in np.arange(0,self.projections_flat.shape[0]) :
            self.projections_flat[idx,:,:]=b2 - self.projections_flat[idx,:,:]

        self.projections_flat[self.projections_flat<0] = 0

        self.histogram = np.histogram(self.projections_flat.flatten(), bins=1000)

        if show:
            _,ax = plt.subplots(1,2,figsize=(10,5))
            ax[0].imshow(self.projections_flat[0],vmin=-0.1,vmax=1)  
            ax[1].plot(self.histogram[1][0:-1],self.histogram[0])
            ax[1].set_yscale('log')


    def flatten_projections(self,amplification=10,stack_window=5, show=False):
        """ Detrends the projection of the cylinder 

        Arguments:
        - proj: a stack of projections
        - amplication: the amplification factor of the median correction to remove the edge effect of the cylinder
        - stack_window: the window size for the max operation in the stack direction
        """
        d = dilation(self.projections,footprint=np.ones((stack_window,1,1)))
        self.projections_flat = d - self.projections
        m=np.median(self.projections_flat,axis=0)
        #flatten projection
        for i in range(self.projections.shape[0]):
            self.projections_flat[i] = self.projections_flat[i] - amplification*m

        self.histogram = np.histogram(self.projections_flat.flatten(), bins=1000)

        if show:
            _,ax = plt.subplots(1,2,figsize=(10,5))
            ax[0].imshow(self.projections_flat[0],vmin=-0.1,vmax=1)  
            ax[1].plot(self.histogram[1][0:-1],self.histogram[0])
            ax[1].set_yscale('log')
            
    

    def show_histogram(self):
        plt.plot(self.histogram[1][0:-1],self.histogram[0])
        plt.yscale('log')
        plt.show()
    
    def mask_beads(self,img,margin=10):
        p = img.mean(axis=0)
        th = (p.max()-p.min() )/4 + p.min()

        w=np.where(th<p)
        res=np.copy(img)
        if len(w[0])!=0:
            minp = np.minimum(0,np.min(w)-margin)
            maxp = np.minimum(img.shape[1],np.max(w)+margin)
        
            res[:,0:minp] = 0
            res[:,maxp:]  = 0
            
        return res  
    
    def threshold_projections(self, threshold, show=False, cleanmethod='median',clearborder=False):
        """
        Apply a threshold to the flattened projections to create a bilevel image.
        
        :param threshold: Threshold value to apply
        """
        self.projections_bilevel = (self.projections_flat > threshold).astype(int)

        # Morphological operations
        kernel_size = 5
        if cleanmethod == 'median':
            for (idx,proj) in enumerate(self.projections_bilevel):
                self.projections_bilevel[idx] = medfilt(proj,kernel_size=(kernel_size,kernel_size))

        elif cleanmethod == 'opening':
            for (idx,proj) in enumerate(self.projections_bilevel):
                self.projections_bilevel[idx] = binary_opening(proj, disk(kernel_size))

        else :
            raise ValueError(f"Invalid cleanmethod: {cleanmethod}")

        for (idx,proj) in enumerate(self.projections_bilevel):
            self.projections_bilevel[idx] = self.mask_beads(proj)

        # Clean items connected to the border
        if clearborder:
            for (idx,proj) in enumerate(self.projections_bilevel):
                self.projections_bilevel[idx] = clear_border(proj)

        if show:
            self.show_bw_projection()

    def show_bw_projection(self,idx=None):
        if idx is None:
            plt.imshow(self.projections_bilevel.max(axis=0),interpolation='none')
        else:
            plt.imshow(self.projections_bilevel[idx],interpolation='none')
        plt.show()

    # def reduce_segments(self,signal):
    #     reduced_signal = np.zeros_like(signal)
    #     if len(signal) == 0:
    #         return reduced_signal
        
    #     reduced_signal[0] = signal[0]
    #     for i in range(1, len(signal)):
    #         if signal[i] == 1 and signal[i-1] == 0:
    #             reduced_signal[i] = 1
    #     return reduced_signal
    
    def reduce_segments(self, signal):
        reduced_signal = np.zeros_like(signal)
        if len(signal) == 0:
            return reduced_signal

        start = -1
        for i in range(len(signal)):
            if signal[i] == 1 and start == -1:
                start = i
            elif signal[i] == 0 and start != -1:
                middle = (start + i - 1) // 2
                reduced_signal[middle] = 1
                start = -1

        # Handle the case where the signal ends with a segment of ones
        if start != -1:
            middle = (start + len(signal) - 1) // 2
            reduced_signal[middle] = 1

        return reduced_signal
    
    def breakup_beads(self, img, threshold=1, lbl=True):
        p = img.sum(axis=1)
        m = p.mean()
        s = p.std()
        g = (1-self.reduce_segments(p<(m-threshold*s))).reshape(-1,1)
    
        if lbl:
            g = label(g)

        g2 =np.matmul(g, np.ones((1,img.shape[1]))).astype(int)

        return img*g2
    
    def find_beads(self, min_distance=5, breakup=False, breakup_threshold=1, show=False):
        self.beads = []
        for (idx,(biproj,flat)) in enumerate(zip(self.projections_bilevel,self.projections_flat)):
            if breakup :
                lbl = self.breakup_beads(biproj,threshold=breakup_threshold,lbl=True)
            else :
                lbl = label(biproj)

            rp = regionprops(lbl,intensity_image=flat)  
            for region in rp:
                if region.area>20:
                    self.beads.append({"idx" : idx, "label" : region.label, "centroid" :np.array([region.centroid_weighted[0],region.centroid_weighted[1]])})
    
        if show:
            self.show_beads()

    def show_beads(self):
        """
        Show the detected beads on the bilevel projection
        """
        plt.imshow(self.projections_bilevel.max(axis=0),interpolation='none')
        for bead in self.beads:
            plt.plot(bead['centroid'][1],bead['centroid'][0],'r.')
        plt.show()

    def prune_trajectories(self,traj, threshold=1):
        """Prune trajectories based on the distance between consecutive points."""
        
        res = []
        for idx,t in enumerate(traj):
            if t.shape[0] > 5:
                tt = np.concatenate((t,t,t))
                bt=np.abs(tt[:,0]-medfilt(tt[:,0],5))
                bt=bt[t.shape[0]:2*t.shape[0]]
                print(t.shape,bt.shape)
                m=np.mean(bt)
                s=np.std(bt)
                
                t=t[bt < m+threshold*s,:]
                res.append(t)

        return res  
    
    def find_trajectories(self, show=False):
        """
        Find trajectories of the beads

        :param show: Show the trajectories

        """

        maxlabel = 0
        for bead in self.beads:
            if bead['label']>maxlabel:
                maxlabel = bead['label']

        self.trajectories = [[] for _ in range(maxlabel+1)]

        for bead in self.beads:
            self.trajectories[bead['label']].append(bead['centroid'])

        for idx in range(len(self.trajectories)):
            self.trajectories[idx] = np.array(self.trajectories[idx])

        self.trajectories = self.prune_trajectories(self.trajectories, threshold=1)

        if show:    
            self.show_trajectories()

    def show_trajectories(self):
        """
        Show the trajectories of the beads
        """
        _, ax = plt.subplots(figsize=(5,6))
        plt.imshow(self.projections_bilevel.max(axis=0),interpolation='none')

        # cmap   = plt.get_cmap('tab20')
        # colors = cmap(np.linspace(0, 1, len(self.trajectories)))
        
        cmap = randomCM(len(self.trajectories),bg=[0,0,0])
        colors = cmap(np.linspace(0, 1, len(self.trajectories)))
        for idx,trajectory in enumerate(self.trajectories):
            if trajectory.shape[0]>5:
                ax.plot(trajectory[:,1],trajectory[:,0],'.', color=colors[idx])

    # converting the coefficients to ellipse parameters
    def fit(self,x, y):
        # Convert x and y to float type if they are not already
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        D1 = np.vstack([x**2, x*y, y**2]).T
        D2 = np.vstack([x, y, np.ones_like(x)]).T
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2
        C1 = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=float)

        # Attempt to invert S3 with safe handling for singular matrices
        try:
            S3_inv = np.linalg.inv(S3)
        except np.linalg.LinAlgError:
            print("Warning: S3 is singular, using pseudo-inverse.")
            S3_inv = np.linalg.pinv(S3)

        M = np.linalg.inv(C1) @ (S1 - S2 @ S3_inv @ S2.T)

        # Eigenvalue decomposition
        eig_vals, eig_vecs = np.linalg.eig(M)
        cond = 4 * eig_vecs[0, :] * eig_vecs[2, :] - eig_vecs[1, :]**2
        valid_idx = np.where(cond > 0)[0]

        if valid_idx.size == 0:
            print("No valid ellipse found.")
            return None

        a1 = eig_vecs[:, valid_idx]
        return np.vstack([a1, np.linalg.solve(-S3, S2.T @ a1)]).flatten()
    
    def errors(self, x, y, coeffs):
        z = np.vstack((x**2, x*y, y**2, x, y, np.ones_like(x)))
        if coeffs.size == 12:
            coeffs = coeffs.reshape(2, 6)
        elif coeffs.size == 6:
            coeffs = coeffs.reshape(1, 6)
        numerator = np.sum(((coeffs @ z) - 1)**2)
        denominator = (len(x) - 6) * np.sum(z**2, axis=1)
        unc = np.sqrt(numerator / denominator)
        return tuple(ufloat(i, j) for i, j in zip(coeffs.flatten(), unc))

    def convert_ellipse_parameters(self,coeffs):
        a,b,c,d,e,f = coeffs
        b /= 2
        d /= 2
        e /= 2
        x0 = (c*d - b*e) / (b**2 - a*c)
        y0 = (a*e - b*d) / (b**2 - a*c)
        center = (x0, y0)
        numerator = 2 * (a*e**2 + c*d**2 + f*b**2 - 2*b*d*e - a*c*f)
        denominator1 = (b*b-a*c)*((c-a)*umath.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        denominator2 = (b*b-a*c)*((a-c)*umath.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        major = umath.sqrt(numerator/denominator1) if numerator/denominator1 > 0 else 0
        minor = umath.sqrt(numerator/denominator2) if numerator/denominator2 > 0 else 0
        phi = .5*umath.atan((2*b)/(a-c))
        major, minor, phi = (major, minor, phi) if major > minor else (minor, major, np.pi/2+phi)
        return center, major, minor, phi

    def fit_ellipses(self, show=False, prune=True):
        """
        Fit ellipses to the trajectories
        
        :param show: Show the ellipses
        :param prune: Prune ellipses with a high error
        """
        self.ellipses = []
        for trajectory in self.trajectories:
            if trajectory.shape[0]>5:
                points = np.copy(trajectory)
                points = points.reshape((-1,1,2)).astype(np.float32)
                e = cv2.fitEllipse(points)
                edict = {"xy" : (e[0][1],e[0][0]), "width" : e[1][1], "height" : e[1][0], "angle" : e[2]} 
                if edict["width"] < edict["height"]:
                    edict["angle"] = edict["angle"]+np.pi/2
                    edict["width"],edict["height"] = edict["height"],edict["width"]

                error = self._compute_ellipse_error(edict, trajectory)
                edict["error"] = error
                self.ellipses.append({"trajectory": trajectory, "ellipse" : edict})
            else:   
                self.ellipses.append({"trajectory": trajectory, "ellipse" : None})

        if prune:
            self._prune_ellipses()


        if show:
            self.show_ellipses()

    def _compute_ellipse_error(self, ellipse, trajectory):
        """
        Compute the lsq error of an ellipse fit to a trajectory

        :param ellipse: Dictionary with ellipse parameters
        :param trajectory: Trajectory to fit

        :return: The error
        """
        center    = np.array(ellipse["xy"])
        angle     = ellipse["angle"] #np.deg2rad(ellipse["angle"])
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        width     = ellipse["width"] / 2.0
        height    = ellipse["height"] / 2.0

        error=0
        for point in trajectory:
            diff = point - center
            x_rot = cos_angle * diff[0] + sin_angle * diff[1]
            y_rot = -sin_angle * diff[0] + cos_angle * diff[1]
            dist = np.sqrt((x_rot / width) ** 2 + (y_rot / height) ** 2)
            error += np.abs(dist - 1)

        return error/len(trajectory)
    
    def _prune_ellipses(self):
        """
        Prune ellipses with a high error
        """
        self.ellipses = [ellipse for ellipse in self.ellipses if ellipse["ellipse"] is not None]

        errors = np.array([ellipse["ellipse"]["error"] for ellipse in self.ellipses])
        mean   = np.mean(errors)
        std    = np.std(errors)

        self.ellipses = [ellipse for ellipse in self.ellipses if ellipse["ellipse"]["error"] < mean + 2*std]

    def show_ellipses(self):
        _, ax = plt.subplots(figsize=(5,6))
        ax.set_xlim(0,self.projections[0].shape[1])
        ax.set_ylim(0,self.projections[0].shape[0])

        # cmap   = plt.get_cmap('tab20')
        # colors = cmap(np.linspace(0, 1, len(self.ellipses)))
        cmap = randomCM(len(self.ellipses),bg=[0,0,0])
        colors = cmap(np.linspace(0, 1, len(self.ellipses)))
        
        for idx,ellipse in enumerate(self.ellipses):
            e = ellipse['ellipse']
            if e is not None:
                ellipse_patch = Ellipse(xy=e["xy"], width=e["width"], height=e["height"], angle=e["angle"], edgecolor=colors[idx], facecolor='none')
                ax.add_patch(ellipse_patch)
                ax.plot(ellipse['trajectory'][:,1],ellipse['trajectory'][:,0],'.', color=colors[idx])
                ax.plot(e["xy"][0],e["xy"][1],'+', color=colors[idx])
        
    def _ellipse_outlier_removal(self):
        """
        Remove ellipses that are outliers
        """
        pass

    def compute_calibration(self, diameter=50, pixelsize=0.139, avgtype='median', remove_outliers=True, ppdegree=3,show=False):
        """
        Compute the calibration
        """
        self._compute_COR()
        self._compute_piercing_point()
        #self._compute_piercing_point2(degree=ppdegree,show=show)
        self._compute_distances(diameter        = diameter, 
                                pixelsize       = pixelsize,
                                remove_outliers = remove_outliers,
                                avgtype         = avgtype)

        if show:
            self.show_calibration()

    def _compute_COR(self):
        """
        Compute the center of rotation and tilt axis
        """
        x = np.array([ellipse["ellipse"]["xy"][0] for ellipse in self.ellipses])
        y = np.array([ellipse["ellipse"]["xy"][1] for ellipse in self.ellipses])
        
                # Perform linear fit
        slope, intercept = np.polyfit(y, x, 1)
        
        # Store the fit parameters
        self.calibration["COR"] = {"center": intercept, "slope": slope, "tilt": np.rad2deg(np.arctan(slope))}

    def _remove_outliers(self,data):
        """
        Remove outliers from the data
        """
        median = np.median(data)
        upper_quartile = np.percentile(data, 75)
        lower_quartile = np.percentile(data, 25)

        iqr         = upper_quartile - lower_quartile
        upper_bound = median + 0.25 * iqr
        lower_bound = median - 0.25 * iqr
        
        return data[(data >= lower_bound) & (data <= upper_bound)]

    def _compute_distances(self, diameter, pixelsize, avgtype='mean',epsilon=1e-7, remove_outliers=True):
        """ 
        Compute the SOD and SDD distances of the cone beam system
        """
        pixelSize = pixelsize   # mm
        D         = diameter    # 50 mm for new calibration sample

        x_centers  = np.array([ellipse["ellipse"]["xy"][0]  for ellipse in self.ellipses])
        y_centers  = np.array([ellipse["ellipse"]["xy"][1]  for ellipse in self.ellipses])
        minor_axes = np.array([ellipse["ellipse"]["height"] for ellipse in self.ellipses])/2 # we want the radius
        major_axes = np.array([ellipse["ellipse"]["width"]  for ellipse in self.ellipses])
        angles     = np.array([ellipse["ellipse"]["angle"]  for ellipse in self.ellipses])   
        hpiercing  = self.calibration["pp"]["y"]  # Piercing point

        # Sample data (assuming y_centers, minor_axes, angles, and vpiercing are defined)
        sod = []
        sdd = []
        mag = []

        for y,minor_axis,angle in zip(y_centers,minor_axes,angles):
            ha = ((y - (minor_axis * np.cos(angle)) - hpiercing)) 
            hb = ((y + (minor_axis * np.cos(angle)) - hpiercing)) 
            est_sod = (hb + ha) * 0.5* D / ((hb - ha) + epsilon)
            sod.append(np.abs(est_sod))
     
        sod = np.array([un.nominal_value(x) for x in sod])

        mag = major_axes *pixelSize / D 
        sdd = np.abs(mag * sod)

        plt.plot(major_axes,y_centers,marker='o',linestyle='-')
        plt.axhline(y=hpiercing,color='r')

        sdd = np.array([un.nominal_value(x) for x in sdd])
        mag = np.array([un.nominal_value(x) for x in mag])
        
        if remove_outliers:
            sod_clean = self._remove_outliers(sod)
            sdd_clean = self._remove_outliers(sdd)
            mag_clean = self._remove_outliers(mag)
        else:
            sod_clean = sod
            sdd_clean = sdd
            mag_clean = mag

        # average the values is ok as the variations are small and often look random
        if avgtype == 'mean':        
            sod = np.mean(sod_clean)
            sdd = np.mean(sdd_clean)
            mag = np.mean(mag_clean)
        else:
            sod = np.median(sod_clean)
            sdd = np.median(sdd_clean)
            mag = np.median(mag_clean)

        self.calibration["SOD"] = sod
        self.calibration["SDD"] = sdd
        self.calibration["MAG"] = mag

    def _compute_piercing_point(self):
        """
        Compute the piercing point of the cone beam system

        To do: Look at the case where the piercing point is outside the field of view. I.e. minidx=0 or minidx=end of the list.

        """

        # Compute vertical piercing point
        y = np.array([ellipse["ellipse"]["xy"][1] for ellipse in self.ellipses])
        minax = np.array([ellipse["ellipse"]["height"] for ellipse in self.ellipses])

        minidx = np.argmin(minax)
        
        res = np.zeros(3)
        for idx in range(3):
             p0,r0,_,_,_ = np.polyfit(y[0:(minidx+idx)],  minax[0:(minidx+idx)],  1, full=True)
             p1,r1,_,_,_ = np.polyfit(y[(minidx+idx+1):], minax[(minidx+idx+1):], 1, full=True)
             r = r0+r1
             res[idx] = np.abs(r0+r1)

        idx = np.argmin(res)
        p0  = np.polyfit(y[0:(minidx+idx)],minax[0:(minidx+idx)],1)
        p1  = np.polyfit(y[(minidx+idx+1):],minax[(minidx+idx+1):],1)

        y0  = (p0[1]-p1[1])/(p1[0]-p0[0])

        # Compute the horizontal piercing point


        # Store the results
        self.calibration["pp"] = {"y" : y0, "x" : p1}

    def _compute_piercing_point2(self,degree=5, show=False):
        """
        Compute the piercing point of the cone beam system

        To do: Look at the case where the piercing point is outside the field of view. I.e. minidx=0 or minidx=end of the list.

        """

        x0     = np.array([ellipse["ellipse"]["xy"][1]  for ellipse in self.ellipses])
        y0     = np.array([ellipse["ellipse"]["xy"][0]  for ellipse in self.ellipses])
        minor_axes  = np.array([ellipse["ellipse"]["height"] for ellipse in self.ellipses])/2
        angles = np.array([ellipse["ellipse"]["angle"]  for ellipse in self.ellipses])

        x1 = x0 + minor_axes * np.cos(angles)
        y1 = y0 + minor_axes * np.sin(angles)

        line_points = np.array([x0,y0,x1,y1]).T

        # Calculate the length of each line segment
        segment_lengths = np.sqrt((line_points[:, 2] - line_points[:, 0])**2 + (line_points[:, 3] - line_points[:, 1])**2)

        # Find the 25th percentile of the length distribution
        percentile_25 = scoreatpercentile(segment_lengths, 25)

        # Filter out line segments in the 1st quarter of the length distribution
        filtered_line_points = line_points[segment_lengths <= percentile_25]

        # Plot the filtered line segments

        if show:
            fig, ax = plt.subplots()

            for x0, y0, x1, y1 in line_points:
                ax.plot([x0, x1], [y0, y1], color='skyblue')

            for x0, y0, x1, y1 in filtered_line_points:
                ax.plot([x0, x1], [y0, y1], color='red')
            # Set Y-axis limits
            # ax.set_ylim(800, 990)  # Adjust the limits as needed

            ax.set_title('Minor Axis, rotated 90 degrees')
            ax.set_xlabel('heights')  # Label for the y-axis
            ax.set_ylabel('X-centers')  # Label for the x-axis

            plt.show()
        
        global_min_point = None
        global_min_value = None


        # Prepare data for fitting
        X = line_points[:, :2]  # Take the starting points of the lines
        Y = minor_axes  # Corresponding minor axis lengths


        # Define the initial guesses for the coefficientss
        num_coeffs = (degree + 1) * (degree + 2) // 2
        initial_guess = np.zeros(num_coeffs)

        def _pp_higher_order_model(xy, *coeffs):
            x, y = xy
            result = np.zeros_like(x)
            index = 0
            for i in range(degree + 1):
                for j in range(degree + 1 - i):
                    result += coeffs[index] * (x ** i) * (y ** j)
                    index += 1
            return result


        # Fit the higher-order polynomial model
        params, params_covariance = curve_fit(lambda xy, *params: _pp_higher_order_model(xy, *params), X.T, Y, p0=initial_guess)

        # Objective function for minimization
   
         # Set bounds for the optimization to realistic values
        bounds = [(X[:,0].min(), X[:,0].max()), (X[:,1].min(), X[:,1].max())]

        def _pp_objective(xy):
            return _pp_higher_order_model(xy, *params)

        # Minimization to find the global minimum
        result = minimize(_pp_objective, x0=[(X[:,0].min() + X[:,0].max()) / 2, (X[:,1].min() + X[:,1].max()) / 2], bounds=bounds)
        global_min_point = result.x
        global_min_value = result.fun

        self.calibration["pp"] = {"x": result.x[0], "y" : result.x[1]}
    
    
    def show_calibration(self):
        x = np.array([ellipse["ellipse"]["xy"][0] for ellipse in self.ellipses])
        y = np.array([ellipse["ellipse"]["xy"][1] for ellipse in self.ellipses])
        minax = np.array([ellipse["ellipse"]["height"] for ellipse in self.ellipses])

        fig,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].plot(x,y,'.',label="Data")
        s = self.calibration["COR"]["slope"]
        c = self.calibration["COR"]["center"]
        ax[0].plot(y*s+c,y,label="y={0:0.02f}+{1:0.06f}x".format(c,s))
        ax[0].set_title("Center of rotation")
        ax[0].legend()
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")

        ax[1].plot(minax,y,'.',label="Data")
        ax[1].axhline(y=self.calibration["pp"]["y"],color='r',label="py={0:0.02f}".format(self.calibration['pp']['y']))
        ax[1].set_title("Piercing point y")
        ax[1].set_xlabel("Minor axis length")
        ax[1].set_ylabel("y")
        ax[1].legend()
        
    