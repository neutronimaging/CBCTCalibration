import numpy as np
from   skimage.filters import median
import matplotlib.pyplot as plt
import subprocess
import json
import sys
import argparse
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import laplace
import sys
sys.path.append('../../scripts/python/')
import amglib.readers as rd
%matplotlib inline

def dict_to_array(d): return np.array(list(d.values()))
def array_to_dict(arr, template): return {k: v for k, v in zip(template.keys(), arr)}

class CBCT_Optimizer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.history = {
            "losses": [],
            "frames": [],
            "pars": []
        }
        
        self.all_pars = {
            "center": 730.781,
            "tiltangle": 0.2127,
            "sod": 193,
            "sdd": 770,
            "pPointx": 1100,
            "pPointy": 750
        }

        self.optimal_pars = None

    def load_model(self, model_path):
        # Load the model from the specified path
        self.model = tf.keras.models.load_model(model_path)

    def set_optimizer(self, optimizer):
        # Set the optimizer for the model
        self.optimizer = optimizer

    def set_loss_function(self, loss_fn):
        # Set the loss function for the model
        self.loss_fn = loss_fn

    def get_recon(self,mask, first, last) :
        return rd.read_images(mask, first=first,last=last)
    
    def total_variation(self,img) :
        gx, gy, gz = np.gradient(img)

        # Compute the absolute gradient (magnitude of gradient vector at each point)
        return np.sqrt(gx**2 + gy**2 + gz**2).mean()

    def std_dev(self,img) :
        return np.std(img)
    
    def gradient_energy(img):
        gx, gy, gz = np.gradient(img)
        return -(gx**2 + gy**2 + gz**2).mean()  # negative for minimization

    def laplacian_energy(img):
        lap = laplace(img)
        return -np.sum(lap**2)  # more edges → more energy
    
    def sparse_gradient(img):
        gx, gy, gz = np.gradient(img)
        return np.sum(np.abs(gx) + np.abs(gy) + np.abs(gz))

    def gradient_entropy(img, bins=100):
        gx, gy, gz = np.gradient(img)
        mag = np.sqrt(gx**2 + gy**2 + gz**2)
        hist, _ = np.histogram(mag, bins=bins, density=True)
        p = hist.astype(float) / np.sum(hist)
        p = hist[hist > 0]
        return -np.sum(p * np.log(p))

    def execute(self, initial_pars, bounds) :
        x0 = dict_to_array(initial_pars)

        # "Nelder-Mead"
        # 'L-BFGS-B'
        # result = minimize(cost_function, x0, method='L-BFGS-B', bounds=bounds, options={"maxiter" :5})
        history ={ "losses" : [], "frames" :[], "pars" : []}
        result = minimize(self.cost_function, x0, method=self.method, bounds=bounds, options={"maxiter" :5})

        # Final optimized parameter set
        optimized_pars = {**array_to_dict(result.x, initial_pars), **fixed_pars}
        print("Optimized Parameters:", optimized_pars)