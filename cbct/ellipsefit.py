import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix

def ellipse_coordinates(xc, yc, a, b, theta, num_points=100):
    """
    Computes the x, y coordinates for an ellipse.
    
    Parameters:
    - xc, yc: Center of the ellipse.
    - a, b: Semi-major and semi-minor axes of the ellipse.
    - theta: Rotation angle of the ellipse in radians.
    - num_points: Number of points to generate along the ellipse.
    
    Returns:
    - x_coords, y_coords: Arrays of x and y coordinates of the ellipse.
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    
    # Parametric equations for the ellipse before rotation
    x = a * np.cos(t)
    y = b * np.sin(t)
    
    # Rotation matrix components
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    
    # Apply rotation and translation
    x_coords = xc + cos_theta * x - sin_theta * y
    y_coords = yc + sin_theta * x + cos_theta * y
    
    return x_coords, y_coords

def ellipse_residuals(params, x, y):
    """Calculate the sum of squared residuals for ellipse fitting."""
    xc, yc, a, b, theta = params
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    
    # Rotate and transform points
    x_rot = cos_theta * (x - xc) + sin_theta * (y - yc)
    y_rot = -sin_theta * (x - xc) + cos_theta * (y - yc)
    
    # Calculate residuals
    residuals = (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1
    return np.sum(residuals ** 2)  # Return scalar sum of squared residuals


def fit_ellipse(x, y, r_min, r_max, initial_guess):
    """Fits an ellipse to x, y coordinates with radius and rotation constraints."""
    # Constraints
    cons = [
        {'type': 'ineq', 'fun': lambda params: params[2] - r_min},  # a >= r_min
        {'type': 'ineq', 'fun': lambda params: r_max - params[2]},  # a <= r_max
        {'type': 'ineq', 'fun': lambda params: params[3] - r_min},  # b >= r_min
        {'type': 'ineq', 'fun': lambda params: r_max - params[3]},  # b <= r_max
    ]
    
    # Fitting the ellipse with constraints
    result = minimize(ellipse_residuals, x0=initial_guess, args=(x, y))
    # result = minimize(ellipse_residuals, initial_guess, args=(x, y))
    # result = minimize(ellipse_residuals, initial_guess, args=(x, y), constraints=cons)
    
    if not result.success:
        raise ValueError("Ellipse fitting did not converge.")
    
    xc, yc, a, b, theta = result.x
    return xc, yc, a, b, theta

# data = np.load("../notebooks/trajectory_02.npy")
data=cal.trajectories[10]
print(data.shape)  # (n, 2)
x = data[:,0]  # Your x coordinates
y = data[:,1]  # Your y coordinates
r_min, r_max = 1.1*(x.max()-x.min())/2, 1.1*(y.max()-y.min())/2    # Example min and max radius constraints
initial_guess = [np.mean(x), np.mean(y), (x.max()-x.min())/2, (y.max()-y.min())/2, 0.0]  # Initial guess for xc, yc, a, b, theta

# Fit ellipse
params = fit_ellipse(x, y, r_min, r_max, initial_guess)
print("Fitted ellipse parameters:", params)
plt.plot(x,y,'.')
x,y = ellipse_coordinates(*params)
plt.plot(x,y)