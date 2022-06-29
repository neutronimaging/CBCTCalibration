import numpy as np
angles = np.linspace(87,93,7)

for angle in angles :
    move('sp2_ry',angle)
    count()
    
    