import numpy as np
    
Nproj       = 626
Nrepeat     = 1

start_angle = 0 
end_angle   = 360

angles = np.repeat(np.linspace(start_angle,end_angle,Nproj),Nrepeat)

scan(sp2_ry,angles.tolist())