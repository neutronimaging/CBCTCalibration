# CBCTCalibration
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15222362.svg)](https://doi.org/10.5281/zenodo.15222362)

## Aim/Object: 
To measure the geometry parameters from a sparse projection data set of a stack of
beads.
In cone-beam tomography, the projections are distorted by the perspective the divergent
beam is producing. This means that geometry is essential to the successful
reconstruction of tomography. The center of rotation and the pixel size (this is only
needed to scale the attenuation coefficients) which is the only information required for
parallel beam reconstruction. For cone-beam reconstruction, you further need to know the
distances from the source to the sample (object), the distance from the source to the detector
(SDD). These parameters define the magnification of the configuration and would in
principle be sufficient for the reconstruction. The beam usually doesn’t hit the detector
perpendicularly, therefore we also need to know the position of the piercing point on the
detector px and py.


## Parameters needed for a clean reconstruction: 

1. COR - Center of Rotation 
2. CTilt - Tilt of the Rotation Axis
3. SOD - Source Object Distance
4. SDD - Source Detector Distance
5. PP - Piercing Point 
