The parameters \( SOD \) (Source-to-Object Distance) and \( SDD \) (Source-to-Detector Distance) are often used in imaging systems, particularly in X-ray and neutron imaging, to describe the geometry of the setup.

### Geometry of the Setup:
1. **\( SOD \)**: This is the horizontal distance from the source to the object (the triangle apex point at \( (0,0) \)).
2. **\( SDD \)**: This is the horizontal distance from the source to the detector plane (where \( T12 \) and \( T22 \) extend their vertical components \( A \) and \( B \)).

### How \( SOD \) and \( SDD \) Relate to the Triangles:
- The two triangles \( T11 \) and \( T12 \) describe rays originating from the source at \( (0,0) \) and intersecting the object and the detector.
- Similarly, \( T21 \) and \( T22 \) describe another set of rays intersecting the object and the detector at different points.
- The parameter \( r \) determines the lateral shift of the object's projection, while \( h \) is the height at which the object intersects the ray paths.

### Determine \( SOD \) and \( SDD \):
1. **From \( T12 \) and \( T22 \):**
   The vertical extents \( A \) and \( B \) are the heights at \( x = SDD \) for the respective rays. Using the slopes of the lines forming these triangles:
   - The slope of the ray in \( T12 \) is given by \( \frac{h}{SOD + r} \), which projects to \( A = h \cdot \frac{SDD}{SOD + r} \).
   - Similarly, for \( T22 \), the ray projects to \( B = h \cdot \frac{SDD}{SOD - r} \).

2. **Equations Relating \( SOD \) and \( SDD \):**
   From the above, you get:
   \[
   A = h \cdot \frac{SDD}{SOD + r}, \quad B = h \cdot \frac{SDD}{SOD - r}.
   \]
   Rearrange these to:
   \[
   SOD + r = h \cdot \frac{SDD}{A}, \quad SOD - r = h \cdot \frac{SDD}{B}.
   \]

3. **Eliminate \( SOD \) to Find \( SDD \):**
   Subtract the equations:
   \[
   2r = h \cdot \left( \frac{SDD}{A} - \frac{SDD}{B} \right),
   \]
   \[
   SDD = \frac{2r \cdot A \cdot B}{h \cdot (B - A)}.
   \]

4. **Find \( SOD \):**
   Substitute \( SDD \) back into one of the equations, for example:
   \[
   SOD = \frac{h \cdot SDD}{A} - r.
   \]

### Summary of Formulas:
- \( SDD = \frac{2r \cdot A \cdot B}{h \cdot (B - A)} \),
- \( SOD = \frac{h \cdot SDD}{A} - r \).

These formulas allow you to compute \( SOD \) and \( SDD \) given \( A \), \( B \), \( r \), and \( h \).
