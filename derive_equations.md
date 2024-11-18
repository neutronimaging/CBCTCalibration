# Finding the CBCT geometry

The parameters $SOD$ (Source-to-Object Distance) and $SDD$ (Source-to-Detector Distance) are often used in imaging systems, particularly in X-ray and neutron imaging, to describe the geometry of the setup.

## Geometry of the Setup:
1. **$SOD$**: This is the horizontal distance from the source to the object (the triangle apex point at $(0,0)$).
2. **$SDD$**: This is the horizontal distance from the source to the detector plane (where $T_{12}$ and $T_{22}$ extend their vertical components $A$ and $B$).

## How $SOD$ and $SDD$ Relate to the Triangles:
- The two triangles $T_{11}$ and $T_{12}$ describe rays originating from the source at $(0,0)$ and intersecting the object and the detector.
- Similarly, $T_{21}$ and $T_{22}$ describe another set of rays intersecting the object and the detector at different points.
- The parameter $r$ determines the lateral shift of the object's projection, while $h$ is the height at which the object intersects the ray paths.

## Determine $SOD$ and $SDD$:
1. **From $T_{12}$ and $T_{22}$:**
   The vertical extents $A$ and $B$ are the heights at $x = SDD$ for the respective rays. Using the slopes of the lines forming these triangles:
   - The slope of the ray in $T12$ is given by $\frac{h}{SOD + r}$, which projects to $A = h \cdot \frac{SDD}{SOD + r}$.
   - Similarly, for $T_{22}$, the ray projects to $B = h \cdot \frac{SDD}{SOD - r}$.

2. **Equations Relating $SOD$ and $SDD$:**
   From the above, you get:

   $$A = h \cdot \frac{SDD}{SOD + r}, \quad B = h \cdot \frac{SDD}{SOD - r}.$$

   Rearrange these to:

   $$SOD + r = h \cdot \frac{SDD}{A}, \quad SOD - r = h \cdot \frac{SDD}{B}.$$

4. **Eliminate $SOD$ to Find $SDD$:**
   Subtract the equations:
   
   $$2r = h \cdot \left( \frac{SDD}{A} - \frac{SDD}{B} \right),$$
   
   $$SDD = \frac{2r \cdot A \cdot B}{h \cdot (B - A)}.$$

5. **Find $SOD$:**
   Substitute $ SDD $ back into one of the equations, for example:
   $$SOD = \frac{h \cdot SDD}{A} - r.$$

## Summary of Formulas:
- $SDD = \frac{2r \cdot A \cdot B}{h \cdot (B - A)}$,
- $SOD = \frac{h \cdot SDD}{A} - r$.

These formulas allow you to compute $SOD$ and $SDD$ given $A$, $B$, $r$, and $h$.

# Making it a least squared problem
To formulate the least squares fit in **matrix form**, we need to express the relationships between $SOD$,$SDD$, and the triangle parameters ($A_i, B_i, r_i, M_i$) in terms of a system of linear or linearized equations. Here's how to do it:

### 1. Relationships Between Variables
From the relationships:

$$A_i = h_i \cdot \frac{SDD}{SOD + r_i}, \quad B_i = h_i \cdot \frac{SDD}{SOD - r_i},$$

we eliminate $h_i$ using magnification $M_i = \frac{SDD}{SOD}$:

$$\frac{A_i}{M_i} \cdot \left( 1 + \frac{r_i \cdot M_i}{SDD} \right) = \frac{B_i}{M_i} \cdot \left( 1 - \frac{r_i \cdot M_i}{SDD} \right).$$

Rearranging gives:

$$\frac{A_i - B_i}{M_i} = -r_i \cdot \frac{A_i + B_i}{SDD}.$$

This can be rewritten in a linearized form for fitting.

---

### 2. Linearized System
Define the following variables:

- $y_i = \frac{A_i - B_i}{M_i}$,
- $x_i = r_i \cdot (A_i + B_i)$.

The equation becomes:

$$y_i = -\frac{x_i}{SDD}.$$

Taking the reciprocal of $SDD$, let:

$$\beta = \frac{1}{SDD}.$$

Then the equation becomes:
$$y_i = -x_i \cdot \beta.$$

In matrix form, for$ N$ triangle pairs, this becomes:

$$\mathbf{y} = -\mathbf{X} \cdot \beta,$$

where:
- $\mathbf{y}$ is an $N \times 1$ vector with entries $y_i = \frac{A_i - B_i}{M_i}$,
- $\mathbf{X}$ is an $N \times 1$ matrix with entries $x_i = r_i \cdot (A_i + B_i)$,
- $\beta = \frac{1}{SDD}$ is the scalar parameter to fit.

---

### 3. Least Squares Solution
The least squares solution minimizes the residuals:
$$\text{Residual}_i = y_i + x_i \cdot \beta.$$

The objective function to minimize is:
$$\| \mathbf{y} + \mathbf{X} \cdot \beta \|^2.$$

The solution is obtained using the normal equation:
$$\beta = -(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}.$$

---

### 4. Recover$ SDD$ and$ SOD$
Once$\beta$ is determined:
$$SDD = \frac{1}{\beta}.$$

Using the magnification $M$ (average over all pairs if $M_i$ varies), $SOD$ is:
$$SOD = \frac{SDD}{M}.$$

---

### Complete Matrix Form
For multiple pairs, the full least squares problem becomes:
1. Construct the vector $\mathbf{y}$ and matrix $\mathbf{X}$:

$$\mathbf{y}=\begin{bmatrix}\frac{A_1-B_1}{M_1}\\ \frac{A_2-B_2}{M_2}\\ \vdots\\ \frac{A_N-B_N}{M_N}\end{bmatrix},\quad\mathbf{X}=\begin{bmatrix}r_1\cdot(A_1+B_1)\\ r_2\cdot(A_2 + B_2)\\ \vdots\\ r_N\cdot(A_N+B_N)\end{bmatrix}.$$
   
3. Solve for $\beta$ using:

$$\beta = -(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}.$$

4. Compute $SDD$ and $SOD$:
  $$SDD = \frac{1}{\beta}, \quad SOD = \frac{SDD}{M}.$$

This matrix formulation ensures a robust least squares fit for $SDD$ and $SOD$.







