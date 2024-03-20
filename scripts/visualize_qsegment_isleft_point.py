# Script to debug whether a point p is left of a quadratic segment.
# Let p_closest be the point on the quadratic segment closest to p.
# Typically, this is tested using the cross product of the tangent vector at p_closest
# and the vector p_closest - p is positive.

import matplotlib.pyplot as plt
import numpy as np

# Input
#######################################################################
# Number of interpolation points to visualize the quadratic segment
ninterp = 1000

# Quadratic segment vertices
p0 = np.array([0, 0])
p1 = np.array([2, 0])
p2 = np.array([2, 1])

# The point of interest
p = np.array([1.97473, -0.100964])
plt.scatter(p[0], p[1], c='b')

# parametric coordinate of the closest point on the quadratic segment 
r = 1

plot_bezier_triangle = True
#######################################################################

# Compute the coefficients of Q(r) = a*r^2 + b*r + c
v02 = p2 - p0
v12 = p2 - p1
c = p0
b = 3 * v02 + v12
a = -2 * (v02 + v12)

# Plot the bounding bezier triangle
#######################################################################
if plot_bezier_triangle:
    def lerp(p0, p1, r):
        return p0 + r * (p1 - p0)

    def get_line(p0, p1):
        rr = np.linspace(0, 1, ninterp)
        line_x = np.zeros(ninterp)
        line_y = np.zeros(ninterp)
        for i in range(ninterp):
            point = lerp(p0, p1, rr[i])
            line_x[i] = point[0]
            line_y[i] = point[1]
        return line_x, line_y 

    # Compute the Bezier control point
    bcp = 2 * p2 - (p1 + p0) / 2
    xx, yy = get_line(p0, p1) 
    plt.plot(xx, yy, 'g')
    xx, yy = get_line(p1, bcp)
    plt.plot(xx, yy, 'g')
    xx, yy = get_line(bcp, p0)
    plt.plot(xx, yy, 'g')

# Plot the quadratic segment
#######################################################################
def plot_quadratic_segment(p0, p1, p2):

    def quadratic_segment_interp(r):
        return c + r * (b + r * a)

    rr = np.linspace(0, 1, ninterp)
    
    q_x = np.zeros(ninterp)
    q_y = np.zeros(ninterp)
    for i in range(ninterp):
        point = quadratic_segment_interp(rr[i])
        q_x[i] = point[0]
        q_y[i] = point[1]
    
    plt.plot(q_x, q_y, 'r')

plot_quadratic_segment(p0, p1, p2)

def quadratic_segment_tangent(r):
    return b + (2 * r) * a
plt.show()
