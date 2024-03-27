# Script to debug the intersection of a quadratic segment and a ray.
# Typically, this means the point of intersection and the point on the
# quadratic segment closest to the intersection point differ.

import matplotlib.pyplot as plt
import numpy as np

# Number of interpolation points to visualize the quadratic segment and the ray
ninterp = 1000

# Quadratic segment vertices
p0 = np.array([-0.215672, 0.0637669])
p1 = np.array([2.13615, -0.179838])
p2 = np.array([3.78916, 2.91062])

# Ray
origin = np.array([1.20349, -0.334424])
direction = np.array([0.637189, 0.770707])

# The point on ray where the quadratic segment intersects
p = np.array([3.86361, 2.8831])
plt.scatter(p[0], p[1], c='k')

# Point on quadratic segment closest to p
q_closest = np.array([3.86364, 2.88308])
plt.scatter(q_closest[0], q_closest[1], c='g')

def quadratic_segment_interp(p0, p1, p2, r):
    w0 = (2 * r - 1) * (r - 1)
    w1 = (2 * r - 1) *  r
    w2 = -4 * r      * (r - 1)
    return w0 * p0 + w1 * p1 + w2 * p2

def ray_interp(origin, direction, r):
    return origin + r * direction

rr = np.linspace(0, 1, ninterp)
tt = np.linspace(0, 5, ninterp)

qpoints_x = np.zeros(ninterp)
qpoints_y = np.zeros(ninterp)
for i in range(ninterp):
    #points = quadratic_segment_interp(a, b, c, rr[i])
    points = quadratic_segment_interp(p0, p1, p2, rr[i])
    qpoints_x[i] = points[0]
    qpoints_y[i] = points[1]

rpoints_x = np.zeros(ninterp)
rpoints_y = np.zeros(ninterp)
for i in range(ninterp):
    points = ray_interp(origin, direction, tt[i])
    rpoints_x[i] = points[0]
    rpoints_y[i] = points[1]

plt.plot(qpoints_x, qpoints_y, 'r')
plt.plot(rpoints_x, rpoints_y, 'b')
plt.show()
