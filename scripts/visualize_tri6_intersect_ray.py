# Script to debug the intersection of a quadratic triangle and a ray. 

import matplotlib.pyplot as plt
import numpy as np

# Number of interpolation points to visualize the quadratic segment and the ray
ninterp = 1000

# Quadratic triangle vertices
p0 = np.array([0.931684, 0.370625])
p1 = np.array([0.642067, 0.463308])
p2 = np.array([-1.46639, -0.0754096])
p3 = np.array([-1.24821, 1.22303])
p4 = np.array([0.161909, 0.917263])
p5 = np.array([-0.142488, 0.221372])

# Ray
origin = np.array([-1.46639, 1.12162])
direction = np.array([0.903913, 0.427717])

# The point on ray where the quadratic segment intersects
p = np.array([-1.25078, 1.22364])
plt.scatter(p[0], p[1], c='k')

plt.scatter(p0[0], p0[1], c='r')
plt.scatter(p1[0], p1[1], c='r')
plt.scatter(p2[0], p2[1], c='r')
plt.scatter(p3[0], p3[1], c='r')
plt.scatter(p4[0], p4[1], c='r')
plt.scatter(p5[0], p5[1], c='r')

def quadratic_segment_interp(p0, p1, p2, r):
    w0 = (2 * r - 1) * (r - 1)
    w1 = (2 * r - 1) *  r
    w2 = -4 * r      * (r - 1)
    return w0 * p0 + w1 * p1 + w2 * p2

def ray_interp(origin, direction, r):
    return origin + r * direction

rr = np.linspace(0, 1, ninterp)
tt = np.linspace(0, 1, ninterp)

qpoints_x = np.zeros(3 * ninterp)
qpoints_y = np.zeros(3 * ninterp)
for i in range(ninterp):
    points = quadratic_segment_interp(p0, p1, p3, rr[i])
    qpoints_x[i] = points[0]
    qpoints_y[i] = points[1]

for i in range(ninterp):
    points = quadratic_segment_interp(p1, p2, p4, rr[i])
    qpoints_x[ninterp + i] = points[0]
    qpoints_y[ninterp + i] = points[1]

for i in range(ninterp):
    points = quadratic_segment_interp(p2, p0, p5, rr[i])
    qpoints_x[2 * ninterp + i] = points[0]
    qpoints_y[2 * ninterp + i] = points[1]

rpoints_x = np.zeros(ninterp)
rpoints_y = np.zeros(ninterp)
for i in range(ninterp):
    points = ray_interp(origin, direction, tt[i])
    rpoints_x[i] = points[0]
    rpoints_y[i] = points[1]

plt.plot(qpoints_x, qpoints_y, 'r')
plt.plot(rpoints_x, rpoints_y, 'b')
plt.show()
