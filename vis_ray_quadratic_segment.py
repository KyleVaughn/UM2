import matplotlib.pyplot as plt
import numpy as np

ninterp = 100

p0 = np.array([0, 0])
p1 = np.array([2, 0])
p2 = np.array([0, 1])

origin = np.array([-0.3625, 0.5])
direction = np.array([0.672188, 0.740381])

def quadratic_segment_interp(p0, p1, p2, r):
    w0 = (2 * r - 1) * (r - 1)
    w1 = (2 * r - 1) *  r
    w2 = -4 * r      * (r - 1)
    return w0 * p0 + w1 * p1 + w2 * p2

def ray_interp(origin, direction, r):
    return origin + r * direction

rr = np.linspace(0, 1, ninterp)

qpoints_x = np.zeros(ninterp)
qpoints_y = np.zeros(ninterp)
for i in range(ninterp):
    points = quadratic_segment_interp(p0, p1, p2, rr[i])
    qpoints_x[i] = points[0]
    qpoints_y[i] = points[1]

rpoints_x = np.zeros(ninterp)
rpoints_y = np.zeros(ninterp)
for i in range(ninterp):
    points = ray_interp(origin, direction, rr[i])
    rpoints_x[i] = points[0]
    rpoints_y[i] = points[1]

plt.plot(qpoints_x, qpoints_y, 'r')
plt.plot(rpoints_x, rpoints_y, 'b')
plt.show()
