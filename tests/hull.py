import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import numpy as np
from time import time

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

rng = np.random.default_rng()
points = rng.random((30000, 3))   # 30 random points in 2-D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

hull = Delaunay(points)

t0 = time()
for i in range(10000):
    in_hull(np.array([0.5, 0.5, 0.5]), hull)

print(in_hull(np.array(np.array([[0.5, 0.5, 0.5], [2, 2, 2]])), points))