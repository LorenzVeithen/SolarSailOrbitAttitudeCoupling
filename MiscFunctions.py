import numpy as np
from itertools import groupby
from numba import jit

@jit(nopython=True, cache=True)
def compute_panel_geometrical_properties(panel_coordinates):
    # Number of attachment points
    number_of_attachment_points = np.shape(panel_coordinates)[0]

    # Compute centroid
    current_panel_centroid = np.zeros(3)
    for i in range(number_of_attachment_points):
        current_panel_centroid[0] += panel_coordinates[i, 0]
        current_panel_centroid[1] += panel_coordinates[i, 1]
        current_panel_centroid[2] += panel_coordinates[i, 2]
    current_panel_centroid /= number_of_attachment_points

    # Compute area using vectorized operations
    ref_point_area_calculation = panel_coordinates[0, :]
    current_panel_coordinates_wrt_ref_point = np.zeros((number_of_attachment_points - 1, 3))
    for i in range(1, number_of_attachment_points):
        current_panel_coordinates_wrt_ref_point[i - 1, 0] = panel_coordinates[i, 0] - ref_point_area_calculation[0]
        current_panel_coordinates_wrt_ref_point[i - 1, 1] = panel_coordinates[i, 1] - ref_point_area_calculation[1]
        current_panel_coordinates_wrt_ref_point[i - 1, 2] = panel_coordinates[i, 2] - ref_point_area_calculation[2]

    # Vectorized cross product sum
    cross_product_sum = np.zeros(3)
    for i in range(number_of_attachment_points - 2):
        u = current_panel_coordinates_wrt_ref_point[i]
        v = current_panel_coordinates_wrt_ref_point[i + 1]
        cross_product_sum[0] += u[1] * v[2] - u[2] * v[1]
        cross_product_sum[1] += u[2] * v[0] - u[0] * v[2]
        cross_product_sum[2] += u[0] * v[1] - u[1] * v[0]

    current_panel_area = 0.5 * np.sqrt(cross_product_sum[0]**2 + cross_product_sum[1]**2 + cross_product_sum[2]**2)

    # Compute surface normal
    norm = np.sqrt(cross_product_sum[0]**2 + cross_product_sum[1]**2 + cross_product_sum[2]**2)
    current_panel_surface_normal = cross_product_sum / norm

    return current_panel_centroid, current_panel_area, current_panel_surface_normal

def quiver_data_to_segments(X, Y, Z, u, v, w, length=1):
    segments = (X, Y, Z, X+v*length, Y+u*length, Z+w*length)
    segments = np.array(segments).reshape(6,-1)
    return [[[x, y, z], [u, v, w]] for x, y, z, u, v, w in zip(*list(segments))]

def closest_point_on_a_segment_to_a_third_point(p1, p2, p3):
    if (all(p3[i] == p1[i] for i in range(len(p3)))):
        return p1
    elif (all(p3[i] == p2[i] for i in range(len(p3)))):
        return p2
    else:
        d = (p2-p1)/np.linalg.norm((p2-p1))
        w = p3-p1
        magnitude = np.dot(w, d)
        p4 = d * np.dot(w, d)
        if (magnitude < 0):
            return p1
        elif (magnitude > 0 and np.linalg.norm(p4-p1) > np.linalg.norm(p2-p1)):
            return p2
        else:
            return p4

def lists_of_arrays_equal(list1, list2):
    if len(list1) != len(list2):
        return False

    for arr1, arr2 in zip(list1, list2):
        if (not np.array_equal(arr1, arr2) and np.amax(abs(arr2-arr1))>1e-15):
            return False

    return True

def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def set_axes_equal(ax):  # from https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    return

def Rx_matrix(theta):
    '''
    Rotation matrix around the x axis by an angle theta
    :param theta: rotation angle in radians
    :return: 3x3 rotation matrix around the x axis
    '''
    return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

def Ry_matrix(theta):
    '''
    Rotation matrix around the y axis by an angle theta
    :param theta: rotation angle in radians
    :return: 3x3 rotation matrix around the y axis
    '''
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])


def Rz_matrix(theta):
    '''
    Rotation matrix around the z axis by an angle theta
    :param theta: rotation angle in radians
    :return: 3x3 rotation matrix around the z axis
    '''
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

def fit_2d_ellipse(x, y):
    """
    https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/
    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()

def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y

@jit(nopython=True)
def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        print(a, b, c, d, f, g)
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi

def sigmoid(x):
    return 1/(1 + np.exp(-x))
