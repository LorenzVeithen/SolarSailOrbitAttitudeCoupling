import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.optimize import minimize
from MiscFunctions import cart_to_pol, fit_2d_ellipse, get_ellipse_pts, compute_panel_geometrical_properties, sigmoid
import pygmo as pg
from constants import W, c_sol, vanes_coordinates_list, vanes_origin_list
import os
import itertools

# Best ellipse optimisation
class constrainedEllipseCoefficientsProblem:
    def __init__(self, convex_hull_points):
        self.convex_hull_points = convex_hull_points
        self.A0 = None

        # Vector of inequality constraints
        D = np.array([0, 0, 0, 0, 0, 0])
        for xp in convex_hull_points[0:, :]:
            D = np.vstack((D, np.array([xp[0] ** 2, xp[0] * xp[1], xp[1] ** 2, xp[0], xp[1], 1])))
        self.D = D[1:, :]

    def fitness(self, x):
        [a, b, c, d, e, f] = x
        if (self.check_inequality_sign(x)):
            ineq_constraint = list(np.dot(-self.D, x))  # all rows < 0
        else:
            ineq_constraint = list(np.dot(self.D, x))  # all rows < 0
        A = self.ellipse_area(x)
        self.previous_A = A
        obj = -A
        return [obj] + ineq_constraint + [b**2 - 4 * a * c]

    def ellipse_area(self, x):
        [a, b, c, d, e, f] = x
        if ((b ** 2 - 4 * a * c) < 0):
            x0, y0, ap, bp, e, phi = cart_to_pol([a, b, c, d, e, f])
            A = np.pi * ap * bp
        else:
            A = self.previous_A
        return A

    def get_bounds(self):
        return ([-20, -21, -22, -23, -24, -25], [20, 21, 22, 23, 24, 25])    # Generic bounds as they are necessary for pygmo

    def get_nic(self):
        return np.shape(self.D)[0] + 1

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)

    def sef_initial_area(self, a0):
        self.A0 = a0

    def check_inequality_sign(self, current_weights):
        inf_point = np.array([1e23, 1e23])
        D_inf = np.array([[inf_point[0] ** 2, inf_point[0] * inf_point[1], inf_point[1] ** 2, inf_point[0], inf_point[1], 1]])
        return np.dot(D_inf, current_weights)[0] > 0     # return True if infinity is outside, False if it is inside

def is_dup_simple(arr):
    u, c = np.unique(arr, axis=0, return_counts=True)
    return (c>1).any()

PLOT = False
COMPUTE_ELLIPSES = False
COMPUTE_ELLIPSE_FORMULA = True
dir = "./AMS/Datasets/Ideal_model/vane_1"

if (COMPUTE_ELLIPSES):
    # Extract data
    all_files_list = os.listdir(dir)
    number_files = len(all_files_list)
    optimised_ellipse_coefficients_stack_without_shadow = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    optimised_ellipse_coefficients_stack_with_shadow = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    for i, dataset_file in enumerate(os.listdir(dir)):
        print(f'{100 * (i/number_files)} %')
        ams_data = np.genfromtxt(dir + "/" + dataset_file, delimiter=',')

        alpha_1, alpha_2 = np.rad2deg(ams_data[:, 0]), np.rad2deg(ams_data[:, 1])
        alpha_sun, beta_sun = np.rad2deg(ams_data[0, 2]), np.rad2deg(ams_data[0, 3])
        point_in_shadow = ams_data[:, 4]    # watch out, for now this is only the torque magnitude

        centroid_body_frame, vane_area, surface_normal_body_frame = compute_panel_geometrical_properties(vanes_coordinates_list[1])
        moment_arm = np.linalg.norm(vanes_origin_list[1])
        Tx, Ty, Tz = ams_data[:, 5].T, ams_data[:, 6].T, ams_data[:, 7].T
        Tx, Ty, Tz = Tx[point_in_shadow == 0], Ty[point_in_shadow == 0], Tz[point_in_shadow == 0]   # Only select points which are not in the shadow
        Tx, Ty, Tz = Tx[..., None], Ty[..., None], Tz[..., None]

        # Compute complex hull
        TxTz_2d_points = np.hstack((Tx, Tz))
        TxTz_2d_hull = ConvexHull(TxTz_2d_points)
        TxTz_2d_hull_points = TxTz_2d_hull.points[TxTz_2d_hull.vertices]

        TyTz_2d_points = np.hstack((Ty, Tz))
        TyTz_2d_hull = ConvexHull(TyTz_2d_points)
        TyTz_2d_hull_points = TyTz_2d_hull.points[TyTz_2d_hull.vertices]

        TxTy_2d_points = np.hstack((Tx, Ty))
        TxTy_2d_hull = ConvexHull(TxTy_2d_points)
        TxTy_2d_hull_points = TxTy_2d_hull.points[TxTy_2d_hull.vertices]

        TxTyTz_3d_points = np.hstack((Tx, Ty, Tz))
        TxTyTz_3d_hull = ConvexHull(TxTyTz_3d_points)
        TxTyTz_3d_hull_points = TxTyTz_3d_hull.points[TxTyTz_3d_hull.vertices]

        # Fitted ellipse
        weightsTyTz = fit_2d_ellipse(TyTz_2d_hull_points[:, 0], TyTz_2d_hull_points[:, 1])
        x0, y0, ap, bp, e, phi = cart_to_pol(weightsTyTz)
        x_TyTz, y_TyTz = get_ellipse_pts((x0, y0, ap, bp, e, phi), npts=500)

        # Find the best ellipse circumscribed by the convex hull (and does not go over the boundary)
        # For TyTz of vane 1, then the coefficients will be reused for other vanes
        myConstrainedEllipseCoefficientsProblem = constrainedEllipseCoefficientsProblem(TyTz_2d_hull_points)

        A0 = myConstrainedEllipseCoefficientsProblem.ellipse_area(weightsTyTz)
        myConstrainedEllipseCoefficientsProblem.sef_initial_area(A0)

        obj0 = myConstrainedEllipseCoefficientsProblem.fitness(weightsTyTz)[0]
        worst_inequality_constraint0 = max(myConstrainedEllipseCoefficientsProblem.fitness(weightsTyTz)[1:])

        prob = pg.problem(myConstrainedEllipseCoefficientsProblem)
        prob.c_tol = 0
        pop = pg.population(prob=prob)
        pop.push_back(x=weightsTyTz)  # Defining the initial guess
        scp = pg.algorithm(pg.scipy_optimize(method="SLSQP"))
        scp.set_verbosity(1)
        result = scp.evolve(pop)

        optimised_ellipse_coefficients = pop.champion_x
        A_final = myConstrainedEllipseCoefficientsProblem.ellipse_area(pop.champion_x)
        obj_final = myConstrainedEllipseCoefficientsProblem.fitness(pop.champion_x)[0]
        worst_inequality_constraint_final = max(myConstrainedEllipseCoefficientsProblem.fitness(pop.champion_x)[1:])

        [x0, y0, ap, bp, e, phi] = cart_to_pol(pop.champion_x)
        x_opt, y_opt = get_ellipse_pts([x0, y0, ap, bp, e, phi], npts=1000)

        if (abs(A_final) > abs(A0)):
            print("Final area is larger than the initial fit")
        if (not prob.feasibility_x(pop.champion_x)):
            print("Constraint violation")
        if ((abs(A_final) > abs(A0)) or (not prob.feasibility_x(pop.champion_x))):
            print(dataset_file)
            print(abs(A_final), abs(A0))
            print(worst_inequality_constraint_final)

        current_stack = np.hstack((np.deg2rad(alpha_sun), np.deg2rad(beta_sun), optimised_ellipse_coefficients))
        if (dataset_file[-8:-4] == 'True'):
            # Store ellipse coefficients and write file
            optimised_ellipse_coefficients_stack_with_shadow = np.vstack((optimised_ellipse_coefficients_stack_with_shadow,
                                                              current_stack))
        else:
            # Store ellipse coefficients and write file
            optimised_ellipse_coefficients_stack_without_shadow = np.vstack((optimised_ellipse_coefficients_stack_without_shadow,
                                                              current_stack))
        if PLOT:
            # Plots
            plt.figure()
            plt.grid(True)
            plt.scatter(Ty, Tz, s=1, label="Vane points")
            for simplex in TyTz_2d_hull.simplices:
                plt.plot(TyTz_2d_points[simplex, 0], TyTz_2d_points[simplex, 1], 'k--')
            plt.plot([], [], 'k--', label="Convex Hull")
            plt.plot(x_TyTz, y_TyTz, 'r', label="Fitted ellipse")
            plt.plot(x_opt, y_opt, 'g', label="Optimised ellipse")
            plt.scatter(TyTz_2d_hull_points[:, 0], TyTz_2d_hull_points[:, 1])
            plt.xlabel("Ty [-]")
            plt.ylabel("Tz [-]")
            plt.title(r"$\alpha$=" + f'{round(alpha_sun, 1)}' +  r" and $\beta$=" + f'{round(beta_sun, 1)}')
            plt.legend()
            plt.savefig(f"./AMS/Plots/Ideal_model/vane_1/Fitted_Ellipses/{dataset_file[:-4]}.png")
            #plt.show()
            plt.close()

            """
            plt.figure()
            plt.grid(True)
            plt.scatter(Tx, Tz, s=1, label="Vane points")
            for simplex in TxTz_2d_hull.simplices:
                plt.plot(TxTz_2d_points[simplex, 0], TxTz_2d_points[simplex, 1], 'r-')
            plt.plot([], [], 'r-', label="Convex Hull")
            plt.xlabel("Tx [-]")
            plt.ylabel("Tz [-]")
            plt.title(r"$\alpha$=" + f'{alpha_sun}' + r" and $\beta$=" + f'{beta_sun}')
            plt.legend()
            plt.close()

            plt.figure()
            plt.grid(True)
            plt.scatter(Tx, Ty, s=1, label="Vane points")
            for simplex in TxTy_2d_hull.simplices:
                plt.plot(TxTy_2d_points[simplex, 0], TxTy_2d_points[simplex, 1], 'r-')
            plt.plot([], [], 'r-', label="Convex Hull")
            plt.xlabel("Tx [-]")
            plt.ylabel("Ty [-]")
            plt.title(r"$\alpha$=" + f'{alpha_sun}' +  r" and $\beta$=" + f'{beta_sun}')
            plt.legend()
            plt.close()
            """

    np.savetxt(
        dir + f"/AMS_ellipse_coefficients_shadow_False.csv",
        optimised_ellipse_coefficients_stack_without_shadow, delimiter=",",
        header='alpha_sun, beta_sun, A, B, C, D, E, F')

    np.savetxt(
        dir + f"/AMS_ellipse_coefficients_shadow_True.csv",
        optimised_ellipse_coefficients_stack_with_shadow, delimiter=",",
        header='alpha_sun, beta_sun, A, B, C, D, E, F')

    print(optimised_ellipse_coefficients_stack_without_shadow)
    print(optimised_ellipse_coefficients_stack_with_shadow)

if (COMPUTE_ELLIPSE_FORMULA):
    ellipse_coefficients_data = np.genfromtxt(dir + "/" + "AMS_ellipse_coefficients_shadow_False.csv", delimiter=',')
    alpha_sun = ellipse_coefficients_data[:, 0]
    beta_sun = ellipse_coefficients_data[:, 1]

    sun_angles = np.stack([alpha_sun, beta_sun], axis=1)

    lst = []
    for i in range(2):
        lst += [i]

    combs = []
    for x in itertools.product(lst, repeat=2):
        combs.append(list(x))

    combs = combs[1:]
    print(combs)
    A = np.zeros((1, 1 + len(combs)*4))
    for xp in sun_angles[0:, :]:
        current_alpha_sun = xp[0]
        current_beta_sun = xp[1]
        Ap = np.array([0])
        for c in combs:
            gamma_1 = np.sin(current_alpha_sun) ** c[0] * np.sin(current_beta_sun) ** c[1]
            gamma_2 = np.cos(current_alpha_sun) ** c[0] * np.sin(current_beta_sun) ** c[1]
            gamma_3 = np.sin(current_alpha_sun) ** c[0] * np.cos(current_beta_sun) ** c[1]
            gamma_4 = np.cos(current_alpha_sun) ** c[0] * np.cos(current_beta_sun) ** c[1]
            Ap = np.hstack((Ap, np.array([gamma_1, gamma_2, gamma_3, gamma_4])))
        A = np.vstack((A, Ap))
    A = A[1:, :]
    print(is_dup_simple(A))
    print(A)
    current_coefficient = ellipse_coefficients_data[:, 2]

    ATA = np.dot(A.T, A)
    ATY = np.dot(A.T, current_coefficient)
    current_fit_coefficients = np.dot(np.linalg.inv(ATA), ATY)
    print(current_fit_coefficients)

