import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.optimize import curve_fit
import pygmo as pg
import os
import itertools
from time import time
from matplotlib import cm

from constants import *
from MiscFunctions import compute_panel_geometrical_properties, find_linearly_independent_rows
from vaneControllerMethods import constrainedEllipseCoefficientsProblem
from vaneControllerMethods import fourierSumFunction, fourierSeriesFunction, combinedFourierFitFunction, cart_to_pol, fit_2d_ellipse, get_ellipse_pts
from vaneControllerMethods import generate_AMS_data, vaneAnglesAllocationProblem
from attitudeControllersClass import sail_attitude_control_systems
from sailCraftClass import sail_craft
import h5py

PLOT = False
COMPUTE_DATA = False
COMPUTE_ELLIPSES = False
COMPUTE_ELLIPSE_FOURIER_FORMULA = True
ALL_HULLS = False
dir = "./AMS/Datasets/Ideal_model/vane_1"
f1 = h5py.File('backupData2.h5', 'w')

if (COMPUTE_DATA):
    # Define solar sail - see constants file
    sh_comp = 1     # Define whether to work with or without shadow effects
    vane_id = 1

    if (sh_comp != 0 and sh_comp != 1):
        raise Exception(
            "Error.AMS generation for both shadow TRUE and FALSE is not supported in AMSFormulaDerivation." +
            "Do one after the other")

    acs_object = sail_attitude_control_systems("vanes", boom_list)
    acs_object.set_vane_characteristics(vanes_coordinates_list, vanes_origin_list, vanes_rotation_matrices_list, 0,
                                        np.array([0, 0, 0]), 0.0045, vanes_rotational_dof)

    current_optical_model_str = "Ideal_model"
    sail = sail_craft("ACS3",
                      len(wings_coordinates_list),
                      len(vanes_coordinates_list),
                      wings_coordinates_list,
                      vanes_coordinates_list,
                      wings_optical_properties,
                      [np.array([0., 0., 1., 1., 0., 0., 2 / 3, 2 / 3, 1., 1.])] * (5),
                      sail_I,
                      sail_mass,
                      sail_mass_without_wings,
                      sail_nominal_CoM,
                      sail_material_areal_density,
                      sail_material_areal_density,
                      acs_object)

    vaneAngleProblem = vaneAnglesAllocationProblem(vane_id,
                                                   ([-np.pi, -np.pi], [np.pi, np.pi]),
                                                   10,
                                                   sail,
                                                   acs_object,
                                                   include_shadow=True)
    vaneAngleProblem.update_vane_angle_determination_algorithm(np.array([0, 0, 0]), np.array([0, 0, -1]),
                                                               vane_variable_optical_properties=True)  # and the next time you can put False

    sun_angles_num, vane_angles_num = 181, 100
    sun_angle_alpha_list = np.linspace(-180, 180, sun_angles_num)
    sun_angle_beta_list = np.linspace(-180, 180, sun_angles_num)
    alpha_1_range = np.linspace(-np.pi, np.pi, vane_angles_num)
    alpha_2_range = np.linspace(-np.pi, np.pi, vane_angles_num)

t0 = time()-1
if (COMPUTE_ELLIPSES):
    # Extract data
    optimised_ellipse_coefficients_stack_without_shadow = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    optimised_ellipse_coefficients_stack_with_shadow = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    if (COMPUTE_DATA):
        loop_list = list(itertools.product(sun_angle_alpha_list, sun_angle_beta_list))
    else:
        all_files_list = os.listdir(dir)
        number_files = len(all_files_list)
        loop_list = os.listdir(dir)
    for i, input in enumerate(loop_list):
        print(time() - t0)
        t0 = time()
        if (COMPUTE_DATA):
            print(f'{100 * (i / len(list(loop_list)))} %')
            ams_data, shadow_list = generate_AMS_data(vane_id, vaneAngleProblem,
                                                  input[0],
                                                  input[1],
                                                  alpha_1_range,
                                                  alpha_2_range,
                                                  optical_model_str=current_optical_model_str,
                                                  savefig=False, savedat=False,
                                                  shadow_computation=sh_comp)
            ams_data = ams_data[0]
        else:   # retrieve data from file
            print(f'{100 * (i / number_files)} %')
            if (os.path.isdir(dir + "/" + input)):
                continue
            ams_data = np.genfromtxt(dir + "/" + input, delimiter=',')
            shadow_list = []

        alpha_1_deg, alpha_2_deg = np.rad2deg(ams_data[:, 0]), np.rad2deg(ams_data[:, 1])
        alpha_sun_rad, beta_sun_rad = np.rad2deg(ams_data[0, 2]), np.rad2deg(ams_data[0, 3])
        point_in_shadow = ams_data[:, 4]    # watch out, for now this is only the torque magnitude

        centroid_body_frame, vane_area, surface_normal_body_frame = compute_panel_geometrical_properties(vanes_coordinates_list[1])
        moment_arm = np.linalg.norm(vanes_origin_list[1])
        Tx, Ty, Tz = ams_data[:, 5].T, ams_data[:, 6].T, ams_data[:, 7].T
        Tx, Ty, Tz = Tx[point_in_shadow == 0], Ty[point_in_shadow == 0], Tz[point_in_shadow == 0]   # Only select points which are not in the shadow
        Tx, Ty, Tz = Tx[..., None], Ty[..., None], Tz[..., None]

        # Compute complex hull
        if (ALL_HULLS):
            TxTz_2d_points = np.hstack((Tx, Tz))
            TxTz_2d_hull = ConvexHull(TxTz_2d_points)
            TxTz_2d_hull_points = TxTz_2d_hull.points[TxTz_2d_hull.vertices]

            TxTy_2d_points = np.hstack((Tx, Ty))
            TxTy_2d_hull = ConvexHull(TxTy_2d_points)
            TxTy_2d_hull_points = TxTy_2d_hull.points[TxTy_2d_hull.vertices]

            TxTyTz_3d_points = np.hstack((Tx, Ty, Tz))
            TxTyTz_3d_hull = ConvexHull(TxTyTz_3d_points)
            TxTyTz_3d_hull_points = TxTyTz_3d_hull.points[TxTyTz_3d_hull.vertices]

        TyTz_2d_points = np.hstack((Ty, Tz))
        TyTz_2d_hull = ConvexHull(TyTz_2d_points)
        TyTz_2d_hull_points = TyTz_2d_hull.points[TyTz_2d_hull.vertices]

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
            print(input)
            print(abs(A_final), abs(A0))
            print(worst_inequality_constraint_final)

        current_stack = np.hstack((np.deg2rad(alpha_sun_rad), np.deg2rad(beta_sun_rad), optimised_ellipse_coefficients))
        if (COMPUTE_DATA):
            array_storage_selection = str(bool(sh_comp))
        else:
            array_storage_selection = input[-8:-4]
        if (array_storage_selection == 'True'):
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
            plt.title(r"$\alpha$=" + f'{round(alpha_sun_rad, 1)}' + r" and $\beta$=" + f'{round(beta_sun_rad, 1)}')
            plt.legend()
            if (not COMPUTE_DATA):
                plt.savefig(f"./AMS/Plots/Ideal_model/vane_1/Fitted_Ellipses/{input[:-4]}.png")
            else:
                plt.savefig(f"./AMS/Plots/Ideal_model/vane_1/Fitted_Ellipses/AMS_alpha_{round(np.rad2deg(alpha_sun_rad), 1)}_beta_{round(np.rad2deg(beta_sun_rad), 1)}_shadow_{str(bool(sh_comp))}.png")
            #plt.show()
            plt.close()

            if (ALL_HULLS):
                plt.figure()
                plt.grid(True)
                plt.scatter(Tx, Tz, s=1, label="Vane points")
                for simplex in TxTz_2d_hull.simplices:
                    plt.plot(TxTz_2d_points[simplex, 0], TxTz_2d_points[simplex, 1], 'r-')
                plt.plot([], [], 'r-', label="Convex Hull")
                plt.xlabel("Tx [-]")
                plt.ylabel("Tz [-]")
                plt.title(r"$\alpha$=" + f'{alpha_sun_rad}' + r" and $\beta$=" + f'{beta_sun_rad}')
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
                plt.title(r"$\alpha$=" + f'{alpha_sun_rad}' + r" and $\beta$=" + f'{beta_sun_rad}')
                plt.legend()
                plt.close()

    if (len(np.shape(optimised_ellipse_coefficients_stack_with_shadow)) == 1):
        optimised_ellipse_coefficients_stack_with_shadow = optimised_ellipse_coefficients_stack_with_shadow[..., None]
    if (len(np.shape(optimised_ellipse_coefficients_stack_without_shadow)) == 1):
        optimised_ellipse_coefficients_stack_without_shadow = optimised_ellipse_coefficients_stack_without_shadow[..., None]
    optimised_ellipse_coefficients_stack_with_shadow = optimised_ellipse_coefficients_stack_with_shadow[1:, :]
    optimised_ellipse_coefficients_stack_without_shadow = optimised_ellipse_coefficients_stack_without_shadow[1:, :]

    if (np.shape(optimised_ellipse_coefficients_stack_with_shadow)[0] >1e2):
        np.savetxt(
            dir + f"/ellipseCoefficients/" + "AMS_ellipse_coefficients_shadow_True.csv",
            optimised_ellipse_coefficients_stack_with_shadow, delimiter=",",
            header='alpha_sun, beta_sun, A, B, C, D, E, F')

    if (np.shape(optimised_ellipse_coefficients_stack_without_shadow)[0] >1e2):
        np.savetxt(
            dir + f"/ellipseCoefficients/" + "AMS_ellipse_coefficients_shadow_False.csv",
            optimised_ellipse_coefficients_stack_without_shadow, delimiter=",",
            header='alpha_sun, beta_sun, A, B, C, D, E, F')



if (COMPUTE_ELLIPSE_FOURIER_FORMULA):
    ellipse_coefficients_data = np.genfromtxt(dir + "/ellipseCoefficients/" + "detailed_AMS_ellipse_coefficients_shadow_False.csv", delimiter=',')
    sorted_indices = np.lexsort((ellipse_coefficients_data[:, 1], ellipse_coefficients_data[:, 0]))
    ellipse_coefficients_data = ellipse_coefficients_data[sorted_indices]

    alpha_sun_rad = ellipse_coefficients_data[:, 0]
    beta_sun_rad = ellipse_coefficients_data[:, 1]

    sun_angles = np.stack([alpha_sun_rad, beta_sun_rad], axis=1)
    Y = ellipse_coefficients_data[:, 2:]    # Take all the ellipse coefficients

    current_fit_coefficients = []
    order_i = 6
    #num_coefficients = (order_i*2-1)**2
    #fourierSumFunc = lambda x, *b, ordi=order_i: fourierSumFunction(x, *b, order=ordi)

    order_m = 4
    order_n = 4
    #num_coefficients = order_n * order_m * 4 + 1
    #fourierSeriesFunc = lambda x, *b, ord_n=order_m, ord_m=order_m: fourierSeriesFunction(x, *b, order_n=ord_n, order_m=ord_m)

    num_coefficients = (order_i*2-1)**2 + order_n * order_m * 4
    fourierSeriesFunc = lambda x, *b, ordi=order_i, ord_n=order_m, ord_m=order_m: combinedFourierFitFunction(x, *b, order=ordi, order_n=ord_n, order_m=ord_m)
    for i in range(np.shape(Y)[1]):
        popt, pcov = curve_fit(fourierSeriesFunc, xdata=sun_angles, ydata=Y[:, i], p0=[1] * (num_coefficients))
        print(popt)
        print(np.linalg.cond(pcov))
        current_fit_coefficients.append(popt)

    # Evaluate the fit quality - start with A
    fourier_ellipse_coefficients = np.zeros(np.shape(ellipse_coefficients_data[:, :]))
    fourier_ellipse_coefficients[:, 0] = alpha_sun_rad
    fourier_ellipse_coefficients[:, 1] = beta_sun_rad
    for i in range(6):
        fourier_ellipse_coefficients[:, 2 + i] = combinedFourierFitFunction(sun_angles, *current_fit_coefficients[i], order=order_i, order_n=order_n, order_m=order_m)
    sorted_indices = np.lexsort((fourier_ellipse_coefficients[:, 1], fourier_ellipse_coefficients[:, 0]))
    fourier_ellipse_coefficients = fourier_ellipse_coefficients[sorted_indices]

    n_data_points = len(alpha_sun_rad)
    x = np.reshape(np.rad2deg(alpha_sun_rad), (int(np.sqrt(n_data_points)), int(np.sqrt(n_data_points))))
    y = np.reshape(np.rad2deg(beta_sun_rad), (int(np.sqrt(n_data_points)), int(np.sqrt(n_data_points))))
    for i in range(6):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.grid(b=True, color='grey',
                linestyle='-.', linewidth=0.3,
                alpha=0.2)
        my_cmap = plt.get_cmap('jet')
        z = np.reshape(fourier_ellipse_coefficients[:, i+2], (int(np.sqrt(n_data_points)), int(np.sqrt(n_data_points))))
        ax.plot_surface(x, y, z,cmap=cm.coolwarm, alpha=0.5, label="Fourier fit")
        ax.scatter(np.rad2deg(alpha_sun_rad), np.rad2deg(beta_sun_rad), ellipse_coefficients_data[:, i + 2],
                   c=ellipse_coefficients_data[:, i + 2], cmap=my_cmap, alpha=0.5, label="Data")
        ax.set_xlabel('alpha')
        ax.set_ylabel('beta')
        ax.set_zlabel(f"Ellipse coefficient {['A', 'B', 'C', 'D', 'E', 'F'][i]}")
        ax.legend()
        plt.close()

    for i in range(6):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        z = 100 * (fourier_ellipse_coefficients[:, i+2]-ellipse_coefficients_data[:, i + 2])/(1+ abs(ellipse_coefficients_data[:, i + 2]))
        id = z.argmax(axis=0)
        #print(z[id])
        #print(fourier_ellipse_coefficients[id, i+2], ellipse_coefficients_data[id, i+2])
        ax.scatter(np.rad2deg(alpha_sun_rad), np.rad2deg(beta_sun_rad), z,
                   c=z, cmap=my_cmap, alpha=0.5, label="Data")

    plt.show()
    #plt.close()

