# Considering a single vane for a given torque, let's determine the right angles of the vanes
from time import time
from matplotlib import cm
import matplotlib.pyplot as plt
import pygmo as pg
from scipy.optimize import minimize, direct

from constants import *
from MiscFunctions import *

from ACS_dynamicalModels import vane_dynamical_model
from sailCraftClass import sail_craft
from attitudeControllersClass import sail_attitude_control_systems
from vaneControllerMethods import vaneAnglesAllocationProblem
from scipy.interpolate import BSpline, make_interp_spline, PPoly, CubicSpline

#sun_angle_alpha = np.deg2rad(-89)
#sun_angle_beta = np.deg2rad(134)

#sun_angle_alpha = np.deg2rad(0)
#sun_angle_beta = np.deg2rad(0)

sun_angle_alpha = np.deg2rad(-116)
sun_angle_beta = np.deg2rad(-100)

n_s = np.array([np.sin(sun_angle_alpha) * np.cos(sun_angle_beta), np.sin(sun_angle_alpha) * np.sin(sun_angle_beta), -np.cos(sun_angle_alpha)])   # In the body reference frame
n_s = n_s/np.linalg.norm(n_s)


# Define solar sail - see constants file
acs_object = sail_attitude_control_systems("vanes", boom_list, sail_I, algorithm_constants)
acs_object.set_vane_characteristics(vanes_coordinates_list,
                                    vanes_origin_list,
                                    vanes_rotation_matrices_list,
                                    0,
                                    np.array([0, 0, 0]),
                                    0.0045,
                                    vanes_rotational_dof,
                                    vane_has_ideal_model,
                                    wings_coordinates_list,
                                    vane_mechanical_rotation_limits,
                                    vanes_optical_properties)

sail = sail_craft("ACS3",
                  len(wings_coordinates_list),
                  len(vanes_coordinates_list),
                  wings_coordinates_list,
                  vanes_coordinates_list,
                  wings_optical_properties,
                  vanes_optical_properties,
                  sail_I,
                  sail_mass,
                  sail_mass_without_wings,
                  sail_nominal_CoM,
                  sail_material_areal_density,
                  sail_material_areal_density,
                  acs_object)

vaneAngleProblem = vaneAnglesAllocationProblem(1,
                                               ([-np.pi, -np.pi], [np.pi, np.pi]),
                                               10,
                                               wings_coordinates_list,
                                               acs_object,
                                               include_shadow=False)

vaneAngleProblem.update_vane_angle_determination_algorithm(np.array([-5e-6, 0, -3e-6]), n_s, vane_variable_optical_properties=True, vane_optical_properties_list=vanes_optical_properties)   # and the next time you can put False
#res = vaneAngleProblem.fitness([np.deg2rad(23), np.deg2rad(67), 1])

#t0 = time()
#for i in range(100000):
#    T = vaneAngleProblem.single_vane_torque([np.deg2rad(12), np.deg2rad(-78)])
#print(time()-t0)

RUN = True
if (RUN):
    t0 = time()
    alpha_1_range = np.linspace(-np.pi, np.pi, 181)
    alpha_2_range = np.linspace(-np.pi, np.pi, 181)
    T_ = np.zeros((len(alpha_1_range), len(alpha_2_range)))
    Tx_ = np.zeros((len(alpha_1_range), len(alpha_2_range)))
    Ty_ = np.zeros((len(alpha_1_range), len(alpha_2_range)))
    Tz_ = np.zeros((len(alpha_1_range), len(alpha_2_range)))
    C_ = np.zeros((len(alpha_1_range), len(alpha_2_range)))
    color2 = np.zeros((len(alpha_1_range), len(alpha_2_range)))
    color1 = np.zeros((len(alpha_1_range), len(alpha_2_range)))
    for i, alpha_1 in enumerate(alpha_1_range):
        for j, alpha_2 in enumerate(alpha_2_range):
            T = vaneAngleProblem.single_vane_torque([alpha_1, alpha_2])
            T_[i, j] = np.linalg.norm(T)
            Tx_[i, j] = T[0]
            Ty_[i, j] = T[1]
            Tz_[i, j] = T[2]
            C_[i, j] = np.sum(T ** 2) * (1 / 3)
            color2[i, j] = 1 if (abs(alpha_2) < 1e-15) else 0
            color1[i, j] = 1 if (abs(alpha_1) < 1e-15) else 0
    print(time()-t0)
    ZT = T_
    BT = T_
    BT[BT < 1e20] = 0
    BT[BT > 1e20] = 1
    ZT[ZT > 1e20] = None
    ZTx = Tx_
    ZTx[ZTx > 1e20] = None
    ZTy = Ty_
    ZTy[ZTy > 1e20] = None
    ZTz = Tz_
    ZTz[ZTz > 1e20] = None
    ZC = C_
    ZC[ZC > 1e20] = None
    alpha_1_range_grid, alpha_2_range_grid = np.meshgrid(alpha_1_range, alpha_2_range)

    PLOT_SURFACE = False
    if PLOT_SURFACE:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        surf = ax.plot_surface(np.rad2deg(alpha_2_range_grid), np.rad2deg(alpha_1_range_grid), BT, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel("alpha_1")
        ax.set_ylabel("alpha_2")

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        surf = ax.plot_surface(np.rad2deg(alpha_2_range_grid), np.rad2deg(alpha_1_range_grid), ZC, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel("alpha_1")
        ax.set_ylabel("alpha_2")

    plt.figure()
    flattened_Tx = ZTx.reshape(-1)
    flattened_Ty = ZTy.reshape(-1)
    flattened_Tz = ZTz.reshape(-1)
    flattened_Txy = np.sqrt(flattened_Tx**2 + flattened_Ty**2)
    flattened_color2 = color2.reshape(-1)
    flattened_color1 = color1.reshape(-1)
    plt.scatter(flattened_Txy, flattened_Tz, s=1)
    #plt.plot(flattened_Ty[flattened_color2==1], flattened_Tz[flattened_color2==1], color='r')
    #plt.plot(flattened_Ty[flattened_color1==1], flattened_Tz[flattened_color1==1], color='g')
    #T00 = vaneAngleProblem.single_vane_torque([np.deg2rad(0), np.deg2rad(180)])
    #plt.scatter(T00[1], T00[2], s=50)
    plt.xlabel("Ty")
    plt.ylabel("Tz")
    #plt.close()

    for alpha_s in [-60]:
        for beta_s in [140]:
            print(f'alpha={alpha_s}, beta={beta_s}')
            sun_angle_alpha = np.deg2rad(alpha_s)
            sun_angle_beta = np.deg2rad(beta_s)
            n_s = np.array([np.sin(sun_angle_alpha) * np.cos(sun_angle_beta), np.sin(sun_angle_alpha) * np.sin(sun_angle_beta),
                            -np.cos(sun_angle_alpha)])  # In the body reference frame
            n_s = n_s / np.linalg.norm(n_s)

            vaneAngleProblem.update_vane_angle_determination_algorithm(np.array([-5e-6, 0, -3e-6]), n_s,
                                                                       vane_variable_optical_properties=False)
            for case in range(2):
                t0 = time()
                start_point = -np.pi
                end_point = np.pi
                n_points = 100
                alpha_range = np.linspace(start_point, end_point, n_points)
                Tx_ = np.zeros(np.shape(alpha_range))
                Ty_ = np.zeros(np.shape(alpha_range))
                Tz_ = np.zeros(np.shape(alpha_range))
                for k, alpha in enumerate(alpha_range):
                    if case==0:
                        T = vaneAngleProblem.single_vane_torque([alpha, np.deg2rad(0)])
                    else:
                        T = vaneAngleProblem.single_vane_torque([np.deg2rad(0), alpha])
                    Tx_[k] = T[0]
                    Ty_[k] = T[1]
                    Tz_[k] = T[2]
                alpha_range = alpha_range[Tx_ < 1e20]
                Tx_ = Tx_[Tx_ < 1e20]
                Ty_ = Ty_[Ty_ < 1e20]
                Tz_ = Tz_[Tz_ < 1e20]

                # find the points where there is a gap
                diff_alpha = np.diff(alpha_range)
                cut_indices = np.where(diff_alpha > 1.1*(end_point-start_point)/n_points)[0]
                alpha_range = alpha_range[..., None]
                alpha_range = alpha_range
                data_points = np.hstack((alpha_range, np.column_stack((Tx_, Ty_, Tz_))))

                # Cut early points
                if (len(cut_indices)!=0):
                    if (cut_indices[0]==0):
                        cut_indices = cut_indices[1:]-1
                        data_points = data_points[1:, :]
                    elif (cut_indices[-1]==(n_points-1)):
                        cut_indices = cut_indices[:-1]
                        data_points = data_points[:-1, :]

                split_arrays = np.split(data_points, cut_indices+1)
                sx_TyTz_list, sy_TyTz_list = [], []
                sx_TxTz_list, sy_TxTz_list = [], []
                fTx_list, fTy_list, fTz_list = [], [], []
                if ((len(cut_indices) == 0) and (abs(alpha_range[0]) == np.pi) and (
                        abs(alpha_range[0]) - abs(alpha_range[-1])) < 1e-15):
                    boundary_type = "periodic"
                else:
                    boundary_type = "not-a-knot"

                for i, sp in enumerate(split_arrays):
                    k_int = min(np.shape(sp)[0]-1, 3)   # aim for cubicsplines but go smaller if necessary
                    fTx = make_interp_spline(sp[:, 0], sp[:, 1], k=k_int, bc_type=boundary_type)
                    fTy = make_interp_spline(sp[:, 0], sp[:, 2], k=k_int, bc_type=boundary_type)
                    fTz = make_interp_spline(sp[:, 0], sp[:, 3], k=k_int, bc_type=boundary_type)
                    fTx_list.append(fTx)
                    fTy_list.append(fTy)
                    fTz_list.append(fTz)


                sx_TyTz_plot_list, sy_TyTz_plot_list = [], []
                sx_TxTz_plot_list, sy_TxTz_plot_list = [], []
                for i in range(len(split_arrays)):
                    c_alpha_list = split_arrays[i][:,0]
                    ca = np.linspace(c_alpha_list[0], c_alpha_list[-1], 1000)
                    sx_TyTz_plot_list.append(fTy_list[i](ca))
                    sy_TyTz_plot_list.append(fTz_list[i](ca))
                    sx_TxTz_plot_list.append(fTx_list[i](ca))
                    sy_TxTz_plot_list.append(fTz_list[i](ca))
                #print(time()-t0)

                for sx, sy in zip(fTy_list, fTz_list):
                    u0 = PPoly.from_spline((sx.t, sx.c + 0.03, 3), extrapolate=False).roots()
                    #print(sx(sx.t), sx.c)
                    #print(u0)
                    #print(sy(u0))


                fig, ax = plt.subplots()
                ax.scatter(Ty_, Tz_)
                for sx_new, sy_new in zip(sx_TyTz_plot_list, sy_TyTz_plot_list):
                    ax.plot(sx_new, sy_new, '-')
                plt.grid(True)

                fig, ax = plt.subplots()
                ax.scatter(Tx_, Tz_)
                for sx_new, sy_new in zip(sx_TxTz_plot_list, sy_TxTz_plot_list):
                    ax.plot(sx_new, sy_new, '-')
                plt.grid(True)
                #plt.savefig(f'/Users/lorenz_veithen/Desktop/Education/03-Master/01_TU Delft/02_Year2/Thesis/02_ResearchProject'+
                #            f'/MSc_Thesis_Source_Python/AMS/Plots/reducedDOF_const/{case}_alpha_s_{round(alpha_s, 1)}_beta_s_{round(beta_s, 1)}.png')
                #n = list(np.round(np.rad2deg(alpha_range), 1))
                #for i, ni in enumerate(n):
                #    plt.annotate(ni, (Ty_[i], Tz_[i]))
                #plt.close()
plt.show()