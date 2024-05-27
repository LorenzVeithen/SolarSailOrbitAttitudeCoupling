from ACS_optimisationProblems import vaneAnglesAllocationProblem
from ACS_dynamicalModels import vane_dynamical_model
from sailCraftClass import sail_craft
from attitudeControllersClass import sail_attitude_control_systems
from constants import *

from time import time
import matplotlib.pyplot as plt
import matplotlib
from multiprocessing import Process
import os


matplotlib.pyplot.switch_backend('Agg')


def generate_AMS(vane_id, sail_obj, acs_obj, sun_angles_num=37, vane_angles_num=100, savefig=True, savedat=False):
    # TODO: WATCH OUT CHANGE FILENAME WRT OPTICAL MODEL USED
    optical_model_str = "Ideal_model"
    xlabels = ["Tx", "Tx", "Ty"]
    ylabels = ["Ty", "Tz", "Tz"]
    vaneAngleProblem = vaneAnglesAllocationProblem(vane_id,
                                                   ([-np.pi, -np.pi], [np.pi, np.pi]),
                                                   10,
                                                   sail_obj,
                                                   acs_obj,
                                                   include_shadow=True)
    vaneAngleProblem.update_vane_angle_determination_algorithm(np.array([0, 0, 0]), np.array([0, 0, -1]),
                                                               vane_variable_optical_properties=True)  # and the next time you can put False

    sun_angle_alpha_list = np.linspace(-180, 180, sun_angles_num)
    sun_angle_beta_list = np.linspace(-180, 180, sun_angles_num)
    alpha_1_range, alpha_2_range = np.linspace(-np.pi, np.pi, vane_angles_num), np.linspace(-np.pi, np.pi, vane_angles_num)
    alpha_1_range_grid, alpha_2_range_grid = np.meshgrid(alpha_1_range, alpha_2_range)
    flattened_alpha_1_range_grid = alpha_1_range_grid.reshape(-1)
    flattened_alpha_2_range_grid = alpha_2_range_grid.reshape(-1)

    for alpha in sun_angle_alpha_list:
        alpha = np.deg2rad(alpha)
        for beta in sun_angle_beta_list:
            beta = np.deg2rad(beta)

            # Compute sun vector in body frame
            n_s = np.array([np.sin(alpha) * np.cos(beta),
                            np.sin(alpha) * np.sin(beta),
                            -np.cos(alpha)])   # In the body reference frame

            vaneAngleProblem.update_vane_angle_determination_algorithm(np.array([0, 0, 0]), n_s,
                                                                       vane_variable_optical_properties=False)  # and the next time you can put False
            t0 = time()
            print(f'vane_id={vane_id}, alpha={np.rad2deg(alpha)}, beta={np.rad2deg(beta)}')
            current_fig1 = plt.figure(1)
            current_fig2 = plt.figure(2)
            current_fig3 = plt.figure(3)
            current_figs = [current_fig1, current_fig2, current_fig3]
            for m in range(2):
                if (m==1):
                    SHADOW_BOOL = True
                    current_fig_label = "With Shadow"
                else:
                    SHADOW_BOOL = False
                    current_fig_label = "No shadow"

                vaneAngleProblem.set_shadow_bool(SHADOW_BOOL)
                vstackT_, vstackTx_, vstackTy_, vstackTz_, vstackC_ = (np.zeros(np.shape(alpha_2_range)),
                                                                       np.zeros(np.shape(alpha_2_range)),
                                                                       np.zeros(np.shape(alpha_2_range)),
                                                                       np.zeros(np.shape(alpha_2_range)),
                                                                       np.zeros(np.shape(alpha_2_range)))
                for alpha_1 in alpha_1_range:
                    hstackT_, hstackTx_, hstackTy_, hstackTz_, hstackC_ = (np.array([0]),
                                                                           np.array([0]),
                                                                           np.array([0]),
                                                                           np.array([0]),
                                                                           np.array([0]))
                    for alpha_2 in alpha_2_range:
                        T = vaneAngleProblem.single_vane_torque([alpha_1, alpha_2])
                        Tx, Ty, Tz = T[0], T[1], T[2]
                        C = np.sum(T ** 2) * (1 / 3)
                        hstackT_ = np.hstack((hstackT_, np.array([np.linalg.norm(T)])))
                        hstackTx_ = np.hstack((hstackTx_, np.array([Tx])))
                        hstackTy_ = np.hstack((hstackTy_, np.array([Ty])))
                        hstackTz_ = np.hstack((hstackTz_, np.array([Tz])))
                        hstackC_ = np.hstack((hstackC_, np.array([C])))
                    vstackT_ = np.vstack((vstackT_, hstackT_[1:]))
                    vstackTx_ = np.vstack((vstackTx_, hstackTx_[1:]))
                    vstackTy_ = np.vstack((vstackTy_, hstackTy_[1:]))
                    vstackTz_ = np.vstack((vstackTz_, hstackTz_[1:]))
                    vstackC_ = np.vstack((vstackC_, hstackC_[1:]))

                ZT = vstackT_[1:, :]
                BT = vstackT_[1:, :]
                ZTx = vstackTx_[1:, :]
                ZTy = vstackTy_[1:, :]
                ZTz = vstackTz_[1:, :]
                ZC = vstackC_[1:, :]

                # Remove the shadow points, if any
                if (SHADOW_BOOL):
                    BT[BT < 1e20] = 0
                    BT[BT > 1e20] = 1
                    ZT[ZT > 1e20] = None
                    ZTx[ZTx > 1e20] = None
                    ZTy[ZTy > 1e20] = None
                    ZTz[ZTz > 1e20] = None
                    ZC[ZC > 1e20] = None

                flattened_BT = BT.reshape(-1)
                flattened_Tx = ZTx.reshape(-1)
                flattened_Ty = ZTy.reshape(-1)
                flattened_Tz = ZTz.reshape(-1)

                # Write data to file for further processing

                if (savedat):
                    array_to_save = np.stack([flattened_alpha_1_range_grid, flattened_alpha_2_range_grid,
                                              alpha * np.ones(np.shape(flattened_alpha_1_range_grid)),
                                              beta * np.ones(np.shape(flattened_alpha_1_range_grid)), flattened_BT,
                                              flattened_Tx, flattened_Ty, flattened_Tz], axis=1)
                    np.savetxt(
                        f"./AMS/Datasets/{optical_model_str}/vane_{vane_id}/AMS_alpha_{np.rad2deg(alpha)}_beta_{np.rad2deg(beta)}_shadow_{str(SHADOW_BOOL)}.csv",
                        array_to_save, delimiter=",",
                        header='alpha_1, alpha_2, alpha_sun, beta_sun, Shadow_bool, Tx, Ty, Tz')

                # Plot the scatter-data
                plt.figure(1)
                plt.scatter(flattened_Tx, flattened_Ty, s=1, label=current_fig_label)

                plt.figure(2)
                plt.scatter(flattened_Tx, flattened_Tz, s=1, label=current_fig_label)

                plt.figure(3)
                plt.scatter(flattened_Ty, flattened_Tz, s=1, label=current_fig_label)

            for i in range(1, 4):
                plt.figure(i)
                plt.title(f'vane {vane_id}: alpha={np.rad2deg(alpha)}, beta={np.rad2deg(beta)}')
                plt.xlabel(xlabels[i-1])
                plt.ylabel(ylabels[i-1])
                plt.legend(loc='lower left')
                if (savefig): plt.savefig(f'./AMS/Plots/{optical_model_str}/vane_{vane_id}/plot_{i}/AMS_{i}_alpha_{np.rad2deg(alpha)}_beta_{np.rad2deg(beta)}.png')
                plt.close(current_figs[i-1])

#generate_AMS(2, sail, acs_object, 37, 100, True, False)

if __name__ == "__main__":

    # Define solar sail - see constants file
    acs_object = sail_attitude_control_systems("vanes", boom_list)
    acs_object.set_vane_characteristics(vanes_coordinates_list, vanes_origin_list, vanes_rotation_matrices_list, 0,
                                        np.array([0, 0, 0]), 0.0045, vanes_rotational_dof)

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

    t0 = time()
    processes = []
    for i in range(4):
        pc = Process(target=generate_AMS, args=(i,sail, acs_object, 37, 100, True, False,))
        pc.start()
        processes.append(pc)

    # Waiting for all threads to complete
    for pc in processes:
        pc.join()
    print(time()-t0)
