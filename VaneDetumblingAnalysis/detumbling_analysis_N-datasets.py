import numpy as np
import matplotlib.pyplot as plt
from generalConstants import Project_directory
import os

selected_combinations = [(5.0, 0.0, 0.0),
                    (0.0, 5.0, 0.0),
                    (0.0, 0.0, 5.0),
                    (5.0, 5.0, 0.0),
                    (0.0, 5.0, 5.0),
                    (5.0, 0.0, 5.0),
                    (5.0, 5.0, 5.0)]

for c_id, c in enumerate(selected_combinations):
    for plot_id in range(8):
        # Focus on

        if (plot_id == 0):
            comparison_name = "optical_model"
            states_history_datasets_list = [
                f"0_GeneratedData/DetumblingAnalysis/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_ACS3_opt_model_shadow_False/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
                f"0_GeneratedData/DetumblingAnalysis/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_False/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
                f"0_GeneratedData/DetumblingAnalysis/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_single_ideal_opt_model_shadow_False/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
            ]
            plot_label = ['ACS3 O-SRP', 'DI-SRP', 'SI-SRP']
        elif (plot_id == 1):
            comparison_name = "inclination_LEO"
            states_history_datasets_list = [
                f"0_GeneratedData/DetumblingAnalysis/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_False/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
                f"0_GeneratedData/DetumblingAnalysis/LEO_ecc_0.0_inc_45.0/NoAsymetry_data_double_ideal_opt_model_shadow_False/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
                f"0_GeneratedData/DetumblingAnalysis/LEO_ecc_0.0_inc_0.0/NoAsymetry_data_double_ideal_opt_model_shadow_False/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
            ]
            plot_label = ['i=98.0°', 'i=45.0°', 'i=0.0°']
        elif (plot_id == 2):
            comparison_name = "inclination_MEO"
            states_history_datasets_list = [
                f"0_GeneratedData/DetumblingAnalysis/MEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_False/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
                f"0_GeneratedData/DetumblingAnalysis/MEO_ecc_0.0_inc_45.0/NoAsymetry_data_double_ideal_opt_model_shadow_False/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
                f"0_GeneratedData/DetumblingAnalysis/MEO_ecc_0.0_inc_0.0/NoAsymetry_data_double_ideal_opt_model_shadow_False/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
            ]
            plot_label = ['i=98.0°', 'i=45.0°', 'i=0.0°']
        elif (plot_id == 3):
            comparison_name = "inclination_GEO"
            states_history_datasets_list = [
                f"0_GeneratedData/DetumblingAnalysis/GEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_False/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
                f"0_GeneratedData/DetumblingAnalysis/GEO_ecc_0.0_inc_45.0/NoAsymetry_data_double_ideal_opt_model_shadow_False/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
                f"0_GeneratedData/DetumblingAnalysis/GEO_ecc_0.0_inc_0.0/NoAsymetry_data_double_ideal_opt_model_shadow_False/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
            ]
            plot_label = ['i=98.0°', 'i=45.0°', 'i=0.0°']
        elif (plot_id == 4):
            comparison_name = "orbital_regime"
            states_history_datasets_list = [
                f"0_GeneratedData/DetumblingAnalysis/LEO_ecc_0.0_inc_0.0/NoAsymetry_data_double_ideal_opt_model_shadow_False/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
                f"0_GeneratedData/DetumblingAnalysis/MEO_ecc_0.0_inc_0.0/NoAsymetry_data_double_ideal_opt_model_shadow_False/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
                f"0_GeneratedData/DetumblingAnalysis/GEO_ecc_0.0_inc_0.0/NoAsymetry_data_double_ideal_opt_model_shadow_False/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
            ]
            plot_label = ['LEO', 'MEO', 'GEO']
        elif (plot_id == 5):
            comparison_name = "sma"
            states_history_datasets_list = [
                f"0_GeneratedData/DetumblingAnalysis/LEO_ecc_0.0_inc_0.0/NoAsymetry_data_double_ideal_opt_model_shadow_False/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
                f"0_GeneratedData/DetumblingAnalysis/MEO_ecc_0.0_inc_0.0/NoAsymetry_data_double_ideal_opt_model_shadow_False/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
                f"0_GeneratedData/DetumblingAnalysis/GEO_ecc_0.0_inc_0.0/NoAsymetry_data_double_ideal_opt_model_shadow_False/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
            ]
            plot_label = ['LEO', 'MEO', 'GEO']
        elif (plot_id == 6):
            comparison_name = "orientation"
            states_history_datasets_list = [
                f"0_GeneratedData/DetumblingAnalysis/OrientationAnalysis/sun_pointing/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_True/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
                f"0_GeneratedData/DetumblingAnalysis/OrientationAnalysis/edge-on-x/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_True/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
                f"0_GeneratedData/DetumblingAnalysis/OrientationAnalysis/edge-on-y/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_True/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
                f"0_GeneratedData/DetumblingAnalysis/OrientationAnalysis/identity_to_inertial/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_True/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
                f"0_GeneratedData/DetumblingAnalysis/OrientationAnalysis/alpha_45_beta_90/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_True/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
                f"0_GeneratedData/DetumblingAnalysis/OrientationAnalysis/alpha_45_beta_0/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_True/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
            ]
            plot_label = ['sun-pointing', 'edge-on-x', 'edge-on-y', 'identity_to_inertial', r'$\alpha_{s} = 45$°, $\beta_{s} = 90$°', r'$\alpha_{s} = 45$°, $\beta_{s} = 0$°']
        elif (plot_id == 7):
            comparison_name = 'vane_shadow'
            states_history_datasets_list = [
                f"0_GeneratedData/DetumblingAnalysis/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_True/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
                f"0_GeneratedData/DetumblingAnalysis/LEO_ecc_0.0_inc_98.0/NoAsymetry_data_double_ideal_opt_model_shadow_False/states_history/state_history_omega_x_{c[0]}_omega_y_{c[1]}_omega_z_{c[2]}.dat",
            ]
            plot_label = ['Shadow constraint', 'No shadow constraint']

        # create plots directory
        if (not os.path.exists(f'{Project_directory}/0_FinalPlots/Detumbling/{comparison_name}/{str(c)}')):
            os.makedirs(f'{Project_directory}/0_FinalPlots/Detumbling/{comparison_name}/{str(c)}')
        save_plots_dir = f'{Project_directory}/0_FinalPlots/Detumbling/{comparison_name}/{str(c)}'

        # obtain the associated dependent variables
        dependent_variable_history_datasets_list = []
        for state_data_path in states_history_datasets_list:
            file_name = state_data_path.split('/')[-1]
            l = str(file_name)[:-4].split('_')
            omega_z_rph = float(l[-1])
            omega_y_rph = float(l[-4])
            omega_x_rph = float(l[-7])
            dependent_variable_history_subdir = f"dependent_variable_history/dependent_variable_history_omega_x_{omega_x_rph}_omega_y_{omega_y_rph}_omega_z_{omega_z_rph}.dat"

            # Construct path
            dependent_variable_path = ''
            for path_component in state_data_path.split('/')[:-2]:
                dependent_variable_path += f'{path_component}/'
            dependent_variable_path += dependent_variable_history_subdir
            dependent_variable_history_datasets_list.append(dependent_variable_path)


        time_arrays_list = []
        kep_arrays_list = []
        omega_deg_s_arrays_list = []
        T_arrays_list = []
        detumbling_time_list = []
        vanes_x_rotation_list, vanes_y_rotation_list = [], []
        for (current_states_path, current_dep_vars_path) in zip(states_history_datasets_list, dependent_variable_history_datasets_list):
            # Extract data
            current_state_history_array = np.loadtxt(f'{Project_directory}/{current_states_path}')
            current_dependent_variable_history_array = np.loadtxt(f'{Project_directory}/{current_dep_vars_path}')

            current_time_array = (current_dependent_variable_history_array[:, 0]-current_dependent_variable_history_array[0, 0])/(24*3600)
            current_kep_array = current_dependent_variable_history_array[:, 1:7]

            # apogee and perigee
            current_apo_array = current_kep_array[:, 0] * (1 + current_kep_array[:, 1])
            current_peri_array = current_kep_array[:, 0] * (1 - current_kep_array[:, 1])

            # rotational velocity history
            omega_x_array_deg_s = np.rad2deg(current_state_history_array[:, 11])
            omega_y_array_deg_s = np.rad2deg(current_state_history_array[:, 12])
            omega_z_array_deg_s = np.rad2deg(current_state_history_array[:, 13])
            omega_deg_s = np.rad2deg(current_state_history_array[:, 11:14])

            # SRP torque history
            current_Tx_array = current_dependent_variable_history_array[:, 11]
            current_Ty_array = current_dependent_variable_history_array[:, 12]
            current_Tz_array = current_dependent_variable_history_array[:, 13]
            T = current_dependent_variable_history_array[:, 11:14]

            # Vane angles history
            current_vanes_x_rotations = np.rad2deg(current_dependent_variable_history_array[:,
                                           21:25])
            current_vanes_y_rotations = np.rad2deg(current_dependent_variable_history_array[:,
                                           25:29])

            # get detumbling time
            list_indices_zero_angles = np.where(np.sum(current_dependent_variable_history_array[:, 21:29], axis=1) == 0)[0]
            if (len(list_indices_zero_angles) != 0):
                current_detumbling_time_hours = (current_state_history_array[list_indices_zero_angles[0], 0]
                                                  - current_state_history_array[0, 0]) / 3600
            else:
                current_detumbling_time_hours = None
            time_arrays_list.append(current_time_array)
            kep_arrays_list.append(current_kep_array)
            omega_deg_s_arrays_list.append(omega_deg_s)
            T_arrays_list.append(T)
            detumbling_time_list.append(current_detumbling_time_hours/24)
            vanes_x_rotation_list.append(current_vanes_x_rotations)
            vanes_y_rotation_list.append(current_vanes_y_rotations)


        custom_xlim = (0, max(detumbling_time_list) * 1.05)
        # Compare the rotational velocity change
        for i, (time_array, om_array) in enumerate(zip(time_arrays_list, omega_deg_s_arrays_list)):
            plt.figure(plot_id*1000 + c_id*100 + 1)
            plt.plot(time_array, om_array[:, 0], label=f'{plot_label[i]}')
            plt.figure(plot_id*1000 + c_id*100 + 2)
            plt.plot(time_array, om_array[:, 1], label=f'{plot_label[i]}')
            plt.figure(plot_id*1000 + c_id*100 + 3)
            plt.plot(time_array, om_array[:, 2], label=f'{plot_label[i]}')

        plt.figure(plot_id*1000 + c_id*100 + 1)
        plt.title(f'{str(c)}: {comparison_name}')
        plt.grid(True)
        plt.xlabel(r"Time, $t$, [days]", fontsize=14)
        plt.ylabel(r"Rotational velocity X-component, $\omega_{x}$, [deg/s]", fontsize=14)
        plt.legend()
        plt.xlim(custom_xlim)
        plt.savefig(f'{save_plots_dir}/omega_x.png', bbox_inches='tight')
        plt.close()

        plt.figure(plot_id*1000 + c_id*100 + 2)
        plt.title(f'{str(c)}: {comparison_name}')
        plt.grid(True)
        plt.xlabel(r"Time, $t$, [days]", fontsize=14)
        plt.ylabel(r"Rotational velocity Y-component, $\omega_{y}$, [deg/s]", fontsize=14)
        plt.legend()
        plt.xlim(custom_xlim)
        plt.savefig(f'{save_plots_dir}/omega_y.png', bbox_inches='tight')
        plt.close()

        plt.figure(plot_id*1000 + c_id*100 + 3)
        plt.title(f'{str(c)}: {comparison_name}')
        plt.grid(True)
        plt.xlabel(r"Time, $t$, [days]", fontsize=14)
        plt.ylabel(r"Rotational velocity Z-component, $\omega_{z}$, [deg/s]", fontsize=14)
        plt.legend()
        plt.xlim(custom_xlim)
        plt.savefig(f'{save_plots_dir}/omega_z.png', bbox_inches='tight')
        plt.close()

        # compare vane angles histories
        for vane_id in range(4):
            for i, (time_array, vane_x_rotation_array, vane_y_rotation_array) in enumerate(zip(time_arrays_list, vanes_x_rotation_list, vanes_y_rotation_list)):
                plt.figure(plot_id*1000 + c_id*100 + 4 + vane_id)
                plt.plot(time_array, vane_x_rotation_array[:, vane_id], label=f'{plot_label[i]}')
                plt.figure(plot_id*1000 + c_id*100 + 5 + vane_id)
                plt.plot(time_array, vane_y_rotation_array[:, vane_id], label=f'{plot_label[i]}')

            plt.figure(plot_id*1000 + c_id*100 + 4 + vane_id)
            plt.title(f'{str(c)}: {comparison_name}')
            plt.grid(True)
            plt.xlabel(r"Time, $t$, [days]", fontsize=14)
            plt.ylabel(fr"Vane {vane_id} x-rotation history, $\theta_1$, [deg]", fontsize=14)
            plt.legend()
            plt.xlim(custom_xlim)
            plt.savefig(f'{save_plots_dir}/vane_{vane_id}_x_hist.png', bbox_inches='tight')
            plt.close()

            plt.figure(plot_id*1000 + c_id*100 + 5 + vane_id)
            plt.title(f'{str(c)}: {comparison_name}')
            plt.grid(True)
            plt.xlabel(r"Time, $t$, [days]", fontsize=14)
            plt.ylabel(fr"Vane {vane_id} y-rotation history, $\theta_2$, [deg]", fontsize=14)
            plt.legend()
            plt.xlim(custom_xlim)
            plt.savefig(f'{save_plots_dir}/vane_{vane_id}_y_hist.png', bbox_inches='tight')
            plt.close()


plt.show()
