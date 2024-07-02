from longTermTumblingNoAsymmetryGeneration import runLLT
from mpi4py import MPI
import numpy as np
import itertools
from MiscFunctions import chunks, divide_list
from longTermTumbling_ACS3Model import analysis_save_data_dir
import os
import sys
from generalConstants import R_E
from longTermTumbling_ACS3Model import a_0, e_0, wings_optical_properties

overwrite_previous_bool = False
print(sys.argv)

# argv[0] is the executable
optical_model_mode = int(sys.argv[1])  # 0: ACS3 optical model, 1: double-sided  ideal model, 2: single-sided ideal model
sma_ecc_combination_mode = int(sys.argv[2])    # see below for the combinations

eccentricities = [0.0, 0.3, 0.6]
sma = ['LEO', 'MEO', 'GEO']
sma_ecc_combinations = [[sma[0], eccentricities[0]],
                        [sma[1], eccentricities[0]],
                        [sma[1], eccentricities[1]],
                        [sma[2], eccentricities[0]],
                        [sma[2], eccentricities[1]],
                        [sma[2], eccentricities[2]]]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
n_processes = size

# generate all combinations of rotational velocity components from a selected subset to analyse
# for later
# [-100, -85, -70, -55, -40, -30, -20, -10, 0, 100, 85, 70, 55, 40, 30, 20, 10]
#omega_x_list = np.arange(-15, 15 + 1, 5)
#omega_y_list = np.arange(-15, 15 + 1, 5)
#omega_z_list = np.arange(-15, 15 + 1, 5)
omega_x_list = np.array([-100, -85, -70, -55, -40, -30, -20, -10, 0, 100, 85, 70, 55, 40, 30, 20, 10])
omega_y_list = np.array([-100, -85, -70, -55, -40, -30, -20, -10, 0, 100, 85, 70, 55, 40, 30, 20, 10])
omega_z_list = np.array([0])

sma = sma_ecc_combinations[sma_ecc_combination_mode][0]
ecc = sma_ecc_combinations[sma_ecc_combination_mode][1]

if (sma == 'LEO'):
    initial_sma = a_0
elif (sma == 'MEO'):
    initial_sma = R_E + 10000e3  # m
elif (sma == 'GEO'):
    initial_sma = R_E + 36000e3  # m

if (ecc == 0):
    initial_ecc = e_0
else:
    initial_ecc = ecc

if (optical_model_mode == 0):
    selected_wings_optical_properties = wings_optical_properties
    save_sub_dir = f'LTT_NoAsymetry_data_ACS3/{sma}_ecc_{ecc}'
elif (optical_model_mode == 1):
    selected_wings_optical_properties = [np.array([0., 0., 1., 1., 0.0, 0.0, 2 / 3, 2 / 3, 1.0, 1.0])] * 4
    save_sub_dir = f'LTT_NoAsymetry_data_double_ideal/{sma}_ecc_{ecc}'
elif (optical_model_mode == 2):
    selected_wings_optical_properties = [np.array([0., 0., 1., 0., 0.0, 0.0, 2 / 3, 2 / 3, 1.0, 1.0])] * 4
    save_sub_dir = f'LTT_NoAsymetry_data_single_ideal/{sma}_ecc_{ecc}'
else:
    raise Exception("Unrecognised optical model mode in LTT propagation")

if (not os.path.exists(analysis_save_data_dir + f'/{save_sub_dir}') and rank == 0):
    os.makedirs(analysis_save_data_dir + f'/{save_sub_dir}/states_history')
    os.makedirs(analysis_save_data_dir + f'/{save_sub_dir}/dependent_variable_history')

all_combinations = list(itertools.product(omega_x_list, omega_y_list, omega_z_list))
if (not overwrite_previous_bool):
    new_combs = []
    for comb in all_combinations:
        initial_rotational_velocity = np.array(
            [comb[0] * 2 * np.pi / 3600., comb[1] * 2 * np.pi / 3600, comb[2] * 2 * np.pi / 3600])
        rotations_per_hour = np.round(initial_rotational_velocity * 3600 / (2 * np.pi), 1)
        tentative_file = analysis_save_data_dir + f'/{save_sub_dir}/states_history/state_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}.dat'
        if (os.path.isfile(tentative_file)):
            # if the file exists, skip this propagation
            continue
        else:
            new_combs.append(comb)
    all_combinations = new_combs

chunks_list = divide_list(all_combinations, n_processes)

print(f"hello from rank {rank}")
runLLT(chunks_list[rank],
       selected_wings_optical_properties,
       initial_sma,
       initial_ecc,
       analysis_save_data_dir + f'/{save_sub_dir}',
       overwrite_previous = overwrite_previous_bool)

