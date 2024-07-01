from vaneDetumblingGeneration import runDetumblingAnalysis
from mpi4py import MPI
import numpy as np
import itertools
from MiscFunctions import chunks, divide_list
from vaneDetumbling_ACS3Model import detumbling_save_data_dir
import os
import sys
from generalConstants import R_E
from vaneDetumbling_ACS3Model import a_0, e_0, i_0, wings_optical_properties

overwrite_previous_bool = False

# argv[0] is the executable
optical_model_mode = int(sys.argv[1])  # 0: ACS3 optical model, 1: double-sided  ideal model, 2: single-sided ideal model
sma_ecc_inc_combination_mode = int(sys.argv[2])    # see below for the combinations
include_shadow_b = int(sys.argv[3])


eccentricities = [0.0, 0.3, 0.6]
inclinations_deg = [i_0, 45.0, 0.0]
sma = ['LEO', 'MEO', 'GEO']
sma_ecc_inc_combinations = [[sma[0], eccentricities[0], inclinations_deg[0]],   # like previous: currently comb_0
                        [sma[1], eccentricities[0], inclinations_deg[0]],       # like previous: currently comb_1
                        [sma[1], eccentricities[1], inclinations_deg[0]],       # like previous: currently comb_2
                        [sma[2], eccentricities[0], inclinations_deg[0]],       # like previous: currently comb_3
                        [sma[0], eccentricities[0], inclinations_deg[1]],
                        [sma[0], eccentricities[0], inclinations_deg[2]],
                        [sma[1], eccentricities[0], inclinations_deg[1]],
                        [sma[1], eccentricities[0], inclinations_deg[2]],
                        [sma[2], eccentricities[0], inclinations_deg[1]],
                        [sma[2], eccentricities[0], inclinations_deg[2]]]
                        # add a GTO to the lot ? The amount of data is slowly getting out of hand...

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
n_processes = size

# generate all combinations of rotational velocity components from a selected subset to analyse
# for later
# [-100, -85, -70, -55, -40, -30, -20, -10, 0, 100, 85, 70, 55, 40, 30, 20, 10]
omega_x_list = np.arange(-15, 15 + 1, 5)
omega_y_list = np.arange(-15, 15 + 1, 5)
omega_z_list = np.arange(-15, 15 + 1, 5)
#omega_x_list = np.array([-100, -85, -70, -55, -40, -30, -20, -10, 0, 100, 85, 70, 55, 40, 30, 20, 10])
#omega_y_list = np.array([-100, -85, -70, -55, -40, -30, -20, -10, 0, 100, 85, 70, 55, 40, 30, 20, 10])
#omega_z_list = np.array([0])

sma = sma_ecc_inc_combinations[sma_ecc_inc_combination_mode][0]
ecc = sma_ecc_inc_combinations[sma_ecc_inc_combination_mode][1]
inc = sma_ecc_inc_combinations[sma_ecc_inc_combination_mode][2]

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

initial_inc = inc

if (optical_model_mode == 0):
    selected_vanes_optical_properties = wings_optical_properties
    save_sub_dir = f'detumbling_NoAsymetry_data_ACS3_opt_model/{sma}_ecc_{ecc}_inc_{initial_inc}'
    raise Exception('optical model not yet available for detumbling mode')
elif (optical_model_mode == 1):
    selected_vanes_optical_properties = [np.array([0., 0., 1., 1., 0.0, 0.0, 2 / 3, 2 / 3, 1.0, 1.0])] * 4
    save_sub_dir = f'detumbling_NoAsymetry_data_double_ideal_opt_model/{sma}_ecc_{ecc}_inc_{initial_inc}'
elif (optical_model_mode == 2):
    selected_vanes_optical_properties = [np.array([0., 0., 1., 0., 0.0, 0.0, 2 / 3, 2 / 3, 1.0, 1.0])] * 4
    save_sub_dir = f'detumbling_NoAsymetry_data_single_ideal_opt_model/{sma}_ecc_{ecc}_inc_{initial_inc}'
    raise Exception('optical model not yet available for detumbling mode')
else:
    raise Exception("Unrecognised optical model mode in detumbling propagation")

if (not os.path.exists(detumbling_save_data_dir + f'/{save_sub_dir}') and rank == 0):
    os.makedirs(detumbling_save_data_dir + f'/{save_sub_dir}/states_history')
    os.makedirs(detumbling_save_data_dir + f'/{save_sub_dir}/dependent_variable_history')

all_combinations = list(itertools.product(omega_x_list, omega_y_list, omega_z_list))
if (not overwrite_previous_bool):
    new_combs = []
    for comb in all_combinations:
        initial_rotational_velocity = np.array(
            [comb[0] * 2 * np.pi / 3600., comb[1] * 2 * np.pi / 3600, comb[2] * 2 * np.pi / 3600])
        rotations_per_hour = np.round(initial_rotational_velocity * 3600 / (2 * np.pi), 1)
        tentative_file = detumbling_save_data_dir + f'/{save_sub_dir}/states_history/state_history_omega_x_{rotations_per_hour[0]}_omega_y_{rotations_per_hour[1]}_omega_z_{rotations_per_hour[2]}.dat'
        if (os.path.isfile(tentative_file)):
            # if the file exists, skip this propagation
            continue
        else:
            new_combs.append(comb)
    all_combinations = new_combs

chunks_list = divide_list(all_combinations, n_processes)

print(f"hello from rank {rank}")
runDetumblingAnalysis(chunks_list[rank],
                      selected_vanes_optical_properties,
                      vane_has_ideal_model_bool=True,
                      sma_0=initial_sma,
                      ecc_0=initial_ecc,
                      i_0_deg=initial_inc,
                      include_shadow_bool=bool(include_shadow_b),
                      save_directory=detumbling_save_data_dir + f'/{save_sub_dir}',
                      overwrite_previous = overwrite_previous_bool)

