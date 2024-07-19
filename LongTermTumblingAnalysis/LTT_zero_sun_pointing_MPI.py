from solarSailPropagationFunction import runPropagationAnalysis
import numpy as np
import itertools
import sys

overwrite_previous_bool = False
mode_combinations = list(itertools.product([0, 1, 2], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0]))


# generate all combinations of rotational velocity components from a selected subset to analyse
# for later
# [-100, -85, -70, -55, -40, -30, -20, -10, 0, 100, 85, 70, 55, 40, 30, 20, 10]
omega_x_list = np.array([0])
omega_y_list = np.array([0])
omega_z_list = np.array([0])

all_combinations = list(itertools.product(omega_x_list, omega_y_list, omega_z_list))


for comb_mode in mode_combinations:
    optical_model_mode = comb_mode[0]
    sma_ecc_inc_combination_mode = comb_mode[1]
    optical_mode_str = ["ACS3_optical_model", "double_ideal_optical_model", "single_ideal_optical_model"][
        optical_model_mode]

    runPropagationAnalysis(all_combinations,
                              optical_mode_str,
                              sma_ecc_inc_combination_mode,
                              0,
                              1,
                              overwrite_previous=False,
                              include_shadow_bool=False,
                              run_mode='LTT_sun_pointing',
                              output_frequency_in_seconds_=100,
                              initial_orientation_str='sun_pointing')

