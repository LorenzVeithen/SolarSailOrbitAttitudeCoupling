from solarSailPropagationFunction import runPropagationAnalysis
import numpy as np
import itertools
import sys

mode_combinations = list(itertools.product([0, 1, 2], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

for mode_combo in mode_combinations:
    print(mode_combo)
    optical_model_mode = mode_combo[0]
    sma_ecc_inc_combination_mode = mode_combo[0]
    optical_mode_str = ["ACS3_optical_model", "double_ideal_optical_model", "single_ideal_optical_model"][optical_model_mode]

    runPropagationAnalysis([(0, 0, 0)],
                           optical_mode_str,
                           sma_ecc_inc_combination_mode,
                           0,
                           1,
                           overwrite_previous=False,
                           include_shadow_bool=False,
                           run_mode='keplerian_LTT',
                           output_frequency_in_seconds_=1)

