from longTermTumblingNoAsymmetryGeneration import runLLT
from mpi4py import MPI
import numpy as np
import itertools
from MiscFunctions import chunks


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
n_processes = size

# generate all combinations of rotational velocity components from a selected subset to analyse
# for later
# [-150, -135, -120, -105, -90, -75, -60, -50, -40, -30, -20, -10, -7, -5, -3, -1, 0,
# 150, 135, 120, 105, 90, 75, 60, 50, 40, 30, 20, 10, 7, 5, 3, 1]
omega_x_list = np.arange(-15, 15 + 1, 5)
omega_y_list = np.arange(-15, 15 + 1, 5)
omega_z_list = np.arange(-15, 15 + 1, 5)

all_combinations = list(itertools.product(omega_x_list, omega_y_list, omega_z_list))

generator_of_chunks = chunks(all_combinations, int(len(all_combinations) / n_processes) + 1)

chunks_list = []
for chunk in generator_of_chunks:
    chunks_list.append(list(chunk))

chunks_list = [[(1, 1, 1)], [(-1, -1, -1)], [(1, -1, -1)], [(1, 1, -1)], [(-1, 1, 1)]]
runLLT(chunks_list[rank])

