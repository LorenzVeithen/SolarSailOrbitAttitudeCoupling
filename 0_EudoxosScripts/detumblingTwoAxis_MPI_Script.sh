#!/bin/bash

# Activate the Conda environment
eval "$(conda shell.bash hook)"
conda activate tudat-bundle

export PYTHONPATH="/home2/lorenz/SourceCode":$PYTHONPATH

# Default to 4 processes if not provided
NUM_PROCESSES=${1:-14}
shift  # Remove the first argument, which is the number of processes

# Run mpiexec with the specified or default number of processes
mpiexec -n $NUM_PROCESSES python "/home2/lorenz/SourceCode/VaneDetumblingAnalysis/detumblingTwoAxis_MPI.py" 0 0 0 &> twoAxis_0_0_0.log 2>&1 &

