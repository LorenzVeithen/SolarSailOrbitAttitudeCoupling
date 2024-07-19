#!/bin/bash

#SBATCH --job-name="LTT_2_4"
#SBATCH --time=15:00:00
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=10GB
#SBATCH --account=education-ae-msc-ae

module load 2023r1
module load openmpi
module load python
module load miniconda3
module load openssh
module load git

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

export PYTHONPATH="/scratch/lveithen/SourceCode":$PYTHONPATH

# Activate conda, run job, deactivate conda
conda activate LV-tudat-bundle
srun python /scratch/lveithen/SourceCode/LongTermTumblingAnalysis/LTT_MPI.py 2 4 > LTT_large_optical_2_combination_4.log
conda deactivate
