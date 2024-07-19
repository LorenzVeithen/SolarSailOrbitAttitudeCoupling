#!/bin/bash

#SBATCH --job-name="DT_2_2_1"
#SBATCH --time=15:00:00
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=15GB
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
srun python /scratch/lveithen/SourceCode/VaneDetumblingAnalysis/detumbling_MPI.py 2 2 1 > DT_2_2_1.log
conda deactivate
