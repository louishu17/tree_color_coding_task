#!/bin/bash 
#SBATCH -e slurm_%A_%a.err
#SBATCH -o slurm_%A_%a.out
#SBATCH --array=1-5000

module load anaconda3

srun python vizualizations.py
