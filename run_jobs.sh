#!/bin/bash
#SBATCH -e slurm_%A_%a.err
#SBATCH -o slurm_%A_%a.out
#SBATCH --array=1-98
#SBATCH -c 1
#SBATCH --mem-per-cpu=1G

srun ./fascia -z -r -g file_lists/${SLURM_ARRAY_TASK_ID}_lst.txt

#!/bin/bash
#SBATCH -e slurm_%A_%a.err
#SBATCH -o slurm_%A_%a.out
#SBATCH --array=1-98
#SBATCH -c 8
#SBATCH --mem-per-cpu=1G

srun ./fascia -z -r -o -g file_lists/${SLURM_ARRAY_TASK_ID}_lst.txt