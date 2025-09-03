#!/bin/bash

#SBATCH --job-name=pydamage
#SBATCH -p shared
#SBATCH --output=pydamage_analysis.out
#SBATCH --error=pydamage_analysis.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --mail-user=<dpmp52@durham.ac.uk>
#SBATCH --mail-type=END,FAIL
#SBATCH -t 3-00:00:00

module load python/3.9.9
module load cuda/11.7.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/adna_env/bin/activate

python /nobackup/dpmp52/adna_files/TWO_CAT_files/pydamage_analysis.py