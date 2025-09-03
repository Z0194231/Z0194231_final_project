#!/bin/bash

#SBATCH --job-name=sampled_filtered_cnn_prediction
#SBATCH -p shared
#SBATCH --output=sampled_filt_cnn_prediction.out
#SBATCH --error=sampled_filt_cnn_prediction.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#SBATCH --mail-user=<dpmp52@durham.ac.uk>
#SBATCH --mail-type=END,FAIL
#SBATCH -t 3-00:00:00

module load python/3.9.9
module load cuda/11.7.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/adna_env/bin/activate

python /nobackup/dpmp52/adna_files/TWO_CAT_files/under_filtered.py
