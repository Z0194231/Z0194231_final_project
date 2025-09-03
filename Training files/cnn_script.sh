#!/bin/bash

#SBATCH --job-name=mtb_twc_cnn
#SBATCH -p bigmem
#SBATCH --output=mtb_cnn_modelling.out
#SBATCH --error=mtb_cnn_modelling.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=800GB
#SBATCH --mail-user=<dpmp52@durham.ac.uk>
#SBATCH --mail-type=START,END,FAIL
#SBATCH -t 3-00:00:00

module load python/3.9.9

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=""
export TF_ENABLE_AUTO_MIXED_PRECISION=0

source ~/adna_env/bin/activate

python /nobackup/dpmp52/adna_files/TWO_CAT_files/mtb_cnn_modelling.py
