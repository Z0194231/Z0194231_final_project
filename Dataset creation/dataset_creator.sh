#!/bin/bash

#SBATCH --job-name=control_dataset_creator
#SBATCH -p shared
#SBATCH --output=control_twc_dataset.out
#SBATCH --error=control_twc_dataset.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --mail-user=<dpmp52@durham.ac.uk>
#SBATCH --mail-type=END,FAIL
#SBATCH -t 3-00:00:00

module load python/3.9.9

source ~/adna_env/bin/activate

python /nobackup/dpmp52/adna_files/TWO_CAT_files/two_cat_dataset.py
