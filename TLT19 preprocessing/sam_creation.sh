#!/bin/bash

#SBATCH --job-name=sam_creator
#SBATCH -p shared
#SBATCH --output=sam_creator.out
#SBATCH --error=sam_creator.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --mail-user=<dpmp52@durham.ac.uk>
#SBATCH --mail-type=END,FAIL
#SBATCH -t 3-00:00:00

module load python/3.9.9
module load cuda/11.7.0

source /home/dpmp52/miniconda3

conda activate samtools_env

bwa aln -l 1024 -n 0.01 full_read_megahit/final.contigs.fa /nobackup/dpmp52/adna_files/COMP_U_files/TLT_19_trimmedmerged.fq > fullTLT19_alignment.sam