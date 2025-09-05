# Archaeogenomics of Tuberculosis: Recovering Lost Genes from Ancient Mycobacterium tuberculosis Complex Genomes via Convolutional-BiLSTM Neural Network prediction
## Github Repository for Student Z0194231

This is a repository containing all written code and generated files that were output as part of the Final Project module for MDS in Digital Humanities program 2024-2025. 

Files too big for GitHub storage were instead stored in the student's Durham University account. Bigger files included:
- **PROJECT DATASETS:** Both built datasets for model training and testing, saved as CSV files.
- **TEST DATA:** Test data split from original dataset, used in model prediction.
- **PREDICTION DATA:** A folder containing predicted categories VS true categories, for graph generation and model performance evaluation. 
- **TLT-19 PRE-PROCESSING DATA:** A folder containing all files generated as part of data pre-processing and TLT-19 *de novo* assembly. Most files are in SAM or BAM format, with generated files via DeDup nested in an additional folder. Initial results from PyDamage calculations are also stored in an additional folder within this section.  

This repository is divided into 4 folders sorted according to the 4 stages of data processing and analysis conducted by this project:
- **DATASET CREATION:** contains relevant script for dataset assembly.
- **GRAPH PLOTTERS:** contains relevant scripts for graph plotting, used in model evaluation. 
- **TLT19 PREPROCESSING:** contains relevant script output for FASTQ and SAM/BAM files *de novo* assembly and cleaning, including generated PyDamage files and MEGAHIT files. Most methods were performed using Bash. The nested PyDamage folder also contains Python scripts for result analysis generation.
- **TESTING FILES:** contains relevant script for all 4 prediction runs conducted for 2 model fits, 2 predictions each. It also contains the generated results of model fit during model training. 
- **TRAINING FILES:** contains relevant script for the 2 model fits, as well as the saved tokenizers, label encoders, and models for further prediction testing.

All written code presented in this GitHub repository has been developed using Python for model development and analysis, or Shell for script submission to HPC. Only 1 fail containing Bash commands for SAM generation during the pre-processing state has been saved, as most work was conducted live on the HPC remote server. 
