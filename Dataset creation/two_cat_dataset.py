import sys
import pandas
import os
import csv
from Bio import SeqIO
import pysam

#Keep track of how many files have been processed 
sys.stdout.write("FILES PROCESSED: ")
files_processed = 0

#This dataset was re-used for the creation of both binary datasets. First SAM file becomes commented for the creation of the experimental dataset
#Both the PyDamage filtered and non-filtered SAM files are used for further fragment differentiation, to ensure that the non-MTB category is sufficiently separated from potential MTB strains
NON_MTB_SAM = "fulldedupTLT19_alignment.sam"
MTB_SAM = "fullTLT19_filtered.sam"
CHECK_SAM = "/nobackup/dpmp52/adna_files/COMP_U_files/TLT_19_udghalf.trimmed.QNAMEfiltered.sam"

# Create sets to store recovered ids for mapped aDNA fragments. 
mapped_ids = set()
filtered_out_ids = set()
check_ids = set()

# After opening each SAM file, read in and normalise mapped MTB ids 
with pysam.AlignmentFile(NON_MTB_SAM, 'r') as sam_file:
  for read in sam_file.fetch(until_eof=True):
    if read.query_sequence and not read.is_unmapped:
      filtered_out_ids.add(read.query_name.strip().split()[0].replace('/1', '').replace('/2', ''))

with pysam.AlignmentFile(MTB_SAM, 'r') as sam_file:
  for read in sam_file.fetch(until_eof=True):
    if read.query_sequence and not read.is_unmapped:
      mapped_ids.add(read.query_name.strip().split()[0].replace('/1', '').replace('/2', ''))
      
with pysam.AlignmentFile(CHECK_SAM, 'r') as sam_file:
  for read in sam_file.fetch(until_eof=True):
    if read.query_sequence and not read.is_unmapped:
      check_ids.add(read.query_name.strip().split()[0].replace('/1', '').replace('/2', ''))
      
# Establish an id check to avoid duplicates
seen_seqs = set()

# Create a CSV dataset - for experimental dataset, name changes to "MTB_twc_dataset.csv"
with open('filtered_twc_dataset.csv', 'w', newline='') as csv_file:

  # Prepare an csv writer and start writing in the headers for all columns
  writer = csv.DictWriter(csv_file, fieldnames=["id", "sequence", "status"])
  writer.writeheader()

  # Parse the raw FASTQ file containing all TLT-19 reads, and iterate over each 
  for read in SeqIO.parse("/nobackup/dpmp52/adna_files/COMP_U_files/TLT_19_trimmedmerged.fq", 'fastq'):
    #Normalise the id for each read
    fid = read.id.strip().split()[0].replace('/1', '').replace('/2', '')

    # If read already seen, skip 
    if fid in seen_seqs:
      continue

    # Add read id to seen ids
    seen_seqs.add(fid)

    # If id is present in Blevins' SAM file, categorise as MTB
    if fid in check_ids:
      writer.writerow({
        "id": fid,
        "sequence": read.seq,
        "status": "MTB" #For experimental dataset, this would be 'mapped MTB'
      })  
    # If otherwise id is present in PyDamage filtered SAM file, categorise as MTB
    elif fid in mapped_ids:
      writer.writerow({
        "id": fid,
        "sequence": read.seq,
        "status": "MTB" #For experimental dataset, this would be 'unmapped MTB'
      }) 
    # Finally, if id is not mapped in either file, NOR it is mapped according to the unfiltered SAM file, categorise as non-MTB
    elif fid not in filtered_out_ids:
      writer.writerow({
        "id": fid,
        "sequence": read.seq,
        "status": "non-MTB"
      }) #For experimental dataset, this last section would be commented out

    # Count read as processed
    files_processed += 1
    sys.stdout.write(str(files_processed) + " ")
    sys.stdout.flush()
