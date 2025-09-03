import sys
import pandas
import os
import csv
from Bio import SeqIO
import pysam

sys.stdout.write("FILES PROCESSED: ")
files_processed = 0

#NON_MTB_SAM = "fulldedupTLT19_alignment.sam"
MTB_SAM = "fullTLT19_filtered.sam"
CHECK_SAM = "/nobackup/dpmp52/adna_files/COMP_U_files/TLT_19_udghalf.trimmed.QNAMEfiltered.sam"

mapped_ids = set()
filtered_out_ids = set()
check_ids = set()

#with pysam.AlignmentFile(NON_MTB_SAM, 'r') as sam_file:
  #for read in sam_file.fetch(until_eof=True):
    #if read.query_sequence and not read.is_unmapped:
      #filtered_out_ids.add(read.query_name.strip().split()[0].replace('/1', '').replace('/2', ''))

with pysam.AlignmentFile(MTB_SAM, 'r') as sam_file:
  for read in sam_file.fetch(until_eof=True):
    if read.query_sequence and not read.is_unmapped:
      mapped_ids.add(read.query_name.strip().split()[0].replace('/1', '').replace('/2', ''))
      
with pysam.AlignmentFile(CHECK_SAM, 'r') as sam_file:
  for read in sam_file.fetch(until_eof=True):
    if read.query_sequence and not read.is_unmapped:
      check_ids.add(read.query_name.strip().split()[0].replace('/1', '').replace('/2', ''))
      

seen_seqs = set()
  
with open('MTB_twc_dataset.csv', 'w', newline='') as csv_file:
  
  writer = csv.DictWriter(csv_file, fieldnames=["id", "sequence", "status"])
  writer.writeheader()
      
  for read in SeqIO.parse("/nobackup/dpmp52/adna_files/COMP_U_files/TLT_19_trimmedmerged.fq", 'fastq'):
    fid = read.id.strip().split()[0].replace('/1', '').replace('/2', '')
    
    if fid in seen_seqs:
      continue
      
    seen_seqs.add(fid)
    
    if fid in check_ids:
      writer.writerow({
        "id": fid,
        "sequence": read.seq,
        "status": "mapped MTB"
      })  
    elif fid in mapped_ids:
      writer.writerow({
        "id": fid,
        "sequence": read.seq,
        "status": "unmapped MTB"
      }) 
    #elif fid not in filtered_out_ids:
      #writer.writerow({
        #"id": fid,
        #"sequence": read.seq,
        #"status": "non-MTB"
      #})
        
    files_processed += 1
    sys.stdout.write(str(files_processed) + " ")
    sys.stdout.flush()