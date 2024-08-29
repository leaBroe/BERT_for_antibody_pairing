install.packages("DescTools")
library(Biostrings)
library(DescTools)
library(dplyr)
#install.packages("stringr")
library(stringr)
# Install tidyr package if not already installed
#install.packages("tidyr")
#library(tidyr)

if (!require("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

# The following initializes usage of Bioc devel
BiocManager::install(version = "devel")
BiocManager::install("pwalign")

BiocManager::install("rBLAST")


# Load the libraries
library(rBLAST)
library(pwalign)

#input data is a csv file of the form:
# "sequence_alignment_aa_light","generated_sequence_light","BLOSUM_score","similarity","perplexity"
# "QSALTQPVSVSGSPGQSIAISCTGTSSDVGGYNSVSWFQQHPGKAPKLMIYDVSNRPSGVSNRFSGSKSGNTASLTISGLQAEDEADYYCSSYTSSSTRLFGGGTKLTVL","DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPYTFGQGTKLEIK",-50,12.037037037037,1.64723944664001

# (input file is generated with the R script: evaluate_B_types.R)

#extracted_data <- read.csv("extracted_data_heavy2light_60_epochs_beam_search_127845.csv")

# https://github.com/mhahsler/rBLAST/blob/devel/INSTALL
# Installation Instructions for Package rBLAST
# 
# 1. Install BLAST+
#   The BLAST+ software needs to be installed on your system. The official
# installation instructions for different plattforms are available at
# https://www.ncbi.nlm.nih.gov/books/NBK569861/
#   
#   Windows
# -------
#   Follow the instructions at https://www.ncbi.nlm.nih.gov/books/NBK52637/
#   
#   Linux/Unix
# ----------
#   Precombiled software package are already available for many Linux
# distributions. The package is typically called ⁠ncbi-blast+. For example
# on Debian/Ubuntu, the package can be installed using the APT
# package manager:
#   apt-get install ncbi-blast+
#   
#   MacOSX
# ------
#   The easiest approach is to use the .dmg installer file from
# https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/
# 
# 2. Checking that the path environment variables are correctly set
# R needs to be able to find the executable. The installers for LINUX/Unix
# and MacOSX set the correct values, so this is is mostly only an issue
# with Windows. After installing the software, try in R
# > Sys.which("blastn")
# 
# If the command returns "" instead of
# the path to the executable, then you need to set the environment variable
# called PATH. In R
# Sys.setenv(PATH = paste(Sys.getenv("PATH"),
#                         "path_to_your_BLAST_installation", sep=.Platform$path.sep))
# 
# More details about setting the environment variables permanently can be
# found in the Windows installation guide at
# https://www.ncbi.nlm.nih.gov/books/NBK52637/
# 
# Sys.setenv(PATH = paste(Sys.getenv("PATH"),
#                         "/usr/local/ncbi/blast", sep=.Platform$path.sep))

Sys.which("blastn")

# Function to calculate similarity
calculate_similarity <- function(seq1, seq2) {
  alignment <- pairwiseAlignment(seq1, seq2)
  score <- pid(alignment)  # Percent Identity
  return(score)
}

# Function to calculate BLOSUM score
calculate_blosum <- function(seq1, seq2) {
  alignment <- pairwiseAlignment(seq1, seq2, substitutionMatrix = "BLOSUM62")
  score <- score(alignment)
  return(score)
}

true_seq <- "QSALTQPVSVSGSPGQSIAISCTGTSSDVGGYNSVSWFQQHPGKAPKLMIYDVSNRPSGVSNRFSGSKSGNTASLTISGLQAEDEADYYCSSYTSSSTRLFGGGTKLTVL"
generated_seq <- "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPYTFGQGTKLEIK"

alignment_test <- pairwiseAlignment(true_seq, generated_seq, substitutionMatrix = "BLOSUM62")
print(alignment_test)

# Create a FASTA file from the sequence to be used as the BLAST database
writeXStringSet(AAStringSet("SSELTQDPDVSVALGQTVRITCQGDTLRIYYTSWYQQKPGRAPVLVFYGYNNRPSGIPDRFSGSSSGNTASLTITGAQAEDEAVYYCNSRDASGNPQVFGRGTKVTVL"), 
                "protein_sequence.fasta")

# Make the BLAST database (note: dbtype = 'prot' for protein sequences)
system("makeblastdb -in protein_sequence.fasta -dbtype prot -out protein_db")

# Initialize BLAST for protein sequences
blastp <- blast(db = "protein_db", type = "blastp")

# Create the query sequence as an AAStringSet object
query_seq <- AAStringSet("DIQMTQSPSSLSASVGDRVTITCRASQGISNYLAWFQQKPGKAPKSLIYAASSLQSGVPSKFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPLTFGGGTKVEIK")

# Perform BLAST search
result <- predict(blastp, query_seq)

# View the result
print(result)

print(result$pident)

calculate_blosum(true_seq, generated_seq)

calculate_similarity(true_seq, generated_seq)

# add progress bar
pb <- txtProgressBar(min = 0, max = nrow(extracted_data), style = 3)

# Initialize columns for the results if not already in the data frame
extracted_data$calculated_blosum <- NA
extracted_data$calculated_similarity <- NA

# Loop over each row and calculate the BLOSUM score and similarity
for (i in 1:nrow(extracted_data)) {
  seq1 <- extracted_data$sequence_alignment_aa_light[i]
  seq2 <- extracted_data$generated_sequence_light[i]
  
  # Calculate BLOSUM score (using global alignment by default)
  extracted_data$calculated_blosum[i] <- calculate_blosum(seq1, seq2)
  
  # Calculate similarity (using global alignment by default)
  extracted_data$calculated_similarity[i] <- calculate_similarity(seq1, seq2)
  
  # Update the progress bar
  setTxtProgressBar(pb, i)
  
}

close(pb)
# Print the updated data frame
print(df)

# save the updated data frame

write.csv(extracted_data, "extracted_data_alignment_heavy2light_60_epochs_beam_search_127845.csv", row.names = FALSE)

# calculate the mean BLOSUM score and similarity
mean_blosum <- mean(extracted_data$calculated_blosum, na.rm = TRUE)
mean_similarity <- mean(extracted_data$calculated_similarity, na.rm = TRUE)
median_blosum <- median(extracted_data$calculated_blosum, na.rm = TRUE)
median_similarity <- median(extracted_data$calculated_similarity, na.rm = TRUE)

print(mean_blosum)
print(mean_similarity)
print(median_blosum)
print(median_similarity)
