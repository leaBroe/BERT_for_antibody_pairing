library(Biostrings)
library(DescTools)
library(dplyr)
#install.packages("stringr")
library(stringr)
# Install tidyr package if not already installed
#install.packages("tidyr")
library(tidyr)


# structure of input data:
# csv file with the following columns:
# sequence_id,sequence,sequence_alignment_aa,germline_alignment_aa,fwr1,fwr1_aa,cdr1,cdr1_aa,fwr2,
# fwr2_aa,cdr2,cdr2_aa,fwr3,fwr3_aa,fwr4,fwr4_aa,cdr3,cdr3_aa,fwr1_start,fwr1_end,cdr1_start,cdr1_end,
# fwr2_start,fwr2_end,cdr2_start,cdr2_end,fwr3_start,fwr3_end,fwr4_start,fwr4_end,cdr3_start,cdr3_end,
# cdr3_aa_length,locus,v_family,j_family,d_family

# structure of rows:
# True_Seq_1
# Generated_Seq_1
# True_Seq_2
# Generated_Seq_2

# Load CSV file
data <- read.csv("full_test_set_true_gen_seqs_all_relevant_cols.csv")

# Remove the column "d_family" from data
data <- data[, -which(names(data) %in% c("d_family"))]

# Sort the data by the last number in "sequence_id" -> to make sure we get the right pairing
data <- data[order(as.numeric(sub(".*_", "", data$sequence_id))),]

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

# Function to process a specific region with data cleaning
process_region <- function(data, region) {
  region_column <- paste0(region, "_aa")
  region_column <- paste0(region_column, "_correct")  # for fwr4_aa_correct
  
  # Clean the data for the specific region
  cleaned_data <- data %>%
    filter(!is.na(.data[[region_column]]), .data[[region_column]] != "")
  
  similarity_scores <- c()
  blosum_scores <- c()
  
  pb <- txtProgressBar(min = 0, max = floor(nrow(cleaned_data) / 2), style = 3)
  
  for (i in seq(1, nrow(cleaned_data) - 1, by = 2)) {
    seq1 <- cleaned_data[[region_column]][i]
    seq2 <- cleaned_data[[region_column]][i + 1]
    
    similarity_scores <- c(similarity_scores, calculate_similarity(seq1, seq2))
    blosum_scores <- c(blosum_scores, calculate_blosum(seq1, seq2))
    
    setTxtProgressBar(pb, i / 2)
  }
  
  close(pb)
  
  # Save the results to CSV files
  write.csv(data.frame(similarity_scores, blosum_scores), paste0(region, "_similarity_blosum_scores.csv"), row.names = FALSE)
  
  # Save the statistics to a CSV file
  write.csv(data.frame(
    mean_similarity = mean(similarity_scores),
    median_similarity = median(similarity_scores),
    sd_similarity = sd(similarity_scores),
    n_similarity = length(similarity_scores),
    mean_blosum = mean(blosum_scores),
    median_blosum = median(blosum_scores),
    sd_blosum = sd(blosum_scores),
    n_blosum = length(blosum_scores)
  ), paste0(region, "_similarity_blosum_scores_stats.csv"), row.names = FALSE)
}

# Process each region separately
regions <- c("fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4")

# for (region in regions) {
#   process_region(data, region)
# }

# cdr1
process_region(data, "fwr4")


# fwr4 region is not processed properly by PyIR (only FG or FGG instead of the full fwr4 seq), 
# which is why we have to manipulate the data further and include the true and generated AA light sequences 
# (from our full eval output file!)

# Read the CSV file
# input file structure:
# column names:
# true_sequence,generated_sequence
# rows: AA sequences without spaces

sequences <- read.csv("full_test_set_true_gen_aa_seqs.csv", header = TRUE, stringsAsFactors = FALSE)

# Reshape the data
reshaped <- sequences %>%
  mutate(id = row_number()) %>%
  pivot_longer(cols = c(true_sequence, generated_sequence),
               names_to = "type",
               values_to = "sequence") %>%
  mutate(type = if_else(type == "true_sequence",
                        paste0("True_Seq_", id),
                        paste0("Generated_Seq_", id))) %>%
  select(-id) %>%
  arrange(type)

# Save the reshaped data to a new CSV file
write.csv(reshaped, "reshaped_sequences.csv", row.names = FALSE, quote = FALSE)

reshaped_seqs <- read.csv("reshaped_sequences.csv", header = TRUE, stringsAsFactors = FALSE)

# Sort the data by the last number in "sequence_id"
reshaped_seqs <- reshaped_seqs[order(as.numeric(sub(".*_", "", reshaped_seqs$type))),]

# change column names to match the original data
colnames(reshaped_seqs) <- c("sequence_id", "aa_seq_from_true_output")

# merge the reshaped data with the original data
merged_data <- merge(data, reshaped_seqs, by = "sequence_id")

# Sort the data by the last number in "sequence_id"
merged_data <- merged_data[order(as.numeric(sub(".*_", "", merged_data$sequence_id))),]


# Process the dataframe to calculate the fwr4_aa_correct column
merged_data_added_fwr4 <- merged_data %>%
  mutate(
    # Calculate the AA start position by dividing by 3 and rounding up
    aa_start = ceiling(fwr4_start / 3),
    
    # Extract the substring starting from the calculated position
    fwr4_aa_correct = str_sub(aa_seq_from_true_output, aa_start)
  )


# Function to process a specific region with data cleaning
process_region_fwr4 <- function(data, region) {
  region_column <- paste0(region, "_aa_correct") # fwr4

  # Clean the data for the specific region
  cleaned_data <- data %>%
    filter(!is.na(.data[[region_column]]), .data[[region_column]] != "")
  
  similarity_scores <- c()
  blosum_scores <- c()
  
  pb <- txtProgressBar(min = 0, max = floor(nrow(cleaned_data) / 2), style = 3)
  
  print(paste0("Processing fwr4 in column:", region_column))
  
  for (i in seq(1, nrow(cleaned_data) - 1, by = 2)) {
    seq1 <- cleaned_data[[region_column]][i]
    seq2 <- cleaned_data[[region_column]][i + 1]
    
    similarity_scores <- c(similarity_scores, calculate_similarity(seq1, seq2))
    blosum_scores <- c(blosum_scores, calculate_blosum(seq1, seq2))
    
    setTxtProgressBar(pb, i / 2)
  }
  
  close(pb)
  
  # Save the results to CSV files
  write.csv(data.frame(similarity_scores, blosum_scores), paste0(region, "_similarity_blosum_scores_2.csv"), row.names = FALSE)
  
  # Save the statistics to a CSV file
  write.csv(data.frame(
    mean_similarity = mean(similarity_scores),
    median_similarity = median(similarity_scores),
    sd_similarity = sd(similarity_scores),
    n_similarity = length(similarity_scores),
    mean_blosum = mean(blosum_scores),
    median_blosum = median(blosum_scores),
    sd_blosum = sd(blosum_scores),
    n_blosum = length(blosum_scores)
  ), paste0(region, "_similarity_blosum_scores_stats_2.csv"), row.names = FALSE)
}


# cdr1
process_region_fwr4(merged_data_added_fwr4, "fwr4")




