install.packages(c('Biostrings', 'DescTools', 'dplyr'))
library(Biostrings)
library(DescTools)
library(dplyr)

# Load CSV file
data <- read.csv("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/h2l_div_beam_search_2_epoch_10_lr_1e-4_wd_0.1/full_test_set_true_gen_seqs_all_relevant_cols.csv")

# Remove the column "d_family" from data
data <- data[, -which(names(data) %in% c("d_family"))]

# Sort the data by the last number in "sequence_id"
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
regions <- c("fwr2", "cdr2", "fwr3", "cdr3", "fwr4")

# "fwr1", "cdr1", already done

for (region in regions) {
  process_region(data, region)
}




