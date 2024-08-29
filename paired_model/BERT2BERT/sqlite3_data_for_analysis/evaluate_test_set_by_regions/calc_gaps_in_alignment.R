library(Biostrings)
library(DescTools)
library(dplyr)
library(stringr)
library(tidyr)

# Load CSV file
data <- read.csv("full_test_set_true_gen_seqs_all_relevant_cols.csv")

# Remove the column "d_family" from data
data <- data[, -which(names(data) %in% c("d_family"))]

# Sort the data by the last number in "sequence_id"
data <- data[order(as.numeric(sub(".*_", "", data$sequence_id))),]

# Function to calculate similarity and count gaps using nindel
calculate_similarity <- function(seq1, seq2) {
  #cat("Calculating similarity for sequences:\n")
  #cat(seq1, "\n")
  #cat(seq2, "\n")
  alignment <- pairwiseAlignment(seq1, seq2)
  score <- pid(alignment)  # Percent Identity
  
  gaps <- nindel(alignment)  # Number of insertion/deletion events
  #print(gaps)
  
  # Handle insertions
  if (nrow(gaps@insertion) > 0) {
    num_insertions <- nrow(gaps@insertion)
    total_insertion_length <- sum(gaps@insertion[,"Length"])
    total_insertion_widthsum <- sum(gaps@insertion[,"WidthSum"])
    avg_insertion_length <- mean(gaps@insertion[,"Length"])
  } else {
    num_insertions <- 0
    total_insertion_length <- 0
    total_insertion_widthsum <- 0
    avg_insertion_length <- 0
  }
  
  # Handle deletions
  if (nrow(gaps@deletion) > 0) {
    num_deletions <- nrow(gaps@deletion)
    total_deletion_length <- sum(gaps@deletion[,"Length"])
    total_deletion_widthsum <- sum(gaps@deletion[,"WidthSum"])
    avg_deletion_length <- mean(gaps@deletion[,"Length"])
  } else {
    num_deletions <- 0
    total_deletion_length <- 0
    total_deletion_widthsum <- 0
    avg_deletion_length <- 0
  }
  
  return(list(
    score = score,
    insertion_info = list(num_insertions = num_insertions, 
                          total_insertion_length = total_insertion_length,
                          total_insertion_widthsum = total_insertion_widthsum,
                          avg_insertion_length = avg_insertion_length),
    deletion_info = list(num_deletions = num_deletions, 
                         total_deletion_length = total_deletion_length,
                         total_deletion_widthsum = total_deletion_widthsum,
                         avg_deletion_length = avg_deletion_length)
  ))
}

# Function to process a specific region and save results
process_region <- function(data, region) {
  region_column <- paste0(region, "_aa")
  
  cleaned_data <- data %>%
    filter(!is.na(.data[[region_column]]), .data[[region_column]] != "")
  
  results <- data.frame(
    sequence_pair = character(),
    true_sequence = character(),
    generated_sequence = character(),
    score = numeric(),
    num_insertions = integer(),
    total_insertion_length = integer(),
    total_insertion_widthsum = integer(),
    avg_insertion_length = numeric(),
    num_deletions = integer(),
    total_deletion_length = integer(),
    total_deletion_widthsum = integer(),
    avg_deletion_length = numeric(),
    stringsAsFactors = FALSE
  )
  
  pb <- txtProgressBar(min = 0, max = floor(nrow(cleaned_data) / 2), style = 3)
  
  for (i in seq(1, nrow(cleaned_data) - 1, by = 2)) {
    seq1 <- cleaned_data[[region_column]][i]
    seq2 <- cleaned_data[[region_column]][i + 1]
    
    similarity_result <- calculate_similarity(seq1, seq2)
    
    results <- rbind(results, data.frame(
      sequence_pair = paste(cleaned_data$sequence_id[i], cleaned_data$sequence_id[i + 1], sep = "_"),
      true_sequence = seq1,
      generated_sequence = seq2,
      score = similarity_result$score,
      num_insertions = similarity_result$insertion_info$num_insertions,
      total_insertion_length = similarity_result$insertion_info$total_insertion_length,
      total_insertion_widthsum = similarity_result$insertion_info$total_insertion_widthsum,
      avg_insertion_length = similarity_result$insertion_info$avg_insertion_length,
      num_deletions = similarity_result$deletion_info$num_deletions,
      total_deletion_length = similarity_result$deletion_info$total_deletion_length,
      total_deletion_widthsum = similarity_result$deletion_info$total_deletion_widthsum,
      avg_deletion_length = similarity_result$deletion_info$avg_deletion_length
    ))
    
    setTxtProgressBar(pb, i / 2)
  }
  
  close(pb)
  
  # Save the detailed results to a CSV file
  write.csv(results, paste0(region, "_detailed_indels.csv"), row.names = FALSE)
  
  # Calculate averages
  avg_insertions <- data.frame(
    avg_num_insertions = mean(results$num_insertions),
    avg_total_insertion_length = mean(results$total_insertion_length),
    avg_total_insertion_widthsum = mean(results$total_insertion_widthsum),
    avg_insertion_length = mean(results$avg_insertion_length)
  )
  
  avg_deletions <- data.frame(
    avg_num_deletions = mean(results$num_deletions),
    avg_total_deletion_length = mean(results$total_deletion_length),
    avg_total_deletion_widthsum = mean(results$total_deletion_widthsum),
    avg_deletion_length = mean(results$avg_deletion_length)
  )
  
  avg_indels <- data.frame(
    avg_num_indels = mean(results$num_insertions + results$num_deletions),
    avg_total_indel_length = mean(results$total_insertion_length + results$total_deletion_length),
    avg_total_indel_widthsum = mean(results$total_insertion_widthsum + results$total_deletion_widthsum)
  )
  
  avg_score <- mean(results$score)
  
  # Combine averages into one data frame
  averages <- cbind(avg_insertions, avg_deletions, avg_indels, avg_score = avg_score)
  
  # Save the averages to a CSV file
  write.csv(averages, paste0(region, "_average_indels.csv"), row.names = FALSE)
}


# Process each region separately
regions <- c("fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4")

# for (region in regions) {
#   process_region(data, region)
# }

small_data <- data[1:10,]
process_region(data, "cdr2")

# Function to calculate similarity and count gaps
check_similarity <- function(seq1, seq2) {
  alignment <- pairwiseAlignment(seq1, seq2)
  score <- pid(alignment)  # Percent Identity
  gaps <- nindel(alignment)  # Number of insertion/deletion events
  pos_indels <- indel(alignment)  # Position of insertion/deletion events
  return(list(score = score, gaps = gaps, pos_indels = pos_indels))
}

# Example usage
ex1 <- "ACGTACGTGTT"
ex2 <- "ACGATACGT"
check_similarity_result <- check_similarity(ex1, ex2)

print(check_similarity_result$score)  # Print similarity score
print(check_similarity_result$gaps)   # Print number of gaps
print(check_similarity_result$pos_indels)   # Print position of gaps

# edit distance: Edit distance, also called Levenshtein distance, 
# is a measure of the number of primary edits that would need to be made to transform one string into another. 
# The R function adist() is used to find the edit distance. https://www.r-bloggers.com/2019/04/natural-language-processing-in-r-edit-distance/
adist(ex1, ex2)




