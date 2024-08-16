# Load necessary libraries
library(dplyr)
library(stringr)

# used model run name: full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1

# Step 1: Read the first CSV file
first_file <- read.csv("Btypes_full_paired_test_data_no_dupl.csv")

# Step 2: Read the second file into a string
second_file_content <- readLines("heavy2light_60_epochs_beam_search_127845.txt")

# Step 3: Split the content into blocks corresponding to each sequence pair
sequence_blocks <- split(second_file_content, cumsum(grepl("^Sequence pair", second_file_content)))

# Step 4: Extract the relevant information from each block using regex
extract_info <- function(block) {
  data.frame(
    sequence_alignment_aa_light = str_extract(paste(block, collapse = " "), "(?<=True Sequence: ).*?(?=Generated Sequence:)") %>% str_trim(),
    generated_sequence_light = str_extract(paste(block, collapse = " "), "(?<=Generated Sequence: ).*?(?=BLOSUM Score:)") %>% str_trim(),
    BLOSUM_score = as.numeric(str_extract(paste(block, collapse = " "), "(?<=BLOSUM Score: )-?\\d+\\.?\\d*")),
    similarity = as.numeric(str_extract(paste(block, collapse = " "), "(?<=Similarity Percentage: )\\d+\\.?\\d*")),
    perplexity = as.numeric(str_extract(paste(block, collapse = " "), "(?<=Perplexity: )\\d+\\.?\\d*"))
  )
}

# Apply the function to each block to create a data frame
extracted_data <- bind_rows(lapply(sequence_blocks, extract_info))
# remove first row as it is empty
extracted_data <- extracted_data[-1,]

# remove the spaces in the sequence_alignment_aa_light column and generated_sequence_light column
extracted_data$sequence_alignment_aa_light <- gsub(" ", "", extracted_data$sequence_alignment_aa_light)
extracted_data$generated_sequence_light <- gsub(" ", "", extracted_data$generated_sequence_light)


# Step 5: Merge with the first file based on 'sequence_alignment_aa_light'
final_data <- first_file %>%
  left_join(extracted_data, by = "sequence_alignment_aa_light")

# remove duplicates in column "sequence_alignment_heavy_sep_light" in final_data
final_data_no_dupl <- final_data[!duplicated(final_data$sequence_alignment_heavy_sep_light),]

# Step 6: Write the final data to a CSV file
#write.csv(final_data_no_dupl, "Btypes_full_paired_test_data_no_dupl_with_predictions.csv", row.names = FALSE)

# print column names of final_data_no_dupl
colnames(final_data_no_dupl)
























