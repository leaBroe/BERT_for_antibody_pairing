library(Biostrings)
library(DescTools)
library(dplyr)
#install.packages("stringr")
library(stringr)
# Install tidyr package if not already installed
#install.packages("tidyr")
library(tidyr)

# Load CSV file
data <- read.csv("full_test_set_true_gen_seqs_all_relevant_cols.csv")

# Remove the column "d_family" from data
data <- data[, -which(names(data) %in% c("d_family"))]

# Sort the data by the last number in "sequence_id" -> to make sure we get the right pairing
data <- data[order(as.numeric(sub(".*_", "", data$sequence_id))),]

# compare each true and generated sequence by the column name locus
colnames(data)

# Step 2: Initialize a vector to store whether the loci match for each pair
locus_matches <- logical(length = nrow(data) / 2)

# Step 3: Loop over the pairs of rows and compare the locus values
for (i in seq(1, nrow(data), by = 2)) {
  true_locus <- data$locus[i]
  generated_locus <- data$locus[i + 1]
  locus_matches[i / 2] <- (true_locus == generated_locus)
}

# Step 4: Calculate the average of the matches
average_match <- mean(locus_matches)

# Print the result
cat("The average match between true and generated sequences for the 'locus' column is:", average_match, "\n")

# do the same for v_family
# Step 2: Initialize a vector to store whether the v_family values match for each pair
v_family_matches <- logical(length = nrow(data) / 2)

# Step 3: Loop over the pairs of rows and compare the v_family values
for (i in seq(1, nrow(data), by = 2)) {
  true_v_family <- data$v_family[i]
  generated_v_family <- data$v_family[i + 1]
  v_family_matches[i / 2] <- (true_v_family == generated_v_family)
}

# Step 4: Calculate the average of the matches
average_v_family_match <- mean(v_family_matches)

# Print the result
cat("The average match between true and generated sequences for the 'v_family' column is:", average_v_family_match, "\n")

# j_family

# Step 2: Initialize a vector to store whether the j_family values match for each pair
j_family_matches <- logical(length = nrow(data) / 2)

# Step 3: Loop over the pairs of rows and compare the j_family values
for (i in seq(1, nrow(data), by = 2)) {
  true_j_family <- data$j_family[i]
  generated_j_family <- data$j_family[i + 1]
  j_family_matches[i / 2] <- (true_j_family == generated_j_family)
}

# Step 4: Calculate the average of the matches
average_j_family_match <- mean(j_family_matches)

# Print the result
cat("The average match between true and generated sequences for the 'j_family' column is:", average_j_family_match, "\n")





