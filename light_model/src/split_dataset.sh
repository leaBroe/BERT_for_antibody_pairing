#!/bin/bash

# To use this script, you simply call it from the command line with the path to your dataset file and the desired percentage split for the training set. 
# For example, to split your dataset with an 80% training set, you would run: 
# ./split_dataset.sh path/to/your/dataset.txt 80

# This script does the following:
# Checks for the correct number of arguments and whether the specified file exists.
# Shuffles the dataset to ensure a random distribution of data.
# Calculates the number of lines that should go into the training set based on the provided percentage.
# Splits the shuffled dataset into training and test files.

# this script assumes that each line in your input file corresponds to one data entry (e.g., one sequence).

# !!!!! SHUFFLE THE FILE BEFOREHAND !!!!!
# shuf input.txt > shuffled_sequences.txt

# Check if an input file was provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_file> <split_percentage_for_training>"
    exit 1
fi

INPUT_FILE=$1
SPLIT_PERCENTAGE=$2

# Check if the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File $INPUT_FILE not found."
    exit 1
fi

# Calculate line count for input file
TOTAL_LINES=$(wc -l < "$INPUT_FILE")
TRAINING_LINES=$(echo "($TOTAL_LINES * $SPLIT_PERCENTAGE + 99) / 100" | bc) # +99 to round up with bc

# Split the file
TRAINING_FILE="/ibmm_data2/oas_database/paired_lea_tmp/light_model/data/training_set_light_seq_70_pident_subset.txt"
TEST_FILE="/ibmm_data2/oas_database/paired_lea_tmp/light_model/data/test_set_light_seq_70_pident_subset.txt"

head -n "$TRAINING_LINES" "$INPUT_FILE" > "$TRAINING_FILE"
tail -n +"$((TRAINING_LINES + 1))" "$INPUT_FILE" > "$TEST_FILE"

echo "Dataset split into training and test sets:"
echo "Training set: $TRAINING_FILE (Approx. $SPLIT_PERCENTAGE%)"
echo "Test set: $TEST_FILE (Approx. $((100 - SPLIT_PERCENTAGE))%)"
