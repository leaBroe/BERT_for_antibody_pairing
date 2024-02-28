import csv

# run this script 3 times

# Path to the original CSV file
# original_csv_path = '/ibmm_data/oas_database/paired/csv_files/with_headers/OAS_db_paired_healthy_naive_b_cells_2024-02-28.csv'
# original_csv_path = '/ibmm_data/oas_database/paired/csv_files/with_headers/OAS_db_paired_healthy_memory_b_cells_2024-02-28.csv'
original_csv_path = '/ibmm_data/oas_database/paired/csv_files/with_headers/OAS_db_paired_healthy_plasma_b_cells_2024-02-28.csv'

# Path to the new CSV file without the header
# new_csv_path = '/ibmm_data/oas_database/paired/csv_files/no_headers/no_header_OAS_db_paired_healthy_naive_b_cells_2024-02-28.csv'
# new_csv_path = '/ibmm_data/oas_database/paired/csv_files/no_headers/no_header_OAS_db_paired_healthy_memory_b_cells_2024-02-28.csv'
new_csv_path = '/ibmm_data/oas_database/paired/csv_files/no_headers/no_header_OAS_db_paired_healthy_plasma_b_cells_2024-02-28.csv'

# Open the original CSV file and create a csv.reader object
with open(original_csv_path, mode='r', newline='') as original_file:
    reader = csv.reader(original_file)
    
    # Open the new CSV file and create a csv.writer object
    with open(new_csv_path, mode='w', newline='') as new_file:
        writer = csv.writer(new_file)
        
        # Skip the header row
        next(reader)
        
        # Write the rest of the rows to the new CSV file
        for row in reader:
            writer.writerow(row)

print(f"CSV file without header has been saved as '{new_csv_path}'.")
