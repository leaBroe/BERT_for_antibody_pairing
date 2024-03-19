centroids_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/centroids_ids_cdrl3_aa_70_human_unpaired.txt'  
data_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light_cdr3_light_seq_3_rowid.txt'  
output_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/matched_rows_cdrl3_unpaired_python_4.txt'

# centroids_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/centroid_test.txt'  
# data_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/test_data.txt'  
# output_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/test_output_extraction.txt'

# Reading line numbers to extract, storing them in a list
with open(centroids_file_path, 'r') as f:
    line_numbers_to_extract = [int(line.strip()) for line in f]

#print(line_numbers_to_extract[:10])

# Extracting the specified lines
with open(data_file_path, 'r') as data_file, open(output_file_path, 'w') as output_file:
    for current_line_number, line in enumerate(data_file, 1):
        if current_line_number in line_numbers_to_extract:
            output_file.write(line)

print(f"Extraction complete. Check {output_file_path} for the extracted lines.")


