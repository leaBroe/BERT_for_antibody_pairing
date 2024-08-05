import pandas as pd

def extract_columns_from_tsv(tsv_file, columns, output_csv):
    # Read the TSV file into a DataFrame
    df = pd.read_csv(tsv_file, delimiter='\t')
    
    # Check if all the specified columns exist in the DataFrame
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in the TSV file")
    
    # Extract the specified columns
    extracted_df = df[columns]
    
    # Save the extracted columns to a CSV file
    extracted_df.to_csv(output_csv, index=False)
    print(f"Extracted columns saved to {output_csv}")


# column names in sqlite3: cdr1_aa_light,cdr1_end_light,cdr1_light,cdr1_start_light,cdr2_aa_light,cdr2_end_light,cdr2_light,cdr2_start_light,cdr3_aa_light,cdr3_end_light,cdr3_light,cdr3_start_light,fwr1_aa_light,fwr1_end_light,fwr1_light,fwr1_start_light,fwr2_aa_light,fwr2_end_light,fwr2_light,fwr2_start_light,fwr3_aa_light,fwr3_end_light,fwr3_light,fwr3_start_light,fwr4_aa_light,fwr4_end_light,fwr4_light,fwr4_start_light,sequence_alignment_aa_light,sequence_alignment_light,sequence_alignment_heavy_sep_light
# column names PyIR: sequence_id	sequence	sequence_alignment_aa	germline_alignment_aa	fwr1	fwr1_aa	cdr1	cdr1_aa	fwr2	fwr2_aa	cdr2	cdr2_aa	fwr3	fwr3_aa	fwr4	fwr4_aa	cdr3	cdr3_aa		fwr1_start	fwr1_end	cdr1_start	cdr1_end	fwr2_start	fwr2_end	cdr2_start	cdr2_end	fwr3_start	fwr3_end	fwr4_start	fwr4_end	cdr3_start	cdr3_end	np1	np1_length	np2	np2_length	cdr3_aa_length

tsv_file = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/full_test_data/full_paired_data_sequence_alignment_light.tsv' 

columns_to_extract = [
    "sequence_id",
    "sequence",
    "sequence_alignment_aa",
    "germline_alignment_aa",
    "fwr1",
    "fwr1_aa",
    "cdr1",
    "cdr1_aa",
    "fwr2",
    "fwr2_aa",
    "cdr2",
    "cdr2_aa",
    "fwr3",
    "fwr3_aa",
    "fwr4",
    "fwr4_aa",
    "cdr3",
    "cdr3_aa",
    "fwr1_start",
    "fwr1_end",
    "cdr1_start",
    "cdr1_end",
    "fwr2_start",
    "fwr2_end",
    "cdr2_start",
    "cdr2_end",
    "fwr3_start",
    "fwr3_end",
    "fwr4_start",
    "fwr4_end",
    "cdr3_start",
    "cdr3_end",
    "np1",
    "np1_length",
    "np2",
    "np2_length",
    "cdr3_aa_length"
]

# columns_to_extract = [
#     "sequence_id",
#     "sequence",
#     "sequence_alignment_aa",
#     "locus",
#     "v_family"
# ]

output_csv = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/extracted_data_from_PyIR/full_extraction_pyir_full_data.csv' 

extract_columns_from_tsv(tsv_file, columns_to_extract, output_csv)

