# 26/02/2024
Notes and backups: [https://github.com/leaBroe/master_thesis](https://github.com/leaBroe/master_thesis)  
Google Docs: [https://docs.google.com/document/d/1fy5TRSL0hYPEu7SyuHWBLyyPqRID307pasIa7cvoT1c/edit](https://docs.google.com/document/d/1fy5TRSL0hYPEu7SyuHWBLyyPqRID307pasIa7cvoT1c/edit)
1. Create Conda environment:  
   export conda environment:  
   [https://stackoverflow.com/questions/39280638/how-to-share-conda-environments-across-platforms](https://stackoverflow.com/questions/39280638/how-to-share-conda-environments-across-platforms)
   [https://arrpitk.medium.com/create-share-python-virtual-environment-using-conda-6dd9112a34f6](https://arrpitk.medium.com/create-share-python-virtual-environment-using-conda-6dd9112a34f6)
```bash
conda env list
conda activate OAS_paired_env
```

To export your Conda environment to a **`.yml`** file using the **`--from-history`** option (which exports only the packages that you have explicitly asked for, not their dependencies):

```bash
conda env export --from-history > environment.yml
```

   Conda Cheat Sheet:  
   [https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
   

# 27/02/2024
1. download the `bulk_download.sh` files from https://opig.stats.ox.ac.uk/webapps/oas/oas_paired. with **human** and **unsorted b-type** filters.  
2. generate the `OAS_db_paired_healthy_2024-02-27.csv` file with the data in the desidered format by running the following command:
`./src/oas_download.sh bulk_download.sh paired_healthy`

or  
```bash
cd /ibmm_data/oas_database/src
./oas_download.sh /ibmm_data/oas_database/paired/bulk_download.sh paired_healthy
# move the csv file to paired folder
```
the scripts (oas_download.sh) are from the repo: [https://github.com/ibmm-unibe-ch/OAS_database](https://github.com/ibmm-unibe-ch/OAS_database)
3. Add `OAS_db_paired_healthy_2024-02-27.csv` to `OAS.db`:  
```bash
sqlite> .open /ibmm_data/oas_database/OAS.db
sqlite> .mode csv
sqlite> .import ./OAS_db_paired_healthy_2024-02-27.csv healthy_paired
sqlite> .tables
healthy_paired     heavy_healthy_iga  heavy_hiv_iga      heavy_hiv_igg    
```

# 28/02/2024
1. EDA docs
   mmmseq2 clustering
   [https://github.com/Ch-rode/snippets/blob/main/mmseqs_clustering(huge_file).sh](https://github.com/Ch-rode/snippets/blob/main/mmseqs_clustering(huge_file).sh)  
   [https://github.com/Ch-rode/snippets/blob/main/mmseqs.md](https://github.com/Ch-rode/snippets/blob/main/mmseqs.md)  
   Documentation mmseq2  
   [https://github.com/soedinglab/MMseqs2](https://github.com/soedinglab/MMseqs2)  
   User guide  
   [https://mmseqs.com/latest/userguide.pdf](https://mmseqs.com/latest/userguide.pdf)  
   Paper  
   [https://www.nature.com/articles/nbt.3988](https://www.nature.com/articles/nbt.3988)

2. Add more B Types to the dataset  
   Download Naive, plasma, memory (and unsorted but already in database) separately and add the csv files to the healthy_paired table in OAS.db

**Naive-B Cells:**

**Your search yielded 161,327 filtered sequences from 1 studies**

**Memory-B-Cells:**

**Your search yielded 207,539 filtered sequences from 2 studies.**

**Plasma-B-Cells:**

**Your search yielded 41,614 filtered sequences from 1 studies.**

## Naive B-Cells:

get /ibmm_data/oas_database/paired/csv_files/OAS_db_paired_healthy_naive_b_cells_2024-02-28.csv file:

```sql
cd /ibmm_data/oas_database/src
```

```sql
./oas_download.sh /ibmm_data/oas_database/paired/bulk_downloads/bulk_download_naive_B_cells.sh paired_healthy_naive_b_cells
```

```sql
# move the csv file to paired folder
```

## Memory B-Cells:

```sql
cd /ibmm_data/oas_database/src
```

```sql
./oas_download.sh /ibmm_data/oas_database/paired/bulk_downloads/bulk_download_memory_B_cells.sh paired_healthy_memory_b_cells
```

```sql
# move the csv file to paired folder
```

## Plasma B-Cells:

```sql
cd /ibmm_data/oas_database/src
```

```sql
./oas_download.sh /ibmm_data/oas_database/paired/bulk_downloads/bulk_download_plasma_B_cells.sh paired_healthy_plasma_b_cells
```

```sql
# move the csv file to paired folder
```

Remove the header of the csv files (naive, memory and plasma) with /ibmm_data/oas_database/paired/remove_header_from_csv.py  

(run the script 3 times)  

Add the 3 csv files (without the header!) to the healthy_paired table in OAS.db  

use for this: /ibmm_data/oas_database/paired/add_csv_to_database_table.py and run the script 3 times    

3. Do EDA Pre-Clustering with all 4 B Cell Types  
   For this, see the file `/ibmm_data/oas_database/paired/OAS_db_EDA_paired.ipynb` or [https://github.com/leaBroe/master_thesis/blob/master/OAS_db_EDA_paired.ipynb](https://github.com/leaBroe/master_thesis/blob/master/OAS_db_EDA_paired.ipynb)

# 29/02/2024

Filter out duplicates in EDA → /ibmm_data/oas_database/paired/OAS_db_EDA_paired.ipynb or [https://github.com/leaBroe/master_thesis/blob/master/OAS_db_EDA_paired.ipynb](https://github.com/leaBroe/master_thesis/blob/master/OAS_db_EDA_paired.ipynb)  

**Whole healthy_paired table:**  

Found 25286 duplicates based on sequence_alignment_aa_heavy and sequence_alignment_aa_light  

**Naive B Cells:**  

Found 335 duplicates based on sequence_alignment_aa_heavy and sequence_alignment_aa_light for naive B cells  

**Memory Cells:**  

Found 13668 duplicates based on sequence_alignment_aa_heavy and sequence_alignment_aa_light for memory B cells  

See /ibmm_data/oas_database/paired/OAS_db_EDA_paired.ipynb or [https://github.com/leaBroe/master_thesis/blob/master/OAS_db_EDA_paired.ipynb](https://github.com/leaBroe/master_thesis/blob/master/OAS_db_EDA_paired.ipynb) for the sqlite3 commands  

1. Add full sequence alignment (to check for duplicates / verify the duplicates) 
2. export sequence_id_heavy, sequence_alignment_aa_heavy, sequence_alignment_aa_light, sequence_alignment_aa_full to .txt file  

```sql
sqlite> .open /ibmm_data/oas_database/paired/oas_database/backup_oas_db/OAS.db
sqlite> .tables
healthy_paired     heavy_healthy_igg  heavy_hiv_igg    
heavy_healthy_iga  heavy_hiv_iga    
sqlite> ALTER TABLE healthy_paired ADD COLUMN sequence_alignment_aa_full TEXT;
sqlite> UPDATE healthy_paired
   ...> SET sequence_alignment_aa_full = sequence_alignment_aa_heavy || sequence_alignment_aa_light;
sqlite> .mode csv
sqlite> .output /ibmm_data/oas_database/paired/txt_files_oas_db/exported_data_full_seq.txt
sqlite> SELECT sequence_id_heavy, sequence_alignment_aa_heavy, sequence_alignment_aa_light, sequence_alignment_aa_full FROM healthy_paired;
sqlite> .output stdout
sqlite> .quit
```

# 07/03/2024

Still waiting for the data  

Clustering: first CDH3 sequence only, then full sequence  

Convert to fasta format  

```bash
awk '{ printf ">%s\n%s\n",$1,$2 }' /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/testfile.txt > /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/outputfile.txt
```

```bash
awk -F, '{print ">" $1 "\n" $NF}' /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/exported_data_full_seq_naive.txt > /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/test2_output.fasta
```

# 12/03/2024

New database: 

```bash
/ibmm_data2/oas_database/OAS.db
```

new tables: Bcells_subset_human_unpaired_heavy  all_human_paired  

1. EDA both paired and unpaired (paired are already in the database)  
2. cluster them (step 1 cdrh3, step 2 full sequence (if paired light+heavy, if unpaired only the light sequence) as they did in the paper  
3. EDA after clustering to check what data we have as the final one  
4. if data are ok train the models: 1. NSP with paired data 2. Bert LIght  

## 1. EDA all_human_paired

Pre Clustering of all_human_paired (with all duplicates)  

used column: cdr3_aa → CDRH3 AA sequence  

SELECT cdr3_aa FROM healthy_paired  

```bash
.open /ibmm_data2/oas_database/OAS.db
```

### Using **`tmux`**:  

1. Log in to your remote cluster.
2. Start a new **`tmux`** session by typing **`tmux`** and pressing Enter.
3. Run your SQLite command to extract the columns. For example:
    
    ```bash
    sqlite3 your_database.db "SELECT column1, column2 FROM your_table;" > output.txt
    ```
    
4. To detach from the **`tmux`** session, press **`Ctrl+b`** then **`d`**.
5. You can now safely log out, and the process will continue to run.
6. To reattach to the **`tmux`** session later, log back in and type **`tmux attach`**.

   

```sql
sqlite> ALTER TABLE all_human_paired ADD COLUMN sequence_id_heavy_light TEXT;
sqlite> UPDATE all_human_paired
		...> SET sequence_id_heavy_light = sequence_id_heavy || sequence_id_light;
sqlite> .mode csv
sqlite> .output /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/all_human_paired_cdr3_aa.txt
sqlite> SELECT sequence_id_heavy_light, cdr3_aa_heavy FROM all_human_paired;
sqlite> .output stdout
sqlite> .quit
```

```sql
sqlite> .open /ibmm_data2/oas_database/OAS.db
sqlite> .table
Bcells_subset_human_unpaired_heavy  all_human_paired                  
sqlite> ALTER TABLE all_human_paired ADD COLUMN sequence_alignment_aa_full TEXT;  
sqlite> UPDATE all_human_paired  
   ...> SET sequence_alignment_aa_full = sequence_alignment_aa_heavy || sequence_alignment_aa_light;  
sqlite> .mode csv
sqlite> .output /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/all_human_paired_full_aa_seq.txt
sqlite> SELECT sequence_id_heavy_light, sequence_alignment_aa_heavy, sequence_alignment_aa_light, sequence_alignment_aa_full FROM all_human_paired;
sqlite> .output stdout
sqlite> .quit
```

Remove duplicates:

```bash
awk -F ',' '!seen[$4]++' /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/all_human_paired_full_aa_seq.txt > /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/all_human_paired_full_aa_seq_no_duplicates.txt
```

Create CDRH3 fasta file

```bash
awk -F, '{print ">" $1 "\n" $NF}' /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/all_human_paired_cdr3_aa.txt > /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/all_human_paired_cdr3_aa.fasta
```

Create full AA sequence fasta file (all_human_paired)

```bash
awk -F, '{print ">" $1 "\n" $NF}' /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/all_human_paired_full_aa_seq_no_duplicates.txt > /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/all_human_paired_full_aa_seq_no_duplicates.fasta
```

# 13/03/2024

### **Running the** mmseqs_linclust_2.sh **Script**

1. **Navigate to the Script's Directory**: Use the **`cd`** command to change to the directory containing your script.
2. **Execute the Script**: Run the script by passing the name of your FASTA file (without the **`.fasta`** extension) as an argument. For example, if your FASTA file is named **`sample.fasta`**, execute:
    
    ```bash
    ./mmseqs_linclust_2.sh sample 
    ```
    
    1. Clustering of CDRH3 with default settings → all_human_paired_EDA.ipynb
    2. results in /ibmm_data2/oas_database/paired_lea_tmp/linclust_mmseq2/linclust_2 → kmer-per-seq=5

# 14/03/2024

Extract the centroid IDs

```sql
awk '/^>/{print substr($0, 2)}' /ibmm_data2/oas_database/paired_lea_tmp/linclust_mmseq2/all_human_paired_cdr3_aa_100_clu_rep.fasta > /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/centroids_ids_cdr3_aa_100.txt
```

filtered sequences: 1’736’057

no filtering: 1’794’739

Check for duplicates in the centroids .txt file:

```sql
awk '{if (++count[$1] > 1) print $1, "count:", count[$1]}' /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/centroids_ids_cdr3_aa_100.txt
```

## Create sqlite3 Index:

```sql
.open /ibmm_data2/oas_database/OAS.db
```

Create index on column cdr3_aa_heavy

```sql
CREATE INDEX IF NOT EXISTS idx_cdr3_aa_heavy ON all_human_paired(cdr3_aa_heavy);
```

## Extract the CDRH3 sequences from the centroids file

```sql
awk '!/^>/' /ibmm_data2/oas_database/paired_lea_tmp/linclust_mmseq2/all_human_paired_cdr3_aa_100_clu_rep.fasta > /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/centroids_seqs_cdr3_aa_100.txt
```

sqlite3 query:

```sql
sqlite> SELECT sequence_id_heavy_light, sequence_alignment_aa_heavy, sequence_alignment_aa_light, sequence_alignment_aa_full, cdr3_aa_heavy FROM all_human_paired WHERE cdr3_aa_heavy IN ({});

```

```sql
#!/bin/bash

CENTROIDS_TXT="/path/to/your/centroids.txt"
SQLITE_DB="/path/to/your/database.sqlite"
OUTPUT_FILE="/path/to/your/output_sequences.txt"

# Prepare the output file
echo "" > "$OUTPUT_FILE"

# Read each line in the centroids text file
while IFS= read -r cdrh3_sequence
do
    # Query the SQLite database for matching full sequences
    sqlite3 "$SQLITE_DB" "SELECT sequence_id_heavy_light, sequence_alignment_aa_heavy, sequence_alignment_aa_light, sequence_alignment_aa_full, cdr3_aa_heavy FROM all_human_paired WHERE cdr3_aa_heavy = '$cdrh3_sequence';" >> "$OUTPUT_FILE"
done < "$CENTROIDS_TXT"

echo "Extraction complete. Full sequences saved to $OUTPUT_FILE."

```

→ /ibmm_data2/oas_database/paired_lea_tmp/filter_seqs_with_sqlite.sh

# 15/03/2024

### EDA Bcells_subset_human_unpaired_light in /ibmm_data2/oas_database/OAS_2.db

```sql
.open /ibmm_data2/oas_database/OAS_2.db
```

```sql
sqlite> .mode csv
sqlite> .output /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light_cdr3_light_seq.txt
sqlite> SELECT rowid, cdr3_aa, sequence_alignment_aa FROM Bcells_subset_human_unpaired_light;
sqlite> .output stdout
sqlite> .quit
```

Used the file: → /ibmm_data2/oas_database/paired_lea_tmp/export_csv.sh

1. **Start a tmux session:** You can start a new tmux session by typing **`tmux`** into your terminal. You can also create a new named session using **`tmux new -s session_name`**.

```bash
tmux new -s export_session
```

1. **Run the script:** Once inside the tmux session, navigate to the directory where your script is located, then run the script just as you would in a regular terminal.

```bash
./export_csv.sh
```

1. **Detach from the tmux session:** Once the script is running (or even before, if you're confident it will run without needing immediate input), you can detach from the tmux session and let it run in the background. To detach, press **`Ctrl+b`** then **`d`**.
2. **Re-attach to the tmux session:** If you want to check on the progress of your script, you can re-attach to the tmux session at any time. Use the command **`tmux attach -t session_name`** to re-attach to your session.

```bash
tmux attach -t export_session
```

1. **Close the tmux session:** Once your script has finished running and you don't need the session anymore, you can exit tmux by typing **`exit`** or pressing **`Ctrl+d`**. If you have detached from the session, you can kill it with **`tmux kill-session -t session_name`**.

```bash
tmux kill-session -t export_session
```

Using tmux is particularly handy for tasks that might take a long time to complete or for tasks that you want to ensure keep running even if you disconnect from the server or close your terminal.

Create CDRL3 fasta file for Bcells_subset_human_unpaired_light

```sql
awk -F, '{print ">" $1 "\n" $2}' /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light_cdr3_light_seq_3_rowid.txt > /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light/cdr3_light_seq_3_rowid.fasta

```

Create light aa sequence fasta file for Bcells_subset_human_unpaired_light

```sql
awk -F, '{print ">" $1 "\n" $NF}' /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light_cdr3_light_seq_3_rowid.txt > /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light/full_light_seq_3_rowid.fasta

```

Do clustering of CDRL3 in /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light/linclust_cdrl3

EDA of Bcells_subset_human_unpaired_light -> /ibmm_data2/oas_database/paired_lea_tmp/Bcells_subset_human_unpaired_light_EDA.ipynb

```sql
tmux new -s linclust
```

Remade clustering (1. 99 PIDENT and 2. All PIDENT) with parameters: 

```bash
-k 6 --spaced-kmer-mode 1 --spaced-kmer-pattern 11011101 --sub-mat VTML40.out --gap-open 16 --gap-extend 2
```

in folder:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light/linclust_100_only
/ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light/linclust_new_params
```

# 19/03/24

```bash
**wc -l /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light_cdr3_light_seq_3_rowid.txt**
```

267’911’345 extracted sequences 

We go with 70 PIDENT

```bash
/ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light/linclust_new_params/cdr3_light_seq_3_rowid_70_clu_rep.fasta
```

```bash
/ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light_cdr3_light_seq_3_rowid.txt
```

extract IDs from centroids:

```sql
awk '/^>/{print substr($0, 2)}' /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light/linclust_new_params/cdr3_light_seq_3_rowid_70_clu_rep.fasta > /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/centroids_ids_cdrl3_aa_70_human_unpaired.txt
```

```bash
grep -Fwf /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/centroids_ids_cdrl3_aa_70_human_unpaired.txt /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light_cdr3_light_seq_3_rowid.txt > /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/matched_rows_cdrl3_unpaired.txt
```

```bash
awk 'NR==FNR{a[$0];next} FNR in a' /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/centroids_ids_cdrl3_aa_70_human_unpaired.txt /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light_cdr3_light_seq_3_rowid.txt > /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/matched_rows_cdrl3_unpaired_awk.txt
```

```bash
awk -v linesfile=/ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/centroids_ids_cdrl3_aa_70_human_unpaired.txt -f extract.awk /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light_cdr3_light_seq_3_rowid.txt > /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/matched_rows_cdrl3_unpaired_awk_2.txt
```

try with csv file:

ibmm_data2/oas_database/unpaired/Light/OAS_db_Bcells_subset_human_unpaired_light_20240307.csv

SELECT cdr3_aa, sequence_alignment_aa where the row number is in the centroid_id.txt file

tmux new -s extract_session

# 20/03/24

Try to split up the large file and do it locally:

```
scp ssh [leab@130.92.121.14](mailto:leab@130.92.121.14):/ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/centroids_ids_cdrl3_aa_70_human_unpaired.txt /Users/leabroennimann/Desktop/Master_Bioinformatik/master_thesis_24_LB/data

```

```
scp ssh [leab@130.92.121.14](mailto:leab@130.92.121.14):/ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light_cdr3_light_seq_3_rowid.txt /Users/leabroennimann/Downloads/master_thesis_datay

```

```
scp ssh [leab@130.92.121.14](mailto:leab@130.92.121.14):/ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light_cdr3_light_seq_3_rowid.txt /Users/leabroennimann/Downloads/master_thesis_datay

```

large file has 267’911’345 lines

centroids file has 16’771’428 lines → same as the EDA plot

```bash
split -b 300M /Users/leabroennimann/Downloads/master_thesis_data/Bcells_subset_human_unpaired_light_cdr3_light_seq_3_rowid.txt /Users/leabroennimann/Downloads/master_thesis_data/smaller_files/file_chunk_
```

# 21-25/03/2024

```bash
split -l 2000000 /Users/leabroennimann/Downloads/master_thesis_data/Bcells_subset_human_unpaired_light_cdr3_light_seq_3_rowid.txt /Users/leabroennimann/Downloads/master_thesis_data/smaller_files/file_chunk_
```

use extract.awk: 

```bash
awk -v linesfile=/Users/leabroennimann/Downloads/master_thesis_data/centroids_ids_cdrl3_aa_70_human_unpaired.txt -f extract.awk /Users/leabroennimann/Downloads/master_thesis_data/smaller_files/file_chunk_aa > /Users/leabroennimann/Downloads/master_thesis_data/output_files/matched_rows_cdrl3_unpaired.txt
```

extracted lines from chunk_aa: 25’455 → correct amount of sequences extracted

```bash
sed -n '1,1999905p' /Users/leabroennimann/Downloads/master_thesis_data/output_files/matched_rows_cdrl3_unpaired.txt > /Users/leabroennimann/Downloads/master_thesis_data/output_files/outputfile_1
```

```bash
awk -v linesfile=/Users/leabroennimann/Downloads/master_thesis_data/centroids_ids_cdrl3_aa_70_human_unpaired.txt -f extract_2.awk /Users/leabroennimann/Downloads/master_thesis_data/smaller_files/file_chunk_aa > /Users/leabroennimann/Downloads/master_thesis_data/output_files/matched_rows_cdrl3_unpaired_2.txt
```

```bash
awk -f extract.awk datafile.txt
```

```bash
awk -v linesfile=/Users/leabroennimann/Downloads/master_thesis_data/centroids_ids_cdrl3_aa_70_human_unpaired.txt -f extract.awk /Users/leabroennimann/Downloads/master_thesis_data/Bcells_subset_human_unpaired_light_cdr3_light_seq_3_rowid.txt > /Users/leabroennimann/Downloads/master_thesis_data/output_files/matched_rows_cdrl3_unpaired_3.txt
```

```bash
awk 'NR==FNR { row_numbers[$1] = 1; next } $1 in row_numbers' /Users/leabroennimann/Downloads/master_thesis_data/centroids_ids_cdrl3_aa_70_human_unpaired.txt /Users/leabroennimann/Downloads/master_thesis_data/smaller_files/file_chunk_aa > /Users/leabroennimann/Downloads/master_thesis_data/output_files/matched_rows_cdrl3_unpaired_5.txt
```

```bash
grep -wf /Users/leabroennimann/Downloads/master_thesis_data/centroids_ids_cdrl3_aa_70_human_unpaired.txt /Users/leabroennimann/Downloads/master_thesis_data/smaller_files/file_chunk_aa | awk 'BEGIN {FS=","} {if ($1 in arr) print $0;}' > /Users/leabroennimann/Downloads/master_thesis_data/output_files/matched_rows_cdrl3_unpaired_5.txt
```

```perl
awk 'NR==FNR{tgts[$1]; next} $1 in tgts' /Users/leabroennimann/Downloads/master_thesis_data/centroids_ids_cdrl3_aa_70_human_unpaired.txt /Users/leabroennimann/Downloads/master_thesis_data/smaller_files/file_chunk_aa > /Users/leabroennimann/Downloads/master_thesis_data/output_files/matched_rows_cdrl3_unpaired_6.txt

```

```bash
join /Users/leabroennimann/Downloads/master_thesis_data/centroids_ids_cdrl3_aa_70_human_unpaired.txt /Users/leabroennimann/Downloads/master_thesis_data/smaller_files/file_chunk_aa.txt > /Users/leabroennimann/Downloads/master_thesis_data/output_files/matched_rows_cdrl3_unpaired_6.txt
```

```bash
./match_key.pl /Users/leabroennimann/Downloads/master_thesis_data/centroids_ids_cdrl3_aa_70_human_unpaired.txt /Users/leabroennimann/Downloads/master_thesis_data/smaller_files/file_chunk_aa.txt > /Users/leabroennimann/Downloads/master_thesis_data/output_files/matched_rows_cdrl3_unpaired_6.txt
```

```bash
awk -F, '{print "ID_" $1 "," $2 "," $3}' /Users/leabroennimann/Downloads/master_thesis_data/smaller_files/file_chunk_aa.txt > /Users/leabroennimann/Downloads/master_thesis_data/file_chunk_aa_ids.txt
```

```bash
awk -F, '{print "ID_" $1}' /Users/leabroennimann/Downloads/master_thesis_data/centroid_IDs_strings.txt > /Users/leabroennimann/Downloads/master_thesis_data/centroid_IDs_strings_2.txt
```

```bash
sort -n /Users/leabroennimann/Downloads/master_thesis_data/test_2_full.txt > /Users/leabroennimann/Downloads/master_thesis_data/test_2_full_sorted.txt
```

final output in → /Users/leabroennimann/Downloads/master_thesis_data/output_julia_6.txt

create fasta file from extracted light sequences (after first clustering):

```bash
awk -F, '{print ">" $1 "\n" $NF}'  /Users/leabroennimann/Downloads/master_thesis_data/output_julia_6.txt > /Users/leabroennimann/Downloads/master_thesis_data/extracted_sequences_from_70_pident_cdrl3.fasta

```

```bash
tmux new -s linclust_step_2
```

Remove duplicates

```bash
awk -F ',' '!seen[$3]++' /Users/leabroennimann/Downloads/master_thesis_data/output_julia_6.txt > /Users/leabroennimann/Downloads/master_thesis_data/output_julia_6_no_duplicates.txt
```

14’354’251 sequences left after duplicate removal of light sequence

made new fasta file:

```bash
awk -F, '{print ">" $1 "\n" $NF}'  /Users/leabroennimann/Downloads/master_thesis_data/output_julia_6_no_duplicates.txt > /Users/leabroennimann/Downloads/master_thesis_data/extracted_sequences_from_70_pident_cdrl3_no_dupl.fasta
```

ount singletons:

```bash
awk '{count[$1]++} END {singletons = 0; for (id in count) {if (count[id] == 1) {singletons++}} print singletons}' /ibmm_data2/oas_database/paired_lea_tmp/linclust_second_step/linclust_2nd_step_no_dupl/extracted_sequences_from_70_pident_cdrl3_no_dupl_70_clu.tsv
```

singletons: 4’811’366

```bash
awk '{count[$1]++} END {singletons = 0; for (id in count) {if (count[id] == 1) {singletons++}} print singletons}' /ibmm_data2/oas_database/paired_lea_tmp/linclust_second_step/linclust_2nd_step_no_dupl/extracted_sequences_from_70_pident_cdrl3_no_dupl_80_clu.tsv
```

singletons: 7’146’428

```bash
awk '{count[$1]++} END {singletons = 0; for (id in count) {if (count[id] == 1) {singletons++}} print singletons}' /ibmm_data2/oas_database/paired_lea_tmp/linclust_second_step/linclust_2nd_step_no_dupl/extracted_sequences_from_70_pident_cdrl3_no_dupl_90_clu.tsv
```

singletons: 9'749'565

```bash
awk '{count[$1]++} END {singletons = 0; for (id in count) {if (count[id] == 1) {singletons++}} print singletons}' /ibmm_data2/oas_database/paired_lea_tmp/linclust_second_step/linclust_2nd_step_no_dupl/extracted_sequences_from_70_pident_cdrl3_no_dupl_100_clu.tsv
```

with duplicates: 

```bash
awk '{count[$1]++} END {singletons = 0; for (id in count) {if (count[id] == 1) {singletons++}} print singletons}' /ibmm_data2/oas_database/paired_lea_tmp/linclust_second_step/extracted_sequences_from_70_pident_cdrl3_70_clu.tsv
```

singletons: 4’436’959

```bash
awk '{count[$1]++} END {singletons = 0; for (id in count) {if (count[id] == 1) {singletons++}} print singletons}' /ibmm_data2/oas_database/paired_lea_tmp/linclust_second_step/extracted_sequences_from_70_pident_cdrl3_80_clu.tsv
```

singletons: 6’801’044

```bash
awk '{count[$1]++} END {singletons = 0; for (id in count) {if (count[id] == 1) {singletons++}} print singletons}' /ibmm_data2/oas_database/paired_lea_tmp/linclust_second_step/extracted_sequences_from_70_pident_cdrl3_90_clu.tsv
```

singletons: 9’248’912

```bash
awk '{count[$1]++} END {singletons = 0; for (id in count) {if (count[id] == 1) {singletons++}} print singletons}' /ibmm_data2/oas_database/paired_lea_tmp/linclust_second_step/extracted_sequences_from_70_pident_cdrl3_100_clu.tsv
```

singletons: 11’143’238

# 25/03/24

## Model Training (light BERT MLM self-supervised)

### **1. Data Preparation**

First, ensure your sequences are in a clean, consistent format. Here's an example:

```python
QVQLVQSGAEVKKPGASVKVSCKASGYTFTDYYI...
EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYA...
...

```

100 PIDENT: 

```bash
/ibmm_data2/oas_database/paired_lea_tmp/linclust_second_step/linclust_2nd_step_no_dupl/extracted_sequences_from_70_pident_cdrl3_no_dupl_100_clu_rep.fasta
```

turn it into required format for BERT model:

```bash
awk '/^>/ {if (seq) print seq; seq=""; next} {seq=seq$0} END {if (seq) print seq}' /ibmm_data2/oas_database/paired_lea_tmp/linclust_second_step/linclust_2nd_step_no_dupl/extracted_sequences_from_70_pident_cdrl3_no_dupl_100_clu_rep.fasta > /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/light_sequences_100_pident.txt
```

tmux attach -t training

# 26/03/24

check if any NVIDIA GPUs are available on your system:

```
lspci | grep -i nvidia
```

This command lists all PCI devices and filters the output for "nvidia". If you see output lines listing NVIDIA devices, your system has NVIDIA GPUs.

```bash
01:00.0 3D controller: NVIDIA Corporation GA100 [A100 PCIe 80GB] (rev a1)
23:00.0 3D controller: NVIDIA Corporation GA100 [A100 PCIe 80GB] (rev a1)
41:00.0 3D controller: NVIDIA Corporation Device 2331 (rev a1)
61:00.0 3D controller: NVIDIA Corporation GA100 [A100 PCIe 80GB] (rev a1)
81:00.0 3D controller: NVIDIA Corporation Device 2331 (rev a1)
a1:00.0 3D controller: NVIDIA Corporation GA100 [A100 PCIe 80GB] (rev a1)
c1:00.0 3D controller: NVIDIA Corporation GA100 [A100 PCIe 80GB] (rev a1)
e1:00.0 3D controller: NVIDIA Corporation GA100 [A100 PCIe 80GB] (rev a1)
```

```bash
nvidia-smi
```

Create new environment for pytorch

```bash
conda create --name pytorch python=3.9
```

```bash
conda activate pytorch
```

```bash
conda create --name pytorch_env python=3.9
-> requirements.txt
```

```bash
conda activate pytorch_cuda_env
```

used OAS_paired_env after all!

# 28/03/24

70 PIDENT:

```bash
awk '/^>/ {if (seq) print seq; seq=""; next} {seq=seq$0} END {if (seq) print seq}' /ibmm_data2/oas_database/paired_lea_tmp/linclust_second_step/linclust_2nd_step_no_dupl/extracted_sequences_from_70_pident_cdrl3_no_dupl_70_clu_rep.fasta > /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/light_sequences_70_pident.txt
```

```bash
shuf /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/light_sequences_70_pident.txt > /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/light_sequences_70_pident_shuffled.txt
```

```bash
head -n 1000000 /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/light_sequences_70_pident_shuffled.txt > /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/light_sequences_70_pident_shuffled_subset.txt 
```

extract rowid and cdrh3 in csv and fasta format (for the clustering)

```bash
sqlite3 -header -csv /ibmm_data2/oas_database/OAS_heavy.db "SELECT ROWID,cdr3_aa FROM Bcells_subset_human_unpaired_heavy WHERE Isotype IN ('IGHG', 'IGHA');" | tee >(awk -F, '{print ">"$1"\n"$2}' > /ibmm_data/rodelc/DALM/HEAVY/CDRH3/LM/data/1_cdrh3.fasta) > /ibmm_data/rodelc/DALM/HEAVY/CDRH3/LM/data/1_cdrh3.csv
```

# 29/03/24

run light model (train.model.py) done on 25.03.24 with 70 PIDENT (appr. 4.4 million sequences):

Using cuda device for training.
Test Loss: 0.513
Test Accuracy: 0.874

## Heavy model data prep:

```bash
wc -l /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/heavy_unpaired_cdrh3.fasta
```

1’266’194’968 lines in cdrh3 fasta file of human unpaired heavy → 633’097’484 sequences

```bash
sqlite3 -header -csv /ibmm_data2/oas_database/OAS_heavy.db "SELECT ROWID,cdr3_aa,sequence_alignment_aa FROM Bcells_subset_human_unpaired_heavy;" | awk -F, '{print ">"$1"\n"$2}' > /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/heavy_unpaired_cdrh3.fasta
```

```bash
tail -n +3 /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/heavy_unpaired_cdrh3.fasta > /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/heavy_unpaired_cdrh3_no_header.fasta
```

# 30/03/24

/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/mmseqs/1.cdrh3/heavy_unpaired_cdrh3_no_header_70_clu_rep.fasta

use 70 pident for 2. clustering of heavy seq unpaired (as in the light model unpaired):

70 pident cdrh3: 115’826’685 sequences

extract IDs from centroids:

```bash
awk '/^>/{print substr($0, 2)}' /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/mmseqs/1.cdrh3/heavy_unpaired_cdrh3_no_header_70_clu_rep.fasta > /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/centroids_ids_cdrh3_aa_70_heavy_unpaired.txt
```

```bash
sort -n /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/centroids_ids_cdrh3_aa_70_heavy_unpaired.txt > /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/centroids_ids_cdrh3_aa_70_heavy_unpaired_sorted.txt
```

```bash
wc -l /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/centroids_ids_cdrh3_aa_70_heavy_unpaired_sorted.txt
```

number of sequence IDs: 115’826’685 IDs

## 2. step clustering

use match_ids.jl

→ /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/filtered_heavy_seqs_from_cdrh3.txt

remove duplicates in full heavy seq:

```bash
awk -F ',' '!seen[$3]++' /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/filtered_heavy_seqs_from_cdrh3.txt > /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/filtered_heavy_seqs_from_cdrh3_no_duplicates.txt
```

Create fasta file for 2. step clustering of full heavy seq unpaired without dupl:

```bash
awk -F, '{print ">" $1 "\n" $NF}'  /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/filtered_heavy_seqs_from_cdrh3_no_duplicates.txt > /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/filtered_heavy_seqs_from_cdrh3_no_duplicates.fasta
```

→ /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/remove_dupl_create_fasta.sh  

```bash
wc -l /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/filtered_heavy_seqs_from_cdrh3_no_duplicates.fasta
```

114’282’257 sequences (no dupl.)  


# 31/03/24

run transformers library (run_mlm.py)

```bash
conda create --name transformers_mlm python=3.9
```

```bash
conda activate transformers_mlm
```

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers/examples/pytorch/language-modeling/
```

```bash
#pip install transformers
pip install git+https://github.com/huggingface/transformers
```

```bash
pip install -r requirements.txt
```

To run on your own training and validation files, use the following command:

Test if code works (with small test datasets of 200 sequences):

```bash
python run_mlm.py \
    --model_name_or_path FacebookAI/roberta-base \
    --train_file /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/training_set_test.txt \
    --validation_file /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/test_set_test.txt \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --line_by_line \
    --overwrite_output_dir \
    --output_dir /ibmm_data2/oas_database/paired_lea_tmp/light_model/outputs/tmp/test-mlm
```

→ /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/run_train_mlm_from_transformers.sh

## Server wiki:

https://github.com/ibmm-unibe-ch/Wiki/wiki/Server

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html 
```

# 02/04/24

## Data Prep Paired model (new db):

/ibmm_data2/oas_database/paired_lea_tmp/paired_model/src/create_fasta_csv_paired_model.sh

ALTER TABLE all_human_paired ADD COLUMN sequence_alignment_aa_full TEXT;

UPDATE all_human_paired  
   ...> SET sequence_alignment_aa_full = sequence_alignment_aa_heavy || sequence_alignment_aa_light; 

mmseqs clustering in folder: /ibmm_data2/oas_database/paired_lea_tmp/paired_model/mmseqs

## try to run run_mlm.py from https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling:

```bash
conda create --name transformers_mlm_2 python=3.9
```

```bash
conda activate transformers_mlm_2
```

```bash
cd transformers/examples/pytorch/language-modeling/
```

```bash
pip install git+https://github.com/huggingface/transformers
```

```bash
pip install -r requirements.txt
```

# 03/04/24

install the following in transformers_mlm_2 env:

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Attempting uninstall: torch
Found existing installation: torch 2.2.2
Uninstalling torch-2.2.2:
Successfully uninstalled torch-2.2.2
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
accelerate 0.28.0 requires torch>=1.10.0, but you have torch 1.9.0+cu111 which is incompatible.

```bash
pip install -q accelerate==0.5.0
```

ImportError: Using the `Trainer` with `PyTorch` requires `accelerate>=0.21.0`: Please run `pip install transformers[torch]` or `pip install accelerate -U`

```bash
pip install -q accelerate==0.21.0
```

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
torchaudio 0.9.0 requires torch==1.9.0, but you have torch 2.2.2 which is incompatible.
torchvision 0.10.0+cu111 requires torch==1.9.0, but you have torch 2.2.2 which is incompatible.

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
accelerate 0.21.0 requires torch>=1.10.0, but you have torch 1.9.0+cu111 which is incompatible.

[Github Issue run_mlm.py](https://www.notion.so/Github-Issue-run_mlm-py-615fa2aa9bc54bcf8d428f4d984418c4?pvs=21)

# new env with requirements.txt (from chiara)

```bash
conda create --name req_ch python=3.9
```

```bash
conda activate req_ch
```

```bash
pip install -r requirements.txt
```

```bash
transformers-cli env 
```

- `transformers` version: 4.35.2
- Platform: Linux-4.18.0-372.26.1.el8_6.x86_64-x86_64-with-glibc2.28
- Python version: 3.9.19
- Huggingface_hub version: 0.20.3
- Safetensors version: 0.4.2
- Accelerate version: 0.26.1
- Accelerate config: not found
- PyTorch version (GPU?): 1.10.2+cu102 (True)
- Tensorflow version (GPU?): not installed (NA)
- Flax version (CPU?/GPU?/TPU?): not installed (NA)
- Jax version: not installed
- JaxLib version: not installed
- Using GPU in script?: <fill in>
- Using distributed or parallel set-up in script?: <fill in>

```
pip install evaluate

```

ImportError: cannot import name 'is_torch_xla_available' from 'transformers' (/home/leab/anaconda3/envs/req_ch/lib/python3.9/site-packages/transformers/**init**.py)

solution: transformers version 4.40.0.dev0

https://github.com/huggingface/transformers/issues/29749

```bash
pip install -q accelerate==0.28.0
```

RuntimeError: CUDA error: no kernel image is available for execution on the device
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.

```bash
conda create --name req_ch_py38 python=3.8.5
```

```bash
pip install -r requirements.txt
```

# 04/04/24

GPU support and run_mlm.py works now with: /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch

```bash
gpustat
```
## Paired model:

1. step clustering full seq
    
    ```bash
    /ibmm_data2/oas_database/paired_lea_tmp/paired_model/mmseqs/1.cdrh3/paired_rowid_cdrh3_100_clu_rep.fasta
    ```
    
    get centroid IDs from 100 PIDENT cdrh3:
    
    ```bash
    awk '/^>/{print substr($0, 2)}' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/mmseqs/1.cdrh3/paired_rowid_cdrh3_100_clu_rep.fasta > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/data_oas/centroids_ids_cdrh3_aa_100_paired.txt
    ```
    
    use /ibmm_data2/oas_database/paired_lea_tmp/match_ids.jl for sequence extraction
    
    remove duplicates in full seq (heavy and light):
    
    ```bash
    awk -F ',' '!seen[$NF]++' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/data_oas/filtered_full_seqs_from_cdrh3_paired.txt > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/data_oas/filtered_full_seqs_from_cdrh3_paired_no_dupl.txt
    ```
    
    Create fasta file for 2. step clustering of full heavy seq unpaired without dupl:
    
    ```bash
    awk -F, '{print ">" $1 "\n" $NF}'  /ibmm_data2/oas_database/paired_lea_tmp/paired_model/data_oas/filtered_full_seqs_from_cdrh3_paired_no_dupl.txt > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/data_oas/filtered_full_seqs_from_cdrh3_paired_no_dupl.fasta
    ```

## Light model: New Clustering approach:

Use 100 pident cdr3 centroids, retrieve the corresponding light sequences, cluster them 70 pident, keep the centroids, cluster the centroids at 50 pident and keep all (so again the previous 70 pident centroids but with some kind of grouping)

### 100 PIDENT CDRl3 (light sequences unpaired model) are already in:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light/linclust_new_params/100_pident/cdr3_light_seq_3_rowid_100_clu_rep.fasta
```

### Cluster light sequences 70 PIDENT based on centroids of cdrl3:

extract the sequences from:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light_cdr3_light_seq_3_rowid.txt
```

get centroid IDs from 100 PIDENT cdrl3:

```bash
awk '/^>/{print substr($0, 2)}' /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light/linclust_new_params/100_pident/cdr3_light_seq_3_rowid_100_clu_rep.fasta > /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/clustering_data/centroid_ids_cdrl3_aa_100_unpaired_light.txt
```

```bash
 wc -l /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/clustering_data/centroid_ids_cdrl3_aa_100_unpaired_light.txt
```

→ 124’659’077 IDs

use /ibmm_data2/oas_database/paired_lea_tmp/match_ids.jl for sequence extraction

remove duplicates in light seq:

```bash
awk -F ',' '!seen[$NF]++' /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/clustering_data/filtered_light_seqs_from_cdrl3_100_pident_unpaired.txt > /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/clustering_data/filtered_light_seqs_from_cdrl3_100_pident_unpaired_no_dupl.txt
```

light sequences left (no duplicates): 97’756’584

Create fasta file for 2. step clustering of full light seq unpaired without dupl:

```bash
awk -F, '{print ">" $1 "\n" $NF}'  /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/clustering_data/filtered_light_seqs_from_cdrl3_100_pident_unpaired_no_dupl.txt > /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/clustering_data/filtered_light_seqs_from_cdrl3_100_pident_unpaired_no_dupl.fasta
```

Then cluster the light sequences 70 PIDENT: 

/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/linclust_70_pident.sh

Do clustering with 50 PIDENT

in folder: /ibmm_data2/oas_database/paired_lea_tmp/light_model/linclust/linclust_50_pident

# 05/04/24

Command used for /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/train_test_val_split.py:

```bash
python train_test_val_split.py --tsv_dataset /ibmm_data2/oas_database/paired_lea_tmp/light_model/linclust/linclust_50_pident/filtered_light_seqs_from_cdrl3_100_pident_unpaired_no_dupl_70_clu_rep_50_clu.tsv --rep_fasta_file /ibmm_data2/oas_database/paired_lea_tmp/light_model/linclust/linclust_50_pident/filtered_light_seqs_from_cdrl3_100_pident_unpaired_no_dupl_70_clu_rep_50_clu_rep.fasta --prefix light_seqs
```

# 08/04/24

Did 50 PIDENt clustering with createseqfiledb and result2flat → /ibmm_data2/oas_database/paired_lea_tmp/light_model/linclust/linclust_50_pident_all_seqs/linclust_50_pident.sh

```bash
python train_test_val_split.py --tsv_dataset /ibmm_data2/oas_database/paired_lea_tmp/light_model/linclust/linclust_50_pident/filtered_light_seqs_from_cdrl3_100_pident_unpaired_no_dupl_70_clu_rep_50_clu.tsv --rep_fasta_file /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/clustering_data/filtered_light_seqs_from_cdrl3_100_pident_unpaired_no_dupl.fasta --prefix light_all_seqs
```

# 10/04/24

remove identifier IDs (for BERT model training)

```bash
awk '/^>/ {if (seq) print seq; seq=""; next} {seq=seq$0} END {if (seq) print seq}' /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/light_all_seqs_train.txt > /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/light_all_seqs_train_no_ids.txt
```

```bash
awk '/^>/ {if (seq) print seq; seq=""; next} {seq=seq$0} END {if (seq) print seq}' /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/light_all_seqs_val.txt > /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/light_all_seqs_val_no_ids.txt
```

```bash
awk '/^>/ {if (seq) print seq; seq=""; next} {seq=seq$0} END {if (seq) print seq}' /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/light_all_seqs_test.txt > /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/light_all_seqs_test_no_ids.txt
```

run light model unpaired MLM

# 11/04/2024

## Heavy Model unpaired MLM

### Data preparation

same clustering approach as for the light ones unpaired: Cluster CDRH3, use 100 pident for 2. clustering of heavy sequence, then cluster them 70 pident and again 50 pident

cdrh3 100 pident already in:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/mmseqs/1.cdrh3/heavy_unpaired_cdrh3_no_header_100_clu_rep.fasta
```

### Cluster heavy sequences 70 PIDENT based on centroids of cdrh3:

extract the sequences from:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/heavy_unpaired_cdrh3_full_heavy.csv
```

get centroid IDs from 100 PIDENT cdrh3:

```bash
awk '/^>/{print substr($0, 2)}' /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/mmseqs/1.cdrh3/heavy_unpaired_cdrh3_no_header_100_clu_rep.fasta > /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/heavy_unpaired_cdrh3_100_pident_centroid_ids.txt
```

```bash
 wc -l /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/heavy_unpaired_cdrh3_100_pident_centroid_ids.txt
```

→ 136’246’321 IDs

use /ibmm_data2/oas_database/paired_lea_tmp/match_ids.jl for sequence extraction →  `julia> include("path/to/script-name.jl")`

check file: 136’246’321 sequences extracted

remove duplicates in heavy seq:

```bash
awk -F ',' '!seen[$NF]++' /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/filtered_heavy_seqs_from_cdrh3_100_pident_unpaired.txt > /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/filtered_heavy_seqs_from_cdrh3_100_pident_unpaired_no_dupl.txt
```

heavy sequences left (no duplicates): 131’190’503

Create fasta file for 2. step clustering of full light seq unpaired without dupl:

```bash
awk -F, '{print ">" $1 "\n" $NF}'  /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/filtered_heavy_seqs_from_cdrh3_100_pident_unpaired_no_dupl.txt > /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/filtered_heavy_seqs_from_cdrh3_100_pident_unpaired_no_dupl.fasta
```

Then cluster the light sequences 70 PIDENT:

in folder: /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/mmseqs/2.heavy_seq_100_pident_cdrh3

/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/mmseqs/2.heavy_seq_100_pident_cdrh3/filtered_heavy_seqs_from_cdrh3_100_pident_unpaired_no_dupl_70_clu_rep.fasta → 123’867’780 heavy sequences

cluster the 70 pident heavy seqs again with 50 pident

# 12/04/24

```bash
python train_test_val_split.py --tsv_dataset /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/mmseqs/3.heavy_seq_50_pident/filtered_heavy_seqs_from_cdrh3_100_pident_unpaired_no_dupl_70_clu_rep_50_clu.tsv --rep_fasta_file /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/mmseqs/2.heavy_seq_100_pident_cdrh3/filtered_heavy_seqs_from_cdrh3_100_pident_unpaired_no_dupl.fasta --prefix heavy_all_seqs
```

remove IDs for BERT training:

```bash
awk '/^>/ {if (seq) print seq; seq=""; next} {seq=seq$0} END {if (seq) print seq}' /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/train_test_val_datasets/heavy_all_seqs_train.txt > /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/train_test_val_datasets/heavy_all_seqs_train_no_ids.txt
```

```bash
awk '/^>/ {if (seq) print seq; seq=""; next} {seq=seq$0} END {if (seq) print seq}' /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/train_test_val_datasets/heavy_all_seqs_test.txt > /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/train_test_val_datasets/heavy_all_seqs_test_no_ids.txt
```

```bash
awk '/^>/ {if (seq) print seq; seq=""; next} {seq=seq$0} END {if (seq) print seq}' /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/train_test_val_datasets/heavy_all_seqs_val.txt > /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/train_test_val_datasets/heavy_all_seqs_val_no_ids.txt
```

# 18/04/24

## Paired model clustering

```bash
python train_test_val_split.py --tsv_dataset /ibmm_data2/oas_database/paired_lea_tmp/paired_model/mmseqs/3.full_sequence_50_pident/filtered_full_seqs_from_cdrh3_paired_no_dupl_70_clu_rep_50_clu.tsv --rep_fasta_file /ibmm_data2/oas_database/paired_lea_tmp/paired_model/mmseqs/2.full_sequence_70_pident/filtered_full_seqs_from_cdrh3_paired_no_dupl_100_clu_rep.fasta --prefix paired_full_seqs
```

```bash
awk '/^>/ {if (seq) print seq; seq=""; next} {seq=seq$0} END {if (seq) print seq}' paired_full_seqs_val.txt > paired_full_seqs_val_no_ids.txt
```

## Paired model: Only MLM

```bash
sqlite> UPDATE all_human_paired
...> SET sequence_alignment_heavy_sep_light = sequence_alignment_aa_heavy || '[SEP]' || sequence_alignment_aa_light;
sqlite>
```

Clustering with new full seq (with SEP token)

cdrh3 100 pident clustering already in 

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/mmseqs/1.cdrh3/paired_rowid_cdrh3_100_clu_rep.fasta
```

get centroid IDs from 100 PIDENT cdrh3:

```bash
awk '/^>/{print substr($0, 2)}' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/mmseqs/1.cdrh3/paired_rowid_cdrh3_100_clu_rep.fasta > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/data_oas/paired_rowid_cdrh3_100_clu_rep_centroid_ids.txt
```

use /ibmm_data2/oas_database/paired_lea_tmp/match_ids.jl for sequence extraction →  `julia> include("path/to/script-name.jl")`

remove duplicates in full seq:

```bash
awk -F ',' '!seen[$NF]++' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/data_oas/filtered_full_seqs_from_cdrh3_100_pident_paired.txt > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/data_oas/filtered_full_seqs_sep_from_cdrh3_100_pident_paired_no_dupl.txt
```

create fasta file:

```bash
awk -F, '{print ">" $1 "\n" $NF}'  /ibmm_data2/oas_database/paired_lea_tmp/paired_model/data_oas/filtered_full_seqs_sep_from_cdrh3_100_pident_paired_no_dupl.txt > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/data_oas/filtered_full_seqs_sep_from_cdrh3_100_pident_paired_no_dupl.fasta
```

cluster full seq with [SEP] token 70 PIDENT

```bash
python train_test_val_split.py --tsv_dataset /ibmm_data2/oas_database/paired_lea_tmp/paired_model/mmseqs/3.1full_seq_50/filtered_full_seqs_sep_from_cdrh3_100_pident_paired_no_dupl_70_clu_rep_50_clu.tsv --rep_fasta_file /ibmm_data2/oas_database/paired_lea_tmp/paired_model/mmseqs/2.1full_seq_sep_70/filtered_full_seqs_sep_from_cdrh3_100_pident_paired_no_dupl.fasta --prefix paired_full_seqs_sep
```

train/val/test datasets with [SEP] token are in

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq
```

```bash
awk '/^>/ {if (seq) print seq; seq=""; next} {seq=seq$0} END {if (seq) print seq}' paired_full_seqs_sep_train.txt > paired_full_seqs_sep_train_no_ids.txt
```

# 24/04/24

turn txt file into right format for BertForPreTraining:

```bash
awk -F'\\[SEP\\]' '{print $1 "\n" $2 "\n"}' paired_full_seqs_sep_val_no_ids.txt > paired_full_seqs_val_for_nsp.txt
```

# 25/04/24

## Fine-tune protBERT as a NSP, with HEAVY <SEP> LIGHT as input

https://huggingface.co/Rostlab/prot_bert_bfd

https://aws.amazon.com/de/blogs/machine-learning/fine-tune-and-deploy-the-protbert-model-for-protein-classification-using-amazon-sagemaker/

https://www.kaggle.com/code/cdeotte/prot-bert-finetune-lb-0-30

# 03/05/24


## output of own script for nsp/mlm paired model:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/logs/paired_model_nsp_mlm/paired_mlm_nsp_full_23540.o
```

specifications used for the databases:

in particular:

- Unsorted B-cells from PBMC, no vaccine, no disease
- PLASMA-B-CELLS, no vaccine, no disease
- MEMORY-B-CELLS, all

visualize attention in NLP models:

https://github.com/jessevig/bertviz#model-view

Update 7. May 2024:

IgBERT classification task with random data (0, unpaired) and paired data (1)

adapter, bert2bert

light model different learning rates 

paired model: 1900000 sequences everything


# 07/05/24

## Paired Model Classification task:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/light_heavy_classification.py
```

classification task with the following input structure:

```bash
heavy,light,label
GLEWIAYIYFSGSTNYNPSLKSRVTLSVDTSKNQFSLKLSSVTAADSAVYYCARDVGPYNSISPGRYYFDYWGPGTLVTVSS,QSALTQPASVSGSPGQSITISCTGTSSDVGNYNLVSWYQHHPGKAPKLMIYEVSKRPSGISNRFSGSKSGNTASLTISGLQADDEADYYCCSYAGSRILYVFGSGTKVTVL,1
QLQLQESGPGLVKPSETLSLTCTVSGGSISSSSYYWGWIRQPPGKGLEWIGNFFYSGSTNYNPSLKSRATISLDTSKNELSLKLSSVTAADTAVYYCASNTLMAEATFDYWGQGTLVTVSS,SYEVTQAPSVSVSPGQTASVTCSGDKLDKKYTSWYQQRPGQSPTVVIYQNNKRPSGIPERFSASKSGNTATLTISGTQAVDEADYYCQAWDDSDGVFGPGTTVTVL,0
QLQLQESGPGLVKPSETLSLTCTVSGGSISSSSYYWGWIRQPPGKGLEWIGNFFYSGSTNYNPSLKSRATISLDTSKNELSLKLSSVTAADTAVYYCASNTLMAEATFDYWGQGTLVTVSS,SYEVTQAPSVSVSPGQTASVTCSGDKLDKKYTSWYQQRPGQSPTVVIYQNNKRPSGIPERFSASKSGNTATLTISGTQAVDEADYYCQAWDDSDGVFGPGTTVTVL,0
GLEWIAYIYFSGSTNYNPSLKSRVTLSVDTSKNQFSLKLSSVTAADSAVYYCARDVGPYNSISPGRYYFDYWGPGTLVTVSS,QSALTQPASVSGSPGQSITISCTGTSSDVGNYNLVSWYQHHPGKAPKLMIYEVSKRPSGISNRFSGSKSGNTASLTISGLQADDEADYYCCSYAGSRILYVFGSGTKVTVL,1
```

training data is already available at: 

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_train_no_ids.txt
```

but with the structure heavy[SEP]light:

```bash
QVQLQESGPGLVKPSETLSLTCTVSGGSISGFYWSWIRQSPGKGLEWIAYIYFSGSTNYNPSLKSRVTLSVDTSKNQFSLKLSSVTAADSAVYYCARDVGPYNSISPGRYYFDYWGPGTLVTVSS[SEP]QSALTQPASVSGSPGQSITISCTGTSSDVGNYNLVSWYQHHPGKAPKLMIYEVSKRPSGISNRFSGSKSGNTASLTISGLQADDEADYYCCSYAGSRILYVFGSGTKVTVL
QVQLQESGPGLVKPSETLSLTCTVSGGSISSYHWSWIRQPPGKGLEWIGYMYYSGSTNYNPSLKSRVTISVDTSKTQFSLKLSSVTTADTAVYYCARGRLIWSADYTGGDYFDPWGQGILVTVSS[SEP]QSALTQPASVSGSPGQSITISCTGSSSDVGSYNLVSWYQQHPGKAPKLMIYEVSKRPSGVSNRFSGSKSGNTASLTISGLQAEDEAQYYCCSYGGRNFHVLFGGGTELTVL
QVVLVQSGAVVEKPGASVKVSCKPSGYSFTQHYIHWVRQAPGQGLEWLGLVNPSGGSTSYAQKFQGRVTMTRDTATSTVYMELSGLTSDDTATYYCTRVSADDGSGYFRYYFEHWGQGTLVTVSS[SEP]DIHMTQSPSTLSASVGERVTISCRASETVNTWVAWYQQKPGEVPKVLIHTASILGTGVPSRFSGSGSGTEFTLTISNLQPSDLATYYCQQYHAYPITFGGGTKV
QVQLVESGGGVVQPGKSLRLSCEASGFTFSYYGIHWVRQTPGKGLEWVAVIWSDGDDSSYADSVKGRFTISRDNSKNTVWLQMNSLRAEDTAVYYCAKDPDAKNSHSHYLDFWGQGTLVTVSS[SEP]YDLTQPPSVSVSPGQTATITCSGERLGEKYVCWYQQKSGQSPVLVIYEDRKRPSGIPERFSGSNSGNTATLTISETQAIDEADYYCKTWDSSGVFGVFGGGTKLTVL
QVQLQESGPRLVKPSETLSLTCAVSGGSISSKNWWSWLRQSPEKGLEWIGEVYETGTANHNPSLTRRLALSVDKSRNQFHLNLSSVTAADTGVYFCARGIVDRRPLYFDNWGQGILVTVSS[SEP]DIQVTQSPSSLSASVGDRVTITCRASQNINTNLNWYQQKAGRAPKVLIHGASTLQSGVPVRFSGSGSGTEFTLTINNMEPEDVATYYCQQSHNSRTFGQGTRVEMK
```

idea: take the real paired data with label 1, add unpaired synthesised data with label 0 (mix the heavy and light sequences randomly), but dont mix the different sequences between the training, validation and test sets!

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/create_and_synthesize_dataset_for_classification.py
```

this script takes a dataset with heavy[SEP]light as input, splits the data on the [SEP] token, adds a 1 as label (paired) and generates random pairings of heavy and light chains and adds a 0 for unpaired. In this way, the sequences are not shared between the training, validation and test files (run the script for each training, test and val file separately). 

## Next steps:

- igBert --> classification task adding syntetic data with the target 0 (maybe try also prot-berf-bfd)
- bert2bert --> encoder model model with cross attention and adapters
- light --> try learning rate because performances are soso
- heavy --> wait until epoch 10
- paired one --> wait until results and play with learning rate

config1_esperberto.json  43.219.225
config2_medium_model.json
config3_roberta.json 85.746.457
config4_smaller_model
config5_antiberty.json 25.549.337

environment for classification taks from transformers:

```bash
conda activate class_env
```

commands used for class_env:

```bash
conda create --name class_env python=3.9
pip install datasets
pip install evaluate
pip install transformers
pip install git+https://github.com/huggingface/transformers
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install accelerate -U
pip install scikit-learn
pip install wandb
```

```bash
config_3_run_5e-4_20epochs
```

# 16/05/2014

light model: appr. 500 epochs → see plateau in val loss

print sequence after tokenization → roberta run_mlm.py → see if its 512 or 20

run heavy model → high priority with new max seq length 512, → terminal, or use same config file 

max_seq_length 512

max_position_embeddings 514

in config: max_length 512 → see if its work

## Ran heavy model config 3 (roberta full) with

in directory: 

```
    --output_dir ./FULL_config_3_roberta_run_lr5e-5_500epochs_max_seq_length_512
```

Output heavy model:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_3_roberta_run_lr5e-5_500epochs_max_seq_length_512-23936.out
```


```bash
             23936      test hea_conf     leab  R       3:28      1 dalcosrv
```

heavy model number: 23936

## Ran light model config 3 (roberta full) with

in directory:

```bash
    --output_dir ./FULL_config_3_roberta_run_lr5e-4_500epochs_max_seq_length_512
```

output file: 

```bash
/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_3_roberta_run_lr5e-4_500epochs_max_seq_length_512-23938.out
```

light model number: 23938

             23938      test light_co     leab  R       1:28      1 dalcosrv

## light model config 5 (antiberty)

             23941      test  li_cf_5     leab  R       0:01      1 dalcosrv

## light model config 4 (smaller model)

             23942      test  li_cf_4     leab  R       1:28      1 dalcosrv

# 21/05/24

## NSP ProtBERT model

# 22/05/24

Light model: Config 3 and config 4 different learning rate since loss and accuracy look weird. 

New learning rate: 5e-5

Light model config 3 with new learning rate 5e-5:

             24037      test  li_cf_3     leab  R       0:01      1 dalcosrv

remove spaces in last column of csv file (after using preprocess_input_data.py)

```bash
awk 'BEGIN { FS=OFS="," } { gsub(/ /, "", $NF); print }' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_val_with_unpaired_small_space_separated.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_val_with_unpaired_small_space_separated_rm.csv
```

# 27/05/24

## light model data summary

### Light model:

Before clustering
267‘911‘345 CDRL3/light sequences (/ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light_cdr3_light_seq_3_rowid.txt)

After cdrl3 100 pident: 124‘659‘077 sequences (no duplicates removed in CDRL3)

Remove duplicates in light sequence: 97’756’584 sequences

After full light seq 70 pident: 28‘278‘253 sequences

# IgBERT2IgBERT

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/src/IgBERT2IgBERT.py
```

## error:

OSError: Can't load weights for 'Exscientia/IgBert'. Make sure that:

- 'Exscientia/IgBert' is a correct model identifier listed on 'https://huggingface.co/models'
- or 'Exscientia/IgBert' is the correct path to a directory containing a file named one of pytorch_model.bin, tf_model.h5, model.ckpt.

/Anaconda3/envs/envName/Lib/site-packages/transformers/

```bash
/home/leab/anaconda3/envs/bert2bert_env/lib/python3.9/site-packages/transformers/configuration_utils.py
```

https://discuss.huggingface.co/t/cant-load-weights-for-facebook-bart-base/10367

If you are trying to run your code on a server or HPC system and you receive this error, one solution is to first run the step that downloads the model outside of the batch system and then use the downloaded model within the batch system. This solution resolved my error which was exactly the same e…

### Step 1: download model locally:

```python
from transformers import EncoderDecoderModel, BertTokenizerFast

# Download the tokenizer and model locally
tokenizer = BertTokenizerFast.from_pretrained("Exscientia/IgBert")
model = EncoderDecoderModel.from_encoder_decoder_pretrained("Exscientia/IgBert", "Exscientia/IgBert")

# Save the tokenizer and model locally
tokenizer.save_pretrained("./Exscientia_IgBert_tokenizer")
model.save_pretrained("./Exscientia_IgBert_model")
```

### **Step 2: Transfer the Files to Your HPC or Server**

Transfer the downloaded model and tokenizer files to your HPC or server system. You can use e.g. **`scp`**, **`rsync`**


config1_esperberto.json  43.219.225
config2_medium_model.json
config3_roberta.json 85.746.457
config4_smaller_model
config5_antiberty.json 25.549.337

OAS db:

**Light**: PBMC/Unsorted B cells/no disease/no vaccine/light: **271,922,318 sequences**

Plasma-B-Cells**/**no disease/no vaccine/light**: 0 sequences**

Memory-B-Cells**/**no disease/no vaccine/light**: 1,061,974 sequences**

Total light sequences from OAS db: 272‘984‘292 sequences

**Effectively in our** **dataset: 267‘911‘345 sequences**

**Heavy:** PBMC/Unsorted B cells/no disease/no vaccine/heavy: **624,783,056 sequences**

Plasma-B-Cells**/**no disease/no vaccine/heavy**:**  **7,847,069 sequences**

Memory-B-Cells**/**no disease/no vaccine/heavy**: 2,765,629 sequences**

**Total heavy sequences in OAS db: 635‘395‘754 sequences**

**Effectively in our** **dataset: 633‘097‘484** sequences

heavy and light

paired everything

why: unsorted: didnt know where the b cells came from, just extracted them, very general, biggest group, no vaccine no disease bc we want un unbiased model (not biased to disease or vaccine)

plasma:

paired: bc we dont have that many data

## Data Clustering Summary

**Light model:**

Before clustering

267‘911‘345 CDRL3/light sequences (/ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light_cdr3_light_seq_3_rowid.txt)

After cdrl3 100 pident: 124‘659‘077 sequences (no duplicates removed in CDRL3)

- > Reduction by round 53%

Remove duplicates in light sequence: 97’756’584 sequences -> reduction by 21.6% (100 - 78.4)

After full light seq 70 pident: 28‘278‘253 sequences ->reduction by  71 % (100-29%)

**Heavy Model:**

Before clustering: 633‘097‘484 sequences

After cdrh3 100 pident: 136‘246‘321 sequences -> 21.5 reduction by 78.5%

Remove duplicates in heavy sequence: 131’190’503 sequences -> 96.3 reduction by 4%

After full heavy seq 70 pident: 123’867’780 heavy sequences -> 94%, reduction by 6%

**Paired** **model**

Before clustering: 1‘946‘101 sequences

After 100%identity CDRH3 1‘611‘242 sequences -> 82.79333909

After duplication removal: 1‘609‘886 -> 82.72366131

After 70 pident clustering: 672‘078 sequences -> 34.53458993

1. **1. Input Representation**
- **Embedding Layer**: Converts each word in the input sentence into a dense vector that captures its semantic meaning
- Example: Words "Thank" and "you" or in our case we would have individual amino acids, are converted into 300-dimensional vectors.
- 
1. **Encoder**
- **Recurrent** **Neural Network (RNN)**: Processes the sequence of embedding vectors one at a time. In this case, an LSTM (Long Short-Term Memory) network is used.
- **Internal State**: Maintains information from the input sequence.
- **Context Vector**: The final state of the LSTM representing the entire input sentence.
1. **Decoder**
- **LSTM**: Generates the translated sentence using the context vector from the encoder as its initial state.
- **Embedding Layer**: Represents output words in Spanish.
- **Start-ofSequence (SOS) Token**: Initial input to the decoder's embedding layer to start generating the output sequence.
- 
1. **Fully Connected Layer and Softmax**
- **Fully Connected Layer**: Transforms the BERT output at each step into a vector matching the size of the vocabulary of the light model (e.g., 6000-dimensional).
- 
- **Softmax** **Function**: Converts this vector into a probability distribution, identifying the most likely next word -> with BERT i think you have GELU as activation function
- 
1. **Sequence Generation**
- The decoder generates the output sequence word-by-word. Each predicted word is fed back into the model until an end-of-sequence (EOS) token is produced, signaling completion.
- 
1. **Training**

Teacher Forcing: When training an encoder-decoder model instead of using the predicted token as input to the decoder we use the known correct token to prevent error accumulation and improve learning.

- 
1. **Inference**
- During inference, the model uses its own predictions as inputs for generating subsequent words, as the target words are unknown.

Activation function: Gaussian error linear unit (GELU) activation operation weights the input by its probability under a Gaussian distribution.

**Softmax** **converts** **numbers** **into** **probabilistic** **distribution** **where** **each** **of** **the** **outputs** **represents a probability** **corresponding** **to a class**.

sequence-to-sequence tutorial:

[https://github.com/bentrevett/pytorch-seq2seq/blob/rewrite/1 - Sequence to Sequence Learning with Neural Networks.ipynb](https://github.com/bentrevett/pytorch-seq2seq/blob/rewrite/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb)

[https://medium.com/@abhinavbhartigoml/unraveling-transformers-a-deep-dive-into-self-attention-and-3e37dc875bea#:~:text=Cross-Attention Mechanism%3A-,Specifically present in the decoder%2C the cross-attention mechanism enables,element in the output sequence](https://medium.com/@abhinavbhartigoml/unraveling-transformers-a-deep-dive-into-self-attention-and-3e37dc875bea#:~:text=Cross%2DAttention%20Mechanism%3A-,Specifically%20present%20in%20the%20decoder%2C%20the%20cross%2Dattention%20mechanism%20enables,element%20in%20the%20output%20sequence)

bert2bert.config explanations:

https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/configuration#transformers.PretrainedConfig

# 07/06/24

rerun heavy model config 3 from checkpoint, stopped → not sure if it really trained from checkpoints

out file: /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/slurm-24262.out

rerun light smaller model (config 4) from checkpoint, seems to work

out file: /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/slurm-24265.out

stopped the run bc I just wanted to see if it worked → config 4 light model was already finished

rerun from checkpoint conifg 3 light model

/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/slurm-24266.out

# 10/06/24

Summary meeting 6. June 2024:

- **Light Smallest size**
    - Epoch 40 is the best, but training it for a few more epochs.
- **Light RoBERTa size**
    - Still training.
- **Heavy Small Model**
    - Almost done.
    - Check the evaluation loss for more epochs before stopping it.
- **NSP Loss**
    - Eval loss looks weird.
    - Double-check how the evaluation loss is computed (use prints if necessary).
- **Seq2Seq Model**
    - Working with IgBert without adapters
    - Try to insert adapters and cross attention between encoder and decoder.
    - Print the model to check if the adapters are included.
    - Remember to set adapters to training and active.
- **Seq2SeqAdapterTrainer**
    - Check if it still exists.
- **Model Saving with the new adapter library**
    - Check how the model is saved: as one unique model or model plus adapters.
    - Check if the config are saved after being updated manually in the code
    
    ## Adapters:
    
    Github:
    
    https://github.com/adapter-hub/adapters
    
    Jupyter notebook:
    
    https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/01_Adapter_Training.ipynb#scrollTo=saGtTGogLPVf
    
    Docs:
    
    https://docs.adapterhub.ml/training.html


- **Model Saving with the new adapter library**
    - Check how the model is saved: as one unique model or model plus adapters.
    - Check if the config are saved after being updated manually in the code
    
    ## Adapters:
    
    Github:
    
    https://github.com/adapter-hub/adapters
    
    Jupyter notebook:
    
    https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/01_Adapter_Training.ipynb#scrollTo=saGtTGogLPVf
    
    Docs:
    
    https://docs.adapterhub.ml/training.html
    
    # 11/06/2024
    
    Rerun models from checkpoints:
    
    ```
         JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
             24266      test  li_cf_3     leab  R 3-19:36:03      1 dalcosrv
             34535      test hea_conf     leab  R       8:30      1 dalcosrv
             34536      test hea_conf     leab  R       0:01      1 dalcosrv
    
    ```
    
    heavy model config 3: 34535
    
    heavy model config 4: 34536
    
    light model config 3: 24266 → Rerun light model roBERTa config 3 from epoch 17: /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/slurm-34556.out
    
    ### light model config 3:
    
    06/07/2024 15:18:07 - INFO - **main** - Training new model from scratch
    Loading cached processed dataset at /home/leab/.cache/huggingface/datasets/text/default-2feabe941ecf72f7/0.0.0/c4a140d10f020282918b5dd1b8a49f0104729c6177f60a6b49ec2a365ec69f34/cache-d7c8a94b8efb3c4c.arrow
    06/07/2024 15:18:08 - INFO - datasets.arrow_dataset - Loading cached processed dataset at /home/leab/.cache/huggingface/datasets/text/default-2feabe941ecf72f7/0.0.0/c4a140d10f020282918b5dd1b8a49f0104729c6177f60a6b49ec2a365ec69f34/cache-d7c8a94b8efb3c4c.arrow
    Loading cached processed dataset at /home/leab/.cache/huggingface/datasets/text/default-2feabe941ecf72f7/0.0.0/c4a140d10f020282918b5dd1b8a49f0104729c6177f60a6b49ec2a365ec69f34/cache-24f21d6723d9da9e.arrow
    06/07/2024 15:18:08 - INFO - datasets.arrow_dataset - Loading cached processed dataset at /home/leab/.cache/huggingface/datasets/text/default-2feabe941ecf72f7/0.0.0/c4a140d10f020282918b5dd1b8a49f0104729c6177f60a6b49ec2a365ec69f34/cache-24f21d6723d9da9e.arrow
    06/07/2024 15:18:09 - WARNING - accelerate.utils.other - Detected kernel version 4.18.0, which is below the recommended minimum of 5.5.0; this can c
    
    ### Heavy Model config 3 roBERTa training from epoch 2:
    
    Loading cached processed dataset at /home/leab/.cache/huggingface/datasets/text/default-c20d4731be61e1a0/0.0.0/c4a140d10f020282918b5dd1b8a49f0104729c6177f60a6b49ec2a365ec69f34/cache-bc61843e26ee880e.arrow
    06/11/2024 10:46:00 - INFO - datasets.arrow_dataset - Loading cached processed dataset at /home/leab/.cache/huggingface/datasets/text/default-c20d4731be61e1a0/0.0.0/c4a140d10f020282918b5dd1b8a49f0104729c6177f60a6b49ec2a365ec69f34/cache-bc61843e26ee880e.arrow
    
    Running tokenizer on dataset line_by_line:   0%|          | 0/12386778 [00:00<?, ? examples/s]Caching processed dataset at /home/leab/.cache/huggingface/datasets/text/default-c20d4731be61e1a0/0.0.0/c4a140d10f020282918b5dd1b8a49f0104729c6177f60a6b49ec2a365ec69f34/cache-051db6b149185d7b.arrow
    06/11/2024 10:46:01 - INFO - datasets.arrow_dataset - Caching processed dataset at /home/leab/.cache/huggingface/datasets/text/default-c20d4731be61e1a0/0.0.0/c4a140d10f020282918b5dd1b8a49f0104729c6177f60a6b49ec2a365ec69f34/cache-051db6b149185d7b.arrow
    
    Running tokenizer on dataset line_by_line:   0%|          | 1000/12386778 [00:00<23:33, 8760.56 examples/s]
    
    …
    
    [[INFO|trainer.py:2031](http://INFO%7Ctrainer.py:2031)] 2024-06-11 11:06:44,641 >> ***** Running training *****
    [[INFO|trainer.py:2032](http://INFO%7Ctrainer.py:2032)] 2024-06-11 11:06:44,641 >>   Num examples = 99,094,224
    [[INFO|trainer.py:2033](http://INFO%7Ctrainer.py:2033)] 2024-06-11 11:06:44,641 >>   Num Epochs = 500
    [[INFO|trainer.py:2034](http://INFO%7Ctrainer.py:2034)] 2024-06-11 11:06:44,641 >>   Instantaneous batch size per device = 16
    [[INFO|trainer.py:2037](http://INFO%7Ctrainer.py:2037)] 2024-06-11 11:06:44,641 >>   Total train batch size (w. parallel, distributed & accumulation) = 16
    [[INFO|trainer.py:2038](http://INFO%7Ctrainer.py:2038)] 2024-06-11 11:06:44,641 >>   Gradient Accumulation steps = 1
    [[INFO|trainer.py:2039](http://INFO%7Ctrainer.py:2039)] 2024-06-11 11:06:44,642 >>   Total optimization steps = 3,096,694,500
    [[INFO|trainer.py:2040](http://INFO%7Ctrainer.py:2040)] 2024-06-11 11:06:44,642 >>   Number of trainable parameters = 86,062,873
    [[INFO|trainer.py:2061](http://INFO%7Ctrainer.py:2061)] 2024-06-11 11:06:44,678 >>   Continuing training from checkpoint, will skip to saved global_step
    **[[INFO|trainer.py:2062](http://INFO%7Ctrainer.py:2062)] 2024-06-11 11:06:44,678 >>   Continuing training from epoch 2**
    [[INFO|trainer.py:2063](http://INFO%7Ctrainer.py:2063)] 2024-06-11 11:06:44,678 >>   Continuing training from global step 12386778
    [[INFO|trainer.py:2065](http://INFO%7Ctrainer.py:2065)] 2024-06-11 11:06:44,678 >>   Will skip the first 2 epochs then the first 0 batches in the first epoch.
    
    ### Heavy model config 4 smaller model:
    
    [[INFO|trainer.py:2031](http://INFO%7Ctrainer.py:2031)] 2024-06-11 11:15:13,772 >> ***** Running training *****
    [[INFO|trainer.py:2032](http://INFO%7Ctrainer.py:2032)] 2024-06-11 11:15:13,773 >>   Num examples = 99,094,224
    [[INFO|trainer.py:2033](http://INFO%7Ctrainer.py:2033)] 2024-06-11 11:15:13,773 >>   Num Epochs = 500
    [[INFO|trainer.py:2034](http://INFO%7Ctrainer.py:2034)] 2024-06-11 11:15:13,773 >>   Instantaneous batch size per device = 16
    [[INFO|trainer.py:2037](http://INFO%7Ctrainer.py:2037)] 2024-06-11 11:15:13,773 >>   Total train batch size (w. parallel, distributed & accumulation) = 16
    [[INFO|trainer.py:2038](http://INFO%7Ctrainer.py:2038)] 2024-06-11 11:15:13,773 >>   Gradient Accumulation steps = 1
    [[INFO|trainer.py:2039](http://INFO%7Ctrainer.py:2039)] 2024-06-11 11:15:13,773 >>   Total optimization steps = 3,096,694,500
    [[INFO|trainer.py:2040](http://INFO%7Ctrainer.py:2040)] 2024-06-11 11:15:13,773 >>   Number of trainable parameters = 13,150,745
    [[INFO|trainer.py:2061](http://INFO%7Ctrainer.py:2061)] 2024-06-11 11:15:13,958 >>   Continuing training from checkpoint, will skip to saved global_step
    **[[INFO|trainer.py:2062](http://INFO%7Ctrainer.py:2062)] 2024-06-11 11:15:13,958 >>   Continuing training from epoch 10**
    [[INFO|trainer.py:2063](http://INFO%7Ctrainer.py:2063)] 2024-06-11 11:15:13,958 >>   Continuing training from global step 61933890
    [[INFO|trainer.py:2065](http://INFO%7Ctrainer.py:2065)] 2024-06-11 11:15:13,958 >>   Will skip the first 10 epochs then the first 0 batches in the first epoch.
    
    ## NSP MLM ProtBERT bfd
    
    file: /ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/train_mlm_nsp_prot_bert.py
    
    ## Evaluation and Training loss debugging
    
    ### Training loss
    
    Starting training...
    Epoch: 0, Step: 0, Loss: 4.346868515014648
    Detailed logging at Epoch: 0, Step: 0
    Input IDs: tensor([[ 2,  4,  8,  ...,  0,  0,  0],
    [ 2,  4,  8,  ...,  0,  0,  0],
    [ 2, 18,  8,  ...,  0,  0,  0],
    ...,
    [ 2, 18,  5,  ..., 11, 12,  3],
    [ 2, 18,  8,  ...,  0,  0,  0],
    [ 2, 18,  8,  ...,  0,  0,  0]], device='cuda:0')
    Attention Mask: tensor([[1, 1, 1,  ..., 0, 0, 0],
    [1, 1, 1,  ..., 0, 0, 0],
    [1, 1, 1,  ..., 0, 0, 0],
    ...,
    [1, 1, 1,  ..., 1, 1, 1],
    [1, 1, 1,  ..., 0, 0, 0],
    [1, 1, 1,  ..., 0, 0, 0]], device='cuda:0')
    Training Loss: 4.346868515014648
    Epoch: 0, Step: 1, Loss: 4.368096351623535
    Epoch: 0, Step: 2, Loss: 4.3253173828125
    Epoch: 0, Step: 3, Loss: 4.2544450759887695
    Epoch: 0, Step: 4, Loss: 4.204248905181885
    Epoch: 0, Step: 5, Loss: 4.1194047927856445
    Epoch: 0, Step: 6, Loss: 4.009517669677734
    Epoch: 0, Step: 7, Loss: 3.8966894149780273
    Epoch: 0, Step: 8, Loss: 3.7549123764038086
    Epoch: 0, Step: 9, Loss: 3.637162208557129
    Epoch: 0, Step: 10, Loss: 3.5147769451141357
    Detailed logging at Epoch: 0, Step: 10
    Input IDs: tensor([[ 2,  9,  9,  ..., 12,  3,  0],
    [ 2, 18,  8,  ...,  0,  0,  0],
    [ 2, 18,  8,  ...,  8,  5,  3],
    ...,
    [ 2, 18,  8,  ...,  0,  0,  0],
    [ 2, 18,  8,  ...,  0,  0,  0],
    [ 2,  4, 18,  ...,  0,  0,  0]], device='cuda:0')
    Attention Mask: tensor([[1, 1, 1,  ..., 1, 1, 0],
    [1, 1, 1,  ..., 0, 0, 0],
    [1, 1, 1,  ..., 1, 1, 1],
    ...,
    [1, 1, 1,  ..., 0, 0, 0],
    [1, 1, 1,  ..., 0, 0, 0],
    [1, 1, 1,  ..., 0, 0, 0]], device='cuda:0')
    Training Loss: 3.5147769451141357
    Epoch: 0, Step: 11, Loss: 3.4040257930755615
    Epoch: 0, Step: 12, Loss: 3.3490748405456543
    Train dataloader length: 13
    Epoch 0 Training Loss: 51.18454027175903

    
    ### Evaluation loss
    
    Epoch 0 Training Loss: 48.31981873512268
    Epoch: 0, Step: 0, Loss: 3.1584718227386475
    Detailed logging at Epoch: 0, Step: 0
    Input IDs: tensor([[ 2,  4,  8,  ...,  0,  0,  0],
    [ 2, 18, 16,  ...,  0,  0,  0],
    [ 2, 18,  8,  ...,  0,  0,  0],
    ...,
    [ 2,  9,  8,  ...,  0,  0,  0],
    [ 2, 18,  8,  ...,  0,  0,  0],
    [ 2, 18,  5,  ...,  0,  0,  0]], device='cuda:0')
    Attention Mask: tensor([[1, 1, 1,  ..., 0, 0, 0],
    [1, 1, 1,  ..., 0, 0, 0],
    [1, 1, 1,  ..., 0, 0, 0],
    ...,
    [1, 1, 1,  ..., 0, 0, 0],
    [1, 1, 1,  ..., 0, 0, 0],
    [1, 1, 1,  ..., 0, 0, 0]], device='cuda:0')
    Evaluation Loss: 3.1584718227386475
    Epoch: 0, Step: 1, Loss: 3.1977410316467285
    Epoch: 0, Step: 2, Loss: 3.1480979919433594
    Epoch: 0, Step: 3, Loss: 3.1726250648498535
    Eval dataloader length: 4
    Epoch 0 Evaluation Loss: 12.676935911178589
  

    ```
    
    # 21/06/2024
    
    Rerun Heavy model configs 3 and 4 and light config 3 from checkpoints
    
    ```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
             34535      test hea_conf     leab  R    4:07:56      1 dalcosrv
             34536      test hea_conf     leab  R    4:07:56      1 dalcosrv
             34556      test  li_cf_3     leab  R    4:07:56      1 dalcosrv
    
    ```
    
    print trainable parameters
    
    # 27/06/2024
    
    ## IgBERT2IgBERT
    
    use pip install git+https://github.com/adapter-hub/adapters.git for installing adapters
    
    see the issue:
    
    https://github.com/adapter-hub/adapters/issues/707
    
    ```python
    conda create --name adap_2 python=3.9
    ```
    
    ## adapter training tutorial
    
    https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/01_Adapter_Training.ipynb#scrollTo=saGtTGogLPVf
    
    ## Generation Config
    
    https://huggingface.co/docs/transformers/v4.41.3/en/generation_strategies
    
    # 28/06/2024
    

```python
conda remove -n adap_2 --all
```

```python
generation_config = GenerationConfig(
    num_return_sequences=1,
    max_length=512,
    min_length=50,
    early_stopping = True,
    
    #length_penalty = -2.0,
    
    num_beams = 3,

    # sampling
    do_sample=True,
    top_k=50,
    
    no_repeat_ngram_size = 2,

    # distribution adjustment
    temperature=0.001,
    repetition_penalty=1,

    vocab_size=model.config.encoder.vocab_size,

    # token ids
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.sep_token_id,
    decoder_start_token_id=tokenizer.cls_token_id,

    # others
    use_cache=True,
    output_logits=True,
    output_scores=True,
    output_hidden_states=True,
    return_dict_in_generate=True, )
```

run bert2bert without adapters as well to have a comparison

print out the parameters / see if its training the adapters 

nsp protbert keep the one running but try different hyperparameters

**HEAVY MODEL**:

- Heavy small model stopped at epoch 19 due to overfitting.

**BERT2BERT**:

- Training is on with half a million data points --> training is very slow, batch size of 64
- [x]  Unfreeze or warm up cross-attention.
- Issue with repeated G and S has been solved.
- [x]  Start the 'old style' fine-tuning as soon as possible.
- [x]  Print what are been training

**NSP MODEL**:

- Code is now fortunately working. --> length of true and predicted vectors are the same.
- [x]  Model performs poorly --> do a hyparameter tuning

# 03/07/2024 - 08/07/2024

Full data model with cross attention (freeze):

/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/src/freeze_FULL_data_cross_attention_with_adapters_batch_size_64_epochs_3_lr_0.0001

## bert2bert with adapters

/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/src/bert2bert_with_adapters_test.py → model with adapters and cross attention

number of trainable parameters: 145’837’440

## bert2bert without adapters

/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/src/bert2bert_without_adapters_final.py

number of trainable parameters: 965’877’790

### bert2bert with adapters finished: → run time around 18 hours for 3 epochs

logs:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/freeze_adaps_108995.o
```

model in path:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/src/freeze_FULL_data_cross_attention_with_adapters_batch_size_64_epochs_3_lr_0.0001
```

Generated sequences have not a huge similarity to true sequences → which metrics to access performance? → How to calculate perplexity?

[bert2bert with adapters full data 3 epochs](https://www.notion.so/bert2bert-with-adapters-full-data-3-epochs-4a402d7fe4db466da620c07bab316871?pvs=21)

### bert2bert without adapters finished: → run time 1d 21h 47m 12s for 3 epochs

logs:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/no_adapters_108997.o
```

model in path:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/src/freeze_FULL_data_cross_attention_without_adapters_batch_size_32_epochs_3_lr_0.0001 
```

A lot of similar generated sequences → same generation config as bert2bert with adapters but sequences are way less unique

[bert2bert without adapters 3 epochs full data](https://www.notion.so/bert2bert-without-adapters-3-epochs-full-data-ee9cdd39298f45a28f129061624e32c1?pvs=21)

# Wandb Sweeps

notebook:

[Google Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb#scrollTo=LWuspUMlnG2p)

Walkthrough:

[Walkthrough | Weights & Biases Documentation](https://docs.wandb.ai/guides/sweeps/walkthrough)

Done some sweeps for small data, sweep for full data still running, useful / takes forever?

## Cancelled Protbert NSP run time: around 3 days

            107234      test protbert     leab  R 3-18:50:50      1 dalcosrv

[ibmm-unibe-ch](https://wandb.ai/ibmm-unibe-ch/paired_model_nsp_mlm_protbert/runs/tykdhfzr?nw=nwuserlea_broe)

because of increasing loss

# light2heavy with encoder: light roberta smaller model, decoder: heavy roberta smaller model

We should probably run the seq2seq models for longer (= 100 epochs?)

example log small data

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/light2heavy_114242.o
```

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/light2heavy_114256.o
```

# Summary

### **Light Model --> done**

- **Current Status**:
    - The Light RoBERTa size model was trained for 25 epochs before training was stopped.
    - Ready to be used
    
    ### **Heavy Model --> done**
    
- **Heavy Small Model**:
    - **Current Status**: Training was showing improvement and was in progress until it was stopped at epoch 19 due to overfitting. Ready to be used.
- **Heavy RoBERTa Model**:
    - **Current Status**: Training is extremely slow. Still on going. → still training, epoch 7
    
    ### **NSP Model**
    
- **Current Status**:
    - Training with a batch size of 16 works without memory issues. → I was able to do batch size 32 with smaller block size (256 instead of 512) → for the sweeps
    - take best three from sweeps also take the standards
    
    → minimize the validation loss
    
    - First model stopped due to loss increasing --> Conduct hyperparameter tuning to improve model performance. --> with sweeps from wandb (she tried some sweeps with smaller datasets and one with the full data is currently running but this could take forever)
    
    → sweep is still running, first run probably about to finish soon → I want to double check if i can somehow improve the speed
    
    → run just stopped → something is not working, not logged (still pending) in wandb and only 1 run
    
    → small data sweeps sometimes always ended up in same nsp accuracy (all 1s or 0s → something wrong or just bc of the small data?)
    
    - Training on going
    
    → reconsider classification task? → protbert, accuracy, f1, precision, recall on validation and test set
    
    → once you have the best nsp do accuracy f1 et on test and validation set
    
    ### **Seq2Seq Model (heavy to light)**
    

4 models are training:

- bert2bert light heavy with adapters → FULL data run 20 epochs run in around 8 hours (appr. 30 min per epoch, very fast, number of trainable parameters with adapters: 4’872’832 (SMALL roberta config → full model: around 13 Mio. parameters)

log output path:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/HEAVY2LIGHT_114277.o
```

heavy2light similarity percentage is way higher than for IgBERT → up to 81% for some sequences, but also quite a big variance: some sequences also have only 4-6% similarity percentage → this metric is just to have an idea of the sequences, for full test set evaluation use PERPLEXITY → How to do this for BERT? → pseudo-perplexity?

→ blosum score also 

→ keep similarity score, add blosum and perplexity on test set

- bert2bert light heavy no adapters → not training with FULL data yet, next step → without adapters, still warm-up the crossattention parameters? → This is what I did for IgBERT2IgBERT without adapters
- bert2bert IgBert with adapters --> 3 epochs in around 18 hours (number of trainable parameters: 145’837’440)
- bert2bert IgBert no adapters --> 3 epochs in around 1 day and 21 hours (number of trainable parameters: 965’877’790)

→ igBERT2IgBERT is running with more epochs now, with adapters: at epoch 2, without adapters: 1 epoch done

- **Current Status**:
    - Training continues with and without adapters in parallel.
    - Training with half a million data points is ongoing but slow, using a batch size of 64.
    
    → training is faster if evaluated only at the epochs
    
    - The models are generating sequences 'with sense' (non the same aa repeated)
- **Next Steps**:
    - Adjust beam search parameters.
    
    do a hyperparameter tuning also for IgBERT or heavy2light (sweeps)? For the smaller models for heavy2light we could probably do this bc its pretty fast.
    
    - Compare training parameters with and without adapters.

**NEXT STEPS**:

- select the best nsp model --> selection based on the val set, then compute the metrics on the test set
- have the 4 bert2bert model trained -->
    1. do comparison of adapters and no adapers training --> both on val and test set
    2. compute perplexity of the generated sequences --> in the test set
    3. use the nsp model as validation method (if it performs well on its val/test set)
    4. compare the igbert-based to ours light and heavy
- start writing the thesis

Vocab.txt?

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/2ndr_run_FULL_data_heavy2light_with_adapters_batch_size_64_epochs_20_lr_0.0001_weight_decay_0.1
```

same size of the sequences?

line plot: x axis: length of the light, y-axis: length of the heavy, if they are correlated try to use this for generation → early stopping, try to not use max_length → plot it in the training set but first try to implement the early stopping

introduction for thesis: EDA, plot length of the data, similarity / clustering before and after, deep exploratory analysis of raw data, niche analysis on the different models. length and similarity, to show also the differences in the test, train and validation sets

### **HEAVY ROBERTA SIZE MODEL:**

- keep training, epoch 7 still val loss decreasing

### **NSP:**

- working with block size 256, batch size 32
- block size is a nextsentenceprediction dataloader argument
- **NEXT STEPS**:
- select the best 3 sweeps models trained with the small dataset and train them with the full data + try standard hyper-parameter training like lr 0.0001 and wd 0.1
- since nsp is performing poorly in terms of metrics, we can resume that classifier (fine tuning prot bert + classification task) as validation method --> resume the training of these and add adapters using IgBert_paired (no point in using protbert)

### **SEQ2SEQ**

- with our devoted models is performing quite well in generating the sequences
- **NEXT STEPS**:
- as metrics for the generated sequences add : perplexity, accuracy/exact similarity, blosum62 score
- plot the light length vs heavy length in the training set to decide the max length of the generated sequence --> if we can do early stopping would be the best
- play a little bit with the decoding technique --> i.e. play with beams
- start the 'old school' training
- **ANALYSIS**:
    1. distinguish how we are performing in generating parts of the loops and part of the framework (very similar to the analysis of the IgBert paper: study which part of the light sequence the model is able to recover properly)
    2. Check maybe how we aligned with the germline sequence
    3. Check if we are correctly generating k and lambda for the light sequences
    
    # 11/07/2024
    

Classification task:

remove space in last column

```python
awk 'BEGIN { FS=OFS="," } { gsub(/ /, "", $NF); print }' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_val_with_unpaired_SPACE_sep.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_val_with_unpaired_SPACE_sep_space_removed.csv
```

IgBERT classification is overfitting

huggingface forum:

[BERT fine tuning low epochs?](https://discuss.huggingface.co/t/bert-fine-tuning-low-epochs/54869/2)

L1/L2 regularization

[L1/L2 regularization in PyTorch](https://stackoverflow.com/questions/42704283/l1-l2-regularization-in-pytorch)

### **HEAVY ROBERTA SIZE MODEL:**

- keep training, epoch 7 still val loss decreasing

### **NSP:**

- working with block size 256, batch size 32
- block size is a nextsentenceprediction dataloader argument
- **NEXT STEPS**:
- select the best 3 sweeps models trained with the small dataset and train them with the full data + try standard hyper-parameter training like lr 0.0001 and wd 0.1
- [x]  since nsp is performing poorly in terms of metrics, we can resume that classifier (fine tuning prot bert + classification task) as validation method --> resume the training of these and add adapters using IgBert_paired (no point in using protbert)

classification task performed better than NSP in the evaluation set → how to properly save the full model? Or are the adapters enough?

prioritize classification task

### **SEQ2SEQ**

- with our devoted models is performing quite well in generating the sequences
- **NEXT STEPS**:
- [x]  as metrics for the generated sequences add : perplexity, accuracy/exact similarity, blosum62 score
- [x]  plot the light length vs heavy length in the training set to decide the max length of the generated sequence --> if we can do early stopping would be the best
- play a little bit with the decoding technique --> i.e. play with beams
- [x]  start the 'old school' training
- **ANALYSIS**:
    1. distinguish how we are performing in generating parts of the loops and part of the framework (very similar to the analysis of the IgBert paper: study which part of the light sequence the model is able to recover properly)
    2. Check maybe how we aligned with the germline sequence
    3. Check if we are correctly generating k and lambda for the light sequences
    
    how to tackle this?
    

# Perplexity heavy2light with adapters

perplexity of model with run name: 

**save_adapter_FULL_data_temperature_0.5_tests_max_length_150_early_stopping_true_heavy2light_with_adapters_batch_size_64_epochs_40_lr_0.0001_weight_decay_0.1**

model path: /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/save_adapter_FULL_data_temperature_0.5

Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Overwriting existing adapter 'heavy2light_adapter'.
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F S F Q T G V P P R F R G S G F G T E F T L T V S N L Q P E D F A T Y Y C L H H N S Y P W T F V Q R T K V E I K
Perplexity: 2.39951229095459
Mean Perplexity: 2.39951229095459

Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Overwriting existing adapter 'heavy2light_adapter'.
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Input Sequence: D I Q M T Q S P S T L S A S V G D R V T I T C R A N Q N I N N W L A W Y Q Q K P G K A P K L L I Y K T S S L E S G V P L R F S D T G S E T E F T F I I S N L Q P D D F A T Y Y C Q H Y N S Y P W A F G Q G T K V E I K
Perplexity: 1.946246862411499
Mean Perplexity: 1.946246862411499

```python
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Overwriting existing adapter 'heavy2light_adapter'.
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F S F Q T G V P P R F R G S G F G T E F T L T V S N L Q P E D F A T Y Y C L H H N S Y P W T F V Q E T K V E I K
Perplexity: 2.160383462905884
Input Sequence: E I V M T Q S P A T L S V S L G E R V T F S C R A S Q N I N S N L A W Y Q Q K P G Q A P R L L I Y G A F T R P T G I P D R F R G S G F G T E F A L T I S S M Q P E D S A V Y Y C Q H Y N N W P P W T V G P Q T K V E V K
Perplexity: 2.6374707221984863
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q N I N N Y L N W Y Q Q K P G K A P K L L I Y A A T T L Q T G V P P R F S G S R S E T D F T F I I S N V Q P E D L G T Y Y C Q H S Y S T P L T S F G G R T K V E I
Perplexity: 2.8814258575439453
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I S N Y L A W Y Q Q K P G K V P K L L I Y A A F T L Q F G V S F R F S G S R S E T D F I L T V N N L R P E D V A T Y Y C Q R Y N T A P W T F A Q E I K I E V K
Perplexity: 2.556856632232666
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F S F Q T G V P P R F R G S G F G T E F T L T V S N L Q P E D F A T Y Y C L H H N S Y P W T F V Q V A K V E I K
Perplexity: 2.254211664199829
Input Sequence: E I V M T Q S P A T L S V S L G E R V T F S C R A S Q T V G S N L A W Y Q Q K P G Q A P R L L I Y G A F T R P T G I P D R F R G G R S G T E F A L T I S S M Q P E D S A V Y Y C Q H Y N N W P P W T C G H G N K V E V K
Perplexity: 2.4018263816833496
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C Q A N Q G I S N Y L N W Y Q Q K P G K A P K L L I Y D A F N L E T G V P P R F S G S R S D T D F T F I I N T L Q P E D V A T Y Y C H Q Y H N F P L F G G W T K V E I K
Perplexity: 1.896089792251587
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F S F Q T G V P P R F R G S G F G T E F T L T V S N L Q P E D F A T Y Y C L H H N S Y P Y T F V Q E T K L E I K
Perplexity: 2.7231974601745605
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I S N Y L A W Y Q Q K P G K V P K L L I Y A A F T L Q T G V S F R F S G S R S E T D F A L T V N S M Q P E D V A T Y Y C Q R Y N T A P R T F G R G T R L E I K
Perplexity: 1.923046588897705
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S R S S F N I G S N Y V Y W Y Q Q L P R T A P K L L I Y R N N Q W P L G V P D R F S D S K S T T S P A L A I N G L R P E D E A D Y Y C A A W D D N L S E V V F G G A T K M T V
Perplexity: 2.4596896171569824
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R T S Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F T L H S G V P P R F S D S R S E T E F I L T V S N L Q P E D F A T Y Y C L E H N S Y P W T F G R G T K V E I K
Perplexity: 2.708217144012451
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I S N Y L A W Y Q Q K P G K V P K L L I Y A A T T L Q F G V S F R F S G S R S E T D F T F I I N N L R P E D V A V Y Y C Q R Y N S D P L T Y G R G T R L E I K
Perplexity: 2.7340242862701416
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F S F Q T G V P P R F R G S G F G T E F T L T V S N L Q P E D F A T Y Y C L H H N S Y P W T F V Q E T K V E I K
Perplexity: 2.073734998703003
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F S F Q T G V P P R F R G S G F G T E F T L T V S N L Q P E D F A T Y Y C L H H N S Y P W T F V Q E T K V E I K
Perplexity: 2.6551403999328613
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C Q A N Q D V S N Y L N W Y Q Q K P G K A P K L L I Y D A F N L E T G V P P R F S G S R S D T D F T F I I S T L Q P E D L A T Y Y C H Q Y E N F P L F G Q G T K V E I K
Perplexity: 2.0922188758850098
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A T S T L H S G V P P R F S D S R S E T E F T F I I S N L Q P E D F G T Y Y C L E H N S Y P W T V A Q A V K V E I K
Perplexity: 3.2397103309631348
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R T S Q G I S N Y L A W Y Q Q K P G K V P K L L I Y A A F T L Q T G V S F R F S G S R S E T D F I L T V N N L R P E D V A T Y Y C Q R Y N S D P W T F G R G T K I E I K
Perplexity: 3.3737213611602783
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F S F Q T G V P P R F R G S G F G T E F T L T V S N L Q P E D F A T Y Y C L H H N S Y P W T F D Q E T K V E I K
Perplexity: 2.2599501609802246
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S E S T S N I G R N T V N W Y Q Q L P R T A P K L L I Y T N N Q W P A G V P D R F S R S K S D T T G S L A I N G L Q A E D E A D Y Y C A A W D D S P N A V V F G G A T K V I V
Perplexity: 2.1203739643096924
Input Sequence: Q S A L T Q P A S V S G S P G Q A I T I S C T G T S S D V G G Y N Y V A W Y Q Q H P D K A P K L M I Y E V T N R P S R V P H R F S N S K S E N T A F L I I A G L Q T E D E A D Y Y C S L Y T T R S T W V F G E G A K V I V L
Perplexity: 2.633336305618286
Input Sequence: Q S A L T Q P A S V S G S P G Q A I T I S C T G T S S D V G G Y N Y V A W Y Q Q H P D K A P K L I I Y D I N N R P S E V P H R F F G F K S R N T A F L S I A G L Q T E D E A D Y Y C N S Y T T R S T W V F A A E T K V T V L
Perplexity: 2.8824052810668945
Input Sequence: D I Q M T Q S P S T L S A S V G D R V T I T C R A T E S I S S W L A W Y Q Q K P G K A P K L L I Y K T S N L E T G V P L R F S G S E F G T D F T F I I N S L Q P D D V A A Y Y C Q H Y N T Y P W T W A Q G Q E V E I K
Perplexity: 1.8542991876602173
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R T S Q G I S N Y L A W Y Q Q K P G K V P K L L I Y A A F T L Q T G V S F R F S G S R S E T D F I L T V N S M Q P E D V A T Y Y C Q R Y N T A P F F G P A I K I D L K
Perplexity: 2.650789260864258
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S R S S F N I G S N Y V Y W Y Q Q L P R T A P K L L I Y R N N Q W P A G V P D R F S D S K S T T S P F L A I T G L R P E D E A D Y Y C A A W D D N L N G W V F G G A T K V A V
Perplexity: 2.0767669677734375
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F S F Q T G V P P R F R G S G F G T E F T L T V S N L Q P E D F A T Y Y C L H H N S Y P W T F V Q E T K V E I K
Perplexity: 1.973576545715332
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R T S Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F T L H S G V P P R F S D S R S E T E F I L T V S N L Q P E D F A T Y Y C L E H N S Y P W T F G Q E S K V E I K
Perplexity: 1.8803857564926147
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I S N Y L A W Y Q Q K P G K V P K L L I Y A A F T L Q T G V S F R F S G S R S E T D F A L T V N N L R P E D V A T Y Y C Q R Y N T A P W T F G R G T K M E I K
Perplexity: 1.8158625364303589
Input Sequence: D I Q M T Q S P S T L S A S V G D R V T I T C R A T E S I S S W L A W Y Q Q K P G K A P K L L I Y K T S N L E T G V P L R F S G S E F G T D F T F I I N S L Q P D D V A A Y Y C Q H Y N T Y P W T V S Q G A K V E I K
Perplexity: 2.149930477142334
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F T L Q T G V P P R F S G S R S E T E F I L T V S N L R P E D F A T Y Y C L H H N S Y P W T F G R G T K V E I K
Perplexity: 2.849351406097412
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I S N Y L A W Y Q Q K P G K V P K L L I Y A A F T L Q T G V S F R F S G S R S E T D F A L T V N S M Q P E D V A T Y Y C Q R Y N T A P R T F G R G T R L E I K
Perplexity: 1.8847190141677856
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I S N Y L A W Y Q Q K P G K V P K L L I Y A A F T L Q F G V S F R F S G S R S E T D F I L T V N S M Q P E D V A T Y Y C Q R Y N T A P R T F A R G T R L E I K
Perplexity: 2.3922975063323975
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F S F Q T G V P P R F R G S G F G T E F T L T V S N L Q P E D F A T Y Y C L H H N S Y P W T F D Q E T K V E I K
Perplexity: 1.8981010913848877
Input Sequence: E I V M T Q S P A T L S V S L G E R V T F S C R A S Q T I S S N L A W Y Q Q K P G Q A P R L L I Y G A F T R P T G I P V R F R G S G F G T E F I L T V N S M Q P E D F A V Y Y C Q H Y N N W P P W T C G P D T K V E V K
Perplexity: 2.2822763919830322
Input Sequence: E I V M T Q S P A T L S V S L G E R A A L F C R T S Q T I S S N L A W Y Q Q K P G Q A P R L L I Y G A S T R V T G I P V R F S G S R S E T E F T F I I N G L Q P E D F A V Y Y C Q H Y N N W P P W A F G H G T K V E V K
Perplexity: 2.1657402515411377
Input Sequence: D I Q M T Q S P S T L S A S V G D R V T I T C R A N Q N I N N W L A W Y Q Q K P G K A P K L L I Y K T S S L E S G V P L R F S D T G S E T E F T F I I S N L Q P D D F A T Y Y C Q H Y N S Y P W A F G Q G T K V E I K
Perplexity: 2.0283823013305664
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F S F Q T G V P P R F R G S G F G T E F T L T V S N L Q P E D F A T Y Y C L H H N S Y P W T F D Q E T K V E I K
Perplexity: 2.621914863586426
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F S F Q T G V P P R F R G S G F G T E F T L T V S N L Q P E D F A T Y Y C L H H N S Y P W T F V Q E T K V E I K
Perplexity: 1.8740241527557373
Input Sequence: Q S A L T Q P A S V S G S P G Q A I T I S C T G T S S D V G G Y N Y V A W Y Q Q H P D K A P K L M I Y E V T N R P S R V P H R F S N S K S E N T A F L I I A G L Q T E D E A D Y Y C S T Y T T R S I W V F G E G A K V I V L
Perplexity: 1.8089773654937744
Input Sequence: E I V M T Q S P A T L S V S L G E R V T F S C R A S Q T I S S N L A W Y Q Q K P G Q A P R L L I Y G A F T R P T G I P D R F R G S G F G T E F I L T V N S M Q P E D F A V Y Y C Q H Y N N W P P W T C G H G N K V E V K
Perplexity: 2.001338243484497
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I S N Y L A W Y Q Q K P G K V P K L L I Y A A T T L Q F G V S F R F S G S R S E T D F T F I I N N L R P E D V A V Y Y C Q R Y N S D P L T Y G R G T R L E I K
Perplexity: 2.659529685974121
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I S N Y L A W Y Q Q K P G K V P K L L I Y A A F T L Q T G V S F R F S G S R S E T D F I L T V N S M Q P E D V A T Y Y C Q R Y N T A P W T F G R G T K I E I K
Perplexity: 2.9938127994537354
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F S F Q T G V P P R F R G S G F G T E F T L T V S N L Q P E D F A T Y Y C L H H N S Y P W T F V Q E T K V E I K
Perplexity: 2.085615634918213
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S R S S F N I G S N Y V Y W Y Q Q L P R T A P K L L I Y R N N Q W P A G V P D R F S D S K S E T S T F L A I T G L R P E D E A D Y Y C A A W D D N L N G W V F G G A T K M T V V
Perplexity: 2.1323788166046143
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q N I N T Y L N W Y Q Q K P G K A P K L L I Y A A T T L Q T G V P P R F S G S R S E T D F T F I I S N V Q P E D V A V Y Y C Q E S Y S T P L T S F G R G T K V E I K
Perplexity: 2.5675363540649414
Input Sequence: E I V M T Q S P A T L S V S L G E R A A L F C R T S Q T V G S N L A W Y Q Q K P G Q A P R L L I Y G A S T R V T G I P V R F S G R G F G T E F T P T I S S M Q P E D F A V Y Y C Q H Y N N W P P W T F V Q G N K V E V K
Perplexity: 2.2129945755004883
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I S N Y L A W Y Q Q K P G K V P K L L I Y A A F T L Q T G V S F R F S G S R S E T D F I L T V N S M Q P E D V A T Y Y C Q R Y N T A P W T F G R G T K M E I K
Perplexity: 2.1346049308776855
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C Q A N Q D V S N Y L N W Y Q Q K P G K A P K L L I Y D A F N L E T G V P P R F S G S R S D T D F T F I I S T L Q P E D L A T Y Y C H Q Y E N F P L F G Q G T K V E I K
Perplexity: 1.5939255952835083
Input Sequence: E I V M T Q S P A T L S V S L G E R V T F S C R A S Q T I S S N L A W Y Q Q K P G Q A P R L L I Y G A F T R P T G I P D R F R G S G F G T E F I L T V N N V Q P E D F A V Y Y C Q H Y N D W P P W T C G R E T K V E V K
Perplexity: 2.080124616622925
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F S F Q T G V P P R F R G S G F G T E F T L T V S N L Q P E D F A T Y Y C L H H N S Y P W T F V Q E T K V E I K
Perplexity: 2.3953604698181152
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S R S S F N I G S N Y V Y W Y Q Q L P R T A P K L L I Y R N N Q W P A G V P D R F S D S K S H T S T T L A I T G L R P E D E A D Y Y C A A W D D N L N G W V F G G A T K V A V
Perplexity: 2.2792935371398926
Mean Perplexity: 2.320219039916992
```

# Perplexity heavy2light without adapters

used checkpoint at epoch 13

run name: 

**FULL_data_heavy2light_without_adapters_batch_size_64_epochs_20_lr_0.0001_weight_decay_0.1**

```python
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I S N Y L A W Y Q Q K P G K V P K L L I Y A A F T L Q P E V S T R F S G S R S E T D F I L T V N S M Q A E D V A T Y Y C Q N H N T A P F A L
Perplexity: 1.9456969499588013
Input Sequence: E I V L T Q S P G T L S L F P A E R A T V S C R T S Q T I S S N Y L A W Y Q Q K P D Q A P R L L I Y G A S T R V T G I P E K F S G S A A G R D F T F I I N R M E P Q D S V V Y Y C Q H Y D T A L P Y
Perplexity: 2.7191503047943115
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A T E T V S T Y L N W Y Q Q K P G K A P K L L I Y A A Y N L Q G G V P P R F S G S R S E S D F T L T F N S Q P E D S F A I F Y C Q E A F G T P Y T S
Perplexity: 2.756075143814087
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q N I R N Y L N W Y Q Q K P G K A P K L L I Y A A F S F Q T G V P P R F R G S G F G T D F T L T V S N L Q P E D S T T Y Y C Q H S Y S I P Y T F
Perplexity: 2.2808845043182373
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q N I D T Y L N W Y Q Q K P G K A P K L L I Y A A T S F Q T G V P P R F S G S R S E T D F T L T V S N L Q P E D S T T F Y C Q H S Y S I P W T W
Perplexity: 1.986305832862854
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A T E S I S T Y L N W Y Q Q K P G K A P K L L I Y A A F T L Q T G V P P R F S G S R S E T D F N L T V S N V Q P E D S F I F Y C Q H S Y S M P Y T F
Perplexity: 2.4829509258270264
Input Sequence: S Y E L T Q P P S V S L S P G Q T A S I T C S G D K L G N K Y A C W Y Q Q K P R Q S S A L V I Y H D S K R P A G I P E R F S D F N S E N T V T L I I S R T E A M D E G E Y Y C Q A W D T S T G V F G
Perplexity: 1.5419617891311646
Input Sequence: E I V L T Q S P A T L S L F P G E R A S V S C R T S Q T V N N Y L A W Y Q Q K P D Q A P R L L I Y D A F N R V T G I P P K F S G S A A G T D F T F I I S S I E P E D S T I F Y C Q H R Y N W P L P F
Perplexity: 2.506362199783325
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q T V S N Y L N W Y Q Q K P G K A P K L L I Y A A F S F Q G G V P P R F R G S G F G T D F T L T F N S M Q P E D S T T Y Y C Q H S Y S I P T W T
Perplexity: 1.9455844163894653
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S R S S P N I G S N Y V Y W Y Q Q L P R T A P K L L I Y R N N Q W P A G V P D R F S D S K S E T S T F L A I N G L R A E D E A D Y Y C A A W D D N L S H
Perplexity: 1.8422603607177734
Input Sequence: E I V L T Q S P A T L S L F P G E R A S V S C R T S Q N V G S Y L A W Y Q Q K P D Q A P R L L I Y D A F N R V T G I P P K F S G R G F G T D F T F I I S S I E P E D S A V Y Y C Q H R S N W P L M Y
Perplexity: 2.33276104927063
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A T E S I S T Y L N W Y Q Q K P G K A P K L L I Y A A Y N L Q T G V P A R F S G S R S E T D F T L T V S N V Q P E D S F I F Y C Q H S Y S D P L A F
Perplexity: 2.9103827476501465
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A T E S I S N Y L N W Y Q Q K P G K A P K L L I Y A A F S F Q T G V P P R F R G S G F G T D F T L T V S D L Q P E D S T T Y Y C Q H S Y S M P Y T F
Perplexity: 1.9517513513565063
Input Sequence: E I V M T Q S P A T L S V S L G E R A I L F C R T S Q T V N S N L A W Y Q Q K P G Q A P R L L I Y G A S T R V T G I P D R F S G S E S A A E F T F I I S S M Q P E D F A V Y Y C Q H Y N N W P P W T
Perplexity: 1.7350525856018066
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C Q A T H D V S N Y L N W Y Q Q K P G K A P K L L I Y D A F N L E T G V P P R F S G S R S E R D F T L T V N S M Q P E D T A I F Y C L Q Y E N F P Y T F
Perplexity: 2.0033199787139893
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q N I N T Y L N W Y Q Q K P G K A P K L L I Y A A F S F Q T G V P P R F R G S G F G T D F T L T V S N L Q P E D S T T F Y C Q H S Y S I P Y T S
Perplexity: 2.7695984840393066
Input Sequence: D I Q M T Q S P S T L S A S V G D R V T I T C R A N Q T V S S W L A W Y Q Q K P G K A P K L L I Y K T S N L E S G V P P R F S C S E F G T E Y T F T N S R L Q P D D F A T Y Y C Q H Y N N Y P W T W
Perplexity: 2.917715072631836
Input Sequence: D I Q M T Q S P S T L S A S V G D R V T I T C R A T E S I S S W L A W Y Q Q K P G K A P K L L I Y K T S N L E T G V P P R F S G S E A G T D F T F I I N S L Q P D D S C T Y Y C Q H Y N N Y P W T V
Perplexity: 2.2201664447784424
Input Sequence: S Y V L T Q P P S V S L A P G Q T A T I T C G G D N I G S K S A H W Y Q Q K P R Q A L V M V V Y D D S D R P A G I P E R F S G A N S E N T G T L I I S R V E A A D E G E Y Y C Q V W D T S S N H V I
Perplexity: 2.012082099914551
Input Sequence: E I V L T Q S P G T L S L F P A E R A T V S C R T S Q T I S S T Y L A W Y Q Q K P D Q A P R L L I Y G A S N R V T G I P E K F S G S A A G R D F T F I I N G L E P Q D S V V Y Y C Q H Y D K T P I T
Perplexity: 2.6078813076019287
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A T E N I D T Y L N W Y Q Q K P G K A P K L L I Y G A F T L Q T G V P P R F S G S R S E T D F A L T V S N L E P E D S T T F Y C Q H S Y S F P W T S
Perplexity: 2.7065787315368652
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A T E S I S N Y L N W Y Q Q K P G K A P K L L I Y A A F S F Q T G V P P R F R G S G F G T D F T L T V S D L Q P E D S T T Y Y C Q H S Y S M P Y T F
Perplexity: 1.8703659772872925
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C Q A T E D V S N Y L N W Y Q Q K P G K A P K L L I Y D A A N L E T G V P L R F S G S R S E R D F T F I I S T L Q P E E I A I F Y C L H Y G N F P Y T S
Perplexity: 1.7572847604751587
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C Q A T H D V S N Y L N W Y Q Q K P G K A P K L L I Y D A F N L E T G V P P R F S G S R S E A D F T F I I S T L Q P E D T A I F Y C L H Y N N V A L T V
Perplexity: 2.027517795562744
Input Sequence: D I Q M T Q S P S T L S A S V G D R V T I T C R A T E S I S S W L A W Y Q Q K P G K A P K L L I Y K T S N L E T G V P L R F S G S E A G T D F T F I I N S L Q P D D S D T Y Y C Q H Y N T H P W T V
Perplexity: 1.7261098623275757
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S R S S F N I G S N Y V Y W Y Q Q L P R T A P K L L I Y R N N Q W P L G V P D R F S D S K S E T S T T L A I N G L R P E D E A D Y Y C A A W D D N L S I
Perplexity: 1.913658618927002
Input Sequence: D I Q M T Q S P S T L S A S V G D R V T I T C R A T E S I S S W L A W Y Q Q K P G K A P K L L I Y K T S N L E T G V P L R F S G S E F G T D F T F I I N S L Q P D D S C T Y Y C Q H Y N N Y P I A F
Perplexity: 1.9327425956726074
Input Sequence: Q S A L T Q P A S V S G S P G Q A I T I S C T G T S S D V G G Y N Y V A W Y Q Q H P D K A P K L M I Y D I N N R P S E V P N H F S R S K S Y Y T A Y L I I A G L Q T E D E A D Y F C S T Y S N S H T
Perplexity: 1.8962825536727905
Input Sequence: E I V L T Q S P A T L S L F P G E R A S V S C R T S Q T V G S S Y L A W Y Q Q K P D Q A P R L L I Y G A F N R P T G I P E K F S G G V A G T D F T F I I S R V E P Q D S A V Y Y C Q H Y D T P P W T
Perplexity: 3.0499684810638428
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q T V S T Y L N W Y Q Q K P G K A P K L L I Y A A F S F Q G G V P P R F T G S G F G T D F N L T F R S Q P E D S D T T S Y F C Q H S H P Y T L G
Perplexity: 1.624510407447815
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C Q A T H D V S N Y L N W Y Q Q K P G K A P K L L I Y D A F N L E T G V P P R F S G S R S E R D F T L T V N S M Q P E D L A I F Y C L Q Y H N F P Y T F
Perplexity: 2.2935357093811035
Input Sequence: E I V L T Q S P G T L S L F P A E R A T V S C R T S Q T I S S N Y L A W Y Q Q K P D Q A P R L L I Y G A S T R V T G I P E K F S G S A A G P H F T F I I N R M E P Q D F A V Y Y C Q H Y D S F G Q G
Perplexity: 1.72189199924469
Input Sequence: E I V L T Q S P G T L S L F P R E R A T V S C R T S Q T I S S N Y L A W Y Q Q K P A Q A P S I L I Y G A S T R V T G I P D R F S G S A A G R D F T F I I N R L E P E D S V V Y Y C Q H Y D T T P W T
Perplexity: 2.1873953342437744
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q D V S N Y L A W Y Q Q K P G K V P K L L I Y A A F T L Q P E V A P R F S G S R S E T D F A L T V N S F Q A E D A T T Y Y C Q N F N G A V W T F
Perplexity: 2.273027181625366
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A T H D V S N Y L A W Y Q Q K P G K V P K L L I Y A A F T L Q P E V A P R F S G S R S E T D F A L T V N S M Q A E D A G T Y Y C Q N H N T A R T F G
Perplexity: 2.1596829891204834
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q T V S T Y L N W Y Q Q K P G K A P K L L I Y A A T S F Q N G V P P R F S G S R S E T D F T L T F N S Q P E D S C A I F Y C Q H S Y S I P W T T
Perplexity: 2.247340679168701
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q T V S T Y L N W Y Q Q K P G K A P K L L I Y A A F S F Q G G V P P R F R G S G F G T D F T L T F N S M Q P E D S D T N Y C Q H S Y S I P W T S
Perplexity: 1.8665261268615723
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S E S S F N I G S N T V N W Y Q Q L P R T A P K L L I Y S D N Q W P A G V P D R F S T S K S R S L A F L V I N G L Q A E D E A D Y Y C A A W D D S M N A
Perplexity: 1.8061959743499756
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S R S S F N I G S N Y V Y W Y Q Q L P R T A P K L L I Y R N N Q W P A G V P D R F S D S K S E T S T T L A I N G L R P E D E A D Y Y C A A W D D N L S I
Perplexity: 1.8577088117599487
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S R S S F N I G S N T V N W Y Q Q L P R T A P K L L I Y S D N Q W P A G V P D R F S T S K S E T T G Y L A I N G L Q F E D E A D Y Y C A A W D D S L N A
Perplexity: 2.242842197418213
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A T E N I R N Y L N W Y Q Q K P G K A P N L L I Y G A F S M Q T G V P P R F R G S G F G T D F T L T V S N V Q P E D S T T Y Y C Q H S Y S I P L A L
Perplexity: 2.990779161453247
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S E S S F N I G S N T V N W Y Q Q L P R T A P K L L I Y S D N Q W P A G V P D R F S T S K S R S L A F L V I N G L Q A E D E A D Y Y C A A W D D S M N A
Perplexity: 2.081883668899536
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S R S S P N I G S N Y V Y W Y Q Q L P R T A P K L L I Y R N N Q W P A G V P D R F S D S K S E T S T T L A I N G L R P E D E A D Y Y C A A W D D N L S I
Perplexity: 1.713417887687683
Input Sequence: E I V L T Q S P A T L S L F P G E R A S V S C R T S Q T V G T Y L A W Y Q Q K P D Q A P R L L I Y D A F N R V T G I P P K F S G S R S E T D F T F I I S S I E P E D S A V Y Y C Q H R R N W P L P F
Perplexity: 2.885409355163574
Input Sequence: Q S A L T Q P R S V S G S P G Q A V T I S C T G T S S D V G G Y N Y V A W Y Q Q H P D K A P K L M I Y D A T K R P S W V P N R F S T F K S E N T A S L S I A G L Q T E D E A D Y Y C C S Y A A R Y T
Perplexity: 1.7429214715957642
Input Sequence: Q S A L T Q P A S V S G S P G Q A I T I S C T G T S S D V G G Y D Y V A W Y Q Q H P D K A P K L L I Y E V T N R P S W V P N H F S T T A A M L V F G D T V Y W L Q T E D E A D F Y C S L Y T R S N T
Perplexity: 2.170325756072998
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C Q A T H D V S N Y L N W Y Q Q K P G K A P K L L I Y D A A N L E T G V P L R F S G S R S E R D F T F I I S T L Q P E D L A I F Y C L H Y N N V L P M Y
Perplexity: 1.5618155002593994
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C Q A T H D V S N Y L N W Y Q Q K P G K A P K L L I Y D A F N L E T G V P P R F S G S R S E R D F T L T V N S M Q P E D S C T Y Y C L Q Y E N V L P L F
Perplexity: 2.0753934383392334
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q T V S T Y L N W Y Q Q K P G K A P K L L I Y A A F S F Q G G V P P R F R G S G F G T D F T L T F N S M Q P E D S C T N Y C Q H Q Y G A L P L F
Perplexity: 2.3075404167175293
Input Sequence: E I V L T Q S P G T L S L F P A E R A T V S C R T S Q T I S S N Y L A W Y Q Q K P D Q A P R L L I Y G A S T R V T G I P E K F S G S A A G R D F T F I I N R M E P Q D S V V Y Y C Q H Y D T A L R I
Perplexity: 1.809640645980835
Mean Perplexity: 2.159364700317383
```

# Perplexity IgBERT2IgBERT with adapters

run name: save_adapter_FULL_data_temperature_0.5_tests_max_length_150_early_stopping_true_heavy2light_with_adapters_batch_size_64_epochs_40_lr_0.0001_weight_decay_0.1

```python
Overwriting existing adapter 'seq2seq_adapter'.
Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See https://huggingface.co/docs/transformers/troubleshooting#incorrect-output-when-padding-tokens-arent-masked.
Input Sequence: E I V M T Q S P A T L S V S L G E R A A L F C R T S Q T V R S N L A W Y Q Q K P G Q A P R L L I Y G A S T R V T G I P D R F S G S A F G T E F T F I I S S M Q P E D F A V Y Y C Q H Y N N W P P W T
Perplexity: 1.8974342346191406
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A T E D V S N Y L A W F Q Q K P G K A P K S F I Y A A F S R Q T G V P P R F R G S G F G T D F T L T V N S I E P E H F A I F Y C Q H Y N T Y P L I L
Perplexity: 3.5856242179870605
Input Sequence: S Y V L T Q P P S V S L A P G Q T A R I T C G G D N I G S K S I H W Y Q Q K P R Q A L V V I V Y D D S D R P A G I P E R F S G T N S E N T S T L I I S R V E A E D E G A Y Y C Q V W D T T G N H V F
Perplexity: 3.333098888397217
Input Sequence: S Y V L T Q P P S V S A A P G Q T A R I T C G G D N I G S K S L H W Y Q Q K P D Q A T V V I I Y D D S D R P A G I P E R F S G A N S E N T G T L S I S T I R V G V E A D A E F D Y Y C S S W D G N D
Perplexity: 3.2893471717834473
Input Sequence: S Y V L T Q P P S V S L A P G Q T A R I T C G G N N I G S K S A H W Y Q Q K P D Q A L L V V I Y D D S D R P A G I P E R F S G A N S E N T V T L I I S R V E A E D E G D Y Y C Q V W D T S S N H V F
Perplexity: 1.9231830835342407
Input Sequence: Q S A L T Q P A S V S G S P G Q A I T I S C T G T S S D V G G Y N Y V A W Y Q Q H P D K A P K L I I Y D I N N R P S E V P N H F S R S K S F N T A F L S I A G L Q T E D E A D Y Y C S T Y T T R N S
Perplexity: 3.1758017539978027
Input Sequence: S Y E L T Q P P S V S E S P G Q T A R I T C S G D A L P K Q Y A Y W Y Q Q K P D Q A P V L V I Y K D S N R P A G I P E R F S A S K S R N T V T L S I S W T H A V D E A D Y Y C Q V W D K T T R V F G
Perplexity: 1.840706706047058
Input Sequence: Q S A L T Q P A S V S G S P G Q A I T I S C T G T S S D V G G Y N Y V A W Y Q Q H P D K A P K L M I Y E V T N R P S R V P H R F S T F K S E N T A F L I I A G L Q T E D E A D Y Y C S L Y T T R S I
Perplexity: 2.6770424842834473
Input Sequence: Q S A L T Q P A S V S G S P G Q A I T I S C T G T S S D V G G Y N Y V A W Y Q Q H P D K A P K L M I Y D I N N R P S E V P N H F S R K K S T Y G D N A T L S I A G L Q T E D E A D Y Y C S L Y T T N
Perplexity: 2.018045425415039
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S R S S P N I G S N Y V Y W Y Q Q L P R T A P K L L I Y R N N Q W P A G V P D R F S D S K S E T S T F L A I T G L R A E D E A D Y Y C A A W D D N L S W
Perplexity: 2.479722023010254
Input Sequence: E I V L T Q S P A T L S L F P G E R A S V S C R T S Q N V D T Y L A W Y Q Q K P D Q A P R L L I Y D A F N R V T G I P P K F S G S A A G T D F T F I I S S M E P E D S T V Y Y C Q H R S N W P L P F
Perplexity: 2.6315135955810547
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S E S S F N I G N N Y V Y W Y Q Q L P E T A P K L L I Y R N D Q W P A G V P D R F S D S K S R T S T F L A I T G L R S Q D E A D Y Y C A A W D D N L S N
Perplexity: 4.372453212738037
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I S N Y L A W Y Q Q K P G K V P K L L I Y A A F T L Q P E V S F R F S G S R S E T D F I L T V N S M Q A E D V A T Y Y C Q N C N N A P W T F
Perplexity: 3.9711415767669678
Input Sequence: E I V M T Q S P A T L S V S L G E R A A L F C R S S Q N V G N N A S W Y Q Q K P G Q A P R L L I Y G A F T R V T G I P D R F S G S A Y E T E F A V A E S T F I A N T V F L H Y Y T T Y R T S R N S H
Perplexity: 3.264819383621216
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S R S S F N I G S N T V N W Y Q Q L P R T A P K L L I Y S D N Q W P A G V P D R F S T S K S E T T L S L A I T G L Q F E D E A D Y Y C A A W D D S M N G
Perplexity: 2.339115858078003
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A N Q N I N N Y L N W Y Q Q K P G K A P K L L I Y A A F S F Q G G V P P R F R G S G F G T D F T L T V S N L Q P E D S T T Y Y C Q H S Y S I P L F A
Perplexity: 4.32484769821167
Input Sequence: D I Q M T Q S P S T L S A S V G D R V T I T C R A N Q N I N T W L A W Y Q Q K P G K A P K L L I Y K T S S L E S G V P P R F S D S R S E T E F T F I I S N L Q P D D F A T Y Y C Q H Y N S Y P W T V
Perplexity: 6.104801654815674
Input Sequence: Q S A L T Q P A S V S G S P G Q A I T I S C T G T S S D V G G Y N Y V A W Y Q Q H P D K A P K L I I Y D I N N R P S R V P N H F S T F K S H N T A Y L S I A G L Q T E D E A D Y Y C S L Y T T R S N
Perplexity: 3.2993197441101074
Input Sequence: S Y V L T Q P P S V S L A P G Q T A R I T C G G N N I G S K S A H W Y Q Q K P D Q A A V V I V Y D D S D R P A G I P E R F S G A N S E N T S T L S I S R V E A E D E G D Y Y C Q V W D T T G E H V F
Perplexity: 2.253957509994507
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q T V S T Y L N W Y Q Q K P G K A P K L L I Y A A T S F Q G G V P P R F S G S R S E T D F T L T F N S M Q P E D S C T H C C A V R Y P T T P F G
Perplexity: 3.6863927841186523
Input Sequence: D I V M T Q S P D S L A V S V G E R A T I N C K S S Q T V L Y S F N N K N Y L S W Y Q Q K P G Q P P K L L I S R T S T R E S G V P E K F S A S E F G T D F T L T F R S I Q A E D V A L F Y C Q H Y Y
Perplexity: 4.088196277618408
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q D V N N Y L N W Y Q Q K P G K A P K L L I Y A A F S F Q T G V P A R F T A T R S G T D F A L T V S N L Q P E D S T T Y Y C Q E S Y S I P W T F
Perplexity: 2.1715328693389893
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C Q A T R D V S N Y L N W Y Q Q K P G K A P K L L I Y D A F N L E T G V P P R F S G S R S E A D F T F I I S T L Q P E D L A I F Y C L K Y G N T P L T W
Perplexity: 2.185368537902832
Input Sequence: Q S A L T Q P A S V S G S P G Q T I T L S C T G T S S D V G G Y N Y V C W Y Q Q H P D K A P K L M I Y D I S N R P S E V P N H F S R S K S T N T A Y L A I A G L Q A E D E A D Y Y C S L Y T T R N S
Perplexity: 3.0733437538146973
Input Sequence: D I Q M T Q S P S T L S A S V G D R V T I T C R A T H S I S S W L A W Y Q Q K P G K A P K L L I Y K T S N L E S G V P L R F S D S E F G T E L T F T V S Q L Q P D D F A I F Y C Q H Y N S Y S R T W
Perplexity: 1.9826772212982178
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S R S S F N I G S N Y V Y W Y Q Q L P R T A P K L L I Y R N N Q W P A G V P D R F S D S K S E T S T F L A I T G L R P E D E A D Y Y C A A W D D N L S W
Perplexity: 2.4393136501312256
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A T H S I S T Y L N W Y Q Q K P G K A P K L L I Y A A F S F Q T G V P L R F R G S G F G T D F T L T V S N L Q P E D S D T F Y C Q E S Y S M P W T S
Perplexity: 1.9838541746139526
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A T H S I S T Y L N W Y Q Q K P G K A P K L L I Y A A F S F Q G G V P P R F T G S G F G T D F A L T V S N L Q P E D S E T F Y C Q E S Y S D P L A I
Perplexity: 2.5871901512145996
Input Sequence: E I V L T Q S P G T L S L F P E E R A T V S C R T S Q T I S S N Y L A W Y Q Q K P D Q A P R L L I Y G A S T R P T G I P A R F S G S A F G R D F T F I I N R V E P V D S V V Y Y C Q H Y D T P P W T
Perplexity: 3.6453113555908203
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q T V S T Y L N W Y Q Q K P G K A P K L L I Y A A F S F Q G G V P L R F R G S G F G T D F T L T F N S M Q P E D S D T H Y C Q E S Y S I P W T W
Perplexity: 1.666273593902588
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A T H S I S T Y L N W Y Q Q K P G K A P K L L I Y A A F S F Q T G V P P R F T A G G S G T D F I L T V S N L Q P E D S E T F Y C Q H T S R T P Y T L
Perplexity: 3.078132390975952
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A T E S I S T Y L N W Y Q Q K P G K A P K L L I Y A A F T L Q T G V P P R F S G S R S E T D F A L T V S N L H P E D V A I F Y C Q H Y Y S F P W T F
Perplexity: 2.165443181991577
Input Sequence: E I V M T Q S P A T L S V S L G E R A I L F C R T S Q T V T N N L A W Y Q Q K P G Q A P R L L I Y G A S T R V A G I P D R F S G S E S A A E F T F I I S S F Q P E D F A V Y Y C Q E Y N D W P P W T
Perplexity: 2.3706321716308594
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I S N Y L A W Y Q Q K P G K V P K L L I Y A A F T L Q P A V S F R F S G S R T D T N F I L T V N S M Q A E D F A T Y Y C Q H Y N T A P L F G
Perplexity: 2.9524741172790527
Input Sequence: E I V L T Q S P G T L S L F P E E R A T F S C R T S Q T V S S N Y L A W Y Q Q K P D Q A P R L L I Y G A S T R V T G I P A R F T A G G S G R D F A L V I S R M E P V D V A V Y Y C Q H Y D S A A Y T
Perplexity: 2.3837287425994873
Input Sequence: D I Q M T Q S P S T L S A S V G D R V T I T C R A N Q N I N T W L A W Y Q Q K P G K A P K L L I Y K T S S L E S G V P P R F S D S R S E T E F T F A I S N L Q P D D F G T Y Y C Q H Y N S Y S W T T
Perplexity: 4.262875080108643
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S R S S P N I G S N Y V Y W Y Q Q L P R T A P K L L I Y R N N Q W P A G V P D R F S D S K S E T S T F L A I T G L R P E D E A D Y Y C A A W D D N L S W
Perplexity: 2.525317907333374
Input Sequence: Q S A L T Q P A S V S G S P G Q A I T I S C T G T S S D V G G Y N Y V A W Y Q Q H P D K A P K L M I Y D I N N R P S E V P N H F S T F K S N S M T T A T L S I A G L Q T E D E A D Y Y C S L Y T R S
Perplexity: 1.757493019104004
Input Sequence: Q S A L T Q P A S V S G S P G Q A I T I S C T G T S S D V G G Y N Y V A W Y Q Q H P D K A P K L M I Y D I N N R P S E V P N H F S T F K S F N T T Y L S I A G L Q T E D E A D Y Y C S L Y T R S N S
Perplexity: 2.5109288692474365
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A T E D V S N Y L A W Y Q Q K P G K V P K L L I Y A A F T L Q T G V A P R F S G S R S E T D F I L T V N S M Q P E H V V I F Y C Q N F D S T P L G F
Perplexity: 2.7451088428497314
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S R S S F N I G S N T V N W Y Q Q L P R T A P K L L I Y S D N Q W P A G V P D R F S T S K S E T T G A L A I T D L Q T E D E A D Y Y C A A W D D S L N G
Perplexity: 5.241646766662598
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S R S S F N I G S N Y V Y W Y Q Q L P R T A P K L L I Y R N N Q W P A G V P D R F S D S K S E T S T F L A I T G L R A E D E A D Y Y C M A W D D N L S I
Perplexity: 2.795682668685913
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S R S S P N I G S N Y V Y W Y Q Q L P R T A P K L L I Y R N N Q W P A G V P D R F S D S K S E T S T F L A I T G L R P E D E A D Y Y C A A W D D N L S I
Perplexity: 2.161007881164551
Input Sequence: E I V L T Q S P A T L S L F P G E R A I L Y C R T S Q T V S S Y L A W Y Q Q K P D Q A P R L L I Y D A S N R V T G I P P K F S G S R S E T D F T F I I S D L E P E D S A V Y Y S H Q R R N W P L G F
Perplexity: 3.25909686088562
Input Sequence: Q S A L T Q P A S V S G S P G Q A I T I S C T G T S S D V G G Y N Y V A W Y Q Q H P D K A P K L M I Y D A T N R P S E V F D R F S R G A F G N A A Y L S I A Q L Q T E D E A D Y Y C C S Y A G R S T
Perplexity: 2.2621779441833496
Input Sequence: E I V L T Q S P A T L S L F P G E R A I L A C R T S Q N V S S Y L V W Y Q Q K P D Q A P R L L I Y D A S N R V T G I P P K F S G S A F G T D F T F I I S R M E P E D S V V Y Y C Q H R S D W P L N T
Perplexity: 2.525146484375
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q T V S N Y L N W Y Q Q K P G K A P K L L I Y A A F S F Q G G V P L R F R G S G F G T D F T L T F N S M Q P E D S T T Y Y C Q H S Y S I P Y T S
Perplexity: 1.5829304456710815
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A N Q D V S N Y L A W F Q Q K P G K A P K S I I Y A A T S F H S G V P P R F S D S R S E T D F T L T V N S M Q P E D L G T Y Y C Q H Y N T F P W T T
Perplexity: 2.458169460296631
Input Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A T H G I S N Y L A W F Q Q K P G K A P K S M I Y A A F T L Q T G V P P R F S G S R S E T D F I L T V S D L R P E D S T T Y Y C Q H Y D T F P L I F
Perplexity: 2.829517364501953
Input Sequence: Q S V L T Q P P S A S G T P G Q R V T I S C S R S S F N I G S N Y V Y W Y Q Q L P R T A P K L L I Y R N N Q W P A G V P D R F S D S K S E T S T F L A I T G L R P E D E A D Y Y C A A W D D N L S W
Perplexity: 3.7720534801483154
Mean Perplexity: 2.87861967086792
```

## IgBERT2IgBERT without adapters

```python
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 15.64394474029541
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 16.854511260986328
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 16.88533592224121
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 17.167909622192383
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 15.061911582946777
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 15.648880004882812
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 16.24404525756836
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 16.847829818725586
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 15.79585075378418
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 17.153621673583984
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 18.295555114746094
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 16.1816463470459
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 18.330467224121094
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 18.0992488861084
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 17.13683319091797
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 17.531631469726562
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 16.718971252441406
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 17.981496810913086
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 16.24318504333496
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 17.249210357666016
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 17.589794158935547
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 15.731803894042969
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 16.89297103881836
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 18.06439781188965
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 16.658260345458984
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 16.321271896362305
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 16.29074478149414
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 18.119281768798828
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 16.814334869384766
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 16.454755783081055
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 18.92850685119629
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 17.897951126098633
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 16.83004379272461
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 17.122936248779297
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 17.741506576538086
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 19.53776741027832
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 16.965482711791992
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 15.842968940734863
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 16.91964340209961
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 18.293500900268555
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 17.847064971923828
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 17.56212043762207
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 16.613588333129883
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 17.0093936920166
Input Sequence: S S G S T S L S Q S A S P S Y S V S D S K S R S F S E S N S C S W S I S G G T G L G Q G A G P G Y G V G D G K G R G F G E T T L T Q T A T P T Y T V T D T K T R T F T E G N G
Perplexity: 19.26214599609375
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 16.797101974487305
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 17.43958282470703
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 17.277013778686523
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 16.52609634399414
Input Sequence: S S G S T S L S Q S A S P S Y S V S I S D S K S R S F S E S N S C S W S G G T G L G Q G A G P G Y G V G I G D G K G R G F G E T T L T Q T A T P T Y T V T I T D T K T R T F T E G
Perplexity: 16.976285934448242
Mean Perplexity: 17.108009338378906
```

[BLOSUM Results](https://www.notion.so/BLOSUM-Results-151d2d07377f4645bca886bcc7357109?pvs=21)

### Summary BLOSUM, similarity and perplexity results (50 sequences only so far (from test set) !!)

|  | heavy2light with adapters | heavy2light without adapters | IgBERT2IgBERT with adapters | IgBERT2IgBERT without adapters |
| --- | --- | --- | --- | --- |
| BLOSUM | 103.68 | 59.74 | 60.1 | -54.8 |
| Avg Perplexity | 2.32 | 2.16 | 2.88 | 17.11 |
| Avg Similarity | 31.68% | 25.7% | 25.9% | 10.478 |

# **18/07/2024**

**CLASSIFICATION TASK** --> accuracy ~70%, epoch 10, maybe training little bit more
**NSP** --> put it on pause (we did not see any major results, ~50% accuracy)
**ROBERTA SIZE HEAVY MODEL**  is too slow in training --> we can stop this now 
**BERT2BERT**:

- IgBert-based translation with and without adapters: the training curves are very shitty, it is like it has not been trained --> try with different learning rate
- Translation with our models --> with adapters always better performance and slower training time --> on a subset of 50 sequences on the test set ~30% identity --> do this on the entire test set for robust metrics --> play with hyper parameters tuning during the week end --> Try maybe a pipeline light big - heavy big / light small - heavy small --> start implement the analysis lile IgBert () --> do the analysis like igbert paper (Fraction of correctly predicted residues by region (frameworks (FW) and CDRs for the test light chains ): in oas database we have already the info of the parts splitted, but do a SANITY CHECK !!! using IgBlast or PyIR (remember to check and to use the imgt numbering method)

```
         JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
         34535      test hea_conf     leab  R 27-13:24:23      1 dalcosrv
        118905      test h2l_adap     leab  R       1:35      1 dalcosrv
        118907      test test_set     leab  R       0:05      1 dalcosrv

```

118907: full test set evaluation heavy2light with adapters run name: **save_adapter_FULL_data_temperature_0.5_tests_max_length_150_early_stopping_true_heavy2light_with_adapters_batch_size_64_epochs_40_lr_0.0001_weight_decay_0.1**

# 19/07/2024

![Screenshot 2024-07-19 at 11.16.49.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/882c1b95-476d-4a7f-9994-d02922cb2063/c7af71d6-32a1-4ce3-9d79-982a31440865/Screenshot_2024-07-19_at_11.16.49.png)

start implementing the analysis lile IgBert () --> do the analysis like igbert paper (Fraction of correctly predicted residues by region (frameworks (FW) and CDRs for the test light chains ): in oas database we have already the info of the parts splitted, but do a SANITY CHECK !!! using IgBlast or PyIR (remember to check and to use the imgt numbering method)

## step 1: Extract all relevant columns from OAS_light database

```python
sqlite3
.open OAS_paired.db
.schema
.quit
```

relevant columns:

```python
"cdr1_aa_light"
"cdr1_end_light"
"cdr1_light"
"cdr1_start_light"
"cdr2_aa_light"
"cdr2_end_light"
"cdr2_light"
"cdr2_start_light"
"cdr3_aa_light"
"cdr3_end_light"
"cdr3_light"
"cdr3_start_light"
"fwr1_aa_light"
"fwr1_end_light"
"fwr1_light"
"fwr1_start_light"
"fwr2_aa_light"
"fwr2_end_light"
"fwr2_light"
"fwr2_start_light"
"fwr3_aa_light"
"fwr3_end_light"
"fwr3_light"
"fwr3_start_light"
"fwr4_aa_light"
"fwr4_end_light"
"fwr4_light"
"fwr4_start_light"
"sequence_alignment_aa_light"
"sequence_alignment_light"
sequence_alignment_heavy_sep_light
```

Relevant columns are highlighted in red 

```python
"ANARCI_numbering_heavy" TEXT, "ANARCI_numbering_light" TEXT, "ANARCI_status_heavy" TEXT, "ANARCI_status_light" TEXT,
 "Age" TEXT, "Author" TEXT, "BSource" TEXT, "BType" TEXT,
 "Chain" TEXT, "Disease" TEXT, "Isotype" TEXT, "Isotype_heavy" TEXT,
 "Isotype_light" TEXT, "Link" TEXT, "Longitudinal" TEXT, "Redundancy_heavy" TEXT,
 "Redundancy_light" TEXT, "Run" TEXT, "Species" TEXT, "Subject" TEXT,
 "Unique sequences" TEXT, "Vaccine" TEXT, "c_region_heavy" TEXT, "c_region_light" TEXT,
 "cdr1_aa_heavy" TEXT, "cdr1_aa_light" TEXT, "cdr1_end_heavy" TEXT, "cdr1_end_light" TEXT,
 "cdr1_heavy" TEXT, "cdr1_light" TEXT, "cdr1_start_heavy" TEXT, "cdr1_start_light" TEXT,
 "cdr2_aa_heavy" TEXT, "cdr2_aa_light" TEXT, "cdr2_end_heavy" TEXT, "cdr2_end_light" TEXT,
 "cdr2_heavy" TEXT, "cdr2_light" TEXT, "cdr2_start_heavy" TEXT, "cdr2_start_light" TEXT,
 "cdr3_aa_heavy" TEXT, "cdr3_aa_light" TEXT, "cdr3_end_heavy" TEXT, "cdr3_end_light" TEXT,
 "cdr3_heavy" TEXT, "cdr3_light" TEXT, "cdr3_start_heavy" TEXT, "cdr3_start_light" TEXT,
 "complete_vdj_heavy" TEXT, "complete_vdj_light" TEXT, "d_alignment_end_heavy" TEXT, "d_alignment_end_light" TEXT,
 "d_alignment_start_heavy" TEXT, "d_alignment_start_light" TEXT, "d_call_heavy" TEXT, "d_call_light" TEXT,
 "d_cigar_heavy" TEXT, "d_cigar_light" TEXT, "d_germline_alignment_aa_heavy" TEXT, "d_germline_alignment_aa_light" TEXT,
 "d_germline_alignment_heavy" TEXT, "d_germline_alignment_light" TEXT, "d_germline_end_heavy" TEXT, "d_germline_end_light" TEXT,
 "d_germline_start_heavy" TEXT, "d_germline_start_light" TEXT, "d_identity_heavy" TEXT, "d_identity_light" TEXT,
 "d_score_heavy" TEXT, "d_score_light" TEXT, "d_sequence_alignment_aa_heavy" TEXT, "d_sequence_alignment_aa_light" TEXT,
 "d_sequence_alignment_heavy" TEXT, "d_sequence_alignment_light" TEXT, "d_sequence_end_heavy" TEXT, "d_sequence_end_light" TEXT,
 "d_sequence_start_heavy" TEXT, "d_sequence_start_light" TEXT, "d_support_heavy" TEXT, "d_support_light" TEXT,
 "download_date" TEXT, "fwr1_aa_heavy" TEXT, "fwr1_aa_light" TEXT, "fwr1_end_heavy" TEXT,
 "fwr1_end_light" TEXT, "fwr1_heavy" TEXT, "fwr1_light" TEXT, "fwr1_start_heavy" TEXT,
 "fwr1_start_light" TEXT, "fwr2_aa_heavy" TEXT, "fwr2_aa_light" TEXT, "fwr2_end_heavy" TEXT,
 "fwr2_end_light" TEXT, "fwr2_heavy" TEXT, "fwr2_light" TEXT, "fwr2_start_heavy" TEXT,
 "fwr2_start_light" TEXT, "fwr3_aa_heavy" TEXT, "fwr3_aa_light" TEXT, "fwr3_end_heavy" TEXT,
 "fwr3_end_light" TEXT, "fwr3_heavy" TEXT, "fwr3_light" TEXT, "fwr3_start_heavy" TEXT,
 "fwr3_start_light" TEXT, "fwr4_aa_heavy" TEXT, "fwr4_aa_light" TEXT, "fwr4_end_heavy" TEXT,
 "fwr4_end_light" TEXT, "fwr4_heavy" TEXT, "fwr4_light" TEXT, "fwr4_start_heavy" TEXT,
 "fwr4_start_light" TEXT, "germline_alignment_aa_heavy" TEXT, "germline_alignment_aa_light" TEXT, "germline_alignment_heavy" TEXT,
 "germline_alignment_light" TEXT, "j_alignment_end_heavy" TEXT, "j_alignment_end_light" TEXT, "j_alignment_start_heavy" TEXT,
 "j_alignment_start_light" TEXT, "j_call_heavy" TEXT, "j_call_light" TEXT, "j_cigar_heavy" TEXT,
 "j_cigar_light" TEXT, "j_germline_alignment_aa_heavy" TEXT, "j_germline_alignment_aa_light" TEXT, "j_germline_alignment_heavy" TEXT,
 "j_germline_alignment_light" TEXT, "j_germline_end_heavy" TEXT, "j_germline_end_light" TEXT, "j_germline_start_heavy" TEXT,
 "j_germline_start_light" TEXT, "j_identity_heavy" TEXT, "j_identity_light" TEXT, "j_score_heavy" TEXT,
 "j_score_light" TEXT, "j_sequence_alignment_aa_heavy" TEXT, "j_sequence_alignment_aa_light" TEXT, "j_sequence_alignment_heavy" TEXT,
 "j_sequence_alignment_light" TEXT, "j_sequence_end_heavy" TEXT, "j_sequence_end_light" TEXT, "j_sequence_start_heavy" TEXT,
 "j_sequence_start_light" TEXT, "j_support_heavy" TEXT, "j_support_light" TEXT, "junction_aa_heavy" TEXT,
 "junction_aa_length_heavy" TEXT, "junction_aa_length_light" TEXT, "junction_aa_light" TEXT, "junction_heavy" TEXT,
 "junction_length_heavy" TEXT, "junction_length_light" TEXT, "junction_light" TEXT, "locus_heavy" TEXT,
 "locus_light" TEXT, "np1_heavy" TEXT, "np1_length_heavy" TEXT, "np1_length_light" TEXT,
 "np1_light" TEXT, "np2_heavy" TEXT, "np2_length_heavy" TEXT, "np2_length_light" TEXT,
 "np2_light" TEXT, "productive_heavy" TEXT, "productive_light" TEXT, "rev_comp_heavy" TEXT,
 "rev_comp_light" TEXT, "sequence_alignment_aa_heavy" TEXT, "sequence_alignment_aa_light" TEXT, "sequence_alignment_heavy" TEXT,
 "sequence_alignment_light" TEXT, "sequence_heavy" TEXT, "sequence_id_heavy" TEXT, "sequence_id_light" TEXT,
 "sequence_light" TEXT, "stop_codon_heavy" TEXT, "stop_codon_light" TEXT, "v_alignment_end_heavy" TEXT,
 "v_alignment_end_light" TEXT, "v_alignment_start_heavy" TEXT, "v_alignment_start_light" TEXT, "v_call_heavy" TEXT,
 "v_call_light" TEXT, "v_cigar_heavy" TEXT, "v_cigar_light" TEXT, "v_frameshift_heavy" TEXT,
 "v_frameshift_light" TEXT, "v_germline_alignment_aa_heavy" TEXT, "v_germline_alignment_aa_light" TEXT, "v_germline_alignment_heavy" TEXT,
 "v_germline_alignment_light" TEXT, "v_germline_end_heavy" TEXT, "v_germline_end_light" TEXT, "v_germline_start_heavy" TEXT,
 "v_germline_start_light" TEXT, "v_identity_heavy" TEXT, "v_identity_light" TEXT, "v_score_heavy" TEXT,
 "v_score_light" TEXT, "v_sequence_alignment_aa_heavy" TEXT, "v_sequence_alignment_aa_light" TEXT, "v_sequence_alignment_heavy" TEXT,
 "v_sequence_alignment_light" TEXT, "v_sequence_end_heavy" TEXT, "v_sequence_end_light" TEXT, "v_sequence_start_heavy" TEXT,
 "v_sequence_start_light" TEXT, "v_support_heavy" TEXT, "v_support_light" TEXT, "vj_in_frame_heavy" TEXT,
 "vj_in_frame_light" TEXT, sequence_alignment_aa_full, sequence_alignment_heavy_sep_light TEXT);
sqlite> 
```

script for data extraction: 

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/src/extract_columns_from_slite_db_analysis.sh
```

```python
head -n 100 /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/full_paired_data_for_analysis.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/small_paired_data_for_analysis.csv
```

column names:

```python
cdr1_aa_light,cdr1_end_light,cdr1_light,cdr1_start_light,cdr2_aa_light,cdr2_end_light,cdr2_light,cdr2_start_light,cdr3_aa_light,cdr3_end_light,cdr3_light,cdr3_start_light,fwr1_aa_light,fwr1_end_light,fwr1_light,fwr1_start_light,fwr2_aa_light,fwr2_end_light,fwr2_light,fwr2_start_light,fwr3_aa_light,fwr3_end_light,fwr3_light,fwr3_start_light,fwr4_aa_light,fwr4_end_light,fwr4_light,fwr4_start_light,sequence_alignment_aa_light,sequence_alignment_light,sequence_alignment_heavy_sep_light
```

```python
cat header.csv full_paired_data_for_analysis.csv > with_header_full_paired_data_for_analysis.csv
```

```python
head -n 100 /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/paired_full_seqs_sep_test_no_ids.txt > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/small_paired_seqs_sep_test_no_ids.txt
```

file used for sequence extraction:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/match_sequences.jl
```

Remove duplicates of last column (heavy[SEP]light column):

```python
awk -F ',' '!seen[$NF]++' full_test_data_extraction.txt > full_test_data_extraction_no_dupl.txt
```

All extracted sequences of the test set (67209 sequences) are in the file

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/full_test_data/full_test_data_extraction_with_header.txt
```

# 22/07/2024

## Heavy2Light model with adapters

Heavy2Light models with small light and small heavy config (SMALL_SMALL)

run name: 

```python
**FULL_data_small_small_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_100_lr_0.0005_wd_0.05**
```

output file:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/h2l_adaps_small_small_125412.o
```

model path:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/FULL_data_small_small_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_100_lr_0.0005_wd_0.05
```

running this model for a bit longer

run name:

```python
FULL_data_small_small_temp_0.4_max_length_150_early_stopping_true_batch_size_64_epochs_100_lr_0.005_wd_0.5
```

output file:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/h2l_adaps_small_small_125415.o
```

model path:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/FULL_data_small_small_temp_0.4_max_length_150_early_stopping_true_batch_size_64_epochs_100_lr_0.005_wd_0.5
```

Heavy2Light models with big light and big heavy config (BIG_BIG)

number of trainable parameters: 32839488
trainable params: 32839488 || all params: 204963673 || trainable%: 16.022101633590456

Both runs are still running

run name: 

```python
**FULL_data_big_big_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1**
```

Overfitting

run name: 

```python
**FULL_data_big_big_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0005_wd_0.1**
```

not overfitting

# Using PyIR:

Github:

https://github.com/crowelab/PyIR

```python
# from PyIR
sequence_id,sequence,sequence_alignment_aa,germline_alignment_aa,fwr1,fwr1_aa,cdr1,cdr1_aa,fwr2,fwr2_aa,cdr2,cdr2_aa,fwr3,fwr3_aa,fwr4,fwr4_aa,cdr3,cdr3_aa,fwr1_start,fwr1_end,cdr1_start,cdr1_end,fwr2_start,fwr2_end,cdr2_start,cdr2_end,fwr3_start,fwr3_end,fwr4_start,fwr4_end,cdr3_start,cdr3_end,np1,np1_length,np2,np2_length,cdr3_aa_length
1,CAGTCTGCCCTGACTCAGCCAGCCTCCGTGTCTGGGTCTCCTGGACAGTCGATCACCATCTCCTGCACTGGAACCAGCAGTGATGTTGGGAATTATAACCTTGTCTCCTGGTACCAACACCACCCAGGCAAAGCCCCCAAACTCATGATTTATGAGGTCAGTAAGCGGCCCTCAGGGATTTCTAATCGCTTCTCTGGCTCCAAGTCTGGCAACACGGCCTCCCTGACAATCTCTGGGCTCCAGGCTGACGACGAGGCTGATTATTACTGCTGCTCATATGCAGGTAGTAGAATCCTTTATGTCTTCGGATCTGGGACCAAGGTCACCGTCCTAG,QSALTQPASVSGSPGQSITISCTGTSSDVGNYNLVSWYQHHPGKAPKLMIYEVSKRPSGISNRFSGSKSGNTASLTISGLQADDEADYYCCSYAGSRILYVFGSGTKVTVL,QSALTQPASVSGSPGQSITISCTGTSSDVGSYNLVSWYQQHPGKAPKLMIYEVSKRPSGVSNRFSGSKSGNTASLTISGLQAEDEADYYCCSYAGSXXXYVFGTGTKVTVL,CAGTCTGCCCTGACTCAGCCAGCCTCCGTGTCTGGGTCTCCTGGACAGTCGATCACCATCTCCTGCACTGGAACC,QSALTQPASVSGSPGQSITISCTGT,AGCAGTGATGTTGGGAATTATAACCTT,SSDVGNYNL,GTCTCCTGGTACCAACACCACCCAGGCAAAGCCCCCAAACTCATGATTTAT,VSWYQHHPGKAPKLMIY,GAGGTCAGT,EVS,AAGCGGCCCTCAGGGATTTCTAATCGCTTCTCTGGCTCCAAGTCTGGCAACACGGCCTCCCTGACAATCTCTGGGCTCCAGGCTGACGACGAGGCTGATTATTACTGC,KRPSGISNRFSGSKSGNTASLTISGLQADDEADYYC,TTCGGATCTGGGACCAAGGTCACCGTCCTAG,FGSGTKVTVL,TGCTCATATGCAGGTAGTAGAATCCTTTATGTC,CSYAGSRILYV,1,75,76,102,103,153,154,162,163,270,304,334,271,303,AATCCT,6,,,11

```

example sequence (coloring based on PyIR, sequence with id: 90):

```python
DIQMTQSPSSLSASVGDRVTFTCRSSQNIGIYLNWYQQKPGRAPTVLIYTASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYFCQQSYSLPYTFGQGARLQIK
fwr1
cdr1
fwr2
cdr2
fwr3
cdr3
fwr4
```

lambda, k chains check if generated sequence is the same predicted

Latent space using encoderdecoder model, see if the model can distinguish between lambda and k light sequences → PCA or t-SNE or u-map to see any groupings of the embeddings

use PyIR instead of OAS.db

use PyIR also on the generated sequence → turn it to nucleotides first → try with AA only first → should work

# 24/07/2024

Full test set evaluation heavy2light model with adapters on 67’209 sequences:

Average BLOSUM Score: 92.78026752369475
Average Similarity Percentage: 30.3151422766621%
Mean Perplexity: 2.1886556148529053

# U-MAP

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/extracted_data_from_PyIR/umap_extraction_from_pyir_small_data.csv
```

remove duplicates in heavy[SEP]light column

```python
awk -F ',' '!seen[$1]++' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_for_umap_pca/small_data_heavyseplight_locus.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_for_umap_pca/small_data_heavyseplight_locus_no_dupl.csv
```

remove spaces in last column

```python
awk 'BEGIN { FS=OFS="," } { gsub(/ /, "", $NF); print }' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_for_umap_pca/small_data_heavyseplight_locus_no_dupl_spaces.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_for_umap_pca/small_data_heavyseplight_locus_no_dupl_spaces_rm.csv
```

# Data Processing steps for creating the U-MAP:

At the end we want to have a csv file of the form:

```python
sequence_alignment_heavy_sep_light,locus
Q V Q L Q E S G P G L V K P S E T L S L T C T V S G G S I S G F Y W S W I R Q S P G K G L E W I A Y I Y F S G S T N Y N P S L K S R V T L S V D T S K N Q F S L K L S S V T A A D S A V Y Y C A R D V G P Y N S I S P G R Y Y F D Y W G P G T L V T V S S [SEP] Q S A L T Q P A S V S G S P G Q S I T I S C T G T S S D V G N Y N L V S W Y Q H H P G K A P K L M I Y E V S K R P S G I S N R F S G S K S G N T A S L T I S G L Q A D D E A D Y Y C C S Y A G S R I L Y V F G S G T K V T V L ,IGL
```

Where IGL = Lambda light chain and IGK = Kappa light chain

We get the IGL/IGK label by using PyIR (column name: “locus”)

## Workflow for the full test dataset with 67’209 sequences:

We need a fasta file with the dna sequence for PyIR

Full extracted sequences from OAS.db:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/full_test_data/full_paired_data_for_analysis.csv
```

all relevant test data sequences from OAS.db with all columns:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/full_test_data/full_test_data_extraction_with_header.txt
```

Use the file:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/convert_to_fasta.jl
```

to convert the file from the OAS.db into a fasta file of the form:

```python
>1
CAGTCTGCCCTGACTCAGCCAGCCTCCGTGTCTGGGTCTCCTGGACAGTCGATCACCATCTCCTGCACTGGAACCAGCAGTGATGTTGGGAATTATAACCTTGTCTCCTGGTACCAACACCACCCAGGCAAAGCCCCCAAACTCATGATTTATGAGGTCAGTAAGCGGCCCTCAGGGATTTCTAATCGCTTCTCTGGCTCCAAGTCTGGCAACACGGCCTCCCTGACAATCTCTGGGCTCCAGGCTGACGACGAGGCTGATTATTACTGCTGCTCATATGCAGGTAGTAGAATCCTTTATGTCTTCGGATCTGGGACCAAGGTCACCGTCCTAG
>2
GAAATTGTGTTGACGCAGTCGCCAGGCACCCTGTCTTTGTCTACAGGGGAAAGAGCCACCCTCTCTTGCAGGGCCGGTCAGACTGTTGACGGCAACTCCTTAGCCTGGTACCAGCACAAACCTGGCCAGGCTCCCAGGCTCCTCATCTTTCGTGCATCTCGTAGGGCCGCTGACATCCCAGACAGGTTCACTGGCAGTGGGTCTGGGACCGACTTCACTCTCACCATTAGCAGACTGGAGGTTGAAGATTTCGCAGTTTATTACTGTCAGCAGTATGGTGCCTCACCAAAAACGTTCGGCCAAGGGACCAAGGTGGAA
>3
CAGTCTGCCCTGACTCAGCCTGCCTCCGTGTCTGGGTCTCCTGGACAGTCGATCACCATCTCCTGCACTGGAAGCAGCAGTGATGTTGGGAGTTATAACCTTGTCTCTTGGTACCAACAGCACCCAGGCAAAGCCCCCAAACTCATGATTTATGAGGTCAGTAAGCGGCCCTCAGGGGTTTCTAATCGCTTCTCTGGCTCCAAGTCTGGCAACACGGCCTCCCTGACAATCTCTGGGCTCCAGGCTGAGGACGAGGCTCAATATTACTGCTGCTCATATGGAGGTAGGAATTTTCATGTGCTATTCGGCGGAGGGACCGAGCTGACCGTCCTAG
>4
CAGTCTGCCCTGACTCAGCCTCCCTCCGCGTCCGGGTCTCTTGGACAGTCAGTCACCATCTCCTGCACTGGAAGTAGTAGTGACGTTGGTGGGTATGCCTATGTCTCCTGGTATCAACAACACCCAGGCAAAGCCCCCAAAGTCGTAATTTATGAGGTCACTAAGCGGCCCTCAGGGGTCCCTGAACGGTTCTCTGGCTCCAAGTCTGGCAACACGGCCTCCCTGACCGTCTCTGGGCTCCAGGCTGAAGATGAGGCTGATTATTACTGCATCTCATATGCCGGCGCCAACAAATTAGGGGTATTCGGCGGAGGGACCAAGCTGACCGTCCTAG
```

where the dna sequence is the dna sequence of the light chain AA sequence

full test data fasta file:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/full_test_data/full_paired_data_sequence_alignment_light.fasta
```

Then  use this fasta file to perform PyIR. For this, use the script:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/PyIR.py
```

67,209 sequences processed in 42.41 seconds, 1,584 sequences / s

```python
gunzip -d /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/full_test_data/full_paired_data_sequence_alignment_light.tsv.gz
```

Then, we need to extract the relevant columns for the u-map

```python
columns_to_extract = [
    "sequence_id",
    "sequence",
    "sequence_alignment_aa",
    "locus"
]
```

using the script:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/extract_relevant_cols_from_tsv.py
```

output file:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/extracted_data_from_PyIR/umap_extraction_from_pyir_full_data.csv
```

Then use the script:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_preprocessing_for_umap.py
```

to merge the csv file with the locus labels and the csv with the  heavy[SEP]light column, 

important: both csv files need to have the same column name: “sequence_alignment_aa” to merge them on.

Then remove the duplicates with:

```python
awk -F ',' '!seen[$1]++' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_for_umap_pca/full_data_heavyseplight_locus.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_for_umap_pca/full_data_heavyseplight_locus_no_dupl.csv
```

final output file with 67’209 sequences:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_for_umap_pca/full_data_heavyseplight_locus_no_dupl.csv
```

To include spaces between the AAs use the script:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/preprocess_input_data.py
```

and to remove the spaces in the last column use:

```python
awk 'BEGIN { FS=OFS="," } { gsub(/ /, "", $NF); print }' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_for_umap_pca/full_data_heavyseplight_locus_no_dupl_spaces.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_for_umap_pca/full_data_heavyseplight_locus_no_dupl_spaces_rm.csv
```

So we end up with the final input file of the form:

```python
sequence_alignment_heavy_sep_light,locus
Q V Q L Q E S G P G L V R P S E T L S L E C S V S G S S L S N D Y Y W G W I R Q P P G K G L Q W I G N I Y H S G T T Y Y N P S L K S R L T M S V D T S R N H F S L Q L D S V T A A D T A V Y Y C A R L I Y T G Y G K R C F D Y W G Q G A L V T V S S [SEP] D I Q M T Q S P P F V S A S V G D S V T I T C R A S Q G I T D W L A W Y Q H K Q G K A P K L L I F A A S T L Q S G V P S R F S G T G S G T D F T L T I T R L Q P E D S A T Y Y C Q Q G Y T F P G G F T F G P G T K V D V K ,IGK
Q V Q L V E S G G G V V Q P G R S L R L S C A A S G F T F S S Y G M H W V R Q A P G K G L E W V G V I W Y D G S K K Y Y S D S V K G R F T I S R D S P N N M L Y L Q M N S L R A E D T A V Y F C A R D D D G S N Q Y G I F E Y W G Q G T V V T V S S [SEP] Q S A L T Q P V S V S G S P G Q S I A I S C T G T S S D V G G Y N S V S W F Q Q H P G K A P K L M I Y D V S N R P S G V S N R F S G S K S G N T A S L T I S G L Q A E D E A D Y Y C S S Y T S S S T R L F G G G T K L T V L ,IGL
```

location:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_for_umap_pca/full_data_heavyseplight_locus_no_dupl_spaces_rm.csv
```

We then can use the file 

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/src/plot_umap.py
```

for the U-map.

# B-Type analysis of generated sequences (similarity) / Comparison of Memory and Naive B Cells

extracted information about the B-Types:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/Btypes_full_paired_data_for_analysis.csv
```

Used the file:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/match_sequences.jl
```

output file:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/Btypes_full_paired_test_data.csv
```

Remove duplicates:

```python
awk -F ',' '!seen[$3]++' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/B_cell_analysis_output.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/B_cell_analysis_output_no_dupl.csv
```

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/Btypes_full_paired_test_data_no_dupl.csv
```

model output heavy2light

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/full_eval_heavy2light_with_adapters125463.o
```

```python
awk -F ',' '!seen[$2]++' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/B_cell_analysis_output.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/B_cell_analysis_output_no_dupl.csv
```

### Results B-Cell analysis of heavy2light model on full test set MEAN:

```python
                                            Perplexity  BLOSUM Score  Similarity Percentage
BType                                                                                      
ASC                                           2.258298     59.026316              25.719750
CD27-memory-and-Plasmablast/Plasma-B-Cells    2.221776     92.713127              30.184347
Memory-B-Cells                                2.416487     93.632222              30.312491
Naive-B-Cells                                 1.857270    101.022147              31.820085
Plasma-B-Cells                                2.333874     94.121564              30.370104
Plasmablast                                   2.310689     88.673410              29.606787
Plasmablast/Plasma-B-Cells                    2.460713     70.237770              27.013723
RV+B-Cells                                    2.087540     48.058824              24.554099
Unsorted-B-Cells                              2.108372     94.692478              30.629889
double-nagative-B-cells                       2.300443     91.584906              30.065477
```

### Results Kappa Lambda analysis MEAN (full test set)

```python
       BLOSUM Score  Similarity Percentage  Perplexity
locus                                                 
IGH      -68.000000               8.488388   26.063198
IGK      167.590051              40.481777    2.042250
IGL        7.685513              18.843335    2.356489
```

### Results Kappa Lambda analysis MEDIAN (full test set)

```python
       BLOSUM Score  Similarity Percentage  Perplexity
locus                                                 
IGH           -70.0               8.035714   25.548422
IGK           110.0              31.775701    1.880061
IGL           -49.0              11.214953    2.166538
```

### Sequences in test set:

IGK / Kappa: 36’729 sequences

IGL / Lambda: 30’475 sequences

IGH: only 5 sequences → exclude them

```python
awk -F ',' '!seen[$1]++' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/kappa_lambda_analysis/umap_k_l_subtypes/full_data_heavyseplight_locus_v_family.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/kappa_lambda_analysis/umap_k_l_subtypes/full_data_heavyseplight_locus_v_family_no_dupl.csv
```

```python
awk 'BEGIN { FS=OFS="," } { gsub(/ /, "", $2); print }' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/kappa_lambda_analysis/umap_k_l_subtypes/full_data_heavyseplight_locus_v_family_no_dupl_spaces_rm.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/kappa_lambda_analysis/umap_k_l_subtypes/full_data_heavyseplight_locus_v_family_no_dupl_spaces_rm2.csv
```

# 29/07/2024

output file best performing model heavy2light:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/full_eval_heavy2light_with_adapters125463.o
```

# AlphaFold Sequence Comparison

Sequence pair 1:
True Sequence: Q S A L T Q P V S V S G S P G Q S I A I S C T G T S S D V G G Y N S V S W F Q Q H P G K A P K L M I Y D V S N R P S G V S N R F S G S K S G N T A S L T I S G L Q A E D E A D Y Y C S S Y T S S S T R L F G G G T K L T V L
Generated Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F S F Q T G V P P R F R G S G F G T E F T L T V S N L Q P E D F A T Y Y C L H H N S Y P W T F V Q E T K V E I K
BLOSUM Score: -102.0
**Similarity Percentage: 6.5420560747663545%**
Perplexity: 2.08941912651062
model is on device cuda:0

use the following notebook (for single sequences):

https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb

alphafold explanation pdf:

https://2023.igem.wiki/npu-china/static/model.pdf

Sequence pair 44650:
True Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A S Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A S S L Q S G V P S R F S G S G S G T E F T L T I S S L Q P E D F A T Y Y C L Q H N S Y P W T F G Q G T K V E I K
Generated Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A Y S F Q T G V P P R F S G S E S R T E F T L T V S N L Q P E D F A T Y Y C L H H N S Y P W T F G H G T K V E I K
BLOSUM Score: 509.0
**Similarity Percentage: 89.7196261682243%**
Perplexity: 1.4944385290145874
model is on device cuda:0

Sequence pair 47533:
True Sequence: L T Q S P G T L S L S P G E R A T L S C R A S P S P F L A W Y Q Q R P G Q A P R L L I Y G A S I R A T G I P D R F S G S G S G P D F T L T I S R L E P E D F A V Y F C Q Q Y D S S R R F T F G P G T K V E L K
Generated Sequence: E I V L T Q S P G T L S L F P R E R A T V S C R T S Q T I S S N Y L A W Y Q Q K P D Q A P K L L I Y G A S T R V T G I P A R F S G S E S A A D F T F I I N R L E P E D V A V Y Y C Q H Y D N S R P Y S F G Q E T K V E V K
BLOSUM Score: -148.0
**Similarity Percentage: 0.0%**
Perplexity: 2.6351206302642822
model is on device cuda:0

# Notes

## big / big heavy2light model fully evaluated

run name: FULL_data_big_big_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_1e-05_wd_0.5

```python
Average BLOSUM Score: 84.97292029341308
Average Similarity Percentage: 29.181855516790897%
Mean Perplexity: 3.4276723861694336
```

## Some reruns with other hyperparameters of small / small heavy2light model gave worse or very similar results

run name: FULL_data_small_small_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.01

```python
Average BLOSUM Score: 93.33248523263254
Average Similarity Percentage: 30.282256685340116%
Mean Perplexity: 2.158348560333252
```

small / small heavy2light model 500 epochs

run name: FULL_data_small_small_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_500_lr_0.0005_wd_0.05

```python
Average BLOSUM Score: -38.845943251647846
Average Similarity Percentage: 12.930573842003863%
Mean Perplexity: 4.30082368850708
```

### Model used so far for all analyses

small/small model:

run name: save_adapter_FULL_data_temperature_0.5_tests_max_length_150_early_stopping_true_heavy2light_with_adapters_batch_size_64_epochs_40_lr_0.0001_weight_decay_0.1

```python
Average BLOSUM Score: 92.78026752369475
Average Similarity Percentage: 30.3151422766621%
Mean Perplexity: 2.1886556148529053
```

So far, the best model is still the red one above

# Classification model

Several runs, couldnt get above 69% accuracy

# Questions

1. kappa lambda → 4 subtypes? 
2. Kappa lambda / 7 region analysis is difficult on generated sequences because of the required conversion from AA to DNA → any alternatives to PyIR? 
3. Same evaluation analyses for IgBERT2IgBERT → state of the art comparison

# To do

- [ ]  IgBERT analysis with 7 regions
- [x]  See if kappa / lambda generated correctly → 60% correct
- [ ]  use another tool instead of PyIR for the missing seqs, drop the missing seqs for the analysis
- [ ]  check how many are missing
- [ ]  alphafold: more sequences —> maybe 10 / 10 / 10 sequences → grid of images → supplemetary information
- [x]  different species in the PCA / umap → OAS db
- [x]  umap with some diseases
- [ ]  select different dataset → see if model is able to cluster specific diseases
- [ ]  double check B cell analysis

rename a folder in git:

```python
git mv paired_model/BERT2BERT/sqlite3_data_for_analysis paired_model/BERT2BERT/test_set_analyses
```

show root git directory:

```python
git rev-parse --show-toplevel
```

# PCA / UMAP Species and Diseases

1. Use 

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/src/extract_columns_from_sqlite_db.sh 
```

to extract the relevant columns from OAS db

resulting output file:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_subgroups_analysis/paired_oas_db_full_extraction.csv
```

1. use

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/match_sequences.jl
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/run_match_sequences.sh
```

to extract the test set sequences from the full paired db

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_subgroups_analysis/full_test_data_extraction_species_diseases.txt
lines: 74218
should be: 67209
```

remove duplicates from heavy[SEP]light column (last column → $NF)

```python
awk -F ',' '!seen[$NF]++' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_subgroups_analysis/full_test_data_extraction_species_diseases.txt > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_subgroups_analysis/full_test_data_extraction_species_diseases_no_dupl.txt
```

output file with 67209 sequences:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_subgroups_analysis/full_test_data_extraction_species_diseases_no_dupl.txt
```

```python
header (manually added):
Species, Disease, BType, Isotype_light, sequence_alignment_aa_light, sequence_alignment_light, sequence_alignment_aa_heavy, sequence_alignment_heavy, sequence_alignment_heavy_sep_light

```

use 

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_diseases_subgroups_analysis/add_spaces_between_AAs.py
```

to add spaces between the sequences

use plot_umap_pca_tSNE_diseases_species.py for plotting

# Alphafold Structure prediction with more sequences

output file: full_eval_heavy2light_with_adapters125463.o

## 10 worst sequences

```python
Sequence pair 30791:
True Sequence: V M T Q S P L S L P V T L G Q P A S I S C R S S R D L V Y K D G N T Y L I W L Q Q R P G Q S P R R L I Y R V S H R D A G V P D R F S G S G A G T D F T L R I N R V E A E D V G I Y F C M Q A T E W P Y T F G Q G T K L E I E
Generated Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F T L Q T G V P P R F S G S R S E T E F I L T V S N L R P E D F A T Y Y C L H H N S Y P W T F G R G T K V E I K
BLOSUM Score: -125.0
Similarity Percentage: 0.0%
Perplexity: 3.4887425899505615
model is on device cuda:0
```

```python
Sequence pair 24839:
True Sequence: V V L T Q S P V T V S L S P G E R A T L S C R A S H I L T C S S Q R L T C S L A W Y H L R P G Q P P R L I I Y D A S K R A A G I P A R F S G S V S G T E F T L T I T N L E P E D F G L Y Y C Q Q R S S F G G G T K L E I
Generated Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A T S T L H S G V P P R F S D S R S E T E F T F I I S N L Q P E D F G T Y Y C L E H N S Y P L T T G G R T K V E I K
BLOSUM Score: -146.0
Similarity Percentage: 0.0%
Perplexity: 4.045881748199463
model is on device cuda:0
```

```python
Sequence pair 53395:
True Sequence: V M T Q S P L S L T V T L G Q P A S I S C R A S Q S L V H S D G N T Y L N W F H Q R P G Q S P R R L I Y K V S K R D S G V P D R F S G S G S G S D F T L N I S W V E A E D V G L Y F C M Q G T H W P P F T F G P G T K V E I K
Generated Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A T S T L E S G V P P R F S D S R S E T E F T F I I S N L Q P E D F G T Y Y C L H H N S Y P W T R G P A L V I
BLOSUM Score: -113.0
Similarity Percentage: 0.9615384615384616%
Perplexity: 3.1583383083343506
model is on device cuda:0
```

```python
Sequence pair 56316:
True Sequence: V M T Q S P L S L P V T P G Q P A S I S C R A S Q S L V Y S D G N S Y L N W F Q Q R P G Q S P R R L I Y K V S K R D S G V P D R F S G S G S G T D F T L K I S R V E A E D V A V Y Y C M Q D V R W P W T F G Q G T K V E I K
Generated Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F S F Q T G V P P R F R G S G F G T E F T L T V S N L Q P E D F A T Y Y C L H H N S Y P W T F V Q E T K V E I K
BLOSUM Score: -117.0
Similarity Percentage: 0.9345794392523363%
Perplexity: 2.585385799407959
model is on device cuda:0
```

```python
Sequence pair 56586:
True Sequence: T Q S P A T L S L S P G E R A T L S C R A S Q S V T N L A W Y Q Q K P G Q P P R L L I Y G A S T R A A G V P D R F T G S G S G T D F T L T I T T L Q S E D F G V Y F C Q Q Y N S W P L T F G G G T K V E I K
Generated Sequence: E I V L T Q S P G T L S L F P R E R A T V S C R T S Q T I S S N Y L A W Y Q Q K P A Q A P K L L I Y G A S T R V T G I P D R F S G S E S A A D F T F I I N R L E P E D V A V Y Y C Q H Y D N S R G Y S F G Q E T K V E V K
BLOSUM Score: -155.0
Similarity Percentage: 0.9803921568627451%
Perplexity: 3.3309006690979004
model is on device cuda:0
```

```python
Sequence pair 63239:
True Sequence: V V M T Q S P L S L T V T L G Q P A S I S C R S S Q S L E Y E G S D Y L T W L H Q R P G Q S P R R L I Y K V S H R D S G V P D R F S G S G S G T D F T L T I N R V E A E D V G V Y Y C L Q G R H W P L T F G P G T K V V I
Generated Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A G Q G I R N D L G W Y Q Q K P G K A P K R L I Y A A F T L H S G V P P R F S D S R S E T E F I L T V S N L Q P E D F A T Y Y C L R H N S Y P W T F G R G T K V E I K
BLOSUM Score: -143.0
Similarity Percentage: 0.9345794392523363%
Perplexity: 2.773519277572632
model is on device cuda:0
```

text generation

https://huggingface.co/docs/transformers/en/main_classes/text_generation

# 08/08/2024

## Running some generation configs from heavy2light

### Diverse Beam search beam size = 5 epoch 10

run name: 

```python
**full_Diverse_beam_search_5_decoding_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1**
```

output

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/x_small_Diverse_beam_search_decoding_127791.o
```

model path:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_Diverse_beam_search_5_decoding_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1
```

full eval output:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/x_full_eval_heavy2light_with_adapters_diverse_beam_search127797.o

# on full test set (67209 seqs):
Average BLOSUM Score: 107.60038090136737
Average Similarity Percentage: 32.95260988077156%
Mean Perplexity: 2.1526780128479004
```

### Diverse Beam Search beam size = 5 epochs 50

run name:

```python
full_Diverse_beam_search_5_decoding_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1
```

output:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/x_small_Diverse_beam_search_decoding_127790.o
```

model path:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_Diverse_beam_search_5_decoding_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1
```

full eval output:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/full_Diverse_beam_search_5_decoding_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1127799.o

# on full test set (67209 seqs):
Average BLOSUM Score: 107.55434540016962
Average Similarity Percentage: 32.940888668254786%
Mean Perplexity: 2.0127155780792236
```

### Diverse Beam search beam size = 2 10 epochs

run name: 

```python
full_diverse_beam_search_2_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1
```

output file:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/x_contrastive_search_decoding_127796.o
```

model path:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_diverse_beam_search_2_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1
```

full eval output:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/full_eval_full_diverse_beam_search_2_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1_127798.o

# on full test set (67209 seqs):
Average BLOSUM Score: 108.25002603817941
Average Similarity Percentage: 33.05800137172086%
Mean Perplexity: 2.1675915718078613
```

## Contrastive search 40 epochs

run name:

```python
full_contrastive_search_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_40_lr_0.0001_wd_0.1
```

model output path:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/x_contrastive_search_decoding_127795.o
```

model path:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_contrastive_search_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_40_lr_0.0001_wd_0.1
```

full eval output path:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/full_contrastive_search_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_40_lr_0.0001_wd_0.1_127808.o

# on full test set (67209 seqs):
Average BLOSUM Score: 97.33711258908777
Average Similarity Percentage: 31.3516276662991%
Mean Perplexity: 1.981735348701477
```

Heavy2Light 

run name: 

```jsx
full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1
```

output path:

```jsx
paired_lea_tmp/paired_model/BERT2BERT/logs/experimenting/beam_search_decoding_127842.o
```

model path:

```jsx
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1
```

# 13/08/2024

## Light Chain Regions Analysis (see IgBERT paper for reference)

IgBERT paper: https://arxiv.org/abs/2403.17889

I’m using the model with run name: full_diverse_beam_search_2_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1

run name = name on wandb: https://wandb.ai/ibmm-unibe-ch/heavy2light_translation?nw=nwuserlea_broe

reference name: h2l_div_beam_search_2_epoch_10_lr_1e-4_wd_0.1 (for directory names)

because this has the best overall metrics (BLOSUM and Similarity %):

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/full_eval_full_diverse_beam_search_2_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1_127798.o

# on full test set (67209 seqs):
Average BLOSUM Score: 108.25002603817941
Average Similarity Percentage: 33.05800137172086%
Mean Perplexity: 2.1675915718078613
```

I used the file

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions.py
```

to first convert the generated and true sequences form the output file of the full evaluation into a DNA fasta file of the form:

```bash
>True_Seq_1
CAATCTGCTTTAACTCAACCTGTTTCTGTTTCTGGTTCTCCTGGTCAATCTATTGCTATTTCTTGTACTGGTACTTCTTCTGATGTTGGTGGTTATAATTCTGTTTCTTGGTTTCAACAACATCCTGGTAAAGCTCCTAAATTAATGATTTATGATGTTTCTAATCGTCCTTCTGGTGTTTCTAATCGTTTTTCTGGTTCTAAATCTGGTAATACTGCTTCTTTAACTATTTCTGGTTTACAAGCTGAAGATGAAGCTGATTATTATTGTTCTTCTTATACTTCTTCTTCTACTCGTTTATTTGGTGGTGGTACTAAATTAACTGTTTTA
>Generated_Seq_1
GATATTCAAATGACTCAATCTCCTTCTTCTTTATCTGCTTCTGTTGGTGATCGTGTTACTATTACTTGTCGTGCTTCTCAATCTATTTCTTCTTATTTAAATTGGTATCAACAAAAACCTGGTAAAGCTCCTAAATTATTAATTTATGCTGCTTCTTCTTTACAATCTGGTGTTCCTTCTCGTTTTTCTGGTTCTGGTTCTGGTACTGATTTTACTTTAACTATTTCTTCTTTACAACCTGAAGATTTTGCTACTTATTATTGTCAACAATCTTATTCTACTCCTCCTACTTTTGGTCAAGGTACTAAAGTTGAAATTAAA
>True_Seq_2
GAAATTGTTTTAACTCAATCTCCTGGTACTTTATCTTTATCTCCTGGTGAAGCTGCTACTTTATCTTGTAAATCTTCTCAACCTGTTGCTACTACTTATTTAGCTTGGTATCAACAAAAACGTGGTCAACCTCCTCGTTTATTAATTTATGGTACTTCTAATCGTGCTGTTGGTATTCCTGATCGTTTTACTGGTTCTGGTTCTCGTACTGATTTTACTTTAACTATTGATCGTTTAGAAGCTGAAGATTTTGGTTTATATTTTTGTCAACAATATGCTACTTCTCCTTATACTTTTGGTCAAGGTACTAATTTAGAA
>Generated_Seq_2
GAAATTGTTTTAACTCAATCTCCTGCTACTTTATCTTTATCTCCTGGTGAACGTGCTACTTTATCTTGTCGTGCTTCTCAATCTGTTTCTTCTTATTTAGCTTGGTATCAACAAAAACCTGGTCAAGCTCCTCGTTTATTAATTTATGATGCTTCTAATCGTGCTACTGGTATTCCTGCTCGTTTTTCTGGTTCTGGTTCTGGTACTGATTTTACTTTAACTATTTCTTCTTTAGAACCTGAAGATTTTGCTGTTTATTATTGTCAACAACGTTCTAATTGGCCTCCTTTAACTTTTGGTGGTGGTACTAAAGTTGAAATTAAA
```

I checked the validity of the generated DNA sequences with the website: https://web.expasy.org/translate/

```bash
#true seq 1 with expasy translate tool (first line) and DNA sequence generated from script (2nd line):
QSALTQPVSVSGSPGQSIAISCTGTSSDVGGYNSVSWFQQHPGKAPKLMIYDVSNRPSGVSNRFSGSKSGNTASLTISGLQAEDEADYYCSSYTSSSTRLFGGGTKLTVL
QSALTQPVSVSGSPGQSIAISCTGTSSDVGGYNSVSWFQQHPGKAPKLMIYDVSNRPSGVSNRFSGSKSGNTASLTISGLQAEDEADYYCSSYTSSSTRLFGGGTKLTVL

#generated seq 1 with expasy translate tool (first line) and DNA sequence generated from script (2nd line):
DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPTFGQGTKVEIK
DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPTFGQGTKVEIK
```

Then, the same script can be used to perform PyIR on the fasta file, which gives you a tsv file with the full PyIR output.

```bash
134,418 sequences successfully split into 1325 pieces
Starting process pool using 64 processors
100%|███████████████████████████████████████████████████████████| 134418/134418 [00:52<00:00, 2539.55seq/s]
134,418 sequences processed in 53.39 seconds, 2,517 sequences / s
```

```bash
134418 / 2 = 67209 -> because each sequence is double (true seq and generated seq)
```

use gunizp -d to unzip the output tsv.gz file

the resulting tsv file has the columns:

```bash
sequence_id     sequence        locus   stop_codon      vj_in_frame     v_frameshift    productive      rev_comp   complete_vdj    v_call  d_call  j_call  sequence_alignment      germline_alignment      sequence_alignment_aa      germline_alignment_aa   v_alignment_start       v_alignment_end d_alignment_start       d_alignment_end    j_alignment_start       j_alignment_end v_sequence_alignment    v_sequence_alignment_aa v_germline_alignment       v_germline_alignment_aa d_sequence_alignment    d_sequence_alignment_aa d_germline_alignment       d_germline_alignment_aa j_sequence_alignment    j_sequence_alignment_aa j_germline_alignment       j_germline_alignment_aa fwr1    fwr1_aa cdr1    cdr1_aa fwr2    fwr2_aa cdr2    cdr2_aa fwr3    fwr3_aa    fwr4    fwr4_aa cdr3    cdr3_aa junction        junction_length junction_aa     junction_aa_lengthv_score  d_score j_score v_cigar d_cigar j_cigar v_support       d_support       j_support       v_identityd_identity       j_identity      v_sequence_start        v_sequence_end  v_germline_start        v_germline_end     d_sequence_start        d_sequence_end  d_germline_start        d_germline_end  j_sequence_start  j_sequence_end   j_germline_start        j_germline_end  fwr1_start      fwr1_end        cdr1_start      cdr1_end   fwr2_start      fwr2_end        cdr2_start      cdr2_end        fwr3_start      fwr3_end        fwr4_start fwr4_end        cdr3_start      cdr3_end        np1     np1_length      np2     np2_length      v_family   d_family        j_family        cdr3_aa_length
```

I then used the script:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/extract_relevant_cols_from_tsv.py
```

to extract the needed columsn from the tsv and store it in a csv file.

output csv file:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/h2l_div_beam_search_2_epoch_10_lr_1e-4_wd_0.1/full_test_set_true_gen_seqs_all_relevant_cols.csv
```

For the actual analysis of the regions, i used an R script and ran it locally (not on the server). 

The R script can still be found at the server:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/evaluate_by_regions.R
```

I used R locally because the data manipulation was easier. 

```python
DIQMTQSPSSLSASVGDRVTFTCRSSQNIGIYLNWYQQKPGRAPTVLIYTASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYFCQQSYSLPYTFGQGARLQIK
fwr1
cdr1
fwr2
cdr2
fwr3
cdr3
fwr4
```

## Results

model reference name: h2l_div_beam_search_2_epoch_10_lr_1e-4_wd_0.1

|  | Mean similarity | Median Similarity | sd similarity | Mean BLOSUM | Median BLOSUM | sd BLOSUM | nbr of seqs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fwr 1 | 57.2534398648183 | 51.7241379310345 | 21.0600614108688 | 66.5928514646548 | 65 | 36.7113144447963 | 66671 |
| cdr 1 | 36.7393301524813 | 33.3333333333333 | 23.8213718770445 | -0.0265100570847904 | -2 | 15.6275317759508 | 66918 |
| fwr 2 | 72.45185 | 70.58824 | 13.81073 | 72.50457 | 72 | 12.81756 | 67132 |
| cdr 2 | 39.30957 | 33.33333 | 31.62589 | 3.224903 | 3 | 6.512473 | 67140 |
| fwr 3 | 69.37579 | 66.66667 | 13.48666 | 128.4014 | 128 | 30.92481 | 67194 |
| cdr 3 | 35.86976 | 33.33333 | 21.28005 | 2.261007 | 1 | 17.46895 | 63868 |
| fwr 4 | 73.48998 | 77.77778 | 21.29164 | 29.40751 | 38 | 45.53655 | 63132 |

# 14/08/2024

## Full evaluation of heavy2light model

**Best model so far!**

run name: 

```bash
full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1
```

output path:

```jsx
paired_lea_tmp/paired_model/BERT2BERT/logs/experimenting/beam_search_decoding_127842.o
```

model path:

```jsx
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1
```

full eval output file:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/heavy2light_60_epochs_beam_search_127845.o
```

```bash
# full test set evaluation:
Average BLOSUM Score: 109.48768766087875
Average Similarity Percentage: 33.21875942607254%
Mean Perplexity: 1.8054779767990112

# global alignment
> print(mean_blosum)
[1] 324.8816
> print(mean_similarity)
[1] 62.9108
> print(median_blosum)
[1] 322
> print(median_similarity)
[1] 61.11111
```

|  | Mean BLOSUM | Mean BLOSUM with alignment | Median BLOSUM with alignment | Mean similarity with global alignment | Median similarity with global alignment | Mean similarity | Mean perplexity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| diverse beam search, beam size = 5 | 109.5 | 325 | 322 | 63% | 61.1% | 33.2% | 1.8 |
|  |  |  |  |  |  |  |  |

# 16/08/2024

## Full evaluation of heavy2light model

run name: 

```bash
full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_100_lr_0.001_wd_0.1
```

model path:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_100_lr_0.001_wd_0.1
```

Full eval output:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/heavy2light_60_epochs_beam_search_127858.o

# full test set evaluation:
Average BLOSUM Score: 109.67296046660418
Average Similarity Percentage: 33.24877984628893%
Mean Perplexity: 1.8666492700576782
```

# Alphafold Structure prediction

model with run name:

```bash
full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1
```


### example light chain (52319):

```bash
DIQMTQSPSSLSASVGDRVTITCQASQDISNYLNWYQQKPGKAPKLLIYDASNLETGVPSRFSGSGSGTDFTFTISSLQPEDIATYYCQQYDNLPLTFGGGTKVEIK
```

DIQMTQSPSSLSASVGDRVTITCQAS → FWR1

QDISNY → CDR1

LNWYQQKPGKAPKLLIY → FWR2

DAS → CDR2

NLETGVPSRFSGSGSGTDFTFTISSLQPEDIATYYC → FWR3

QQYDNLPLT → CDR3

FGGGTKVEIK → FWR4


# questions:

test set evaluation of classification task

test set evaluation of heavy and light model

aphafold structure prediction superimposing

perplexity different regions

Similarity quantitative measure gen / true seq?

How to tell whether the input heavy seq actually matters? We know it generates meaningful light sequences, but do we know whether the heavy sequence as input actually has an influence on the generation?

include negative / neutral results in thesis (appendix)?

# To-Do

- [x]  Perplexity calculations per region
- [x]  alphafold: superimposition for each sequence
- [x]  New similarity measurement using proper alignment

```bash
185000 ^ -3 = 1.5793734e-16
(185000 ^ -3)^(-1/203) = 1.19629965167
```


# Perplexity calculation for each region

**implementing perplexity for a bert2bert encoder-decoder model for each region of the light sequence**

We have 7 regions

fwr1

cdr1

fwr2

cdr2

fwr3

cdr3

fwr4

in the light chain and we want to calculate the perplexity for each region.

e.g. if we want to calculate perplexity for fwr2 we consider the sequence up to fwr2

- > fwr1 cdr1 fwr2 and put the logits of fwr1 cdr1 to 1
1. generate the sequence -> take the logits (sum to 1= -> 1 position vector -> n
2. No argmax due to beam search -> probability for each residue in the generated seq -> len(seq) = n
3. Extract parts from PyIR
4. i.e. you know where cdr1 is stopped

example:

[FWR1] [CDR1]

ABCD    EFG

logits: 0.1 0.2 0.3 0.1 0.4 0.3 0.1

but since we are interested in CDR+ in this example, we set ABCD logits to 1

n = len(FWR1 + CDR1)

1. Apply they formula

(\prod_E^G P(x_i | x_{<i}))^(1/n)

= (0.4 * 0.9 * 0.1)^{-1/7}

first only make 1 example with the generated sequence:

DIQMTQSPSSLSASVGDRVTFTCRSSQNIGIYLNWYQQKPGRAPTVLIYTASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYFCQQSYSLPYTFGQGARLQIK

where

DIQMTQSPSSLSASVGDRVTFTCRSS -> fwr1

QNIGIY -> cdr1

LNWYQQKPGRAPTVLIY -> fwr2

TAS -> cdr2

SLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYFC -> fwr3

QQSYSLPYT -> cdr3

FGQGARLQIK -> fwr4

```bash
# on full test data
fwr1    1.122398
cdr1    1.091885
fwr2    1.032957
cdr2    1.010399
fwr3    1.020283
cdr3    1.028448
fwr4    1.011886
```


Fully evaluating model with run name: full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1

# MEETING WITH THOMAS - NEXT STEPS:

- check if the alignment has gaps in the cdr loops (ideally we should see the fwr should be more conserved; if we have a lot of gaps in the fwr it can be problematic. It is ok to have more gaps in the cdrs)
- perplexity is lower than a standsrd protein model because the complexity is lower
- https://www.nature.com/articles/s41586-022-05371-z --> check what dataset they used and how they extracted ligh coherence, check what they did. How they measure the difference? i.e. what k or lambda was the subpairs. Check if we can make sense from their dataset. Check if their data are available and if they are nice.
- in our project: plot the distribution (median,sd) of the data already collected (memory cell, b cell etc) for the light coherence

To compare to igbert:

- use igbert mask the entire light chain and ask the model to generate (to compare to out translation task)
- do the recovery on the light chains with the light_lm, do the recovery on the heavy chains with the heavy_lm
- do single analysys light chain in light_chain_lm, heavy chain in the heavy_chain_lm, and heavy through the encoder-decoder model

Analysis: 
If diseases analysis are not working we can put them as negative results (in the thesis not in the paper), the reason can be that there is always a pair of heavy and light preferred.
Do the reconstruction/recovery for each subtypes.
Do a non-human one as an out of distribution analysis --> did it learn something specific about humans?

For the aa --> codons translation you can do codon optimization, but not sure if it is worthed. Maybe checked the predicted parts by the PyIr, plot them with pymol and check that everything is ok.  Check what IgBERT did to find the different parts. Maybe in the sabdab database have a pipeline to do it since they did not do it manually.

For the thesis report classification model in the main thesis as previous step for the encoder decoder (translation tasks) model, NSP model goes in the appendix.

# Meeting 27/08/2024

- Do the embeddings for the memory and naive only
- Do also the non human species analysis (as negative result) --> Do a non-human one as an out of distribution analysis --> did it learn something specific about humans?
- Do embeddings for the heavy chain model (only the heavy chain model) and check if you can identify different genes
- check if one specific heavy chain gene has a prefered light chain gene (confusion matrix heavy genes vs light genes) --> from the zurich group they saw that there is always a pair that is preferred for i.e. HIV
- use broad v gene categories (and then if needed you can go with the smaller ones )
- check if the alignment has gaps in the cdr loops (ideally we should see the fwr should be more conserved; if we have a lot of gaps in the fwr it can be problematic. It is ok to have more gaps in the cdrs while framework should be conserved) --> check the error rate in the frameworks compared to the cdrs (framework should have a lower error rate compare to cdrs)
- we can also try to give the starting point of the framework during the translation
- the embeddings analysis we did (light chain through the encoder decoder model) are 'wrong', we have to re-do them like this:
- heavy chain into encoder-decoder model (extract the embedding before the beam-search-decoding part)
- heavy chain into the heavy model
- light model into the light model
- Do the recovery on the light chains with the light_lm, do the recovery on the heavy chains with the heavy_lm

To compare to igbert:

- use igbert, with him mask the entire light chain and ask the model to generate (to compare to our translation task)

# Number of Gaps in global alignment

|  | avg_total_insertion_length | avg_total_insertion_widthsum | avg_insertion_length | avg_total_deletion_length | avg_total_deletion_widthsum | avg_deletion_length | avg_total_indel | avg_total_indel_widthsum | avg_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fwr1 | 0.2191358 | 0.2722323 | 0.2191358 | 0.2233505 | 0.2816967 | 0.2233505 | 0.4424862 | 0.553929 | 57.25344 |
| cdr1 | 0.2333901 | 0.5489554 | 0.2333901 | 0.2475418 | 0.551197 | 0.2475418 | 0.4809319 | 1.100152 | 36.73933 |
| fwr2 | 0.001430018 | 0.004662456 | 0.001430018 | 0.00138533 | 0.00595841 | 0.00138533 | 0.002815349 | 0.01062087 | 72.45185 |
| cdr2 | 0.002144772 | 0.008385463 | 0.002144772 | 0.000893655 | 0.003321418 | 0.000893655 | 0.003038427 | 0.01170688 | 39.30957 |

# full evaluation IgBERT2IgBERT

```bash
Average BLOSUM Score: -51.87956970048654
Average Similarity Percentage: 10.5584562033619%
Mean Perplexity: 16.85335922241211
```

# full evaluation IgBERT MLM with 100% masking of light sequence

on 2000 sequences

```bash
--- Averages Across All Sequences ---
Average Perplexity: 136.04740077114104
Average Global Alignment Similarity: 26.6641
Average Global Alignment BLOSUM Score: -24.77275
Average 'Hard' Similarity: 7.02%
Average 'Hard' BLOSUM Score: -73.1935
```

do umap only with the memory and naive cells

use h100 with tmux → CUDA_VISIBLE_DEVICES=0

```bash
CUDA_VISIBLE_DEVICES=0, python ....
```

# 04/09/2024

Data Processing for UMAP/t-SNE and PCA plots for heavy sequences (heavy MLM model)

/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/heavy_model_vdj_gene_analysis

used /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/src/extract_columns_from_sqlite_db.sh

output:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/heavy_vdj_genes_paired_oas_db_full_extraction.csv
```

used 

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/match_sequences.jl
```

to extract the test set sequences

add header

```bash
"Species,Disease,BType,Isotype_light,locus_heavy,locus_light,v_call_heavy,d_call_heavy,j_call_heavy,sequence_alignment_aa_light,sequence_alignment_light,sequence_alignment_aa_heavy,sequence_alignment_heavy,sequence_alignment_heavy_sep_light"

```

remove duplicates from heavy[SEP]light column (last column → $NF)

```bash
awk -F ',' '!seen[$NF]++' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/heavy_vdj_genes_paired_oas_db_test_set_extraction.txt > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/heavy_vdj_genes_paired_oas_db_test_set_extraction_no_dupl.txt
```

use 

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_diseases_subgroups_analysis/add_spaces_between_AAs.py
```

to add spaces between the amino acids

Create a file of the format: sequence_alignment_heavy_sep_light, v_call_heavy,d_call_heavy,j_call_heavy

```bash
awk -F, 'BEGIN {OFS=","} {print $7, $8, $9, $14}' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/heavy_model_vdj_gene_analysis/spaces_heavy_vdj_genes_paired_oas_db_test_set_extraction_no_dupl.txt > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/heavy_model_vdj_gene_analysis/relevant_cols_spaces_heavy_vdj_genes_paired_oas_db_test_set_extraction_no_dupl.txt
```

**Explanation:**

•	-F, tells awk to treat commas as field separators (since it’s a CSV file).

•	OFS="," ensures that the output fields are separated by commas.

•	{print $7, $8, $9, $14}:

•	$7 corresponds to v_call_heavy.

•	$8 corresponds to d_call_heavy.

•	$9 corresponds to j_call_heavy.

•	$14 corresponds to sequence_alignment_heavy_sep_light.

 v_call_heavy,d_call_heavy,j_call_heavy have the form IGHV4-38-2*01,IGHD5-12*01,IGHJ4*02, only take the first part until the first - and make a new column with this informaton called v_call_fewer_heavy,d_call_fewer_heavy,j_call_fewer_heavy

use for this the script:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/heavy_model_vdj_gene_analysis/extract_relevant_gene_names.py
```

# Used models

|  | Architecture | Base Model | Used with Adapters | Number of trainable parameters | Number of trainable parameters from adapter | Dataset | Task | Reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Heavy MLM model | RoBERTa | Trained drom Scratch | No | 13’150’745 | - | Heavy Chains Unpaired | MLM |  |
| Light MLM model | RoBERTa | Trained from Scratch | No | 13’150’745 | - | Light Chains Unpaired | MLM |  |
| NSP model | BERT | ProtBERT bfd | No | 419’931’136 | - | Paired Sequences | MLM & NSP | Rost et al. (2020) |
| Classification model | BERT | IgBERT | Yes | 427’862’786 | 7’931’650 | Paired Sequences | Sequence Classification |  |
| Encoder-Decoder model “Heavy2Light” | RoBERTa | Encoder: Heavy MLM model Decoder: Light MLM model  | Yes | 31’173’273 | 4’872’832 | Paired Sequences | Text Generation |  |
| Encoder-Decoder model “IgBERT2IgBERT” | BERT  | Encoder: IgBERT Decoder: IgBERT | Yes | 985’701’790 | 145’837’440 | Paired Sequences | Text Generation |  |
| IgBERT MLM model | BERT | IgBERT | No | 418’881’536 | - | Paired Sequences | MLM with 100% masking of light chain |  |

# Heavy and Light MLM models Results

|  | Heavy MLM model small conf. | Heavy MLM model big conf. | Light MLM model small conf. | Light MLM model big conf. |
| --- | --- | --- | --- | --- |
| Epochs | 19 | 9 | 44 | 25 |
| Learning Rate | 5e-5 | 5e-5 | 5e-5 | 5e-5 |
| Weight Decay | 0.1 | 0.1 | 0.1 | 0.1 |
| Train loss | 0.4537 | 0.4353 | 0.3968 | 0.361 |
| Eval Loss | 0.4359 | 0.4235 | 0.6291 | 0.5449 |
| Accuracy | 88.95% | 89.12% | 84.3683 % | 86.52% |
|  |  |  |  |  |

```bash
Running full evaluation of test set for: FULL_DATA_lr_2e-06_batch_64_epochs_40_weight_decay_0.3_warmup_steps_1000_max_grad_norm_1.0
Length of test_heavy: 134418
Length of test_light: 134418
Length of test_labels: 134418
Test set results: {'eval_loss': 0.7342624664306641, 'eval_accuracy': 0.5, 'eval_f1': 0.6666666666666666, 'eval_precision': 0.5, 'eval_recall': 1.0, 'eval_runtime': 1436.6958, 'eval_samples_per_second': 93.561, 'eval_steps_per_second': 1.462}
Predicted pairing for heavy: EVQLVESGGGLVKPGGSLRLSCAASGFTFSSYSMNWVRQAPGKGLEWVSSISSSSSYIYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARDEGVPMVTPPYYYYYMDVWGKGTTVTVSS and light: SSELTQDPAVSVALGQTVRITCQGDSLRSYYASWYQQKPGQAPVLVIYGKNNRPSGIPDRFSGSSSGNTASLTITGAQAEDEADYYCNSRDSSGNHWVFGGGTKLTVL: paired
Predicted pairing for heavy: EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARDLSGRVPEFDYWGQGTLVTVSS and light: SYELTQPPSVSVSPGQTASITCSGDKLGDKYACWYQQKPGQSPVLVIYQDSKRPSGIPERFSGSNSGNTATLTISGTQAMDEADYYCQAWDSSTVVFGGGTKLTVL: paired
Predicted pairing for heavy: QVQLVQSGAEVKKPGSSVRVSCKASGGTFSRYAISWVRQAPGQGLEWMGGIVPTLGPANYAQKFQGRVTILADESTTTAYMELSSLRSDDTAMYYCATDRSGVFDYWGQGTLVTVSS and light: DIQMTQSPLSLSASMGDRVSLTCRASQSISVYLNWYQQKPGKAPKLLIYAAFSLQSGVPSRFSGSGSGTDFTLTISSLQPEDSATYYCQQSYSTPWTFGQGTKVEIK: paired
Predicted pairing for heavy: QVQLVQSGADEKKPGASVKVSCKASGFTFISYYFHWVRQAPGHGLEWMGVINPNGGRASYAQKFQGRITMTTDTSTSTVYMELSSLRSEDTAVYYCARENRVWGQGTRVTVSS and light: DIQMTQSPSSLSASVGDSVTITCQASQDINNYLNWFQQKPGEAPKLLIYGASNLETGVPSRFSGGGSGTDFTFTITSLQPEDIATYYCQQYDNLFSFGPGTKVDFK: paired
Predicted pairing for heavy: QVQLVESGGGVVQPGKSLRLSCAASRFTFSNYGMHWVRQAPGKGLEWVAVVSYDETYKYYADSVKGRFTISRDNSKNTLYLQMDSLRADDSAVYYCAKEEYSARWYEEGYYFDYWGQGTLVTVSS and light: DIQMTQSPSTLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGSGSGTEFTLTISSLQPDDFATYYCQQYNSYWTFGQGTKVEIK: paired
```

```bash
Running full evaluation of test set for: adapters_FULL_data_lr_2e-06_batch_64_epochs_10_weight_decay_0.3_warmup_steps_1000_max_grad_norm1.0
Length of test_heavy: 134418
Length of test_light: 134418
Length of test_labels: 134418
Test set results: {'eval_loss': 0.7002736330032349, 'eval_accuracy': 0.5, 'eval_f1': 0.6666666666666666, 'eval_precision': 0.5, 'eval_recall': 1.0, 'eval_runtime': 1435.9076, 'eval_samples_per_second': 93.612, 'eval_steps_per_second': 1.463}
Predicted pairing for heavy: EVQLVESGGGLVKPGGSLRLSCAASGFTFSSYSMNWVRQAPGKGLEWVSSISSSSSYIYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARDEGVPMVTPPYYYYYMDVWGKGTTVTVSS and light: SSELTQDPAVSVALGQTVRITCQGDSLRSYYASWYQQKPGQAPVLVIYGKNNRPSGIPDRFSGSSSGNTASLTITGAQAEDEADYYCNSRDSSGNHWVFGGGTKLTVL: paired
Predicted pairing for heavy: EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARDLSGRVPEFDYWGQGTLVTVSS and light: SYELTQPPSVSVSPGQTASITCSGDKLGDKYACWYQQKPGQSPVLVIYQDSKRPSGIPERFSGSNSGNTATLTISGTQAMDEADYYCQAWDSSTVVFGGGTKLTVL: paired
Predicted pairing for heavy: QVQLVQSGAEVKKPGSSVRVSCKASGGTFSRYAISWVRQAPGQGLEWMGGIVPTLGPANYAQKFQGRVTILADESTTTAYMELSSLRSDDTAMYYCATDRSGVFDYWGQGTLVTVSS and light: DIQMTQSPLSLSASMGDRVSLTCRASQSISVYLNWYQQKPGKAPKLLIYAAFSLQSGVPSRFSGSGSGTDFTLTISSLQPEDSATYYCQQSYSTPWTFGQGTKVEIK: paired
Predicted pairing for heavy: QVQLVQSGADEKKPGASVKVSCKASGFTFISYYFHWVRQAPGHGLEWMGVINPNGGRASYAQKFQGRITMTTDTSTSTVYMELSSLRSEDTAVYYCARENRVWGQGTRVTVSS and light: DIQMTQSPSSLSASVGDSVTITCQASQDINNYLNWFQQKPGEAPKLLIYGASNLETGVPSRFSGGGSGTDFTFTITSLQPEDIATYYCQQYDNLFSFGPGTKVDFK: paired
Predicted pairing for heavy: QVQLVESGGGVVQPGKSLRLSCAASRFTFSNYGMHWVRQAPGKGLEWVAVVSYDETYKYYADSVKGRFTISRDNSKNTLYLQMDSLRADDSAVYYCAKEEYSARWYEEGYYFDYWGQGTLVTVSS and light: DIQMTQSPSTLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGSGSGTEFTLTISSLQPDDFATYYCQQYNSYWTFGQGTKVEIK: paired

```

```bash
Running full evaluation of test set for: FULL_DATA_lr_2e-05_batch_64_epochs_10_weight_decay_0.1_warmup_steps_500_max_grad_norm_1.0
Length of test_heavy: 134418
Length of test_light: 134418
Length of test_labels: 134418
Test set results: {'eval_loss': 0.849236011505127, 'eval_accuracy': 0.5, 'eval_f1': 0.0, 'eval_precision': 1.0, 'eval_recall': 0.0, 'eval_runtime': 1427.6909, 'eval_samples_per_second': 94.151, 'eval_steps_per_second': 1.472}
Predicted pairing for heavy: EVQLVESGGGLVKPGGSLRLSCAASGFTFSSYSMNWVRQAPGKGLEWVSSISSSSSYIYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARDEGVPMVTPPYYYYYMDVWGKGTTVTVSS and light: SSELTQDPAVSVALGQTVRITCQGDSLRSYYASWYQQKPGQAPVLVIYGKNNRPSGIPDRFSGSSSGNTASLTITGAQAEDEADYYCNSRDSSGNHWVFGGGTKLTVL: not paired
Predicted pairing for heavy: EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARDLSGRVPEFDYWGQGTLVTVSS and light: SYELTQPPSVSVSPGQTASITCSGDKLGDKYACWYQQKPGQSPVLVIYQDSKRPSGIPERFSGSNSGNTATLTISGTQAMDEADYYCQAWDSSTVVFGGGTKLTVL: not paired
Predicted pairing for heavy: QVQLVQSGAEVKKPGSSVRVSCKASGGTFSRYAISWVRQAPGQGLEWMGGIVPTLGPANYAQKFQGRVTILADESTTTAYMELSSLRSDDTAMYYCATDRSGVFDYWGQGTLVTVSS and light: DIQMTQSPLSLSASMGDRVSLTCRASQSISVYLNWYQQKPGKAPKLLIYAAFSLQSGVPSRFSGSGSGTDFTLTISSLQPEDSATYYCQQSYSTPWTFGQGTKVEIK: not paired
Predicted pairing for heavy: QVQLVQSGADEKKPGASVKVSCKASGFTFISYYFHWVRQAPGHGLEWMGVINPNGGRASYAQKFQGRITMTTDTSTSTVYMELSSLRSEDTAVYYCARENRVWGQGTRVTVSS and light: DIQMTQSPSSLSASVGDSVTITCQASQDINNYLNWFQQKPGEAPKLLIYGASNLETGVPSRFSGGGSGTDFTFTITSLQPEDIATYYCQQYDNLFSFGPGTKVDFK: not paired
Predicted pairing for heavy: QVQLVESGGGVVQPGKSLRLSCAASRFTFSNYGMHWVRQAPGKGLEWVAVVSYDETYKYYADSVKGRFTISRDNSKNTLYLQMDSLRADDSAVYYCAKEEYSARWYEEGYYFDYWGQGTLVTVSS and light: DIQMTQSPSTLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGSGSGTEFTLTISSLQPDDFATYYCQQYNSYWTFGQGTKVEIK: not paired

```

### Alternative Approach for classification task? SVM: Workflow for Classifying Paired and Non-Paired Antibody Chains

### 1. **Dataset Structure**

You will have pairs of sequences, each consisting of:

- A heavy chain sequence.
- A light chain sequence.
- A binary label: `1` (paired) or `0` (not paired).

Example dataset:

| Heavy Chain Sequence | Light Chain Sequence | Label |
| --- | --- | --- |
| EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIK... | SYELTQPPSVSVSPGQTASITCSGDKLGDKYACWYQQKPGQSPVLVIYQD... | 1 |
| QVQLVQSGAEVKKPGSSVRVSCKASGGTFSRYAISWVRQAPGQGLEWMGGI... | DIQMTQSPLSLSASMGDRVSLTCRASQSISVYLNWYQQKPGKAPKLLIYA... | 0 |

Each row in the dataset consists of a heavy chain, a light chain, and a label indicating whether they are paired.

### 2. **Feature Engineering**

Since SVM models require numerical inputs, you need to convert the heavy and light chain sequences into meaningful numerical features that can capture the relationship between the two chains. Here are a few approaches for feature extraction:

### a. **Pairwise Amino Acid Composition (AAC)**

You can compute the amino acid composition for each chain separately, then concatenate them to form a single feature vector for the heavy and light chain pair. This would give you a feature vector of length 40 (20 features for the heavy chain and 20 for the light chain).

### b. **Pairwise Dipeptide Composition**

Similarly, you can calculate the dipeptide composition (pairwise frequency of amino acids) for both the heavy and light chains and concatenate the results. This would give a feature vector of length 800 (400 features for the heavy chain and 400 for the light chain).

### c. **Sequence Alignment or Similarity Scores**

One advanced feature could be the **sequence alignment score** (e.g., using BLOSUM62 or PAM matrices) between the heavy and light chains to capture how well they are paired at a structural level. You could generate a similarity score as a feature.

### d. **Physicochemical Properties**

For each chain, calculate properties like:

- Hydrophobicity
- Molecular weight
- Isoelectric point
- Polarizability

Concatenate these properties for the heavy and light chains to form a feature vector.

You can choose one or a combination of the above methods to extract features from the sequences.

### Example of Feature Extraction Code (AAC in Python using `biopython`):

```python
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def calculate_aac(sequence):
    """Calculate the amino acid composition (AAC) of a given protein sequence."""
    analysed_seq = ProteinAnalysis(sequence)
    return analysed_seq.get_amino_acids_percent()

# Example: Compute AAC for heavy and light chains and concatenate them
heavy_chain = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIK..."
light_chain = "SYELTQPPSVSVSPGQTASITCSGDKLGDKYACWYQQKPGQSPVLVIYQDS..."

heavy_aac = calculate_aac(heavy_chain)
light_aac = calculate_aac(light_chain)

# Combine both feature vectors
features = list(heavy_aac.values()) + list(light_aac.values())
print(features)  # This can be fed into the SVM model

```

### 3. **Preparing the Dataset**

Once the feature extraction is complete for each pair of heavy and light chains, you should have a matrix of features (X) and corresponding labels (y).

- **X**: The feature matrix, where each row is a pair of sequences with extracted features.
- **y**: The binary label for each pair (1 = paired, 0 = not paired).

### 4. **Train/Test Split**

You’ll need to split your data into training and test sets to train your SVM model and evaluate its performance. You can use `train_test_split` from `scikit-learn`.

```python
from sklearn.model_selection import train_test_split

# X is the feature matrix, y is the label (paired or not)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

### 5. **Training the SVM Model**

Now you can train an SVM model using `scikit-learn`:

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM classifier
svm = SVC(kernel='linear', C=1)  # You can also experiment with 'rbf' or 'poly' kernels
svm.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Detailed classification report
print(classification_report(y_test, y_pred))

```

### 6. **Evaluating the Model**

- **Accuracy**: A simple metric showing the percentage of correctly classified pairs.
- **Precision, Recall, F1-Score**: These metrics help you understand how well the model is performing, especially in handling false positives and false negatives.

### 7. **Hyperparameter Tuning (Optional)**

To improve the model’s performance, you can tune the SVM hyperparameters such as the regularization parameter `C` and the kernel type (linear, polynomial, or RBF) using cross-validation and grid search.

```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly']}

# Perform grid search with cross-validation
grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5)
grid.fit(X_train_scaled, y_train)

# Best parameters and best model evaluation
print(f"Best Parameters: {grid.best_params_}")
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_best))

```

# U-MAP of individual MLM models with heavy / light seqs unpaired

.schema of /ibmm_data2/oas_database/OAS_heavy_part1.db (same for part2):

```bash
CREATE TABLE IF NOT EXISTS "human_unpaired_novaccine_nodisease_heavy"(
"sequence" TEXT, "locus" TEXT, "stop_codon" TEXT, "vj_in_frame" TEXT,
 "v_frameshift" TEXT, "productive" TEXT, "rev_comp" TEXT, "complete_vdj" TEXT,
 "v_call" TEXT, "d_call" TEXT, "j_call" TEXT, "sequence_alignment" TEXT,
 "germline_alignment" TEXT, "sequence_alignment_aa" TEXT, "germline_alignment_aa" TEXT, "v_alignment_start" TEXT,
 "v_alignment_end" TEXT, "d_alignment_start" TEXT, "d_alignment_end" TEXT, "j_alignment_start" TEXT,
 "j_alignment_end" TEXT, "v_sequence_alignment" TEXT, "v_sequence_alignment_aa" TEXT, "v_germline_alignment" TEXT,
 "v_germline_alignment_aa" TEXT, "d_sequence_alignment" TEXT, "d_sequence_alignment_aa" TEXT, "d_germline_alignment" TEXT,
 "d_germline_alignment_aa" TEXT, "j_sequence_alignment" TEXT, "j_sequence_alignment_aa" TEXT, "j_germline_alignment" TEXT,
 "j_germline_alignment_aa" TEXT, "fwr1" TEXT, "fwr1_aa" TEXT, "cdr1" TEXT,
 "cdr1_aa" TEXT, "fwr2" TEXT, "fwr2_aa" TEXT, "cdr2" TEXT,
 "cdr2_aa" TEXT, "fwr3" TEXT, "fwr3_aa" TEXT, "fwr4" TEXT,
 "fwr4_aa" TEXT, "cdr3" TEXT, "cdr3_aa" TEXT, "junction" TEXT,
 "junction_length" TEXT, "junction_aa" TEXT, "junction_aa_length" TEXT, "v_score" TEXT,
 "d_score" TEXT, "j_score" TEXT, "v_cigar" TEXT, "d_cigar" TEXT,
 "j_cigar" TEXT, "v_support" TEXT, "d_support" TEXT, "j_support" TEXT,
 "v_identity" TEXT, "d_identity" TEXT, "j_identity" TEXT, "v_sequence_start" TEXT,
 "v_sequence_end" TEXT, "v_germline_start" TEXT, "v_germline_end" TEXT, "d_sequence_start" TEXT,
 "d_sequence_end" TEXT, "d_germline_start" TEXT, "d_germline_end" TEXT, "j_sequence_start" TEXT,
 "j_sequence_end" TEXT, "j_germline_start" TEXT, "j_germline_end" TEXT, "fwr1_start" TEXT,
 "fwr1_end" TEXT, "cdr1_start" TEXT, "cdr1_end" TEXT, "fwr2_start" TEXT,
 "fwr2_end" TEXT, "cdr2_start" TEXT, "cdr2_end" TEXT, "fwr3_start" TEXT,
 "fwr3_end" TEXT, "fwr4_start" TEXT, "fwr4_end" TEXT, "cdr3_start" TEXT,
 "cdr3_end" TEXT, "np1" TEXT, "np1_length" TEXT, "np2" TEXT,
 "np2_length" TEXT, "c_region" TEXT, "Redundancy" TEXT, "ANARCI_numbering" TEXT,
 "ANARCI_status" TEXT, "Age" TEXT, "Author" TEXT, "BSource" TEXT,
 "BType" TEXT, "Chain" TEXT, "Disease" TEXT, "Isotype" TEXT,
 "Link" TEXT, "Longitudinal" TEXT, "Run" TEXT, "Species" TEXT,
 "Subject" TEXT, "Total sequences" TEXT, "Unique sequences" TEXT, "Vaccine" TEXT,
 "download_date" TEXT);
```

```bash
COLUMNS="locus, v_call, d_call, j_call, sequence_alignment_aa, fwr1_aa, cdr1_aa, fwr2_aa, cdr2_aa, fwr3_aa, fwr4_aa, cdr3_aa, BType, Disease, Age, Species, Vaccine"
```

used the script

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/src/extract_columns_from_sqlite_db.sh
```

.schema of  /ibmm_data2/oas_database/OAS_light.db:

```bash
CREATE TABLE IF NOT EXISTS "human_unpaired_novaccine_nodisease_light"(
"sequence" TEXT, "locus" TEXT, "stop_codon" TEXT, "vj_in_frame" TEXT,
 "v_frameshift" TEXT, "productive" TEXT, "rev_comp" TEXT, "complete_vdj" TEXT,
 "v_call" TEXT, "d_call" TEXT, "j_call" TEXT, "sequence_alignment" TEXT,
 "germline_alignment" TEXT, "sequence_alignment_aa" TEXT, "germline_alignment_aa" TEXT, "v_alignment_start" TEXT,
 "v_alignment_end" TEXT, "d_alignment_start" TEXT, "d_alignment_end" TEXT, "j_alignment_start" TEXT,
 "j_alignment_end" TEXT, "v_sequence_alignment" TEXT, "v_sequence_alignment_aa" TEXT, "v_germline_alignment" TEXT,
 "v_germline_alignment_aa" TEXT, "d_sequence_alignment" TEXT, "d_sequence_alignment_aa" TEXT, "d_germline_alignment" TEXT,
 "d_germline_alignment_aa" TEXT, "j_sequence_alignment" TEXT, "j_sequence_alignment_aa" TEXT, "j_germline_alignment" TEXT,
 "j_germline_alignment_aa" TEXT, "fwr1" TEXT, "fwr1_aa" TEXT, "cdr1" TEXT,
 "cdr1_aa" TEXT, "fwr2" TEXT, "fwr2_aa" TEXT, "cdr2" TEXT,
 "cdr2_aa" TEXT, "fwr3" TEXT, "fwr3_aa" TEXT, "fwr4" TEXT,
 "fwr4_aa" TEXT, "cdr3" TEXT, "cdr3_aa" TEXT, "junction" TEXT,
 "junction_length" TEXT, "junction_aa" TEXT, "junction_aa_length" TEXT, "v_score" TEXT,
 "d_score" TEXT, "j_score" TEXT, "v_cigar" TEXT, "d_cigar" TEXT,
 "j_cigar" TEXT, "v_support" TEXT, "d_support" TEXT, "j_support" TEXT,
 "v_identity" TEXT, "d_identity" TEXT, "j_identity" TEXT, "v_sequence_start" TEXT,
 "v_sequence_end" TEXT, "v_germline_start" TEXT, "v_germline_end" TEXT, "d_sequence_start" TEXT,
 "d_sequence_end" TEXT, "d_germline_start" TEXT, "d_germline_end" TEXT, "j_sequence_start" TEXT,
 "j_sequence_end" TEXT, "j_germline_start" TEXT, "j_germline_end" TEXT, "fwr1_start" TEXT,
 "fwr1_end" TEXT, "cdr1_start" TEXT, "cdr1_end" TEXT, "fwr2_start" TEXT,
 "fwr2_end" TEXT, "cdr2_start" TEXT, "cdr2_end" TEXT, "fwr3_start" TEXT,
 "fwr3_end" TEXT, "fwr4_start" TEXT, "fwr4_end" TEXT, "cdr3_start" TEXT,
 "cdr3_end" TEXT, "np1" TEXT, "np1_length" TEXT, "np2" TEXT,
 "np2_length" TEXT, "c_region" TEXT, "Redundancy" TEXT, "ANARCI_numbering" TEXT,
 "ANARCI_status" TEXT, "Age" TEXT, "Author" TEXT, "BSource" TEXT,
 "BType" TEXT, "Chain" TEXT, "Disease" TEXT, "Isotype" TEXT,
 "Link" TEXT, "Longitudinal" TEXT, "Run" TEXT, "Species" TEXT,
 "Subject" TEXT, "Total sequences" TEXT, "Unique sequences" TEXT, "Vaccine" TEXT,
 "download_date" TEXT);
```

```bash
COLUMNS="locus, v_call, d_call, j_call, sequence_alignment_aa, fwr1_aa, cdr1_aa, fwr2_aa, cdr2_aa, fwr3_aa, fwr4_aa, cdr3_aa, BType, Disease, Age, Species, Vaccine"
```

test set light sequences 

```bash
/ibmm_data2/oas_database/paired_lea_tmp/light_model/data/train_test_val_datasets/light_all_seqs_test_no_ids.txt
```

output file extracted test set sequences

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/extracted_seqs_light_model_test_set.txt
```

remove duplicates

```bash
awk -F ',' '!seen[$5]++' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/extracted_seqs_light_model_test_set.txt > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/extracted_seqs_light_model_test_set_no_dupl.txt

```

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/extracted_seqs_light_model_test_set_no_dupl.txt
```

```bash
locus,v_call,d_call,j_call,sequence_alignment_aa,fwr1_aa,cdr1_aa,fwr2_aa,cdr2_aa,fwr3_aa,fwr4_aa,cdr3_aa,BType,Disease,Age,Species,Vaccine
```

# Summary of Results different decoding strategies heavy2light translation with global alignment results:

| Decoding Strategy | Run Name | Epochs | Beam Size | Mean BLOSUM | Mean BLOSUM with alignment | Median BLOSUM with alignment | Mean similarity with global alignment | Median similarity with global alignment | Mean similarity | Mean perplexity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diverse beam search | full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1 | 60 | 5 | 109.5 | 325 | 322 | 63% | 61.1% | 33.2% | 1.8 |
| diverse beam search | full_Diverse_beam_search_5_decoding_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1 | 50 | 5 | 107.55 | 324.9 | 325 | 62.863 | 61.6% | 32.94% | 2.01 |
| diverse beam search | full_Diverse_beam_search_5_decoding_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1 | 10 | 5 | 107.6 | 324.94 | 323 | 62.92591 | 61.46789 | 32.95% | 2.15 |
| diverse beam search | full_diverse_beam_search_2_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1 | 10 | 2 | 108.25 | 324.55 | 323 | 62.89% | 61.47% | 33.06% | 2.17 |
| Contrastive Search  | full_contrastive_search_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_40_lr_0.0001_wd_0.1 | 40 | - | 97.34 | 323.3 | 325 | 62.43% | 61.61% | 31.35% | 1.98 |
| Beam Search  |  save_adapter_FULL_data_temperature_0.5_tests_max_length_150_early_stopping_true_heavy2light_with_adapters_batch_size_64_epochs_40_lr_0.0001_weight_decay_0.1 | 40 | 3 (number of beams) | 92.78 | 261.3103 | 249 | 53.35381 | 50 | 30.32% | 2.19 |

| Decoding Strategy | Epochs | Beam Size | Mean BLOSUM with alignment | Median BLOSUM with alignment | Mean similarity with global alignment | Median similarity with global alignment | Mean perplexity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| diverse beam search | 60 | 5 | 325 | 322 | 63% | 61.1% | 1.8 |
| diverse beam search | 50 | 5 | 324.9 | 325 | 62.863 | 61.6% | 2.01 |
| diverse beam search | 10 | 5 | 324.94 | 323 | 62.92591 | 61.46789 | 2.15 |
| diverse beam search | 10 | 2 | 324.55 | 323 | 62.89% | 61.47% | 2.17 |
| Contrastive Search  | 40 | - | 323.3 | 325 | 62.43% | 61.61% | 1.98 |
| Beam Search  | 40 | 3 (number of beams) | 261.3103 | 249 | 53.35381 | 50 | 2.19 |

•	**Beam Search** focuses on maximizing probability, often resulting in safer, more generic outputs.

•	**Diverse Beam Search** adds diversity by penalizing similarity across beams.

•	**Contrastive Search:** strikes a balance between fluency and novelty by promoting distinctiveness in the generated text.

# Remaking Kappa Lambda Gene Analysis with GLOBAL ALIGNMENT

used model with run name: full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1

output file with the global alignment results:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/kappa_lambda_analysis/extracted_data_alignment_heavy2light_60_epochs_beam_search_127845.csv
```

column names: 

"sequence_alignment_aa_light","generated_sequence_light","BLOSUM_score","similarity","perplexity","calculated_blosum","calculated_similarity"

used script: /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/kappa_lambda_analysis/kappa_lambda_subtypes_analysis.ipynb

| Subtype | Calculated Blosum |  |  |  | Calculated Similarity |  |  |  | Perplexity |  |  |  | Subtype Size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Mean | Median | Min | Max | Mean | Median | Min | Max | Mean | Median | Min | Max | Size |
| IGHV3 | 39.0 | 30.0 | 23 | 73 | 31.66581336775645 | 31.418085731062803 | 30.7692307692308 | 33.0578512396694 | 37.53665494918825 | 37.5018405914307 | 29.9767284393311 | 45.1662101745605 | 4 |
| IGHV4 | 31.0 | 31.0 | 31 | 31 | 31.4516129032258 | 31.4516129032258 | 31.4516129032258 | 31.4516129032258 | 33.1371688842773 | 33.1371688842773 | 33.1371688842773 | 33.1371688842773 | 1 |
| IGKV1 | 357.217578405164 | 375.0 | -118 | 556 | 68.16099239507976 | 67.5925925925926 | 26.5486725663717 | 100.0 | 1.8969033790037328 | 1.77949285507202 | 1.26024842262268 | 43.854061126709 | 15337 |
| IGKV1/OR10 | 148.0 | 148.0 | 148 | 148 | 45.2830188679245 | 45.2830188679245 | 45.2830188679245 | 45.2830188679245 | 3.47773241996765 | 3.47773241996765 | 3.47773241996765 | 3.47773241996765 | 1 |
| IGKV1/OR2 | 435.0 | 435.0 | 435 | 435 | 83.0188679245283 | 83.0188679245283 | 83.0188679245283 | 83.0188679245283 | 2.03114461898804 | 2.03114461898804 | 2.03114461898804 | 2.03114461898804 | 1 |
| IGKV1D | 351.6896551724138 | 377.0 | 8 | 497 | 67.08290494300607 | 67.8899082568807 | 40.3669724770642 | 92.4528301886792 | 1.794725377258213 | 1.66050255298615 | 1.27454233169556 | 24.9570751190186 | 435 |
| IGKV2 | 283.58861834654584 | 300.0 | -118 | 565 | 55.43348855982863 | 57.1428571428571 | 37.1681415929204 | 95.5357142857143 | 1.7266618775484381 | 1.55056530237198 | 1.25970184803009 | 44.0683135986328 | 3532 |
| IGKV2D | 282.546835443038 | 297.0 | 6 | 516 | 55.40585917608005 | 57.1428571428571 | 38.9380530973451 | 86.6071428571429 | 1.8151850467995752 | 1.63188314437866 | 1.31514930725098 | 31.3265476226807 | 395 |
| IGKV3 | 364.9057468255007 | 378.0 | -106 | 556 | 69.52731476466566 | 69.1588785046729 | 33.0 | 100.0 | 1.9468552131649086 | 1.76577198505402 | 1.23484826087952 | 46.1852378845215 | 7639 |
| IGKV3/OR2 | 336.0 | 344.5 | 234 | 421 | 68.02845720870192 | 68.3454482519903 | 49.1071428571429 | 86.3157894736842 | 2.443458914756775 | 2.615604639053345 | 1.67163586616516 | 2.87099051475525 | 4 |
| IGKV3D | 356.55598455598454 | 375.0 | 140 | 529 | 68.05739359880934 | 68.2242990654206 | 36.697247706422 | 96.2962962962963 | 2.039557228677521 | 1.95140099525452 | 1.29369747638702 | 10.4619483947754 | 259 |
| IGKV4 | 309.1385068762279 | 333.0 | -88 | 581 | 61.327901364355455 | 64.2857142857143 | 35.3448275862069 | 98.2300884955752 | 1.9627242472410673 | 1.70947229862213 | 1.30272448062897 | 35.76025390625 | 1018 |
| IGKV5 | 252.72727272727272 | 289.0 | 141 | 351 | 49.43470794181 | 53.2710280373832 | 33.3333333333333 | 61.6822429906542 | 3.601048066399314 | 3.29003000259399 | 2.56631302833557 | 9.64686298370361 | 55 |
| IGKV6 | 297.0583153347732 | 333.0 | 145 | 391 | 59.4418662379835 | 63.5514018691589 | 40.9090909090909 | 71.9626168224299 | 1.685407367174878 | 1.5714727640152 | 1.32166576385498 | 3.66992211341858 | 463 |
| IGKV6D | 296.4855072463768 | 337.0 | 170 | 387 | 59.35455219658042 | 63.8888888888889 | 41.5929203539823 | 70.0934579439252 | 1.6998955097751345 | 1.538156390190125 | 1.35501098632812 | 4.11395215988159 | 138 |
| IGLV1 | 303.05360956856043 | 249.0 | -124 | 573 | 59.82280888712472 | 52.2522522522523 | 36.036036036036 | 98.1981981981982 | 1.9223815071333274 | 1.754107415676115 | 1.29373645782471 | 37.6474227905273 | 4682 |
| IGLV10 | 259.03320683111957 | 312.0 | -121 | 557 | 52.47838651236252 | 58.1818181818182 | 25.6880733944954 | 94.5454545454545 | 2.0428326727090793 | 1.8589757680892949 | 1.47560703754425 | 33.8190841674805 | 1054 |
| IGLV2 | 354.1783487625165 | 363.0 | -106 | 574 | 66.68284140704169 | 65.7657657657658 | 31.1926605504587 | 100.0 | 1.931942092140388 | 1.76536846160889 | 1.3056218624115 | 40.2229423522949 | 5293 |
| IGLV3 | 265.0635241301908 | 246.0 | -132 | 560 | 53.45651183092402 | 50.9090909090909 | 27.0 | 99.0825688073395 | 1.910066246531494 | 1.778349101543425 | 1.2342312335968 | 50.2256126403809 | 8910 |
| IGLV4 | 210.8562091503268 | 211.0 | -74 | 532 | 49.20979376870044 | 49.1228070175439 | 31.7757009345794 | 90.0900900900901 | 2.105661759384317 | 1.819890797138215 | 1.33181214332581 | 34.7513198852539 | 612 |
| IGLV5 | 228.31034482758622 | 237.0 | 128 | 587 | 52.792658408919834 | 53.448275862069 | 41.025641025641 | 96.5217391304348 | 2.2877695411838967 | 2.17309641838074 | 1.55946803092957 | 6.63831233978271 | 377 |
| IGLV6 | 278.12074303405575 | 312.0 | 22 | 387 | 56.28985831982575 | 59.8214285714286 | 40.1785714285714 | 69.0265486725664 | 1.6901337200827642 | 1.537551045417785 | 1.31663620471954 | 22.7751502990723 | 1292 |
| IGLV7 | 236.83401221995928 | 239.0 | -14 | 542 | 48.736199696800995 | 48.2142857142857 | 33.6448598130841 | 95.4128440366973 | 2.0516732769682546 | 1.85242086648941 | 1.40236115455627 | 48.1054153442383 | 982 |
| IGLV8 | 259.3545751633987 | 284.0 | -113 | 378 | 51.1933869664186 | 52.7272727272727 | 25.0 | 67.2727272727273 | 1.9125142101368873 | 1.677620947360995 | 1.3618575334549 | 35.2711639404297 | 612 |
| IGLV9 | 174.11363636363637 | 180.0 | 103 | 218 | 44.148081936519155 | 44.117231638418104 | 35.042735042735 | 50.4347826086956 | 3.259393147446893 | 2.8065845966339102 | 1.98493111133575 | 6.19665622711182 | 44 |

| Subtype | Calculated Blosum |  | Calculated Similarity |  | Perplexity |  | Subtype Size |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | Mean | Median | Mean | Median | Mean | Median |  |
| IGKV1 | 357.217578405164 | 375.0 | 68.16099239507976 | 67.5925925925926 | 1.8969033790037328 | 1.77949285507202 | 15337 |
| IGKV1D | 351.6896551724138 | 377.0 | 67.08290494300607 | 67.8899082568807 | 1.794725377258213 | 1.66050255298615 | 435 |
| IGKV2 | 283.58861834654584 | 300.0 | 55.43348855982863 | 57.1428571428571 | 1.7266618775484381 | 1.55056530237198 | 3532 |
| IGKV2D | 282.546835443038 | 297.0 | 55.40585917608005 | 57.1428571428571 | 1.8151850467995752 | 1.63188314437866 | 395 |
| IGKV3 | 364.9057468255007 | 378.0 | 69.52731476466566 | 69.1588785046729 | 1.9468552131649086 | 1.76577198505402 | 7639 |
| IGKV3D | 356.55598455598454 | 375.0 | 68.05739359880934 | 68.2242990654206 | 2.039557228677521 | 1.95140099525452 | 259 |
| IGKV4 | 309.1385068762279 | 333.0 | 61.327901364355455 | 64.2857142857143 | 1.9627242472410673 | 1.70947229862213 | 1018 |
| IGKV5 | 252.72727272727272 | 289.0 | 49.43470794181 | 53.2710280373832 | 3.601048066399314 | 3.29003000259399 | 55 |
| IGKV6 | 297.0583153347732 | 333.0 | 59.4418662379835 | 63.5514018691589 | 1.685407367174878 | 1.5714727640152 | 463 |
| IGKV6D | 296.4855072463768 | 337.0 | 59.35455219658042 | 63.8888888888889 | 1.6998955097751345 | 1.538156390190125 | 138 |
| IGLV1 | 303.05360956856043 | 249.0 | 59.82280888712472 | 52.2522522522523 | 1.9223815071333274 | 1.754107415676115 | 4682 |
| IGLV10 | 259.03320683111957 | 312.0 | 52.47838651236252 | 58.1818181818182 | 2.0428326727090793 | 1.8589757680892949 | 1054 |
| IGLV2 | 354.1783487625165 | 363.0 | 66.68284140704169 | 65.7657657657658 | 1.931942092140388 | 1.76536846160889 | 5293 |
| IGLV3 | 265.0635241301908 | 246.0 | 53.45651183092402 | 50.9090909090909 | 1.910066246531494 | 1.778349101543425 | 8910 |
| IGLV4 | 210.8562091503268 | 211.0 | 49.20979376870044 | 49.1228070175439 | 2.105661759384317 | 1.819890797138215 | 612 |
| IGLV5 | 228.31034482758622 | 237.0 | 52.792658408919834 | 53.448275862069 | 2.2877695411838967 | 2.17309641838074 | 377 |
| IGLV6 | 278.12074303405575 | 312.0 | 56.28985831982575 | 59.8214285714286 | 1.6901337200827642 | 1.537551045417785 | 1292 |
| IGLV7 | 236.83401221995928 | 239.0 | 48.736199696800995 | 48.2142857142857 | 2.0516732769682546 | 1.85242086648941 | 982 |
| IGLV8 | 259.3545751633987 | 284.0 | 51.1933869664186 | 52.7272727272727 | 1.9125142101368873 | 1.677620947360995 | 612 |
| IGLV9 | 174.11363636363637 | 180.0 | 44.148081936519155 | 44.117231638418104 | 3.259393147446893 | 2.8065845966339102 | 44 |

| Locus | Calculated Blosum |  |  |  | Calculated Similarity |  |  |  | Perplexity |  |  |  | Locus Size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Mean | Median | Min | Max | Mean | Median | Min | Max | Mean | Median | Min | Max | Size |
| IGK | 346.1323906137924 | 358.0 | -118 | 581 | 66.3405493364566 | 65.4867256637168 | 26.5486725663717 | 100.0 | 1.889390157285958 | 1.74144816398621 | 1.23484826087952 | 46.1852378845215 | 29277 |
| IGL | 289.2826724788331 | 250.0 | -132 | 587 | 57.36146994975642 | 51.8181818181818 | 25.0 | 100.0 | 1.9306571228484795 | 1.76997798681259 | 1.2342312335968 | 50.2256126403809 | 23858 |

| Locus | Calculated Blosum |  | Calculated Similarity |  | Perplexity |  | Locus Size |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | Mean | Median | Mean | Median | Mean | Median | Size |
| IGK | 346.1323906137924 | 358.0 | 66.3405493364566 | 65.4867256637168 | 1.889390157285958 | 1.74144816398621 | 29277 |
| IGL | 289.2826724788331 | 250.0 | 57.36146994975642 | 51.8181818181818 | 1.9306571228484795 | 1.76997798681259 | 23858 |

LDA embeddings as input, label with j genes

path to used script

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/test_classification_models.py
```

used models:

1. FULL_DATA_lr_2e-06_batch_64_epochs_40_weight_decay_0.3_warmup_steps_1000_max_grad_norm_1.0

adapters and model path:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/checkpoints_light_heavy_classification/FULL_DATA_lr_2e-06_batch_64_epochs_40_weight_decay_0.3_warmup_steps_1000_max_grad_norm_1.0
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/checkpoints_light_heavy_classification/FULL_DATA_lr_2e-06_batch_64_epochs_40_weight_decay_0.3_warmup_steps_1000_max_grad_norm_1.0/checkpoint-672080/class_adap
```

1. adapters_FULL_data_lr_2e-06_batch_64_epochs_10_weight_decay_0.3_warmup_steps_1000_max_grad_norm1.0

only adapters available

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/checkpoints_light_heavy_classification/adapters_FULL_data_lr_2e-06_batch_64_epochs_10_weight_decay_0.3_warmup_steps_1000_max_grad_norm1.0
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/checkpoints_light_heavy_classification/adapters_FULL_data_lr_2e-06_batch_64_epochs_10_weight_decay_0.3_warmup_steps_1000_max_grad_norm1.0/class_adap
```

1. FULL_DATA_lr_2e-05_batch_64_epochs_10_weight_decay_0.1_warmup_steps_500_max_grad_norm_1.0

adapter and model available

used adapters at epoch 3 bc of overfitting after

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/checkpoints_light_heavy_classification/FULL_DATA_lr_2e-05_batch_64_epochs_10_weight_decay_0.1_warmup_steps_500_max_grad_norm_1.0/checkpoint-50406
```

# IgBERT 100% Masking light Chain Generation results on full test set

```bash
output file: /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/igbert_mlm_FULL_144577.o
--- Averages Across All Sequences ---
Average Perplexity: 157.3379544786811
Average Global Alignment Similarity: 27.0515
Average Global Alignment BLOSUM Score: -22.69960868336086
Average 'Hard' Similarity: 7.24%
Average 'Hard' BLOSUM Score: -71.55635405972414
```

# full evaluation IgBERT2IgBERT

run name: FULL_data_cross_attention_with_adapters_batch_size_64_epochs_20_lr_0.0005_weight_decay_0.05

```bash
output file: /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/x_full_eval_IgBERT2IgBERT_127813.o
Average BLOSUM Score: -51.87956970048654
Average Similarity Percentage: 10.5584562033619%
Mean Perplexity: 16.85335922241211

> print(mean_blosum)
[1] -84.3587
> print(mean_similarity)
[1] 25.35582
> print(median_blosum)
[1] -80
> print(median_similarity)
[1] 25.4386
```

|  | Mean Similarity | Mean BLOSUM score | Perplexity |
| --- | --- | --- | --- |
| IgBERT2IgBERT | 0.2536 | -84.3587 | 16.85335 |
| IgBERT 100% Masking MLM | 0.2705 | -22.6996 | 157.3379 |

# Gap Analysis in Global Alignment Explanation of Approach

For the Gap analysis, Biopython (version 1.84)  and Python (version 3.10.11) was used. 

Explanation:

```bash
target            0 EIVLTQSPG-TLSLSPGE-AATLSCK--SS 26
                  0 ||||||||--||||||||-|-||||---|- 30
query             0 EIVLTQSP-ATLSLSPGERA-TLSC-RAS- 26

Total gaps: 8
Alignment length: 30
normalized gaps: 8 / 30 = 0.2667
Gap Density: 0.2667

#This was done for each region and the full test set 
```

Results:

| Region (column_name) | Average Normalized Gaps |
| --- | --- |
| FWR 1 (fwr1_aa) | 0.5156 |
| CDR 1 (cdr1_aa) | 0.6682 |
| FWR 2 (fwr2_aa) | 0.3943 |
| CDR 2 (cdr2_aa) | 0.6551 |
| FWR 3 (fwr3_aa) | 0.4144 |
| CDR 3 (cdr3_aa) | 0.7130 |
| FWR 4 (fwr4_aa_correct) | 0.4139 |

# MLM big/big conf Model evaluation

|  | Mean BLOSUM | Median BLOSUM | Mean Similarity | Median Similarity | Perplexity | Encoder | Decoder |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Heavy2Light model | 260.2032 | 248 | 0.5288 | 49.091 | 3.4277 | Heavy MLM model with big configuration | Light MLM with big configuration |

Big configuration: number of trainable parameters: 32’839’488 trainable params: 32’839’488 || all params: 204’963’673 || trainable%: 16.022101633590456

### 

| Locus | BLOSUM Mean | Similarity Mean | Perplexity Mean |
| --- | --- | --- | --- |
| κ (IGK) | 346.13 | 0.6634 | 1.89 |
| λ (IGL) | 289.28 | 0.5736 | 1.93 |

|  | Mean Similarity | Mean BLOSUM score | Perplexity |
| --- | --- | --- | --- |
| IgBERT2IgBERT | 0.2536 | -84.3587 | 16.85335 |
| IgBERT 100% Masking MLM | 0.2705 | -22.6996 | 157.3379 |

|  | Architecture | Base Model | Trainable Parameters | Parameters from Adapter | Task |
| --- | --- | --- | --- | --- | --- |
| Heavy MLM Model Unpaired | RoBERTa | Trained from Scratch | 13'150'745 | - | MLM |
| Light MLM Model Unpaired | RoBERTa | Trained from Scratch | 13'150'745 | - | MLM |
| NSP Model | BERT | ProtBert-BFD | 419'931'136 | - | MLM & NSP |
| Classification Model | BERT | IgBERT | 427'862'786 | 7'931'650 | Sequence Classification |
| Encoder-Decoder Model 'Heavy2Light' | RoBERTa | Encoder: Heavy MLM model Decoder: Light MLM model | 31'173'273 | 4'872'832 | Text Generation |
| Encoder-Decoder Model 'IgBERT2IgBERT' | BERT | Encoder: IgBERT Decoder: IgBERT | 985'701'790 | 145'837'440 | Text Generation |
| IgBERT MLM Model | BERT | IgBERT | 418'881'536 | - | MLM with 100% masking of light chain |

| **IgBERT** | **Val. Set** | **0.5723** | **0.7457** | **0.7072** | **0.6589** | **0.8589** |
| --- | --- | --- | --- | --- | --- | --- |
| IgBERT | Test Set | 0.5470 | 0.7512 | 0.7286 | 0.6935 | 0.8194 |