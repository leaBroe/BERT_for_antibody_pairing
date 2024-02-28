import csv
import sqlite3

# Explanation of the code: I accidentally added the header of the naive B cells csv (added the wrong file) and then deleted the specific row where the header was (row 154845)

# # Path to your SQLite database
# db_path = '/ibmm_data/oas_database/OAS.db'
# # Row number you want to query
# row_id = 154845

# # Connect to the SQLite database
# conn = sqlite3.connect(db_path)
# cursor = conn.cursor()

# # SQL statement to select the row with the specific ROWID
# query = f"SELECT * FROM healthy_paired WHERE ROWID = {row_id};"

# # Execute the query
# cursor.execute(query)

# # Fetch the result
# row = cursor.fetchone()

# # Check if the row exists and print it
# if row:
#     print(f"Values of row number {row_id}:", row)
# else:
#     print(f"Row number {row_id} does not exist in the table.")

# # Close the connection
# conn.close()

######## DELETE SPECIFIC ROW in table ########################################################

# # Path to your SQLite database
# db_path = '/ibmm_data/oas_database/OAS.db'

# # Connect to the SQLite database
# conn = sqlite3.connect(db_path)
# cursor = conn.cursor()

# # SQL statement to delete the row with the specific ROWID
# row_id = 154845  # The ROWID of the row you want to delete
# delete_query = f"DELETE FROM healthy_paired WHERE ROWID = {row_id};"

# # Execute the delete query
# cursor.execute(delete_query)

# # Commit the changes to the database
# conn.commit()

# # Check if the row was deleted successfully
# if cursor.rowcount == 1:
#     print(f"Row number {row_id} has been successfully deleted from the table.")
# else:
#     print(f"Row number {row_id} could not be found or deleted.")

# # Close the connection
# conn.close()


######## ADD CSV Files to healthy_paired table in  /ibmm_data/oas_database/OAS.db ########################################################


# Path to your SQLite database
db_path = '/ibmm_data/oas_database/OAS.db'
# Name of the table you're inserting data into
table_name = 'healthy_paired'
# Path to your CSV file without the header
# csv_file_path = '/ibmm_data/oas_database/paired/csv_files/no_headers/no_header_OAS_db_paired_healthy_naive_b_cells_2024-02-28.csv'
# csv_file_path = '/ibmm_data/oas_database/paired/csv_files/no_headers/no_header_OAS_db_paired_healthy_memory_b_cells_2024-02-28.csv'
# csv_file_path = '/ibmm_data/oas_database/paired/csv_files/no_headers/no_header_OAS_db_paired_healthy_plasma_b_cells_2024-02-28.csv'


# Connect to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get the number of columns in the table
cursor.execute(f"PRAGMA table_info({table_name});")
columns_info = cursor.fetchall()
number_of_columns = len(columns_info)
#print(number_of_columns)

# Open the CSV file
with open(csv_file_path, mode='r') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file)
    
    # Prepare a SQL query for inserting data. The placeholders (?) will be replaced by row data
    insert_query = f'INSERT INTO {table_name} VALUES ({",".join(["?"] * number_of_columns)})'
    
    # Execute the insert query for each row in the CSV file
    for row in csv_reader:
        cursor.execute(insert_query, row)

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Data from the CSV file has been added to the SQLite table.")
