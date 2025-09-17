# data_cleaning.py
import pandas as pd
import os

def clean_data_chunked(input_file='metadata.csv', output_file='cleaned_data.csv', chunksize=10000):
    """
    Cleans the CORD-19 metadata by reading it in chunks to manage memory:
    - Handles missing values
    - Converts publish_time to datetime and extracts year
    - Creates new columns (e.g., title word count)
    - Saves the cleaned data.
    Assumes 'title' is present to check for essential data.
    """
    print(f"Starting cleaning process for '{input_file}' using chunking (chunksize={chunksize})...")
    
    first_chunk = True  # Flag to handle header writing for the output CSV
    total_rows_processed = 0
    total_rows_kept = 0

    try:
        # --- Read the file in chunks ---
        chunk_iter = pd.read_csv(input_file, chunksize=chunksize, low_memory=False)
        
        for i, chunk_df in enumerate(chunk_iter):
            print(f"Processing chunk {i+1} (rows ~{i*chunksize + 1} to {(i+1)*chunksize})...")
            total_rows_processed += len(chunk_df)
            
            # --- Cleaning Process for the current chunk ---
            
            # 1. Drop rows where 'title' is missing (often crucial)
            # Store initial count for this chunk
            initial_chunk_rows = len(chunk_df) 
            chunk_df.dropna(subset=['title'], inplace=True)
            rows_after_drop_title = len(chunk_df)
            print(f"  - Dropped {initial_chunk_rows - rows_after_drop_title} rows (missing 'title') in this chunk.")

            # 2. Fill missing 'abstract' with empty string
            chunk_df['abstract'] = chunk_df['abstract'].fillna('')
            # print("  - Filled missing 'abstract' with empty strings.") # Too verbose

            # 3. Convert publish_time to datetime and extract year
            # print("  - Converting 'publish_time' to datetime...") # Too verbose
            chunk_df['publish_time'] = pd.to_datetime(chunk_df['publish_time'], errors='coerce')
            chunk_df['year'] = chunk_df['publish_time'].dt.year
            # print("  - Extracted 'year' from 'publish_time'.") # Too verbose

            # 4. Create a word count for titles (handle potential NaN in title if any slipped through)
            chunk_df['title_word_count'] = chunk_df['title'].fillna('').str.split().str.len()
            # print("  - Created 'title_word_count' column.") # Too verbose

            # 5. Drop columns with excessive missing values (done once, applies to all chunks conceptually)
            # We handle this by simply not selecting them if we knew beforehand,
            # or by dropping them from the final concatenated result.
            # For chunking, it's easier to just not write them if we don't need them,
            # or drop them from the *final* dataframe if needed after concatenation.
            # Let's assume we want to keep most columns for flexibility, 
            # and just not write the ones we identified as problematic *if they exist* in the chunk.
            # However, dropping columns chunk-by-chunk is less efficient than selecting needed ones.
            # Let's stick to processing and write all processed columns for now.
            # If memory is still tight, specify `usecols` in pd.read_csv or drop here.
            
            # Example of dropping specific columns *if they exist in this chunk*:
            cols_to_drop_if_present = ['journal', 'pmcid'] # Example
            cols_actually_dropped = [col for col in cols_to_drop_if_present if col in chunk_df.columns]
            if cols_actually_dropped:
                chunk_df.drop(columns=cols_actually_dropped, inplace=True)
                # print(f"  - Dropped columns: {cols_actually_dropped}") # Too verbose

            # --- Write the cleaned chunk to the output file ---
            rows_to_write = len(chunk_df)
            total_rows_kept += rows_to_write
            
            if rows_to_write > 0:
                # Use mode='a' (append) for subsequent chunks, and avoid writing header again
                write_header = first_chunk
                chunk_df.to_csv(output_file, mode='a', index=False, header=write_header)
                first_chunk = False # Set to False after the first chunk is written
                print(f"  - Wrote {rows_to_write} rows to '{output_file}'.")
            else:
                 print(f"  - No rows to write for this chunk after cleaning.")

        print(f"\n--- Cleaning Process Complete ---")
        print(f"Total rows processed: {total_rows_processed}")
        print(f"Total rows kept/written: {total_rows_kept}")
        print(f"Cleaned data saved to '{output_file}'.")

        # Optional: Load the final file to get its shape (this might also cause memory issues if file is huge)
        # But for verification, we can just report the count.
        # final_df = pd.read_csv(output_file)
        # print(f"Final cleaned data shape: {final_df.shape}")
        # return final_df
        
        # Safer: Just return the count or None
        if os.path.exists(output_file):
            return total_rows_kept # Or just return True/None to indicate success
        else:
            print(f"Warning: Output file '{output_file}' was not created successfully.")
            return None

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_file}' is empty.")
        return None
    except MemoryError as me:
        print(f"MemoryError encountered during chunking: {me}")
        print("This is unexpected with chunking. Consider reducing chunksize or checking system resources.")
        return None
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred during cleaning: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # --- Important: Remove any existing output file before starting ---
    output_file = 'cleaned_data.csv'
    if os.path.exists(output_file):
        print(f"Removing existing '{output_file}'...")
        os.remove(output_file)
    
    # You can experiment with different chunk sizes depending on your RAM.
    # Smaller chunksize uses less memory but might be slower.
    # Start with 10000, try 5000 or 20000 if needed.
    result = clean_data_chunked(chunksize=10000) 
    if result:
        print("Data cleaning script finished successfully (using chunking).")
    else:
        print("Data cleaning script failed.")
