# analysis_and_viz.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud # pip install wordcloud
import re
from collections import Counter # For efficient word counting
import os

def analyze_and_visualize_chunked(cleaned_data_file='cleaned_data.csv', chunksize=20000): # Adjust chunksize as needed
    """
    Performs basic analysis and creates visualizations from the cleaned data,
    processing it in chunks to manage memory usage.
    """
    print(f"Starting analysis and visualization for '{cleaned_data_file}' using chunking (chunksize={chunksize})...")
    
    # --- Aggregators for results across chunks ---
    year_counts_agg = pd.Series(dtype='int64') # Use appropriate dtype
    source_counts_agg = pd.Series(dtype='int64')
    all_words = Counter() # Efficiently count words
    title_word_counts_list = [] # Collect title word counts for histogram

    try:
        print("Loading cleaned data for analysis (chunk by chunk)...")
        chunk_iter = pd.read_csv(cleaned_data_file, chunksize=chunksize, low_memory=False)
        
        total_rows_processed = 0
        for i, chunk_df in enumerate(chunk_iter):
            print(f"Analyzing chunk {i+1}...")
            total_rows_processed += len(chunk_df)
            
            # --- Analysis on the current chunk ---
            
            # 1. Aggregate paper counts by publication year
            if 'year' in chunk_df.columns:
                chunk_year_counts = chunk_df['year'].value_counts()
                # Add counts for overlapping years
                year_counts_agg = year_counts_agg.add(chunk_year_counts, fill_value=0).astype('int64')
            
            # 2. Aggregate top sources
            if 'source_x' in chunk_df.columns: # Adjust column name if needed
                chunk_source_counts = chunk_df['source_x'].value_counts()
                source_counts_agg = source_counts_agg.add(chunk_source_counts, fill_value=0).astype('int64')

            # 3. Collect words for frequent word analysis
            if 'title' in chunk_df.columns:
                # Combine titles in this chunk, handling potential NaNs
                titles_text_chunk = ' '.join(chunk_df['title'].dropna().astype(str))
                # Simple cleaning and word extraction for this chunk
                words_chunk = re.findall(r'\b[a-zA-Z]{3,}\b', titles_text_chunk.lower())
                # Update the overall word counter
                all_words.update(words_chunk)

            # 4. Collect title word counts for distribution
            if 'title_word_count' in chunk_df.columns:
                 # Drop NaNs and convert to list to append
                 title_word_counts_list.extend(chunk_df['title_word_count'].dropna().tolist())
                 
        print(f"\nFinished processing {total_rows_processed} rows in total.")

        # --- Finalize aggregations after all chunks ---
        year_counts_final = year_counts_agg.sort_index()
        top_sources_final = source_counts_agg.head(10)
        top_words_final = pd.Series(all_words).head(50) # Convert Counter to Series for top 50
        print("\n--- Final Aggregated Results ---")
        print("Paper counts by year (top 10):")
        print(year_counts_final.head(10))
        print("\nTop 10 sources:")
        print(top_sources_final)
        print("\nMost frequent words in titles (top 10):")
        print(top_words_final.head(10))

        # --- Generate Visualizations ---
        print("\n--- Generating Visualizations ---")
        
        # Ensure output directory exists
        os.makedirs('visualizations', exist_ok=True)

        # 1. Plot number of publications over time
        plt.figure(figsize=(12, 6))
        if not year_counts_final.empty:
            plt.plot(year_counts_final.index, year_counts_final.values, marker='o', markersize=4)
            plt.title('Number of Publications by Year')
            plt.xlabel('Year')
            plt.ylabel('Number of Papers')
            plt.grid(True, linestyle='--', alpha=0.5)
            # Improve x-axis if there are many years
            if len(year_counts_final) > 20:
                plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('visualizations/publications_by_year.png')
            print("Saved plot: visualizations/publications_by_year.png")
        else:
            print("No data to plot for publications by year.")
        plt.close() # Close the figure to free memory

        # 2. Bar chart of top sources
        plt.figure(figsize=(10, 6))
        if not top_sources_final.empty:
            sns.barplot(x=top_sources_final.values, y=top_sources_final.index, palette='viridis')
            plt.title('Top Publishing Sources')
            plt.xlabel('Number of Papers')
            plt.ylabel('Source')
            plt.tight_layout()
            plt.savefig('visualizations/top_sources.png')
            print("Saved plot: visualizations/top_sources.png")
        else:
             print("No data found for top sources plot.")
        plt.close()

        # 3. Word cloud of paper titles
        plt.figure(figsize=(12, 6))
        # Combine all collected words into a single string, weighted by frequency
        # WordCloud can take a frequency dictionary directly
        if all_words:
            wordcloud = WordCloud(width=1200, height=600, background_color='white', max_words=200).generate_from_frequencies(all_words)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title('Word Cloud of Paper Titles')
            plt.tight_layout()
            plt.savefig('visualizations/title_wordcloud.png')
            print("Saved plot: visualizations/title_wordcloud.png")
        else:
             print("No title text available for word cloud.")
        plt.close()

        # 4. Distribution of title word counts
        plt.figure(figsize=(10, 6))
        if title_word_counts_list:
            plt.hist(title_word_counts_list, bins=50, edgecolor='black', alpha=0.7) # Increase bins for detail
            plt.title('Distribution of Title Word Counts')
            plt.xlabel('Number of Words in Title')
            plt.ylabel('Frequency')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('visualizations/title_word_count_dist.png')
            print("Saved plot: visualizations/title_word_count_dist.png")
        else:
             print("No title word count data available for distribution plot.")
        plt.close()

        print("\nAnalysis and visualization complete. Plots saved in 'visualizations/' folder.")

    except FileNotFoundError:
        print(f"Error: Cleaned data file '{cleaned_data_file}' not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: Cleaned data file '{cleaned_data_file}' is empty.")
    except MemoryError as me:
        print(f"MemoryError encountered during chunked analysis: {me}")
        print("This is unexpected if chunking is working. Consider reducing chunksize further.")
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred during analysis/visualization: {e}")
        print("Traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    # Run the chunked analysis
    analyze_and_visualize_chunked(chunksize=15000) # You can experiment with chunksize (e.g., 10000, 20000)
