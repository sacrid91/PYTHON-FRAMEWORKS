# app.py (Loading a Sample)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

# --- Configuration ---
DATA_FILE = 'cleaned_data.csv'
VIZ_DIR = 'visualizations'
# --- Sample Configuration ---
SAMPLE_FRAC = 0.05  # Load 5% of the data. Adjust this (e.g., 0.01 for 1%, 0.1 for 10%) based on your RAM and desired responsiveness.
SAMPLE_SEED = 42    # For reproducible sampling

# --- Optimized Helper Functions ---
@st.cache_data # Cache the *sampled* data loading for performance
def load_sampled_data():
    """Loads a random sample of the cleaned data to manage memory."""
    try:
        if not os.path.exists(DATA_FILE):
             st.error(f"Data file '{DATA_FILE}' not found. Please run the cleaning script first.")
             return pd.DataFrame()

        # --- Estimate total lines (optional, for info) ---
        # This can be slow for very large files, but gives an idea.
        # total_lines = sum(1 for line in open(DATA_FILE, 'r', encoding='utf-8'))
        # st.info(f"Total lines estimated in '{DATA_FILE}': ~{total_lines:,}")
        # lines_to_skip = sorted(random.sample(range(1, total_lines), int(total_lines * (1 - SAMPLE_FRAC))))
        # data = pd.read_csv(DATA_FILE, skiprows=lines_to_skip)

        # --- Simpler approach: Read all, then sample (still needs to read once into memory briefly) ---
        # This might still fail if even reading the file header uses too much memory in one go with 1.5GB.
        # Let's try the chunking approach for sampling.

        # --- Chunking approach for sampling ---
        print(f"Loading a sample ({SAMPLE_FRAC*100:.1f}%) of data from '{DATA_FILE}'...")
        st.info(f"Loading a sample ({SAMPLE_FRAC*100:.1f}%) of the data for interactive exploration...")

        chunks = []
        chunk_count = 0
        total_rows_read = 0
        # Read in chunks and sample from each
        for chunk in pd.read_csv(DATA_FILE, chunksize=10000, low_memory=False):
            chunk_count += 1
            total_rows_read += len(chunk)
            # Sample the chunk
            sampled_chunk = chunk.sample(frac=SAMPLE_FRAC, random_state=SAMPLE_SEED + chunk_count) # Vary seed slightly per chunk
            chunks.append(sampled_chunk)
            # Optional: Add a stop condition if you know the approximate size and want to limit read time
            # if total_rows_read > 500000: # e.g., stop after reading ~500k rows
            #     st.warning("Stopped reading early for sampling.")
            #     break

        if not chunks:
            st.error("No data chunks were read.")
            return pd.DataFrame()

        # Concatenate the sampled chunks
        data = pd.concat(chunks, ignore_index=True)
        # Re-sample the final concatenated dataframe to get the exact overall fraction
        # This step helps ensure the final sample size is closer to the desired fraction
        # especially if chunk sizes vary or reading was stopped early.
        final_sample_size = int(len(data) * SAMPLE_FRAC)
        if final_sample_size > 0:
             data = data.sample(n=final_sample_size, random_state=SAMPLE_SEED)
        else:
             # If the concatenated sample is tiny, just use it
             pass

        print(f"Sample loaded. Shape: {data.shape}")
        st.success(f"Sample loaded successfully! (Sample size: {len(data):,} rows)")

        # --- Optimize Data Types (Example) ---
        # Convert 'year' to a smaller integer type if values fit
        if 'year' in data.columns:
             data['year'] = pd.to_numeric(data['year'], errors='coerce', downcast='integer')

        # Convert 'source_x' (or your source column) to categorical if it has repeated strings
        if 'source_x' in data.columns:
            data['source_x'] = data['source_x'].astype('category')

        # Potentially convert 'title_word_count' if it's always small integers
        # Check range first: print(data['title_word_count'].describe())
        # if 'title_word_count' in data.columns and data['title_word_count'].max() < 32767:
        #    data['title_word_count'] = data['title_word_count'].astype('int16')

        print("Data types optimized.")
        return data

    except FileNotFoundError:
        st.error(f"Data file '{DATA_FILE}' not found. Please run the cleaning script first.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        st.error(f"Data file '{DATA_FILE}' is empty.")
        return pd.DataFrame()
    except MemoryError as me:
        st.error("MemoryError: Even sampling failed, likely due to extremely large file size or limited system RAM. Consider reducing SAMPLE_FRAC or checking system resources.")
        st.write(f"Details: {me}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading/optimizing sampled data: {e}")
        # Optionally print traceback for debugging in development
        # import traceback
        # st.write("Traceback:")
        # st.code(traceback.format_exc())
        return pd.DataFrame()

def load_image(image_path):
    """Helper to load images if saved by analysis script."""
    if os.path.exists(image_path):
        return plt.imread(image_path)
    else:
        # Don't warn for missing images if analysis hasn't run yet
        # st.warning(f"Image not found: {image_path}") 
        return None

# --- Streamlit App ---
st.set_page_config(page_title="CORD-19 Explorer (Sample)", layout="wide") # Note the (Sample) in title

st.title("CORD-19 Research Papers Explorer (Sample)")
st.write(f"This app explores a **{SAMPLE_FRAC*100:.1f}% sample** of the CORD-19 metadata dataset to manage performance.")

# Load sampled data
df = load_sampled_data()

if not df.empty:
    # Sidebar for filters
    st.sidebar.header("Filters")

    # Filter by Year
    valid_years = df['year'].dropna()
    if not valid_years.empty:
        min_year = int(valid_years.min())
        max_year = int(valid_years.max())
        # Handle case where min/max are the same or very close
        if min_year == max_year:
            st.sidebar.info(f"All papers are from {min_year}. No year filter available.")
            year_range = (min_year, max_year)
            filtered_df = df.copy() # No filtering possible
        else:
            default_start = max(min_year, max_year - 5) # Last 5 years or min if less
            year_range = st.sidebar.slider("Select Publication Year Range:", min_year, max_year, (default_start, max_year))
            filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    else:
        st.sidebar.warning("No valid years found in the sampled data.")
        filtered_df = df.copy() # Show all if no year filter possible

    # Filter by Source
    if 'source_x' in df.columns:
        unique_sources = df['source_x'].cat.categories if hasattr(df['source_x'], 'cat') else df['source_x'].dropna().unique()
        if len(unique_sources) > 0:
            # Default to top categories in the sample or first few
            source_counts = df['source_x'].value_counts()
            default_sources = source_counts.head(5).index.tolist()
            # Ensure defaults are in the unique list (handles potential type issues)
            default_sources = [s for s in default_sources if s in unique_sources]
            if not default_sources:
                 default_sources = unique_sources[:min(3, len(unique_sources))].tolist() # Fallback

            selected_sources = st.sidebar.multiselect("Select Sources:", options=unique_sources, default=default_sources)
            if selected_sources:
                 filtered_df = filtered_df[filtered_df['source_x'].isin(selected_sources)]
        else:
             st.sidebar.info("No unique sources available for filtering in the sample.")
    else:
        st.sidebar.info("Source column not found in loaded sample data.")


    st.header("Data Overview")
    st.write(f"Showing data for years {year_range[0]} to {year_range[1]} (from the sample)")
    st.write(f"Total papers in filtered sample: {len(filtered_df):,}")
    st.subheader("Sample Data (First 10 Rows)")
    display_cols = ['title', 'abstract', 'year', 'source_x']
    if 'title_word_count' in filtered_df.columns:
        display_cols.append('title_word_count')
    st.dataframe(filtered_df[display_cols].head(10))


    st.header("Visualizations")

    # --- Load and Display Pre-generated Plots ---
    # Note: These plots were generated from the *full* dataset.
    # The sample in the app might show different patterns.
    st.markdown("**Note:** The plots below are based on the *full* dataset analysis. The sample data above might show different distributions.**")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Publications by Year")
        img_year = load_image(os.path.join(VIZ_DIR, 'publications_by_year.png'))
        if img_year is not None:
            st.image(img_year, caption="", use_column_width=True)
        else:
             st.info("Plot not available. Run `analysis_and_viz.py` first.")

        st.subheader("Title Word Count Distribution")
        img_dist = load_image(os.path.join(VIZ_DIR, 'title_word_count_dist.png'))
        if img_dist is not None:
            st.image(img_dist, caption="", use_column_width=True)
        else:
             st.info("Plot not available. Run `analysis_and_viz.py` first.")


    with col2:
        st.subheader("Top Publishing Sources")
        img_sources = load_image(os.path.join(VIZ_DIR, 'top_sources.png'))
        if img_sources is not None:
            st.image(img_sources, caption="", use_column_width=True)
        else:
             st.info("Plot not available. Run `analysis_and_viz.py` first.")

        st.subheader("Word Cloud of Titles")
        img_wc = load_image(os.path.join(VIZ_DIR, 'title_wordcloud.png'))
        if img_wc is not None:
            st.image(img_wc, caption="", use_column_width=True)
        else:
             st.info("Plot not available. Run `analysis_and_viz.py` first.")


    # --- Create Plots Dynamically in Streamlit (from the Sample) ---
    st.markdown("---")
    st.subheader("Dynamic Plots (Based on Sample Filters)")

    # Dynamic Year Plot
    if not filtered_df.empty and 'year' in filtered_df.columns:
        plot_data_year = filtered_df.dropna(subset=['year'])
        if not plot_data_year.empty:
            year_counts_filtered = plot_data_year['year'].value_counts().sort_index()
            if not year_counts_filtered.empty:
                fig1, ax1 = plt.subplots(figsize=(8, 4))
                ax1.plot(year_counts_filtered.index, year_counts_filtered.values, marker='o')
                ax1.set_title('Filtered Publications by Year (Sample)')
                ax1.set_xlabel('Year')
                ax1.set_ylabel('Number of Papers (Sample)')
                ax1.grid(True)
                st.pyplot(fig1)
            else:
                st.write("No data available for the selected year range/source in the sample (after filtering).")
        else:
             st.write("No data with valid years available for dynamic year plot in the sample.")
    elif not filtered_df.empty:
        st.write("Year column not available in filtered sample data for dynamic plot.")
    else:
        st.write("No data available in the sample for dynamic plots.")

    # Dynamic Top Sources Plot
    if not filtered_df.empty and 'source_x' in filtered_df.columns:
        plot_data_source = filtered_df.dropna(subset=['source_x'])
        if not plot_data_source.empty:
            top_sources_filtered = plot_data_source['source_x'].value_counts().head(10)
            if not top_sources_filtered.empty:
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                sns.barplot(x=top_sources_filtered.values, y=top_sources_filtered.index, ax=ax2, palette='viridis')
                ax2.set_title('Top Sources (Filtered Sample)')
                ax2.set_xlabel('Number of Papers (Sample)')
                ax2.set_ylabel('Source')
                st.pyplot(fig2)
            else:
                 st.write("No source data available for the selected filters in the sample (after filtering).")
        else:
             st.write("No data with valid sources available for dynamic source plot in the sample.")
    elif not filtered_df.empty:
         st.write("Source column not available in filtered sample data for dynamic plot.")
    else:
        st.write("No data available in the sample for dynamic plots.")


else:
    st.info("Please ensure the data cleaning step is completed and 'cleaned_data.csv' exists.")
