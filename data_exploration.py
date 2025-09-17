# data_exploration.py
import pandas as pd

def load_and_explore(file_path='metadata.csv'):
    """
    Loads the CORD-19 metadata, performs basic exploration,
    and prints findings to the console.
    """
    try:
        print("Loading data...")
        df = pd.read_csv(file_path)
        print("Data loaded successfully.\n")

        print("--- Data Overview ---")
        print(f"Shape (rows, columns): {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
        print("\nColumn information:")
        print(df.info())
        print("\nMissing values per column:")
        print(df.isnull().sum())
        print("\nBasic statistics for numerical columns:")
        print(df.describe())

        return df # Return the dataframe for potential use in other scripts

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please check the path.")
        return None
    except Exception as e:
        print(f"An error occurred during exploration: {e}")
        return None

if __name__ == "__main__":
    df = load_and_explore()
    # You can save basic info to a file if needed, but printing is sufficient for exploration