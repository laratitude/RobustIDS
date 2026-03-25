import pandas as pd
import os

def preprocess_data(input_dir, output_path):
    all_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    dfs = []

    for file in all_files:
        file_path = os.path.join(input_dir, file)
        print("Loading:", file_path)
        df = pd.read_csv(file_path)
        dfs.append(df)

    print("Combining files...")
    combined_df = pd.concat(dfs, ignore_index=True)

    print("Saving processed data...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    print("Done.")

if __name__ == "__main__":
    preprocess_data("../../data/raw", "../../data/processed/data.csv")
