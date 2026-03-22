import pandas as pd
import glob
import os

def load_and_merge_data(path):
    files = glob.glob(os.path.join(path, "*.csv"))
    df_list = []

    for file in files:
        print(f"Loading {file}")
        df = pd.read_csv(file, low_memory=False)
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def clean_data(df):
    # Remove spaces in column names
    df.columns = df.columns.str.strip()

    # Replace infinity values
    df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)

    # Drop missing values
    df = df.dropna()

    return df

def main():
    raw_path = "data/raw"
    output_path = "data/processed/data.csv"

    df = load_and_merge_data(raw_path)
    df = clean_data(df)

    # Convert labels to binary
    df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

    # Save
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)

    print("✅ Data preprocessing complete!")

if __name__ == "__main__":
    main()
