import pandas as pd

def load_cicids(path):
    df = pd.read_csv(path)
    df = df.dropna()
    return df
  
