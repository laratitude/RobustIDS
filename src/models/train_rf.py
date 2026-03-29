from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

def train_rf(data_path):
    df = pd.read_csv(data_path)

    # clean column names
    df.columns = df.columns.str.strip()

    print("Data shape before cleaning:", df.shape)

    # remove duplicates
    df = df.drop_duplicates()

    # handle invalid values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    print("Data shape after cleaning:", df.shape)

    # target
    y = df["Label"]

    # drop non-useful columns
    X = df.drop(columns=[
        "Label",
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp"
    ], errors="ignore")

    # keep numeric only
    X = X.select_dtypes(include=["number"])

    # split (sequential)
    split = int(0.8 * len(X))

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    print("Train:", X_train.shape, "Test:", X_test.shape)

    # 🔥 FIX: handle class imbalance
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    train_rf("../../data/processed/data.csv")
