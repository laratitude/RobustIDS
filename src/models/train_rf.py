from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

def train_rf(data_path):
    df = pd.read_csv(data_path)

    df.columns = df.columns.str.strip()

    print("Data shape before cleaning:", df.shape)

    df = df.drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    print("Data shape after cleaning:", df.shape)

    y = df["Label"]

    X = df.drop(columns=[
        "Label",
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp"
    ], errors="ignore")

    X = X.select_dtypes(include=["number"])

    split = int(0.8 * len(X))

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    print("Before balancing:")
    print(y_train.value_counts())

    # 🔥 Oversample minority classes
    train_df = pd.concat([X_train, y_train], axis=1)

    max_size = train_df["Label"].value_counts().max()

    balanced_df = train_df.groupby("Label").apply(
        lambda x: x.sample(max_size, replace=True)
    ).reset_index(drop=True)

    print("\nAfter balancing:")
    print(balanced_df["Label"].value_counts())

    X_train_balanced = balanced_df.drop("Label", axis=1)
    y_train_balanced = balanced_df["Label"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train_balanced, y_train_balanced)

    preds = model.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    train_rf("../../data/processed/data.csv")
