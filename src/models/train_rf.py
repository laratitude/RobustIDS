from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

def train_rf(data_path):
    df = pd.read_csv(data_path)

    print(df.columns)
    print(df.head())

    print("Data shape before cleaning:", df.shape)

    df = df.drop_duplicates()

    print("Data shape after removing duplicates:", df.shape)

    # label
    y = df["Label"]

    # remove identifiers and high leakage features
    drop_cols = [
        "Label",
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp",
        "Flow Bytes/s",
        "Flow Packets/s"
    ]

    X = df.drop(columns=drop_cols, errors="ignore")

    # keep numeric only
    X = X.select_dtypes(include=["number"])

    # sequential split
    split_index = int(0.8 * len(X))

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("Train:", X_train.shape, "Test:", X_test.shape)

    # model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    train_rf("../../data/processed/data.csv")
    
