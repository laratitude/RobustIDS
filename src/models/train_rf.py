from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

def train_rf(data_path):
    df = pd.read_csv(data_path)

    # fix column names
    df.columns = df.columns.str.strip()

    print("Data shape:", df.shape)

    # remove duplicates
    df = df.drop_duplicates()

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

    # keep only numeric
    X = X.select_dtypes(include=["number"])

    # simple split
    split = int(0.8 * len(X))

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    print("Train:", X_train.shape, "Test:", X_test.shape)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    train_rf("../../data/processed/data.csv")
