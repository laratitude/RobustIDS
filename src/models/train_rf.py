from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

def train_rf(data_path):
    df = pd.read_csv(data_path)

    print(df.columns)
    print(df.head())

    print("Data shape before cleaning:", df.shape)

    df = df.drop_duplicates()

    print("Data shape after removing duplicates:", df.shape)

    y = df["Label"]
    X = df.drop(columns=["Label"])

    X = X.select_dtypes(include=["number"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Train:", X_train.shape, "Test:", X_test.shape)

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
    
