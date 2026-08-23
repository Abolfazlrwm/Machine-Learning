import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_dataset(path=None):
    if path is None:
        path = DATA_DIR / "human_vs_ai_text.csv"

    df = pd.read_csv(path)
    # original csv has weird column names, fixing them
    df = df.rename(columns={"Data": "text", "Labels": "label"})

    # drop empty rows and dupes, there were a bunch when I first checked
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip().str.len() > 0]
    df = df.drop_duplicates(subset=["text"])
    df = df.reset_index(drop=True)

    return df


def split_dataset(df, test_size=0.2, random_state=42):
    # stratify so we dont end up with an unbalanced test set
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"]
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # quick sanity check
    d = load_dataset()
    print(d.shape)
    print(d["label"].value_counts())
