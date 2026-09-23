from pathlib import Path
import re
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from kaggle.api.kaggle_api_extended import KaggleApi

nltk.download("stopwords", quiet=True)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "dineshpiyasamara/sentiment-analysis-dataset"
MODEL_PATH = BASE_DIR / "model.joblib"
VECTORIZER_PATH = BASE_DIR / "vectorizer.joblib"
STOPWORDS = set(stopwords.words("english"))

def download_kaggle_dataset():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET, path=str(DATA_DIR), unzip=True, quiet=False)

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    cols = {c.lower(): c for c in df.columns}
    text_col = None
    label_col = None

    for c in ["text", "review", "statement", "comment", "tweet", "content", "sentence"]:
        if c in cols:
            text_col = cols[c]
            break

    for c in ["label", "sentiment", "status", "class", "rating", "category"]:
        if c in cols:
            label_col = cols[c]
            break

    if text_col is None or label_col is None:
        if df.shape[1] >= 2:
            text_col = df.columns[0]
            label_col = df.columns[1]
        else:
            raise ValueError(f"Could not infer columns from: {list(df.columns)}")

    out = df[[text_col, label_col]].copy()
    out.columns = ["text", "label"]
    return out

def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = df["text"].fillna("").astype(str).map(clean_text)
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    label_map = {
        "positive": 1,
        "pos": 1,
        "1": 1,
        "4": 1,
        "negative": 0,
        "neg": 0,
        "0": 0,
        "2": 0,
        "neutral": 0
    }

    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.len() > 0].copy()
    return df

def main():
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        download_kaggle_dataset()
        csv_files = list(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV file found in data/raw")

    df = load_dataset(csv_files[0])
    df = normalize_labels(df)

    if df.empty:
        raise ValueError("No training data available after preprocessing")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.5)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Saved {MODEL_PATH} and {VECTORIZER_PATH}")

if __name__ == "__main__":
    main()