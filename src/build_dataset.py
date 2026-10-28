"""Build data/messages_v2.csv = synthetic dataset + a cleaned sample of the Kaggle 'Phishing Email Detection' corpus.

Usage (from the project root):
    python -m src.build_dataset --input path/to/Phishing_Email.csv --n 2000 --phish-ratio 0.3

Steps: normalise labels -> clean text -> mask personal data -> truncate to the first N words (the synthetic
messages are short) -> drop empty / very short / duplicate texts -> stratified sample -> merge with a `source` column.
The split into train/test is NOT done here: it stays in src/pipeline.py and happens before any fitting.
"""
import argparse
import re

import pandas as pd

SEED = 42
LABELS = {"phishing email": "phish", "safe email": "legit"}
EMAIL_RE = re.compile(r"[\w.+-]+\s?@\s?[\w-]+(?:\s?\.\s?[\w-]+)+")
PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")
HTTP_RE = re.compile(r"(https?)\s*:\s*/\s*/\s*", re.I)


def clean(text: str, max_words: int) -> str:
    text = HTTP_RE.sub(r"\1://", str(text))      # "http : / / x" -> "http://x"
    text = EMAIL_RE.sub(" emailtok ", text)      # personal data: e-mail addresses
    text = PHONE_RE.sub(" numtok ", text)        # personal data: phone numbers
    words = text.split()                         # also collapses newlines / repeated spaces
    return " ".join(words[:max_words])


def load_kaggle(path: str, max_words: int, min_words: int) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw.rename(columns={"Email Text": "text", "Email Type": "type"})
    raw = raw.dropna(subset=["text", "type"])
    raw["label"] = raw["type"].str.strip().str.lower().map(LABELS)
    raw = raw.dropna(subset=["label"])
    raw["text"] = raw["text"].map(lambda t: clean(t, max_words))
    raw = raw[raw["text"].str.split().str.len() >= min_words]
    raw = raw.drop_duplicates(subset="text")
    return raw[["text", "label"]]


def sample(df: pd.DataFrame, n: int, phish_ratio: float) -> pd.DataFrame:
    n_phish = min(int(n * phish_ratio), (df["label"] == "phish").sum())
    n_legit = min(n - int(n * phish_ratio), (df["label"] == "legit").sum())
    parts = [df[df["label"] == "phish"].sample(n_phish, random_state=SEED),
             df[df["label"] == "legit"].sample(n_legit, random_state=SEED)]
    return pd.concat(parts).sample(frac=1, random_state=SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Kaggle CSV (columns: Email Text, Email Type)")
    ap.add_argument("--synthetic", default="data/messages.csv")
    ap.add_argument("--output", default="data/messages_v2.csv")
    ap.add_argument("--n", type=int, default=2000, help="number of Kaggle emails to keep")
    ap.add_argument("--phish-ratio", type=float, default=0.3)
    ap.add_argument("--max-words", type=int, default=30)
    ap.add_argument("--min-words", type=int, default=3)
    a = ap.parse_args()

    kag = sample(load_kaggle(a.input, a.max_words, a.min_words), a.n, a.phish_ratio)
    kag["source"] = "kaggle_phishing_email_detection"
    syn = pd.read_csv(a.synthetic)[["text", "label"]]
    syn["source"] = "synthetic"

    out = pd.concat([syn, kag], ignore_index=True).drop_duplicates(subset="text")
    out.to_csv(a.output, index=False)
    print(f"Wrote {a.output}: {len(out)} rows")
    print(out.groupby(["source", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()