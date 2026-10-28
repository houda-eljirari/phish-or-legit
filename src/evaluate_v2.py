"""Compare models trained / tested on the synthetic data, the Kaggle sample, and the merged dataset.

Run from the project root:   python -m src.evaluate_v2
Requires data/messages_v2.csv (see src/build_dataset.py).

Protocol: each source is split 75/25 (stratified, seed 42) BEFORE any fitting; the test parts are never used for
training.  'merged' = synthetic-train + kaggle-train.  Threshold fixed at 0.5 for all rows (no tuning on test).
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from src.pipeline import SEED, make_pipeline, global_top_features, load_data


def split_source(df, test_size=0.25):
    return train_test_split(df["text"], df["y"], test_size=test_size, stratify=df["y"], random_state=SEED)


def score(pipe, X, y, thr=0.5):
    p = pipe.predict_proba(X)[:, 1]
    pr, rc, f1, _ = precision_recall_fscore_support(y, (p >= thr).astype(int), average="binary", zero_division=0)
    return dict(precision=pr, recall=rc, f1=f1, brier=brier_score_loss(y, p))


def main(path="data/messages_v2.csv"):
    df = load_data(path)
    parts = {}
    for name, sub in (("synthetic", df[df["source"] == "synthetic"]),
                      ("kaggle", df[df["source"] != "synthetic"])):
        parts[name] = split_source(sub)                     # (Xtr, Xte, ytr, yte)

    train_sets = {
        "synthetic": (parts["synthetic"][0], parts["synthetic"][2]),
        "kaggle": (parts["kaggle"][0], parts["kaggle"][2]),
        "merged": (pd.concat([parts["synthetic"][0], parts["kaggle"][0]]),
                   pd.concat([parts["synthetic"][2], parts["kaggle"][2]])),
    }
    test_sets = {"synthetic": (parts["synthetic"][1], parts["synthetic"][3]),
                 "kaggle": (parts["kaggle"][1], parts["kaggle"][3])}

    print("sizes  train:", {k: (len(v[0]), f"{v[1].mean():.0%} phish") for k, v in train_sets.items()})
    print("       test :", {k: (len(v[0]), f"{v[1].mean():.0%} phish") for k, v in test_sets.items()})
    print("\ntrain on   test on    model  precision  recall    F1   Brier")
    rows, merged_lr = [], None
    for kind, label in (("lr", "LR"), ("nb", "NB")):
        for tr_name, (Xtr, ytr) in train_sets.items():
            pipe = make_pipeline(kind).fit(Xtr, ytr)
            if kind == "lr" and tr_name == "merged":
                merged_lr = pipe
            for te_name, (Xte, yte) in test_sets.items():
                m = score(pipe, Xte, yte)
                rows.append(dict(train=tr_name, test=te_name, model=label, **m))
                print(f"{tr_name:<10} {te_name:<10} {label:<5} {m['precision']:9.3f} {m['recall']:7.3f} "
                      f"{m['f1']:6.3f} {m['brier']:7.3f}")

    # Shortcut check: do the strongest features look like phishing cues, or like source artefacts?
    phish, legit = global_top_features(merged_lr, k=12)
    print("\nMerged LR - strongest 'phish' features:", phish)
    print("Merged LR - strongest 'legit' features:", legit)

    os.makedirs("results", exist_ok=True)
    with open("results/metrics_v2.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nSaved results/metrics_v2.json")


if __name__ == "__main__":
    main()