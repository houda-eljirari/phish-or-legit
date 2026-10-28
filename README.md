# Phish or Legit? (Lab 04)

A small web application that classifies a message as **phishing** or **legitimate**, shows the
classifier score, lets you change the decision threshold live, and explains each decision with
per-feature contributions.

**Live app:** 

> This tool supports a human decision and does not block messages by itself. The data is synthetic.

## What it does

- TF-IDF features (word unigrams and bigrams) after replacing URLs, amounts and numbers by placeholder tokens.
- Two classifiers compared on the same features: Multinomial Naive Bayes and logistic regression.
- The app uses logistic regression. Each feature contributes `TF-IDF value x coefficient` to the log-odds,
  which the app displays as "Increase the score" / "Decrease the score".
- A decision threshold slider (not fixed at 0.5), a selector for the training data, and four example
  messages (two phishing, two legitimate) with the ground truth shown after classification.
- Stratified train/test split done before any fitting; the vectoriser and the classifier are fitted
  inside a single `Pipeline` on the training set only.

## Data

| Dataset | Content |
|---|---|
| `data/messages.csv` | 2,000 synthetic short messages (20 % phishing), fictional brands |
| `data/messages_v2.csv` | synthetic data (deduplicated) + a cleaned sample of 2,000 emails from the public Kaggle corpus *Phishing Email Detection* |

The Kaggle emails were truncated to 30 words, e-mail addresses and phone numbers were masked, duplicates were
removed, and each row keeps a `source` column. Its "phishing" label is closer to spam than to credential phishing,
and its legitimate class is dominated by Enron emails, so a model can partly learn the style of the source
(the word "enron" is one of the strongest "legit" features). Results must not be read as performance on real traffic.

## Results (logistic regression, test set, threshold 0.5)

Main evaluation on the synthetic data (`python -m src.evaluate`):

| Classifier | Precision | Recall | F1 | Brier |
|---|---|---|---|---|
| Naive Bayes | 0.611 | 0.800 | 0.693 | 0.104 |
| Logistic regression | 0.747 | 0.710 | 0.728 | 0.071 |

Effect of the new dataset (`python -m src.evaluate_v2`, F1):

| Trained on | Tested on synthetic | Tested on Kaggle sample |
|---|---|---|
| synthetic only | 0.798 | 0.013 |
| merged | 0.804 | 0.715 |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Rebuild the merged dataset from the Kaggle CSV (not stored in this repository) and re-run the evaluations:

```bash
python -m src.build_dataset --input data/Phishing_Email.csv --n 2000 --phish-ratio 0.3
python -m src.evaluate
python -m src.evaluate_v2
```

## Project structure

```
app.py                 Streamlit application
src/pipeline.py        data loading, split, pipelines, contributions
src/evaluate.py        metrics, calibration, failure cases (synthetic data)
src/build_dataset.py   builds data/messages_v2.csv from the Kaggle corpus
src/evaluate_v2.py     cross-source comparison (synthetic / Kaggle / merged)
data/                  datasets and the script that generated the synthetic one
results/               metrics, figures, failure cases
```
