"""Lab 04 - Phish or Legit? (Streamlit UI). Run:  streamlit run app.py"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from src.pipeline import SEED, load_data, split, make_pipeline, contributions, word_weights

st.set_page_config(page_title="Phish or Legit?")
st.title("Phish or Legit?")

# Visible notice required by the lab
st.warning("This tool supports a human decision and does not block messages by itself. "
           "The data is synthetic.")
st.caption("TF-IDF + logistic regression. The original lab data is SYNTHETIC (fictional brands); labels are "
           "'phishing-style' vs 'legit', not real-world ground truth.")

# ---- Example messages: (text, true label). Two phishing, two legitimate. Whether the model gets them right
# depends on the selected training data and the threshold: the app shows the verdict after classification.
EXAMPLES = {
    "(write your own message)": ("", None),
    "Example 1 - phishing": (
        "ALERT: BlueFreight: your package is waiting. Confirm your details. This is your final notice. "
        "http://secure-2kcy-verify.com/j8n1ws", "phish"),
    "Example 2 - phishing (subtle)": (
        "ALERT: ParcelGo: your package is waiting. http://tinyurl.com/p7s5e1b Thanks", "phish"),
    "Example 3 - legitimate": (
        "Reminder: class stats tomorrow at 9. Open the official app to view it.", "legit"),
    "Example 4 - legitimate (looks suspicious)": (
        "Northban: there is a message about your account. Log in to confirm. https://docs.example.com/d/7estf1",
        "legit"),
}
TRUTH = {t: lab for t, lab in EXAMPLES.values() if lab}

V2 = "data/messages_v2.csv"
MODEL_CHOICES = (["Merged (synthetic + Kaggle sample)", "Synthetic only (original lab dataset)"]
                 if os.path.exists(V2) else ["Synthetic only (original lab dataset)"])


@st.cache_resource
def get_model(choice):
    """Split first, then fit the whole pipeline on the TRAIN part only (no leakage)."""
    if choice.startswith("Merged"):
        df = load_data(V2)
        tr, te = [], []
        for is_syn in (True, False):            # each source is split separately, as in src/evaluate_v2.py
            sub = df[(df["source"] == "synthetic") == is_syn]
            tr.append(train_test_split(sub["text"], sub["y"], test_size=0.25, stratify=sub["y"],
                                       random_state=SEED))
        Xtr = pd.concat([t[0] for t in tr])
        ytr = pd.concat([t[2] for t in tr])
        Xte = pd.concat([t[1] for t in tr])
        yte = pd.concat([t[3] for t in tr])
    else:
        Xtr, Xte, ytr, yte = split(load_data())
    model = make_pipeline("lr").fit(Xtr, ytr)
    return model, model.predict_proba(Xte)[:, 1], np.asarray(yte)


choice = st.selectbox("Training data", MODEL_CHOICES)
model, p_test, y_test = get_model(choice)
if choice.startswith("Merged"):
    st.caption("Merged mode adds a cleaned sample of a public Kaggle email corpus (real emails; its "
               "'phishing' label is closer to spam). Texts were truncated and personal data masked.")


def load_example():
    st.session_state["msg"] = EXAMPLES[st.session_state["example"]][0]
    st.session_state.pop("classified", None)


st.selectbox("Load an example:", list(EXAMPLES), key="example", on_change=load_example)
text = st.text_area("Paste a message:", key="msg", height=110)

threshold = st.slider("Decision threshold (phishing if score >= threshold)",
                      min_value=0.05, max_value=0.95, value=0.50, step=0.05)

if st.button("Classify", type="primary"):
    if not text.strip():
        st.error("The message is empty - paste some text to classify.")
        st.session_state.pop("classified", None)
    else:
        st.session_state["classified"] = text

# The result is computed from the stored message, so moving the slider updates it live.
if "classified" in st.session_state:
    msg = st.session_state["classified"]
    score = float(model.predict_proba([msg])[0][1])
    if score >= threshold:
        st.error(f"⚠️ Predicted class: PHISHING  ·  score {score:.2f}  ≥  threshold {threshold:.2f}")
    else:
        st.success(f"✅ Predicted class: LEGIT  ·  score {score:.2f}  <  threshold {threshold:.2f}")
    if msg in TRUTH:
        is_phish = TRUTH[msg] == "phish"
        ok = (score >= threshold) == is_phish
        st.info(f"Ground truth: {'phishing' if is_phish else 'legitimate'} -> the model is "
                f"{'correct' if ok else 'WRONG'} at threshold {threshold:.2f}.")
    st.caption("The score is the model output for the phishing class. It is not guaranteed to be a "
               "calibrated probability.")

    # Highlighted text
    weights = word_weights(model, msg)
    parts = []
    for tok in re.findall(r"\w+|\W+", msg):
        w = weights.get(tok.lower(), 0.0)
        if re.match(r"\w+", tok) and abs(w) > 0.3:
            parts.append(f":red[**{tok}**]" if w > 0 else f":green[**{tok}**]")
        else:
            parts.append(tok)
    st.markdown("".join(parts))
    st.caption("red = pushes toward phishing, green = pushes toward legit (strong words only)")

    # Per-feature explanation: contribution to the log-odds = tf-idf value x coefficient
    contrib = contributions(model, msg)
    up = [(f, v) for f, v in contrib.items() if v > 0][:8]
    down = [(f, v) for f, v in sorted(contrib.items(), key=lambda kv: kv[1]) if v < 0][:8]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Increase the score")
        if up:
            st.table({"feature": [f for f, _ in up], "contribution": [round(v, 2) for _, v in up]})
        else:
            st.write("No feature pushes toward phishing.")
    with col2:
        st.subheader("Decrease the score")
        if down:
            st.table({"feature": [f for f, _ in down], "contribution": [round(v, 2) for _, v in down]})
        else:
            st.write("No feature pushes toward legit.")
    st.caption("Contribution to the log-odds = TF-IDF value x logistic-regression coefficient. "
               "Features such as 'urltok' stand for URLs replaced by a placeholder.")

# ---- Optional: precision / recall versus threshold
with st.expander("Precision / recall versus threshold (test set)"):
    ths = np.arange(0.05, 0.96, 0.05)
    prec, rec = [], []
    for t in ths:
        pr, rc, _, _ = precision_recall_fscore_support(y_test, (p_test >= t).astype(int),
                                                       average="binary", zero_division=0)
        prec.append(pr)
        rec.append(rc)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(ths, prec, marker="o", label="precision")
    ax.plot(ths, rec, marker="s", label="recall")
    ax.axvline(threshold, color="grey", linestyle="--", label="current threshold")
    ax.set_xlabel("threshold")
    ax.set_ylim(0, 1.05)
    ax.legend()
    st.pyplot(fig)
    st.caption("Computed on the held-out test set for illustration. Pick the final threshold with a "
               "stated policy (or on validation data), not by maximising the test score.")