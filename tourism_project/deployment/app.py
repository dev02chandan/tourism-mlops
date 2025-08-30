import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Purchase Predictor", page_icon="🧭", layout="centered")

REPO_ID = os.environ.get("HF_MODEL_REPO", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", None)  # set as a Secret in your Space if the model is private
assert REPO_ID, "HF_MODEL_REPO env var is empty. Set it to your model repo id, e.g. 'username/tourism-wellness-model'."

MODEL_LOCAL = hf_hub_download(
    repo_id=REPO_ID, filename="best_model.joblib", repo_type="model", token=HF_TOKEN
)
model = joblib.load(MODEL_LOCAL)

# derive feature names from the fitted sklearn pipeline
pre = model.named_steps["preprocess"]
num_cols = list(pre.transformers_[0][2])
cat_cols = list(pre.transformers_[1][2])

st.title("🏝️ Wellness Tourism – Purchase Prediction")
st.caption(f"Model: **{REPO_ID}**")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.01)

st.subheader("Enter customer details")

inputs = {}
cols = st.columns(2)
# numeric on left, categorical on right
with cols[0]:
    for c in num_cols:
        # sensible defaults; users can change
        default = 0.0
        if c.lower().endswith(("id",)):  # none here, just in case
            default = 0
        inputs[c] = st.number_input(c, value=float(default))

with cols[1]:
    for c in cat_cols:
        inputs[c] = st.text_input(c, value="")

X = pd.DataFrame([inputs])
st.write("Preview:", X)

if st.button("Predict"):
    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= threshold)
    st.metric("Purchase probability", f"{proba:.3f}")
    st.metric("Predicted class", pred)

    # log request
    row = {"probability": proba, "prediction": pred, "threshold": threshold, **inputs}
    log_path = "/tmp/inputs_log.csv"
    pd.DataFrame([row]).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
    st.caption(f"Saved input to {log_path}")
