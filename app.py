import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Success Predictor", page_icon="🎓", layout="centered")


@st.cache_resource
def load():
    try:
        return joblib.load("model.joblib"), json.load(open("meta.json"))
    except Exception:  # e.g. scikit-learn version mismatch on the host: retrain
        import subprocess, sys
        subprocess.run([sys.executable, "train.py"], check=True)
        return joblib.load("model.joblib"), json.load(open("meta.json"))


model, meta = load()
FEATURES = meta["features"]
med = meta["medians"]

LABELS = {
    "Submission.Actions": "Submission actions (assignments submitted)",
    "Quiz.Actions": "Quiz actions",
    "Forum.Actions": "Forum actions",
    "Create.Actions": "Create actions",
    "Read.Actions": "Read actions (resources/pages viewed)",
    "Update.Actions": "Update actions",
    "Delete.Actions": "Delete actions",
    "Days.Logged.In": "Days logged in",
}

st.title("🎓 Student Success Predictor")
st.write(
    "Enter a student's learning-platform activity to predict whether they are "
    "likely to **Pass** or **Fail**."
)

with st.form("student"):
    cols = st.columns(2)
    values = {}
    for i, f in enumerate(FEATURES):
        with cols[i % 2]:
            values[f] = st.number_input(LABELS.get(f, f), min_value=0, value=int(med[f]), step=1)
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    row = pd.DataFrame([values])[FEATURES]
    proba = model.predict_proba(row)[0]
    p_pass = float(proba[list(model.classes_).index(1)])
    if p_pass >= 0.5:
        st.success(f"Prediction: **PASS** (High)  —  confidence {p_pass:.0%}")
    else:
        st.error(f"Prediction: **FAIL** (Low)  —  confidence {1 - p_pass:.0%}")
    st.progress(p_pass, text=f"Probability of passing: {p_pass:.0%}")
    if values["Read.Actions"] > 5000:
        st.warning("Read actions is far above what the model saw in most training data; treat this result with caution.")

with st.expander("About the model"):
    st.write(
        f"Random Forest classifier trained on 3,602 student records. "
        f"Estimated accuracy on unseen students: about {meta['cv_accuracy']:.0%}. "
        "This is a screening aid, not a final judgment about any student."
    )
    imp = pd.Series(meta["importances"]).sort_values()
    imp.index = [LABELS.get(i, i).split(" (")[0] for i in imp.index]
    st.bar_chart(imp)
