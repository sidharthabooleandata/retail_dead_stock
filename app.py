# streamlit_app.py
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

MODEL_OUT = "model_pipeline.joblib"
META_OUT = "model_metadata.json"

st.set_page_config(page_title="Prediction Accelerator Demo", layout="centered")
st.title("Prediction Accelerator — Demo")
st.markdown("This app loads a pre-trained model. Provide the top inputs and get a prediction.")

if not Path(META_OUT).exists() or not Path(MODEL_OUT).exists():
    st.error("Run `python build_model.py` first.")
    st.stop()

with open(META_OUT, "r") as f:
    meta = json.load(f)

test_accuracy = meta.get("test_accuracy")
top_features = meta.get("top_features_asked", [])
final_feature_cols = meta.get("final_feature_cols", top_features)
cat_features = meta.get("categorical_features", [])
num_features = meta.get("numeric_features", [])
categories_map = meta.get("categories_map", {})

with st.sidebar:
    # Logo
    st.image(
        "https://booleandata.ai/wp-content/uploads/2022/09/Boolean-logo_Boolean-logo-USA-1.png",
        use_container_width=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # # Model accuracy
    # st.subheader("📊 Model Accuracy")
    # st.metric("Hold-out Accuracy", f"{holdout_acc:.2%}")

    # Spacer
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

    # About Us
    st.markdown("""
<div style="font-size:15px; line-height:1.4; color:#333;text-align: center;">
<h5 style="font-size:18px;">🚀 About Us</h5>
We leverage Snowflake to plan and design emerging data architectures that facilitate incorporation of high-quality and flexible data. 
<br><br>
These solutions lower costs and enhance output, designed to transform smoothly as your enterprise, and your data continue to increase over time.
</div>
    """, unsafe_allow_html=True)

    # Social media links
    st.markdown("""
<div style="text-align:center; display:flex; justify-content:center; gap:15px; margin-top:10px;">
<a href="https://booleandata.ai/" target="_blank">🌐</a>
<a href="https://www.facebook.com/Booleandata" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/24/1384/1384005.png" width="24">
</a>
<a href="https://www.youtube.com/channel/UCd4PC27NqQL5v9-1jvwKE2w" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/24/1384/1384060.png" width="24">
</a>
<a href="https://www.linkedin.com/company/boolean-data-systems" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/24/145/145807.png" width="24">
</a>
</div>
    """, unsafe_allow_html=True)

model = joblib.load(MODEL_OUT)

st.subheader("Enter feature values")
with st.form("input_form"):
    user_input = {}
    for feat in top_features:
        if feat in cat_features:
            options = categories_map.get(feat, [])
            if not options:
                options = ["Option1", "Option2"]
            val = st.selectbox(f"{feat}", options)
        else:
            val = st.number_input(f"{feat}", value=0.0, step=1.0)
        user_input[feat] = val
    submitted = st.form_submit_button("Predict")

if submitted:
    row = {c: user_input.get(c, np.nan) for c in final_feature_cols}
    Xuser = pd.DataFrame([row])

    try:
        pred = model.predict(Xuser)[0]
        st.success(f"Predicted class: **{pred}**")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(Xuser)[0]
            st.write("Class probabilities:")
            dfp = pd.DataFrame([probs], columns=model.named_steps["clf"].classes_)
            st.table(dfp.T.rename(columns={0: "probability"}))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
