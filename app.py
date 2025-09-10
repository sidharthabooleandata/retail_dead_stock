# streamlit_app.py
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

MODEL_OUT = "model_pipeline.joblib"
META_OUT = "model_metadata.json"

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Prediction Accelerator Demo", layout="centered")
st.title("Prediction Accelerator — Demo")
st.markdown("This app loads a pre-trained model. Provide feature values and get a prediction.")

# -------------------------
# Load model + metadata
# -------------------------
if not Path(META_OUT).exists() or not Path(MODEL_OUT).exists():
    st.error("Run `python build_model.py` first to generate the model and metadata.")
    st.stop()

with open(META_OUT, "r") as f:
    meta = json.load(f)

final_feature_cols = meta.get("final_feature_cols", [])
cat_features = meta.get("categorical_features", [])
num_features = meta.get("numeric_features", [])
categories_map = meta.get("categories_map", {})
test_accuracy = meta.get("test_accuracy", None)

# -------------------------
# Friendly labels mapping
# -------------------------
friendly_labels = {
    # Product Info
    "category": "Product Category",
    "brand": "Brand / Sub-brand",
    # Pricing
    "price": "Current Selling Price (INR)",
    "cost": "Cost per Unit (INR)",
    "competitor_price_index": "Price vs Competitor Index",
    # Inventory
    "on_hand": "Units on Hand",
    "lead_time_days": "Supplier Lead Time (days)",
    "shelf_life_days": "Shelf Life (days)",
    # Sales
    "days_since_first_sale": "Days Since First Sale",
    # Promotions
    "promo_flag_14d": "Promotion Active Last 14 Days",
    "promo_depth": "Promotion Depth (%)",
    "price_elasticity_est": "Estimated Price Sensitivity",
    # Store
    "seasonality_index": "Seasonality Index",
    "store_footfall": "Store Footfall (Daily Customers)",
    "store_income_bracket": "Store Neighborhood Income Level",
    "region": "Store Region",
    # Stock Quality
    "days_out_of_stock_365d": "Days Out of Stock Last Year",
    "returns_rate": "Returns Rate (%)",
    "is_new_launch": "Is New Product Launch",
    "holiday_flag": "Snapshot Near Holiday"
}

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    # Logo
    st.image(
        "https://booleandata.ai/wp-content/uploads/2022/09/Boolean-logo_Boolean-logo-USA-1.png",
        use_container_width=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if test_accuracy is not None:
        st.subheader("📊 Model Accuracy")
        st.metric("Hold-out Accuracy", f"{test_accuracy:.2%}")

    # Spacer
    st.markdown("<br><br>", unsafe_allow_html=True)

    # About
    st.markdown("""
<div style="font-size:15px; line-height:1.4; color:#333;text-align: center;">
<h5 style="font-size:18px;">🚀 About Us</h5>
We leverage Snowflake to plan and design emerging data architectures that facilitate incorporation of high-quality and flexible data. 
<br><br>
These solutions lower costs and enhance output, designed to transform smoothly as your enterprise, and your data continue to increase over time.
</div>
    """, unsafe_allow_html=True)

    # Social links
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

# -------------------------
# Load model
# -------------------------
model = joblib.load(MODEL_OUT)

# -------------------------
# Group features by category for input form
# -------------------------
feature_groups = {
    "Product Information": ["category", "brand"],
    "Pricing & Cost": ["price", "cost", "competitor_price_index"],
    "Inventory & Supply": ["on_hand", "lead_time_days", "shelf_life_days"],
    "Sales Performance": ["days_since_first_sale"],
    "Promotions & Pricing": ["promo_flag_14d", "promo_depth", "price_elasticity_est"],
    "Store Attributes": ["seasonality_index", "store_footfall", "store_income_bracket", "region"],
    "Stock Quality": ["days_out_of_stock_365d", "returns_rate", "is_new_launch", "holiday_flag"]
}

# -------------------------
# Input form
# -------------------------
st.subheader("Enter feature values")
with st.form("input_form"):
    user_input = {}
    for group_name, feats in feature_groups.items():
        st.markdown(f"### {group_name}")
        for feat in feats:
            if feat not in final_feature_cols:
                continue
            label = friendly_labels.get(feat, feat)
            if feat in cat_features:
                options = categories_map.get(feat, [])
                if not options:
                    options = ["Option1", "Option2"]
                val = st.selectbox(f"{label}", options)
            else:
                val = st.number_input(f"{label}", value=0.0, step=1.0)
            user_input[feat] = val
    submitted = st.form_submit_button("Predict")

# -------------------------
# Prediction
# -------------------------
if submitted:
    Xuser = pd.DataFrame([{c: user_input.get(c, np.nan) for c in final_feature_cols}])
    try:
        pred = model.predict(Xuser)[0]
        st.success(f"Predicted class: **{pred}**")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(Xuser)[0]
            dfp = pd.DataFrame([probs], columns=model.named_steps["clf"].classes_)
            st.write("Class probabilities:")
            st.table(dfp.T.rename(columns={0: "probability"}))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
