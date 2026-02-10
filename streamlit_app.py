import os
import re
import base64
import joblib
import gdown
import numpy as np
import pandas as pd
import streamlit as st

# Run:  streamlit run streamlit_app.py

# App Config
st.set_page_config(page_title="HDB Resale Price Predictor", page_icon="🏠", layout="centered")


# Background Image
BG_IMAGE_PATH = "Background.png"

def set_background(image_path):
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{data}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stMainBlockContainer {{
            background-color: rgba(255, 255, 255, 0.92);
            border-radius: 12px;
            padding: 2rem;
            margin-top: 1rem;
            max-width: 900px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

if os.path.exists(BG_IMAGE_PATH):
    set_background(BG_IMAGE_PATH)

MODEL_PATH = "hdb_best_rs_rf_model.pkl"
GDRIVE_FILE_ID = "1GTi5vOlOsOr3GpumiD-Z4wQ4j-pKMPpy"

# Full town list for user-friendly input (model may use drop_first dummy encoding)
ALL_TOWNS = [
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH",
    "BUKIT PANJANG", "BUKIT TIMAH", "CENTRAL AREA", "CHOA CHU KANG",
    "CLEMENTI", "GEYLANG", "HOUGANG", "JURONG EAST", "JURONG WEST",
    "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "PUNGGOL",
    "QUEENSTOWN", "SEMBAWANG", "SENGKANG", "SERANGOON", "TAMPINES",
    "TOA PAYOH", "WOODLANDS", "YISHUN"
]

FLAT_TYPE_MAP = {
    "1 ROOM": 1,
    "2 ROOM": 2,
    "3 ROOM": 3,
    "4 ROOM": 4,
    "5 ROOM": 5,
    "EXECUTIVE": 6,
    "MULTI-GENERATION": 7
}

# All original flat models from the dataset
ALL_FLAT_MODELS = [
    "Model A", "Improved", "New Generation", "Simplified", "Standard", 
    "Apartment", "Model A2", "2-room",
    "Premium Apartment", "Premium Apartment Loft", "DBSS", "Premium Maisonette",
    "Maisonette", "Model A-Maisonette", "Improved-Maisonette",
    "Adjoined flat", "Multi Generation", "3Gen",
    "Terrace", "Type S1", "Type S2"
]

# Mapping from flat_model to flat_model_class (as used in model training)
FLAT_MODEL_TO_CLASS = {
    # Standard Category
    "Model A": "Standard",
    "Improved": "Standard",
    "New Generation": "Standard",
    "Simplified": "Standard",
    "Standard": "Standard",
    "Apartment": "Standard",
    "Model A2": "Standard",
    "2-room": "Standard",
    # Premium Category
    "Premium Apartment": "Premium",
    "Premium Apartment Loft": "Premium",
    "DBSS": "Premium",
    "Premium Maisonette": "Premium",
    # Others Category
    "Maisonette": "others",
    "Model A-Maisonette": "others",
    "Improved-Maisonette": "others",
    "Adjoined flat": "others",
    "Multi Generation": "others",
    "3Gen": "others",
    "Terrace": "others",
    "Type S1": "others",
    "Type S2": "others",
}



# Helpers

## Build storey range labels
def build_storey_ranges(max_floor=51, step=3):
    labels = []
    for start in range(1, max_floor + 1, step):
        end = min(start + step - 1, max_floor)
        labels.append(f"{start:02d} TO {end:02d}")
    return labels

## Convert storey range label to average storey value
def storey_range_to_avg(label):
    m = re.match(r"(\d+)\s*TO\s*(\d+)", label)
    if not m:
        return np.nan
    low = float(m.group(1))
    high = float(m.group(2))
    return (low + high) / 2.0

## Load Model with Caching
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)

## Build feature row for prediction
def build_feature_row(model, user_inputs):
    cols = list(model.feature_names_in_)
    row = pd.DataFrame([np.zeros(len(cols), dtype=float)], columns=cols)

    # Numeric engineered features
    if "floor_area_sqm" in row.columns:
        row.loc[0, "floor_area_sqm"] = user_inputs["floor_area_sqm"]

    if "remaining_lease_years" in row.columns:
        row.loc[0, "remaining_lease_years"] = user_inputs["remaining_lease_years"]

    if "storey_avg" in row.columns:
        row.loc[0, "storey_avg"] = user_inputs["storey_avg"]

    if "flat_type_ordinal" in row.columns:
        row.loc[0, "flat_type_ordinal"] = user_inputs["flat_type_ordinal"]

    # Town one-hot (if selected town column exists, set 1; else all-zero baseline)
    town_col = f"town_{user_inputs['town']}"
    if town_col in row.columns:
        row.loc[0, town_col] = 1

    # Flat model class dummies (engineered from flat_model)
    flat_model_class = FLAT_MODEL_TO_CLASS.get(user_inputs["flat_model"], "Standard")
    for cls in ["Standard", "Premium", "others"]:
        c = f"flat_model_class_{cls}"
        if c in row.columns:
            row.loc[0, c] = 1 if flat_model_class == cls else 0

    return row



# Load Model
try:
    model = load_model()
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()


# UI
st.title("HDB Resale Price Prediction")
st.caption("Input flat details to estimate resale price using your tuned Random Forest model trained with the latest data.")

## Input Form
with st.form("predict_form"):
    town_selected = st.selectbox("Town", ALL_TOWNS, index=ALL_TOWNS.index("TAMPINES"))
    flat_type_selected = st.selectbox("Flat Type", list(FLAT_TYPE_MAP.keys()), index=3)

    storey_options = build_storey_ranges()
    storey_range_selected = st.selectbox("Storey Range", storey_options, index=3)

    floor_area_selected = st.slider(
        "Floor Area (sqm)",
        min_value=30.0,
        max_value=400.0,
        value=95.0,
        step=1.0
    )

    lease_years = st.slider("Remaining Lease (Years)", min_value=40, max_value=99, value=75, step=1)

    flat_model_selected = st.selectbox(
        "Flat Model",
        ALL_FLAT_MODELS,
        index=0,
        help="Select the flat model type. This will be categorized into Standard/Premium/Others for prediction."
    )

    submitted = st.form_submit_button("Predict HDB Price")

if submitted:
    remaining_lease_years = float(lease_years)
    storey_avg_value = storey_range_to_avg(storey_range_selected)

    user_inputs = {
        "town": town_selected,
        "flat_type_ordinal": FLAT_TYPE_MAP[flat_type_selected],
        "storey_avg": storey_avg_value,
        "floor_area_sqm": floor_area_selected,
        "remaining_lease_years": remaining_lease_years,
        "flat_model": flat_model_selected,
    }

    X_input = build_feature_row(model, user_inputs)
    pred = float(model.predict(X_input)[0])

    st.markdown(
        f"""
        <div style="background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 8px; padding: 1.2rem; margin: 1rem 0;">
            <p style="font-size: 1.4rem; font-weight: 600; color: #155724; margin: 0;">
                Predicted Resale Price: SGD {pred:,.2f}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    flat_model_class = FLAT_MODEL_TO_CLASS.get(flat_model_selected, "Standard")
## Show Prediction Details
    with st.expander("Show prediction details"):
        st.subheader("Your Inputs")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Town:** {town_selected}")
            st.markdown(f"**Flat Type:** {flat_type_selected}")
            st.markdown(f"**Storey Range:** {storey_range_selected}")
            st.markdown(f"**Floor Area:** {floor_area_selected} sqm")
        with col2:
            st.markdown(f"**Remaining Lease:** {lease_years} years")
            st.markdown(f"**Flat Model:** {flat_model_selected}")
            st.markdown(f"**Flat Model Category:** {flat_model_class}")

        st.subheader("Transformed Model Input")
        # Only show non-zero features for clarity
        feature_df = X_input.T.rename(columns={0: "value"})
        non_zero = feature_df[feature_df["value"] != 0]
        st.dataframe(non_zero, width="stretch")
