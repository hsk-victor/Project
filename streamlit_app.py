import os
import re
import joblib
import gdown
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="HDB Resale Price Predictor", page_icon="🏠", layout="centered")

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

FLAT_MODEL_CLASS_OPTIONS = ["Standard", "Premium", "others"]


# ----------------------------
# Helpers
# ----------------------------
def build_storey_ranges(max_floor=51, step=3):
    labels = []
    for start in range(1, max_floor + 1, step):
        end = min(start + step - 1, max_floor)
        labels.append(f"{start:02d} TO {end:02d}")
    return labels


def storey_range_to_avg(label):
    m = re.match(r"(\d+)\s*TO\s*(\d+)", label)
    if not m:
        return np.nan
    low = float(m.group(1))
    high = float(m.group(2))
    return (low + high) / 2.0


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)


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

    # Flat model class dummies
    for cls in FLAT_MODEL_CLASS_OPTIONS:
        c = f"flat_model_class_{cls}"
        if c in row.columns:
            row.loc[0, c] = 1 if user_inputs["flat_model_class"] == cls else 0

    return row


# ----------------------------
# Load Model
# ----------------------------
try:
    model = load_model()
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

# ----------------------------
# UI
# ----------------------------
st.title("HDB Resale Price Prediction")
st.caption("Input flat details to estimate resale price using your tuned Random Forest model.")

with st.form("predict_form"):
    town_selected = st.selectbox("Town", ALL_TOWNS, index=ALL_TOWNS.index("TAMPINES"))
    flat_type_selected = st.selectbox("Flat Type", list(FLAT_TYPE_MAP.keys()), index=3)

    storey_options = build_storey_ranges()
    storey_range_selected = st.selectbox("Storey Range", storey_options, index=3)

    floor_area_selected = st.slider(
        "Floor Area (sqm)",
        min_value=30.0,
        max_value=250.0,
        value=95.0,
        step=1.0
    )

    lease_years = st.slider("Remaining Lease - Years", min_value=40, max_value=99, value=75, step=1)
    lease_months = st.selectbox("Remaining Lease - Months", list(range(0, 12)), index=0)

    flat_model_class_selected = st.selectbox(
        "Flat Model Class",
        FLAT_MODEL_CLASS_OPTIONS,
        index=0,
        help="Engineered category used in your final model."
    )

    submitted = st.form_submit_button("Predict HDB Price")

if submitted:
    remaining_lease_years = lease_years + (lease_months / 12.0)
    storey_avg_value = storey_range_to_avg(storey_range_selected)

    user_inputs = {
        "town": town_selected,
        "flat_type_ordinal": FLAT_TYPE_MAP[flat_type_selected],
        "storey_avg": storey_avg_value,
        "floor_area_sqm": floor_area_selected,
        "remaining_lease_years": remaining_lease_years,
        "flat_model_class": flat_model_class_selected,
    }

    X_input = build_feature_row(model, user_inputs)
    pred = float(model.predict(X_input)[0])

    st.success(f"Predicted Resale Price: SGD {pred:,.2f}")

    with st.expander("Show transformed model input"):
        st.dataframe(X_input.T.rename(columns={0: "value"}))
