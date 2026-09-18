"""
Sydney Housing Price Predictor (SIT307 Task 8.1D, Part 5).

Loads the Ridge pipeline saved in the notebook (housing_model.joblib) and
predicts sale prices for Chatswood, Castle Hill and Blacktown properties,
either one at a time (form) or in bulk (CSV upload).

Run:  streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

SUBURBS = ["Blacktown", "Castle Hill", "Chatswood"]
PROPERTY_TYPES = ["house", "townhouse", "unit"]
REQUIRED_COLS = ["suburb", "property_type", "bedrooms",
                 "bathrooms", "car_spaces", "land_size_m2"]
CAUTION_PRICE = 3_500_000   # above this, out-of-fold errors grew sharply


@st.cache_resource
def load_artifacts(path="housing_model.joblib"):
    """Load the fitted pipeline and per-suburb error once, then cache it."""
    return joblib.load(path)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the SAME feature engineering as the notebook (Cell 4).

    Units have no private land (0); unknown land stays NaN and is
    imputed by the pipeline. Land is log-transformed with log1p.
    """
    df = raw.copy()
    df["suburb"] = df["suburb"].str.strip().str.title()
    df["property_type"] = df["property_type"].str.strip().str.lower()
    df.loc[df["property_type"] == "unit", "land_size_m2"] = 0
    df["log_land"] = np.log1p(df["land_size_m2"])
    return df[["suburb", "property_type", "bedrooms",
               "bathrooms", "car_spaces", "log_land"]]


def predict_prices(model, raw: pd.DataFrame) -> np.ndarray:
    """Model predicts log(price); convert back to dollars."""
    return np.exp(model.predict(build_features(raw)))


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Sydney Housing Price Predictor")
st.title("Sydney housing price predictor")
st.write("Estimates the sale price of a property in Chatswood, Castle Hill "
         "or Blacktown, based on 102 recent sales.")

try:
    artifacts = load_artifacts()
except FileNotFoundError:
    st.error("housing_model.joblib not found. Run the notebook to create it, "
             "then put it in the same folder as app.py.")
    st.stop()

model = artifacts["model"]
suburb_error = artifacts["suburb_error"]

single_tab, batch_tab = st.tabs(["Single property", "Upload a CSV"])

# ---- Single property
with single_tab:
    col1, col2 = st.columns(2)
    suburb = col1.selectbox("Suburb", SUBURBS)
    ptype = col2.selectbox("Property type", PROPERTY_TYPES)
    bedrooms = col1.number_input("Bedrooms", 1, 10, 3)
    bathrooms = col2.number_input("Bathrooms", 1, 8, 2)
    car_spaces = col1.number_input("Car spaces", 0, 10, 1)

    if ptype == "unit":
        land = 0.0
        col2.caption("Units have no private land, so land size is set to 0.")
    else:
        land_known = col2.checkbox("I know the land size", value=True)
        land = (col2.number_input("Land size (m²)", 50, 5000, 550)
                if land_known else np.nan)

    if st.button("Estimate price", type="primary"):
        row = pd.DataFrame([{
            "suburb": suburb, "property_type": ptype, "bedrooms": bedrooms,
            "bathrooms": bathrooms, "car_spaces": car_spaces,
            "land_size_m2": land,
        }])
        price = predict_prices(model, row)[0]
        err = suburb_error[suburb] / 100

        st.metric("Estimated sale price", f"${price:,.0f}")
        # \\$ escapes the dollar sign so Markdown doesn't render it as LaTeX
        st.write(f"Typical range for {suburb}: "
                 f"\\${price * (1 - err):,.0f} to \\${price * (1 + err):,.0f} "
                 f"(median error in testing: {suburb_error[suburb]}%).")

        if suburb == "Chatswood" or price > CAUTION_PRICE:
            st.warning("Treat this as a rough guide. Testing showed large "
                       "errors for Chatswood and for homes above $3.5M, where "
                       "condition, views and lot type matter more than the "
                       "features this model uses.")

# ---- Batch upload
with batch_tab:
    st.write("Upload a CSV with these columns: " + ", ".join(REQUIRED_COLS)
             + ". Leave land_size_m2 blank if unknown.")
    upload = st.file_uploader("CSV file", type="csv")

    if upload is not None:
        data = pd.read_csv(upload)
        missing = [c for c in REQUIRED_COLS if c not in data.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            bad_suburbs = set(data["suburb"].str.strip().str.title()) - set(SUBURBS)
            if bad_suburbs:
                st.warning("Unsupported suburbs will be less reliable: "
                           + ", ".join(sorted(bad_suburbs)))
            data["predicted_price"] = predict_prices(model, data).round(-3)
            st.dataframe(data, use_container_width=True)
            st.download_button("Download predictions",
                               data.to_csv(index=False),
                               "predictions.csv", "text/csv")