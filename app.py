import joblib
import pandas as pd
import streamlit as st

from src.features import FEATURES, add_features

st.set_page_config(page_title="Bucharest Apartment Price Estimator")


@st.cache_resource
def load_model():
    return joblib.load("models/price_model.joblib")


model = load_model()

st.title("Bucharest Apartment Price Estimator")
st.write(
    "Estimate the price of an apartment in Bucharest. "
    "The model was trained on listings from March 2019, "
    "so estimates reflect 2019 price levels."
)

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sqm)", min_value=15.0, max_value=350.0, value=60.0, step=1.0)
    rooms = st.number_input("Rooms", min_value=1, max_value=9, value=2)
    sector = st.selectbox("Sector", [1, 2, 3, 4, 5, 6])

with col2:
    total_floors = st.number_input("Total floors in building", min_value=1, max_value=24, value=8)
    floor = st.number_input("Floor (0 = ground floor)", min_value=-1, max_value=24, value=2)
    score = st.selectbox("Zone score (1 = best, 5 = worst)", [1, 2, 3, 4, 5], index=2)

if st.button("Estimate price"):
    if floor > total_floors:
        st.error("Floor cannot be higher than the total number of floors.")
    else:
        # Raw inputs only; engineered features come from the shared function
        apartment = pd.DataFrame([{
            "rooms": rooms,
            "area": area,
            "floor": floor,
            "total_floors": total_floors,
            "sector": sector,
            "score": score,
        }])
        apartment = add_features(apartment)[FEATURES]

        price = model.predict(apartment)[0]

        st.metric("Estimated price", f"€{price:,.0f}")
        st.caption(
            f"Typical error is about ±16%: "
            f"roughly €{price * 0.84:,.0f} – €{price * 1.16:,.0f}."
        )

        if price > 150_000:
            st.warning(
                "The model is less reliable above €150,000 "
                "and tends to underestimate these apartments."
            )