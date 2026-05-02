import streamlit as st
import pandas as pd
import numpy as np
import joblib

from geopy.geocoders import Nominatim
from sklearn.neighbors import BallTree

# =========================
# App Config
# =========================
st.set_page_config(
    page_title="Selangor House Price Prediction",
    layout="wide"
)

MODEL_PATH = "house_price_model_compressed.pkl"
POI_PATH = "selangor_pois_clean.csv"
EARTH_RADIUS_KM = 6371
MODEL_ACCURACY = 92.25

# =========================
# Load Model and POI
# =========================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_pois():
    return pd.read_csv(POI_PATH)

model = load_model()
pois = load_pois()

# =========================
# Helper Functions
# =========================
@st.cache_data(show_spinner=False)
def geocode_address(address):
    geolocator = Nominatim(
        user_agent="selangor_house_price_prediction_app",
        timeout=15
    )

    location = geolocator.geocode(address)

    if location:
        return location.latitude, location.longitude

    return None, None


@st.cache_resource
def build_balltree(category):
    poi_cat = pois[
        (pois["poi_category"] == category) &
        (pois["poi_lat"].notnull()) &
        (pois["poi_lon"].notnull())
    ]

    if len(poi_cat) == 0:
        return None

    coords = np.radians(poi_cat[["poi_lat", "poi_lon"]].values)
    return BallTree(coords, metric="haversine")


def nearest_distance(lat, lon, category):
    tree = build_balltree(category)

    if tree is None:
        return np.nan

    house_coord = np.radians([[lat, lon]])
    dist, _ = tree.query(house_coord, k=1)

    return round(dist[0][0] * EARTH_RADIUS_KM, 3)


def count_within_radius(lat, lon, category, radius_km):
    tree = build_balltree(category)

    if tree is None:
        return 0

    house_coord = np.radians([[lat, lon]])
    radius_rad = radius_km / EARTH_RADIUS_KM

    count = tree.query_radius(
        house_coord,
        r=radius_rad,
        count_only=True
    )

    return int(count[0])


def hospital_group(count):
    if count <= 2:
        return "0-2"
    elif count <= 5:
        return "3-5"
    elif count <= 10:
        return "6-10"
    elif count <= 20:
        return "11-20"
    else:
        return "20+"


def align_input_to_model(input_df, model):
    """
    Ensures input columns match the training columns used by the pipeline.
    Missing columns are added as NaN.
    Extra columns are removed.
    """
    expected_cols = list(model.feature_names_in_)

    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = np.nan

    input_df = input_df[expected_cols]

    return input_df


# =========================
# UI
# =========================
st.title("🏠 Selangor House Price Prediction")
st.caption(f"Model: Random Forest Regressor | Model Accuracy: {MODEL_ACCURACY:.2f}%")

st.write(
    "Enter property details and address. The system will geocode the address, "
    "calculate nearby facilities using POI data, and predict the expected house price."
)

col1, col2 = st.columns(2)

with col1:
    property_type = st.selectbox(
        "Property Type",
        [
            "Terrace House",
            "Semi Detached",
            "Detached",
            "Low-Cost House",
            "Cluster House",
            "Town House",
            "Apartment",
            "Condominium"
        ]
    )

    district = st.selectbox(
        "District",
        [
            "Petaling",
            "Gombak",
            "Hulu Langat",
            "Klang",
            "Sepang",
            "Kuala Langat",
            "Kuala Selangor",
            "Hulu Selangor",
            "Sabak Bernam"
        ]
    )

    tenure = st.selectbox(
        "Tenure",
        ["Freehold", "Leasehold"]
    )

    transaction_date = st.date_input("Transaction Date")

with col2:
    land_area = st.number_input(
        "Land Area",
        min_value=1.0,
        value=150.0
    )

    main_floor_area = st.number_input(
        "Main Floor Area",
        min_value=1.0,
        value=120.0
    )

    address = st.text_input(
        "Property Address",
        "Shah Alam, Selangor, Malaysia"
    )

# =========================
# Prediction
# =========================
if st.button("Predict House Price", type="primary"):

    full_address = address

    if "selangor" not in full_address.lower():
        full_address = full_address + ", Selangor, Malaysia"

    with st.spinner("Finding latitude and longitude using Nominatim..."):
        lat, lon = geocode_address(full_address)

    if lat is None or lon is None:
        st.error("Address could not be geocoded. Please enter a more complete Selangor address.")
        st.stop()

    categories = [
        "school",
        "hospital",
        "clinic",
        "mall",
        "supermarket",
        "park",
        "station"
    ]

    with st.spinner("Calculating nearby POI features..."):

        feature_dict = {
            "Property Type": property_type,
            "District": district,
            "Tenure": tenure,
            "LandArea": land_area,
            "MainFloorArea": main_floor_area,
            "TransactionDate": str(transaction_date),
            "TransactionYear": transaction_date.year,
            "TransactionMonth": transaction_date.month
        }

        # Nearest distance features
        for category in categories:
            feature_dict[f"dist_nearest_{category}_km"] = nearest_distance(
                lat,
                lon,
                category
            )

        # Count features for 1km, 2km, 5km
        for radius in [1, 2, 5]:
            for category in categories:
                feature_dict[f"count_{category}_within_{radius}km"] = count_within_radius(
                    lat,
                    lon,
                    category,
                    radius
                )

        # Grouped hospital feature
        feature_dict["Number_of_hospital_within_2km"] = hospital_group(
            feature_dict["count_hospital_within_2km"]
        )

        input_df = pd.DataFrame([feature_dict])
        input_df = align_input_to_model(input_df, model)

        log_prediction = model.predict(input_df)[0]
        predicted_price = np.expm1(log_prediction)

    # =========================
    # Results
    # =========================
    st.subheader("Prediction Result")

    st.metric(
        label="Estimated House Price",
        value=f"RM {predicted_price:,.2f}"
    )

    st.info(f"Model Accuracy: {MODEL_ACCURACY:.2f}% based on Random Forest R² performance.")

    st.subheader("Detected Location")

    st.write(f"Latitude: `{lat}`")
    st.write(f"Longitude: `{lon}`")

    st.subheader("Nearest Facility Distance")

    distance_cols = [
        col for col in input_df.columns
        if col.startswith("dist_nearest")
    ]

    distance_df = input_df[distance_cols].T.reset_index()
    distance_df.columns = ["Facility", "Distance (km)"]

    distance_df["Facility"] = (
        distance_df["Facility"]
        .str.replace("dist_nearest_", "", regex=False)
        .str.replace("_km", "", regex=False)
        .str.replace("_", " ")
        .str.title()
    )

    st.dataframe(distance_df, use_container_width=True)

    st.subheader("Facility Count within Radius")

    count_cols = [
        col for col in input_df.columns
        if col.startswith("count_")
    ]

    count_df = input_df[count_cols].T.reset_index()
    count_df.columns = ["Facility Count Feature", "Count"]

    count_df["Facility Count Feature"] = (
        count_df["Facility Count Feature"]
        .str.replace("count_", "", regex=False)
        .str.replace("_within_", " within ", regex=False)
        .str.replace("_", " ")
        .str.title()
    )

    st.dataframe(count_df, use_container_width=True)

    st.subheader("Model Input Used")

    st.dataframe(input_df, use_container_width=True)