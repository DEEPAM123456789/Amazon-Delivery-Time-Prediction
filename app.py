
import streamlit as st
import pandas as pd
import joblib
import numpy as np

MODEL_PATH = "xgb_pipeline_final.joblib"

@st.cache_resource
def load_model(path=MODEL_PATH):
    return joblib.load(path)

pipe = load_model()

st.title("Amazon Delivery Time Predictor")
st.markdown("Fill the form, then click Predict. **Make sure categorical choices match training spellings.case.**")

# ---- Inputs ----
col1, col2 = st.columns(2)

with col1:
    Agent_Age = st.number_input("Agent Age", min_value=16, max_value=60, value=30, step=1)
    Agent_Rating = st.number_input("Agent Rating (0-5)", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
    Weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Fog", "Windy", "Sandstorms", "Stormy"])
    Traffic = st.selectbox("Traffic", ["Low ", "Medium ", "High ", "Jam "])
    Vehicle = st.selectbox("Vehicle", ["motorcycle ", "scooter ", "van"])
    Area = st.selectbox("Area", ["Urban ", "Semi-Urban ", "Metropolitan ", "Other"])

with col2:
    Category = st.selectbox("Category", ['Clothing', 'Electronics', 'Sports', 'Cosmetics', 'Toys', 'Snacks', 'Shoes', 'Apparel', 'Jewelry', 'Outdoors', 'Grocery', 'Books', 'Kitchen', 'Pet Supplies', 'Skincare', 'Home'])
    distance_km = st.number_input("Distance (km)", min_value=0.0, max_value=22.0, value=5.0, step=0.1)
    order_dayofweek = st.selectbox("Order Day (0=Mon .. 6=Sun)", list(range(7)), index=4)
    order_month = st.selectbox("Order Month", list(range(1,13)), index=2)
    order_hour = st.slider("Order Hour (0-23)", 0, 23, 12)

# ---- Derived Features (exact names from X_train data) ----
is_weekend = int(order_dayofweek >= 5) # 5=Sat, 6=Sun
is_rush_hour = int((8 <= order_hour <= 10) or (18 <= order_hour <= 21))

traffic_weather = f"{Traffic.strip()}_{Weather.strip()}"
vehicle_traffic = f"{Vehicle.strip()}_{Traffic.strip()}"
vehicle_category = f"{Vehicle.strip()}_{Category.strip()}"

# Build Dataframe exactly in the same column order
cols = ['Agent_Age', 'Agent_Rating', 'Weather', 'Traffic', 'Vehicle', 'Area',
        'Category', 'distance_km', 'order_dayofweek', 'order_month',
        'is_weekend', 'order_hour', 'is_rush_hour', 'traffic_weather',
        'vehicle_traffic', 'vehicle_category']

input_df = pd.DataFrame([{
    'Agent_Age': int(Agent_Age),
    'Agent_Rating': float(Agent_Rating),
    'Weather': Weather.strip(),
    'Traffic': Traffic.strip(),
    'Vehicle': Vehicle.strip(),
    'Area': Area.strip(),
    'Category': Category.strip(),
    'distance_km': float(distance_km),
    'order_dayofweek': int(order_dayofweek),
    'order_month': int(order_month),
    'is_weekend': int(is_weekend),
    'order_hour': int(order_hour),
    'is_rush_hour': int(is_rush_hour),
    'traffic_weather': traffic_weather,
    'vehicle_traffic': vehicle_traffic,
    'vehicle_category': vehicle_category
}], columns=cols)

st.write('### Input Preview')

# ---- Predict ----
if st.button("Predict Delivery Time"):
    try:
        pred = pipe.predict(input_df)
        st.success(f"Predicted delivery time: {pred[0]:.2f} minutes")
    except Exception as e:
        st.error("Prediction failed - see debug info below.")
        st.write("Error", e)
        # helpful debug info
        st.write("Pipeline steps:", list(pipe.named_steps.keys()))
        # try to get feature names from preprocessor (if available)
        pre = None
        for name, step in pipe.named_steps.items():
            if "ColumnTransformer" in str(type(step)) or hasattr(step, "transform"):
                pre = step
                break
        try:
            st.write("Preprocessor features names (if available):")
            st.write(pre.get_feature_names_out())
        except Exception:
            st.write("Could not get feature names from preprocessor. Inspect pipeline in notebook if needed.")
