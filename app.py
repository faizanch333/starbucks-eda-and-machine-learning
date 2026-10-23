import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(
    page_title="Starbucks Calorie Predictor",
    page_icon="☕",
    layout="centered"
)

st.title("☕ Starbucks Calorie Predictor")
st.write("Explore calorie predictions from your Decision Tree model.")

MODEL_PATH = "starbucks_decision_tree.pkl"
DATA_PATH = "starbucks.csv"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Upload the .pkl file to GitHub.")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("Upload your Starbucks dataset as starbucks.csv.")
    st.stop()

MODEL_PATH = "starbucks_decision_tree.pkl"
DATA_PATH = "starbucks.csv"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

model = load_model()
df = pd.read_csv(DATA_PATH)

# The saved pipeline should retain the feature columns
if not hasattr(model, "feature_names_in_"):
    st.error(
        "The model does not expose its input feature names. "
        "Check the columns used during training."
    )
    st.stop()

feature_cols = list(model.feature_names_in_)
missing = [col for col in feature_cols if col not in df.columns]

if missing:
    st.error(f"Dataset is missing required model features: {missing}")
    st.stop()

# Find a beverage-name column for the dropdown
name_candidates = [
    "Beverage", "Beverage_name", "Drink",
    "Item", "Product", "Beverage"
]
name_col = next(
    (col for col in name_candidates if col in df.columns),
    None
)

if name_col is None:
    st.error("Could not find a beverage-name column in the CSV.")
    st.write("Available columns:", list(df.columns))
    st.stop()

drink_names = df[name_col].fillna("Unknown").astype(str).tolist()
selected_index = st.selectbox(
    "Choose a Starbucks beverage",
    range(len(drink_names)),
    format_func=lambda i: drink_names[i]
)

if st.button("Predict Calories", type="primary"):
    input_data = df.iloc[[selected_index]][feature_cols]
    prediction = model.predict(input_data)[0]

    st.metric("Predicted Calories", f"{prediction:.1f} kcal")

    if "Calories" in df.columns:
        actual = df.iloc[selected_index]["Calories"]
        st.write(f"Dataset calories: **{actual} kcal**")

    st.caption(
        "This demo uses the selected beverage's dataset features "
        "as model input; it does not estimate a new, custom recipe."
    )
