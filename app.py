import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

# ---------- LOAD ----------
model = load_model("model/lstm_model.h5", compile=False)
scaler = joblib.load("model/scaler.save")

st.set_page_config(page_title="JARVIS CNC AI", layout="wide")

st.title("🤖 CNC Predictive Intelligence System")

# ---------- AUGMENT ----------
def augment_data(X):
    return X + np.random.normal(0, 0.01, X.shape)

# ---------- PROCESS ----------
def process_input(X):
    X = augment_data(X)
    X = scaler.transform(X)
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    return X

# ---------- CSV ----------
st.header("📂 Upload CSV")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())

    if st.button("Predict CSV"):
        if "tool_wear" in df.columns:
            X = df.drop("tool_wear", axis=1).values
        else:
            X = df.values

        X = process_input(X)
        preds = model.predict(X)

        df["Prediction"] = preds
        st.dataframe(df)

        if "tool_wear" in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=df["tool_wear"], name="Actual"))
            fig.add_trace(go.Scatter(y=df["Prediction"], name="Prediction"))
            st.plotly_chart(fig)

# ---------- MANUAL ----------
st.header("⚙️ Manual Input")

vibration = st.number_input("Vibration", 0.0)
temperature = st.number_input("Temperature", 30.0)
force = st.number_input("Force", 50.0)
spindle = st.number_input("Spindle Speed", 2000.0)
feed = st.number_input("Feed Rate", 200.0)

if st.button("Predict Single"):
    X = np.array([[vibration, temperature, force, spindle, feed]])
    X = process_input(X)
    pred = model.predict(X)
    st.success(f"Tool Wear: {pred[0][0]:.4f}")

# ---------- LIVE ----------
st.header("📡 Live Simulation")

run = st.button("Start Monitoring")
placeholder = st.empty()

if run:
    for i in range(30):
        X = np.random.rand(1,5)
        X = process_input(X)
        pred = model.predict(X)[0][0]
        placeholder.metric("Live Tool Wear", f"{pred:.4f}")
        time.sleep(0.5)
