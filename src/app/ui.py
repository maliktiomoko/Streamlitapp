import streamlit as st
from .pipeline_tabular import run_tabular_forecasting
from .pipeline_time_series import run_time_series_forecasting

def render_dataset_tab():
    st.subheader("📁 Upload and Describe Dataset")
    st.text("Coming soon: unified upload + config handler")

def render_forecasting_tab():
    st.subheader("📈 Forecasting Pipeline")
    # Call run_tabular_forecasting or run_time_series_forecasting here with placeholder values
    st.text("Forecasting logic will be routed based on problem type")

def render_feedback_tab():
    st.subheader("🧠 Feedback Phase")
    st.text("Placeholder for reinforcement-based feedback loop")