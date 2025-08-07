import subprocess
import os
import streamlit as st

def submit_to_kaggle(submission_file, competition_name):
    st.info("📤 Submitting file to Kaggle...")
    result = subprocess.run([
        "kaggle", "competitions", "submit",
        "-c", competition_name,
        "-f", submission_file,
        "-m", "Auto submission from Streamlit"
    ], capture_output=True, text=True)
    st.code(result.stdout)

    if "Successfully submitted" in result.stdout:
        st.success("✅ Submitted successfully!")
    else:
        st.error("❌ Submission failed. See logs above.")