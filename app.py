import streamlit as st
import joblib

# Load your trained AM-PM model
model = joblib.load("am_pm_progress_model.pkl")

st.title("📊 Student AM-PM Progress Tracker")

st.markdown("""
This app predicts a student's daily progress level by comparing their morning (AM) plans
to their evening (PM) outcomes.
""")

# AM and PM inputs
am_text = st.text_area("Enter AM (morning) check-in:", placeholder="E.g. Today I will work on my Personal Project")
pm_text = st.text_area("Enter PM (evening) check-in:", placeholder="E.g. I completed the Scene for my Personal Project")

# Predict on click
if st.button("Analyze Daily Progress"):
    if am_text.strip() and pm_text.strip():
        combined_text = f"AM: {am_text.strip()} PM: {pm_text.strip()}"
        prediction = model.predict([combined_text])[0]
        st.success(f"Predicted Progress Level: **{prediction}**")
    else:
        st.warning("Please enter both AM and PM check-ins to analyze.")

st.markdown("""
---
✅ **Powered by your own AM-PM check-in dataset and trained ML model.**  
Uses logistic regression on TF-IDF features to classify daily progress.
""")

