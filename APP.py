import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("herbal_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit App UI
st.title("🌿China Herbal Hospital Medicine Recommender")

st.write("""
### Enter your symptoms, and we’ll recommend the best herbal medicine!
""")

# Input field for symptoms
symptoms_input = st.text_area("Enter symptoms (e.g., high blood pressure, cough, malaria)")

if st.button("Recommend Medicine"):
    if symptoms_input.strip():
        # Convert input symptoms to vectorized form
        X_input = vectorizer.transform([symptoms_input])

        # Predict the best herbal medicine
        prediction = model.predict(X_input)[0]

        # Display result
        st.success(f"✅ Recommended Herbal Medicine: **{prediction}**")
    else:
        st.warning("⚠️ Please enter symptoms to get a recommendation.")
