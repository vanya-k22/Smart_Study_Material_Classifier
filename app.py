import streamlit as st
import PyPDF2
import numpy as np
from train_model import train_model
from preprocess import clean_text

# Page config
st.set_page_config(
    page_title="Smart Study Material Classifier",
    page_icon="📘",
    layout="centered"
)

# Title
st.title("📘 Smart Study Material Classifier")
st.write("Upload your study notes PDF to predict the subject.")

# Train model once
model, vectorizer = train_model()

# File upload
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    if text.strip() != "":
        # Preprocess text
        cleaned_text = clean_text(text)
        vec_text = vectorizer.transform([cleaned_text])

        # Prediction
        prediction = model.predict(vec_text)[0]

     
        probs = model.predict_proba(vec_text)[0]
        sorted_probs = np.sort(probs)

        confidence = (sorted_probs[-1] / (sorted_probs[-1] + sorted_probs[-2])) * 100
        confidence = round(confidence, 2)
        # --------------------------------

      
        st.success(f"📚 Predicted Subject: **{prediction}**")
        st.info(f"🔍 Confidence Score: **{confidence}%**")

    else:
        st.error("❌ Could not extract readable text from the PDF.")
