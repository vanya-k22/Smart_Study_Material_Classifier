# 📘 Smart Study Material Classifier

## 📌 Project Description
The Smart Study Material Classifier is a Machine Learning based application that automatically
classifies uploaded study material PDFs into academic subjects such as **Math, Operating Systems (OS),
Database Management Systems (DBMS), and Artificial Intelligence (AI)**.

The system uses **Natural Language Processing (NLP)** techniques to extract and preprocess text from
PDF files and applies a **Naive Bayes classification model** to predict the most relevant subject
along with a confidence score.

---

## 🛠️ Technologies Used
- Python
- Streamlit (Web Interface)
- Scikit-learn
- Natural Language Processing (TF-IDF, Stopword Removal)
- PyPDF2
- NumPy
- NLTK

---

## ⚙️ Core Features
- Upload study material in PDF format
- Automatic text extraction from PDF
- Text preprocessing using NLP techniques
- Subject classification into predefined categories
- Confidence score based on relative probability dominance
- Simple and interactive web interface

---

## 📂 Project Structure
Smart_Study_Material_Classifier/

 -app.py # Streamlit web application
 -train_model.py # Model training logic
 -preprocess.py # Text cleaning and preprocessing
 -requirements.txt # Project dependencies
 -README.md # Project documentation
 -data/ # Training data
 math.txt
 os.txt
 dbms.txt
 ai.txt
screenshot_home.png # Application home screen
screenshot_result.png # Prediction output screen

## Confidence Score Logic

Naive Bayes models tend to produce conservative probability estimates when trained on limited
datasets. To address this, the confidence score is calculated using relative probability dominance
between the top two predicted classes, providing a more meaningful confidence representation.
