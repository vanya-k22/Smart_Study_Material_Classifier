import os
from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def load_data():
    texts = []
    labels = []

    data_path = "data"

    for file in os.listdir(data_path):
        label = file.replace(".txt", "").upper()
        with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
            content = f.read()
            content = clean_text(content)
            texts.append(content)
            labels.append(label)

    return texts, labels


def train_model():
    texts, labels = load_data()

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    model = MultinomialNB()
    model.fit(X, labels)

    return model, vectorizer
