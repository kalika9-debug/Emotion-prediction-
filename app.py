import streamlit as st
import pandas as pd
import numpy as np
import string
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# -------------------------------
# TEXT CLEANING FUNCTION
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("training.csv")
    df['clean_text'] = df['text'].apply(clean_text)
    return df

phrases = load_data()

# -------------------------------
# MODEL TRAINING
# -------------------------------
@st.cache_resource
def train_model(df):
    X = df['clean_text']
    y = df['emotion']

    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1,2),
        min_df=2,
        sublinear_tf=True
    )

    X_vector = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vector, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LinearSVC(C=2, class_weight='balanced')
    model.fit(X_train, y_train)

    return model, vectorizer

model, vectorizer = train_model(phrases)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Emotion AI", page_icon="🧠")

st.title("🧠 Emotion Prediction App")
st.write("Enter a sentence and detect the emotion")

user_input = st.text_area("Enter text here:")

if st.button("Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)

        st.success(f"Predicted Emotion: {prediction[0]}")
