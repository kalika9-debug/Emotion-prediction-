import streamlit as st
import pandas as pd
import numpy as np
import string
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Emotion AI", page_icon="🧠", layout="centered")

# -------------------------------
# TEXT CLEANING
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

    # detect columns
    text_col = 'text' if 'text' in df.columns else 'sentence'
    label_col = 'emotion' if 'emotion' in df.columns else 'label'

    df['clean_text'] = df[text_col].apply(clean_text)

    return df, text_col, label_col

df, text_col, label_col = load_data()

# -------------------------------
# LABEL FIX (IMPORTANT)
# -------------------------------
# If labels are numbers → convert to meaningful text
label_map = {
    0: "sad",
    1: "happy",
    2: "angry",
    3: "anxious",
    4: "love",
    5: "surprise",
    6: "neutral"
}

# If numeric → map them
if df[label_col].dtype != 'object':
    df[label_col] = df[label_col].map(label_map)

# Ensure labels are string
df[label_col] = df[label_col].astype(str)

# -------------------------------
# TRAIN MODEL
# -------------------------------
@st.cache_resource
def train_model(df, label_col):
    X = df['clean_text']
    y = df[label_col]

    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1,2))
    X_vector = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vector, y, test_size=0.2, random_state=42, stratify=y
    )

    base_model = LinearSVC(C=2, class_weight='balanced')
    model = CalibratedClassifierCV(base_model)
    model.fit(X_train, y_train)

    return model, vectorizer

model, vectorizer = train_model(df, label_col)

# -------------------------------
# SIMPLE UI
# -------------------------------
st.title("🧠 Emotion Detection")

user_input = st.text_area("Enter text:")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter text")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]
        prediction = str(prediction).lower()

        probs = model.predict_proba(vector)[0]
        confidence = np.max(probs) * 100

        # -------------------------------
        # OUTPUT
        # -------------------------------
        st.subheader("Emotion")
        st.success(prediction)

        st.write(f"Confidence: {confidence:.2f}%")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("AI Emotion Detection App")
