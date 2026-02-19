# Fake News Detection App (Fixed Version for Kaggle Fake.csv & True.csv)

import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")
st.title("📰 Fake News & Misinformation Detection App")

# -------------------------------
# Load Dataset (Robust Fix)
# -------------------------------
@st.cache_data
def load_data():
    fake = pd.read_csv("Fake.csv", encoding="utf-8", on_bad_lines="skip")
    true = pd.read_csv("True.csv", encoding="utf-8", on_bad_lines="skip")

    fake["label"] = 0
    true["label"] = 1

    # Combine & reset index (IMPORTANT FIX)
    data = pd.concat([fake, true], ignore_index=True)

    # Select only required columns
    data = data[["title", "text", "label"]]

    # Handle missing + convert to string
    data["title"] = data["title"].fillna("").astype(str)
    data["text"] = data["text"].fillna("").astype(str)

    # Combine text safely
    data["content"] = data["title"] + " " + data["text"]

    return data

data = load_data()

# -------------------------------
# Clean Text (Safe Version)
# -------------------------------
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["content"] = data["content"].apply(clean_text)

# -------------------------------
# Train Model
# -------------------------------
@st.cache_resource
def train_model(data):
    X = data["content"].astype(str)
    y = data["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    return model, vectorizer, y_test, y_pred, acc

model, vectorizer, y_test, y_pred, acc = train_model(data)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Options")
if st.sidebar.checkbox("Show Dataset Preview"):
    st.dataframe(data.head())

if st.sidebar.checkbox("Show Model Performance"):
    st.subheader("📊 Accuracy")
    st.write(f"Accuracy: {acc:.2f}")

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# -------------------------------
# User Prediction
# -------------------------------
st.subheader("🔍 Enter News Text")
user_input = st.text_area("Paste news article or headline here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter news text.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.success("✅ This appears to be REAL NEWS")
        else:
            st.error("🚨 This appears to be FAKE NEWS")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("Fake News Detection using TF-IDF + Logistic Regression")
