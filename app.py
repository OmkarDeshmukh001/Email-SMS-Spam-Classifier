import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.porter import PorterStemmer

# -------------------------------
# Load model and vectorizer
# -------------------------------
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# -------------------------------
# Text preprocessing setup
# -------------------------------
ps = PorterStemmer()
stop_words = ENGLISH_STOP_WORDS


def transform_text(text):
    text = text.lower()

    # Remove non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    tokens = text.split()

    filtered_words = []
    for word in tokens:
        if word not in stop_words:
            filtered_words.append(ps.stem(word))

    return " ".join(filtered_words)


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📩 Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 Spam")
        else:
            st.success("✅ Not Spam")
