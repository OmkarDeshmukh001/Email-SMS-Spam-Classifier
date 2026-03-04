from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import streamlit as st
import pickle

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')

# 1 text transformation
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    filtered_words = []

    for word in tokens:
        if word.isalnum() and word not in stop_words:
            filtered_words.append(ps.stem(word))

    return " ".join(filtered_words)


input_sms = st.text_area('Enter the message')
# 2 predicting the message
if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')
