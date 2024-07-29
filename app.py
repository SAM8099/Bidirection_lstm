import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
import tensorflow as tf
import streamlit as st 
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

model = load_model("models/BI_LSTM_model.h5")
model.load_weights("models/model_weights.weights.h5")
word_index = imdb.get_word_index()

def preprocess_text(text):
    words = text.lower().split()
    words = [word_index.get(word, 0) for word in words]
    return pad_sequences([words], maxlen=100)


st.title('Bidirection lstm model')
review = st.text_area('Enter your movie review:')

if st.button('Analyze Sentiment'):
    if review:
        processed_review = preprocess_text(review)
        prediction = model.predict(processed_review)
        sentiment = 'Positive' if prediction[0] > 0.5 else 'Negative'
        st.write(f'Sentiment: {sentiment}')
    else:
        st.write('Please enter a review.')