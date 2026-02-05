import streamlit as st
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load model + tokenizer
model = load_model("lstm_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 50  # same length you used in training

st.title("Fake News Detector")

text = st.text_area("Enter news text:")

if st.button("Predict"):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded)[0][0]

    if pred > 0.5:
        st.write("❌ Fake News")
    else:
        st.write("✅ Real News")
