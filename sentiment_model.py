import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SentimentModel:

    def __init__(self, model_path, tokenizer_path, max_len=100):
        self.model = load_model(model_path)
        self.tokenizer = pickle.load(open(tokenizer_path, "rb"))
        self.max_len = max_len

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def predict(self, text):

        text = self.clean_text(text)

        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.max_len)

        pred = float(self.model.predict(padded, verbose=0)[0][0])

        positive = pred
        negative = 1 - pred

        # -------------------------
        # FIXED FINAL LOGIC
        # -------------------------
        if pred >= 0.99:
            label = "Strong Positive 😊"
        elif pred >= 0.98:
            label = "Positive 🙂"
        elif pred >= 0.25:
            label = "Negative 😕"
        else:
            label = "Strong Negative 😞"

        return {
            "label": label,
            "positive": round(positive * 100, 2),
            "negative": round(negative * 100, 2),
            "raw_score": pred
        }