from pymongo import MongoClient
import os
import re
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from datetime import datetime

app = Flask(__name__)

# ---------------------------
# MongoDB Connection
# ---------------------------
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI not set in environment variables")

client = MongoClient(MONGO_URI)

db = client["sentiment_db"]
collection = db["reviews"]

# ---------------------------
# Load Trained Model
# ---------------------------
model = load_model("model/sentiment_model.keras")

# ---------------------------
# Load Tokenizer
# ---------------------------
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ---------------------------
# Load Accuracy
# ---------------------------
with open("model/accuracy.txt", "r") as f:
    accuracy = f.read().strip()

max_len = 100

# ---------------------------
# CLEAN TEXT (CRITICAL FIX)
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------
# Welcome Page
# ---------------------------
@app.route("/")
def welcome():
    return render_template("welcome.html")

# ---------------------------
# Prediction Page
# ---------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():

    result = ""
    review = ""
    stars = ""
    positive = 0
    negative = 0

    if request.method == "POST":

        raw_review = request.form["review"]

        # Empty Input Check
        if raw_review.strip() == "":
            result = "Please enter a review."
            return render_template(
                "index.html",
                result=result,
                review=raw_review,
                stars=stars,
                positive=positive,
                negative=negative,
                accuracy=accuracy
            )

        try:
            # ✅ CLEAN TEXT
            review = clean_text(raw_review)

            # Convert text to sequence
            seq = tokenizer.texts_to_sequences([review])

            # ✅ HANDLE UNKNOWN WORDS
            if len(seq[0]) == 0:
                result = "Cannot determine"
                return render_template(
                    "index.html",
                    result=result,
                    review=raw_review,
                    stars=stars,
                    positive=positive,
                    negative=negative,
                    accuracy=accuracy
                )

            # Padding
            padded = pad_sequences(seq, maxlen=max_len)

            # Prediction
            pred = model.predict(padded, verbose=0)[0][0]

            positive = float(round(pred * 100, 2))
            negative = float(round((1 - pred) * 100, 2))

            # Sentiment Decision
            if pred >= 0.5:
                result = "Positive 😊"
            else:
                result = "Negative 😞"

            # ---------------------------
            # Save to MongoDB
            # ---------------------------
            collection.insert_one({
                "review": raw_review,
                "prediction": result,
                "positive_score": positive,
                "negative_score": negative,
                "time": str(datetime.now())
            })

        except Exception as e:
            result = "Prediction Error"
            print(e)

    return render_template(
        "index.html",
        result=result,
        review=raw_review if request.method == "POST" else "",
        stars=stars,
        positive=positive,
        negative=negative,
        accuracy=accuracy
    )

# ---------------------------
# Run Flask App
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
