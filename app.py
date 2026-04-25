from flask import Flask, render_template, request
from pymongo import MongoClient
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import re
import pickle
from datetime import datetime

app = Flask(__name__)

# ---------------------------
# MongoDB Setup (safe)
# ---------------------------
MONGO_URI = os.getenv("MONGO_URI")

collection = None

if MONGO_URI:
    try:
        client = MongoClient(MONGO_URI)
        db = client["sentiment_db"]
        collection = db["reviews"]
        print("MongoDB connected successfully")
    except Exception as e:
        print("MongoDB connection failed:", e)

# ---------------------------
# Lazy loading (IMPORTANT for Render stability)
# ---------------------------
model = None
tokenizer = None
accuracy = "N/A"
max_len = 100


def load_assets():
    global model, tokenizer, accuracy

    if model is None:
        print("Loading model...")
        model = load_model("model/sentiment_model.keras")

    if tokenizer is None:
        print("Loading tokenizer...")
        with open("model/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

    if accuracy == "N/A":
        try:
            with open("model/accuracy.txt", "r") as f:
                accuracy = f.read().strip()
        except:
            accuracy = "Unknown"

    return model, tokenizer


# ---------------------------
# Text cleaning
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def welcome():
    return render_template("welcome.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    result = ""
    review = ""
    stars = ""
    positive = 0
    negative = 0

    if request.method == "POST":

        raw_review = request.form.get("review", "")

        if raw_review.strip() == "":
            return render_template(
                "index.html",
                result="Please enter a review.",
                review="",
                stars="",
                positive=0,
                negative=0,
                accuracy=accuracy
            )

        try:
            # Load model + tokenizer safely
            model, tokenizer = load_assets()

            # Clean input
            review = clean_text(raw_review)

            # Convert to sequence
            seq = tokenizer.texts_to_sequences([review])

            if len(seq[0]) == 0:
                return render_template(
                    "index.html",
                    result="Cannot determine sentiment",
                    review=raw_review,
                    stars="",
                    positive=0,
                    negative=0,
                    accuracy=accuracy
                )

            # Padding
            padded = pad_sequences(seq, maxlen=max_len)

            # Prediction
            pred = model.predict(padded, verbose=0)[0][0]

            positive = round(float(pred * 100), 2)
            negative = round(float((1 - pred) * 100), 2)

            result = "Positive 😊" if pred >= 0.5 else "Negative 😞"

            # Save to MongoDB (if available)
            if collection is not None:
                try:
                    collection.insert_one({
                        "review": raw_review,
                        "prediction": result,
                        "positive_score": positive,
                        "negative_score": negative,
                        "time": str(datetime.now())
                    })
                except Exception as e:
                    print("Mongo insert failed:", e)

        except Exception as e:
            print("Prediction error:", e)
            result = "Prediction Error"

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
# Run locally only
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
