from flask import Flask, render_template, request
from pymongo import MongoClient
import os
import re
import pickle
from datetime import datetime

app = Flask(__name__)

# ---------------------------
# Paths (IMPORTANT FIX)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sentiment_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "model", "tokenizer.pkl")
ACC_PATH = os.path.join(BASE_DIR, "model", "accuracy.txt")

# ---------------------------
# MongoDB Setup
# ---------------------------
MONGO_URI = os.getenv("MONGO_URI")
collection = None

if MONGO_URI:
    try:
        client = MongoClient(MONGO_URI)
        db = client["sentiment_db"]
        collection = db["reviews"]
    except Exception as e:
        print("MongoDB error:", e)

# ---------------------------
# Lazy-loaded assets
# ---------------------------
model = None
tokenizer = None
accuracy = "N/A"
max_len = 100


def load_assets():
    global model, tokenizer, accuracy

    # TensorFlow imported ONLY when needed (important for Render stability)
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    if model is None:
        print("Loading model...")
        model = load_model(MODEL_PATH)

    if tokenizer is None:
        print("Loading tokenizer...")
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)

    if accuracy == "N/A":
        try:
            with open(ACC_PATH, "r") as f:
                accuracy = f.read().strip()
        except:
            accuracy = "Unknown"

    return model, tokenizer, pad_sequences


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

    raw_review = ""
    result = ""
    stars = ""
    positive = 0
    negative = 0

    if request.method == "POST":

        raw_review = request.form.get("review", "")

        if raw_review.strip() == "":
            return render_template("index.html", result="Please enter a review.")

        try:
            model, tokenizer, pad_sequences = load_assets()

            review = clean_text(raw_review)
            seq = tokenizer.texts_to_sequences([review])

            if len(seq[0]) == 0:
                return render_template(
                    "index.html",
                    result="Cannot determine sentiment",
                    review=raw_review,
                    positive=0,
                    negative=0,
                    accuracy=accuracy
                )

            padded = pad_sequences(seq, maxlen=max_len)
            pred = model.predict(padded, verbose=0)[0][0]

            positive = round(float(pred * 100), 2)
            negative = round(float((1 - pred) * 100), 2)

            result = "Positive 😊" if pred >= 0.5 else "Negative 😞"

            if collection:
                collection.insert_one({
                    "review": raw_review,
                    "prediction": result,
                    "positive_score": positive,
                    "negative_score": negative,
                    "time": str(datetime.now())
                })

        except Exception as e:
            print("Error:", e)
            result = "Prediction Error"

    return render_template(
        "index.html",
        result=result,
        review=raw_review,
        positive=positive,
        negative=negative,
        accuracy=accuracy
    )


# ---------------------------
# Run locally
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
