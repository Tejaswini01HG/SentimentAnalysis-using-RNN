from flask import Flask, render_template, request
from pymongo import MongoClient
import os
import re
import pickle
from datetime import datetime

app = Flask(__name__)

# ---------------------------
# BASE PATHS
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sentiment_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "model", "tokenizer.pkl")
ACC_PATH = os.path.join(BASE_DIR, "model", "accuracy.txt")

# ---------------------------
# SAFE FILE CHECK
# ---------------------------
def safe_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

# ---------------------------
# MONGODB (OPTIONAL)
# ---------------------------
MONGO_URI = os.getenv("MONGO_URI")
collection = None

if MONGO_URI:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client["sentiment_db"]
        collection = db["reviews"]
        print("MongoDB connected")
    except Exception as e:
        print("MongoDB error:", e)

# ---------------------------
# ML ASSETS (LAZY LOAD)
# ---------------------------
model = None
tokenizer = None
pad_sequences = None
accuracy = "N/A"
max_len = 100


def load_assets():
    global model, tokenizer, pad_sequences, accuracy

    # TensorFlow import only when needed (IMPORTANT for Render)
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences as ps

    pad_sequences = ps

    # Load model
    if model is None:
        safe_exists(MODEL_PATH)
        model = load_model(MODEL_PATH)

    # Load tokenizer
    if tokenizer is None:
        safe_exists(TOKENIZER_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)

    # Load accuracy
    if accuracy == "N/A":
        try:
            safe_exists(ACC_PATH)
            with open(ACC_PATH, "r") as f:
                accuracy = f.read().strip()
        except:
            accuracy = "Unknown"

    return model, tokenizer, pad_sequences

# ---------------------------
# TEXT CLEANING
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------
# ROUTES
# ---------------------------
@app.route("/")
def home():
    return render_template("welcome.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    raw_review = ""
    result = ""
    positive = 0.0
    negative = 0.0

    if request.method == "POST":
        raw_review = request.form.get("review", "").strip()

        if not raw_review:
            return render_template(
                "index.html",
                result="Please enter a review.",
                review="",
                positive=0,
                negative=0,
                accuracy=accuracy
            )

        try:
            model, tokenizer, pad_sequences = load_assets()

            review = clean_text(raw_review)
            seq = tokenizer.texts_to_sequences([review])

            if not seq or len(seq[0]) == 0:
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

            # MongoDB safe insert
            if collection:
                try:
                    collection.insert_one({
                        "review": raw_review,
                        "prediction": result,
                        "positive_score": positive,
                        "negative_score": negative,
                        "time": datetime.utcnow().isoformat()
                    })
                except Exception as e:
                    print("Mongo insert failed:", e)

        except Exception as e:
            print("Prediction error:", e)
            result = "Server Error"

    return render_template(
        "index.html",
        result=result,
        review=raw_review,
        positive=positive,
        negative=negative,
        accuracy=accuracy
    )

# ---------------------------
# RUN (RENDER SAFE)
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
