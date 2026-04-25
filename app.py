from flask import Flask, render_template, request
from pymongo import MongoClient
import os
import re
import pickle
from datetime import datetime

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences as ps

app = Flask(__name__)

# ---------------------------
# PATHS
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sentiment_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "model", "tokenizer.pkl")
ACC_PATH = os.path.join(BASE_DIR, "model", "accuracy.txt")

# ---------------------------
# GLOBAL VARIABLES
# ---------------------------
model = None
tokenizer = None
accuracy = "N/A"
max_len = 100

# ---------------------------
# MONGO CONNECTION
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
        print(" MongoDB error:", e)

# ---------------------------
# MODEL LOADING (YOUR FINAL VERSION)
# ---------------------------
def get_model():
    global model, tokenizer, accuracy

    if model is None:
        print(" Loading model on first request only...")

        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences as ps

        model = load_model(MODEL_PATH, compile=False)

        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)

        try:
            with open(ACC_PATH, "r") as f:
                accuracy = f.read().strip()
        except:
            accuracy = "Unknown"

        print(" Model loaded")

    return model, tokenizer, ps

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

    result = ""
    review_text = ""
    positive = 0.0
    negative = 0.0

    if request.method == "POST":

        review_text = request.form.get("review", "").strip()

        if not review_text:
            return render_template(
                "index.html",
                result="Please enter text",
                review="",
                positive=0,
                negative=0,
                accuracy=accuracy
            )

        try:
            # LOAD MODEL (LAZY)
            model, tokenizer, pad_sequences = get_model()

            # CLEAN TEXT
            review = clean_text(review_text)

            # TOKENIZE
            seq = tokenizer.texts_to_sequences([review])

            if not seq or len(seq[0]) == 0:
                return render_template(
                    "index.html",
                    result="Cannot predict sentiment",
                    review=review_text,
                    positive=0,
                    negative=0,
                    accuracy=accuracy
                )

            # PAD
            padded = pad_sequences(seq, maxlen=max_len)

            # PREDICT
            pred = model.predict(padded, verbose=0)[0][0]

            positive = round(pred * 100, 2)
            negative = round((1 - pred) * 100, 2)

            result = "Positive 😊" if pred >= 0.5 else "Negative 😞"

            # SAVE TO MONGO
            if collection:
                try:
                    collection.insert_one({
                        "review": review_text,
                        "prediction": result,
                        "positive": positive,
                        "negative": negative,
                        "time": datetime.utcnow().isoformat()
                    })
                except Exception as e:
                    print("Mongo error:", e)

        except Exception as e:
            print("🔥 ERROR:", e)
            result = "Server Error"

    return render_template(
        "index.html",
        result=result,
        review=review_text,
        positive=positive,
        negative=negative,
        accuracy=accuracy
    )

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
