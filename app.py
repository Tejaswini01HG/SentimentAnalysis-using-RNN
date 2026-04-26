import os
from flask import Flask, render_template, request
from pymongo import MongoClient
from datetime import datetime

from sentiment_model import SentimentModel

# -----------------------
# APP SETUP
# -----------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "sentiment_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "model", "tokenizer.pkl")
ACCURACY_PATH = os.path.join(BASE_DIR, "model", "accuracy.txt")

# Load model once
sentiment = SentimentModel(MODEL_PATH, TOKENIZER_PATH)

# -----------------------
# MONGO SAFE CONNECTION
# -----------------------
MONGO_URI = os.getenv("MONGO_URI")
collection = None

if MONGO_URI:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client["sentiment_db"]
        collection = db["reviews"]
        print("MongoDB connected successfully")
    except Exception as e:
        print("MongoDB connection failed:", e)
        collection = None


# -----------------------
# ACCURACY READER
# -----------------------
def get_accuracy():
    try:
        with open(ACCURACY_PATH, "r") as f:
            return f.read().strip()
    except:
        return "N/A"


# -----------------------
# ROUTES
# -----------------------
@app.route("/")
def home():
    return render_template("welcome.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    result = ""
    review_text = ""
    accuracy = get_accuracy()

    # SAFE DEFAULTS (IMPORTANT FOR PRODUCTION)
    positive = 0
    negative = 0

    try:

        if request.method == "POST":

            review_text = request.form.get("review", "").strip()

            if not review_text:
                return render_template(
                    "index.html",
                    result="Enter text",
                    review="",
                    positive=0,
                    negative=0,
                    accuracy=accuracy
                )

            # -----------------------
            # YOUR EXISTING LOGIC (UNCHANGED)
            # -----------------------
            output = sentiment.predict(review_text)

            result = output["label"]
            positive = output["positive"]
            negative = output["negative"]

            # -----------------------
            # MONGO SAVE SAFE
            # -----------------------
            if collection is not None:
                try:
                    collection.insert_one({
                        "review": review_text,
                        "prediction": result,
                        "positive": positive,
                        "negative": negative,
                        "raw_score": output["raw_score"],
                        "time": datetime.utcnow()
                    })
                except Exception as e:
                    print("MongoDB insert error:", e)

    except Exception as e:
        print("Prediction error:", e)
        result = "Server Error"

    return render_template(
        "index.html",
        result=result,
        review=review_text,
        positive=positive,
        negative=negative,
        accuracy=accuracy
    )


# -----------------------
# RUN (PRODUCTION READY SWITCH)
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
