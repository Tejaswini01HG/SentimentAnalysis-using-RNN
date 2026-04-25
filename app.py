from pymongo import MongoClient
import os
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

# Safety check (IMPORTANT)
if not MONGO_URI:
    raise ValueError("MONGO_URI not set in environment variables")


client = MongoClient(MONGO_URI)

db = client["sentiment_db"]
collection = db["reviews"]


# ---------------------------
# Load Trained RNN Model
# ---------------------------
model = load_model("model/sentiment_model.h5", compile=False)
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

        review = request.form["review"].strip().lower()

        # Empty Input Check
        if review == "":
            result = "Please enter a review."
            return render_template(
                "index.html",
                result=result,
                review=review,
                stars=stars,
                positive=positive,
                negative=negative,
                accuracy=accuracy
            )

        try:
            # Convert text to sequence
            seq = tokenizer.texts_to_sequences([review])

            # Padding
            padded = pad_sequences(seq, maxlen=max_len)

            # Prediction
            pred = model.predict(padded)[0][0]

            positive = float(round(float(pred) * 100, 2))
            negative = float(round((1 - float(pred)) * 100, 2))

            # Sentiment Decision
            if pred >= 0.5:
                result = "Positive 😊"
            
            else:
                result = "Negative 😞"

            

            # ---------------------------
            # Save to MongoDB
            # ---------------------------
            collection.insert_one({
                "review": review,
                "prediction": result,
                "positive_score": float(positive),
                "negative_score": float(negative),
                "time": str(datetime.now())
            })

        except Exception as e:
            result = "Prediction Error"
            print(e)

    return render_template(
        "index.html",
        result=result,
        review=review,
        stars=stars,
        positive=positive,
        negative=negative,
        accuracy=accuracy
    )


# ---------------------------
# Run Flask App
# ---------------------------
#if __name__ == "__main__":
    #app.run(debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
