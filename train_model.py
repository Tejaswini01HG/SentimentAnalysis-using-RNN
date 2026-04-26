import os
import re
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------
# CONFIG
# -----------------------
MAX_WORDS = 10000
MAX_LEN = 100

# -----------------------
# CLEAN TEXT
# -----------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv("dataset.csv", low_memory=False)

df = df[['reviews.text', 'reviews.rating']]
df.dropna(inplace=True)

# -----------------------
# FIXED LABELING (UNCHANGED LOGIC)
# -----------------------
def label_sentiment(rating):
    if rating >= 4:
        return 1
    elif rating <= 2:
        return 0
    else:
        return None

df["sentiment"] = df["reviews.rating"].apply(label_sentiment)
df = df.dropna()

df["clean"] = df["reviews.text"].apply(clean_text)

X = df["clean"].values
y = df["sentiment"].values

# -----------------------
# TOKENIZER
# -----------------------
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X)

X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=MAX_LEN)

# -----------------------
# SPLIT
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y, test_size=0.2, random_state=42
)

# -----------------------
# MODEL (UNCHANGED)
# -----------------------
model = Sequential([
    Embedding(MAX_WORDS, 128),

    LSTM(128, return_sequences=True),
    Dropout(0.3),

    LSTM(64),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ⭐ ONLY ADDITION (FIX SUMMARY BOX)
model.build(input_shape=(None, MAX_LEN))

# -----------------------
# SHOW MODEL STRUCTURE
# -----------------------
print("\n========== MODEL ARCHITECTURE ==========")
model.summary()

# -----------------------
# TRAIN
# -----------------------
model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# -----------------------
# EVALUATE
# -----------------------
loss, acc = model.evaluate(X_test, y_test)
print("\nTEST ACCURACY:", acc)

# -----------------------
# PREDICTIONS
# -----------------------
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# -----------------------
# METRICS
# -----------------------
print("\n========== MODEL PERFORMANCE ==========")

print("ACCURACY:", accuracy_score(y_test, y_pred))
print("PRECISION:", precision_score(y_test, y_pred))
print("RECALL:", recall_score(y_test, y_pred))
print("F1 SCORE:", f1_score(y_test, y_pred))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

# -----------------------
# SAVE MODEL
# -----------------------
os.makedirs("model", exist_ok=True)

model.save("model/sentiment_model.h5")

with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("model/accuracy.txt", "w") as f:
    f.write(str(acc))

print("\nMODEL SAVED SUCCESSFULLY")
