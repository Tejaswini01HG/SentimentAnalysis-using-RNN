import os
import pickle
import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

os.makedirs("model", exist_ok=True)

max_words = 10000
max_len = 100

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df = pd.read_csv("dataset.csv", low_memory=False)
df = df[['reviews.text', 'reviews.rating']]
df.dropna(inplace=True)

def label_sentiment(rating):
    return 1 if rating >= 4 else 0

df["sentiment"] = df["reviews.rating"].apply(label_sentiment)
df["clean"] = df["reviews.text"].apply(clean_text)

X_text = df["clean"].values
y = df["sentiment"].values

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_text)

X = tokenizer.texts_to_sequences(X_text)
X = pad_sequences(X, maxlen=max_len)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {0: weights[0], 1: weights[1]}

model = Sequential([
    Input(shape=(max_len,)),
    Embedding(max_words, 128),

    LSTM(128, return_sequences=True),
    Dropout(0.4),

    LSTM(64),
    Dropout(0.4),

    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.build(input_shape=(None, max_len))
print(model.summary())

early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early_stop]
)

pred = model.predict(X_test)
y_pred = (pred > 0.5).astype(int)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

model.save("model/sentiment_model.keras")

with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("MODEL SAVED")
