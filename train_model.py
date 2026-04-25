import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# CREATE MODEL FOLDER
# =====================================================
os.makedirs("model", exist_ok=True)

# =====================================================
# LOAD DATASET
# =====================================================
df = pd.read_csv("dataset.csv", low_memory=False)

df = df[['reviews.text', 'reviews.rating']]
df.dropna(inplace=True)

# =====================================================
# SENTIMENT LABEL (BINARY ONLY)
# =====================================================
def sentiment_label(rating):
    return 1 if rating >= 4 else 0   # 1=Positive, 0=Negative

df["sentiment"] = df["reviews.rating"].apply(sentiment_label)

# =====================================================
# INPUT / OUTPUT
# =====================================================
X_text = df["reviews.text"].astype(str)
y = df["sentiment"].values

# =====================================================
# TOKENIZATION
# =====================================================
max_words = 5000
max_len = 100

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_text)

X = tokenizer.texts_to_sequences(X_text)
X = pad_sequences(X, maxlen=max_len)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# BUILD RNN MODEL
# =====================================================
model = Sequential([
    Input(shape=(max_len,)),
    Embedding(input_dim=max_words, output_dim=128),
    SimpleRNN(128),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nRNN MODEL SUMMARY")
model.summary()

# =====================================================
# TRAIN MODEL
# =====================================================
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

# =====================================================
# PREDICTIONS
# =====================================================
pred = model.predict(X_test)
y_pred = (pred > 0.5).astype(int)

# =====================================================
# METRICS
# =====================================================
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("Accuracy :", round(acc * 100, 2), "%")
print("Precision:", round(pre * 100, 2), "%")
print("Recall   :", round(rec * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")

# =====================================================
# SAVE MODEL
# =====================================================
model.save("model/sentiment_model.h5")

with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("model/accuracy.txt", "w") as f:
    f.write(str(round(acc * 100, 2)))

with open("model/metrics.txt", "w") as f:
    f.write(f"Accuracy: {round(acc*100,2)}%\n")
    f.write(f"Precision: {round(pre*100,2)}%\n")
    f.write(f"Recall: {round(rec*100,2)}%\n")
    f.write(f"F1 Score: {round(f1*100,2)}%\n")

# =====================================================
# CONFUSION MATRIX
# =====================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
plt.show()

# =====================================================
# ACCURACY GRAPH
# =====================================================
plt.figure(figsize=(6, 4))
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy Graph")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("model/accuracy_graph.png")
plt.show()

print("\n✔ MODEL TRAINING COMPLETE")
print("✔ All files saved inside /model folder")