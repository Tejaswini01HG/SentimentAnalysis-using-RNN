import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = r"C:\Users\LENOVO\OneDrive\Desktop\SentimentalAnyalisis-RNN\Sentiment_project\model\sentiment_model.h5"
TOKENIZER_PATH = r"C:\Users\LENOVO\OneDrive\Desktop\SentimentalAnyalisis-RNN\Sentiment_project\model\tokenizer.pkl"

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

tests = [
    "worst",
    "worst product ever",
    "this is the worst experience",
    "excellent product",
    "I love this",
    "this awful",
    "the product is bad"
]

for t in tests:
    review = clean_text(t)
    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, maxlen=100)

    pred = model.predict(padded, verbose=0)[0][0]

    print(f"{t} → {pred}")