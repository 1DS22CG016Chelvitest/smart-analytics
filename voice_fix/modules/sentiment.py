import os
import pandas as pd
from transformers import pipeline

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
REVIEWS_PATH = os.path.join(BASE_DIR, "data", "reviews.csv")
REVIEWS_LABELED_PATH = os.path.join(BASE_DIR, "data", "reviews_labeled.csv")

def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

def analyze_single(text, model):
    try:
        result = model(str(text)[:512])[0]
        return result['label'], round(result['score'] * 100, 1)
    except:
        return "NEUTRAL", 50.0

def run_sentiment_on_all():
    print("Loading BERT model... (first time takes 2 mins)")
    model = load_sentiment_model()

    df = pd.read_csv(REVIEWS_PATH)
    print(f"Analyzing {len(df)} reviews...")

    labels, scores = [], []

    # ✅ FIXED COLUMN NAME HERE
    for i, text in enumerate(df['Text']):
        label, score = analyze_single(text, model)
        labels.append(label)
        scores.append(score)

        if i % 100 == 0:
            print(f"Done {i}/{len(df)}")

    df['sentiment'] = labels
    df['confidence'] = scores

    df.to_csv(REVIEWS_LABELED_PATH, index=False)
    print("Saved to reviews_labeled.csv")

    return df['sentiment'].value_counts()

def get_sentiment_summary():
    try:
        df = pd.read_csv(REVIEWS_LABELED_PATH)
        return df['sentiment'].value_counts()
    except:
        return run_sentiment_on_all()
