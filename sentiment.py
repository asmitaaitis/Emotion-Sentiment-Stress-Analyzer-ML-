import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. LOAD AND CLEAN DATASET
# -----------------------------

def load_data():
    # Load CSV and Parquet files
    print("🔄 Loading datasets...")
    df_csv = pd.read_csv("sentiment.csv")
    df_parquet = pd.read_parquet("train-00000-of-00001.parquet")

    # Mapping dictionary to convert numbers to actual emotion words
    emotion_map = {
        '0': 'sadness/stress',
        '1': 'joy',
        '2': 'love',
        '3': 'anger',
        '4': 'fear/anxiety',
        '5': 'surprise'
    }

    def normalize(df, source_name):
        df = df.copy()
        text_col = None
        label_col = None

        # Auto-detect column names
        for col in df.columns:
            c_low = col.lower()
            if c_low in ["text", "sentence", "content"]: text_col = col
            if c_low in ["label", "sentiment", "emotion"]: label_col = col

        if not text_col or not label_col:
            raise ValueError(f"Could not find columns in {source_name}")

        df = df[[text_col, label_col]]
        df.columns = ["text", "label"]
        
        # Standardize labels: Convert to string, then map numbers to words
        df["label"] = df["label"].astype(str).str.lower()
        df["label"] = df["label"].replace(emotion_map)
        
        return df

    df_csv = normalize(df_csv, "CSV")
    df_parquet = normalize(df_parquet, "Parquet")

    # Combine datasets
    df = pd.concat([df_csv, df_parquet], ignore_index=True)
    df.dropna(inplace=True)
    
    print(f"✅ Data Loaded. Total rows: {len(df)}")
    return df

# -----------------------------
# 2. TRAIN AND SAVE MODEL
# -----------------------------

def train_and_save(df):
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline using Scikit-Learn only
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", MultinomialNB())
    ])

    print("🧠 Training the model... please wait.")
    model.fit(X_train, y_train)

    # Performance Report
    y_pred = model.predict(X_test)
    print(f"\n📊 Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\n📌 Classification Report:\n", classification_report(y_test, y_pred))

    # Save using built-in pickle (No joblib needed)
    with open('sentiment_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("💾 Model saved to 'sentiment_model.pkl'")

    return model

# -----------------------------
# 3. INTERACTIVE CHAT
# -----------------------------

if __name__ == "__main__":
    try:
        # Load and Train
        data = load_data()
        trained_model = train_and_save(data)

        print("\n--- Sentiment Analyzer Ready ---")
        while True:
            user_text = input("\nEnter text (or 'exit'): ")
            if user_text.lower() == 'exit':
                break
            
            if user_text.strip():
                prediction = trained_model.predict([user_text])[0]
                print(f"🔍 Analysis: {prediction.upper()}")

    except Exception as e:
        print(f"❌ Error: {e}")