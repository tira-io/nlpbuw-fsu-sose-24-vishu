from tira.rest_api_client import Client
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk

nltk.download('punkt')

# Function to preprocess text
def preprocess(text):
    text = text.str.lower().str.replace(r'[^a-z\s]', '', regex=True)
    return text

# Function to calculate sentence embeddings
def calculate_embeddings(sentence1, sentence2, model):
    embeddings1 = model.encode(sentence1.tolist())
    embeddings2 = model.encode(sentence2.tolist())
    return embeddings1, embeddings2

# Function to calculate cosine similarity
def cosine_similarity(embeddings1, embeddings2):
    cos_sim = np.sum(embeddings1 * embeddings2, axis=1) / (np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1))
    return cos_sim

# Function to calculate Levenshtein distance
def levenshtein_distance(df: pd.DataFrame):
    distances = df.apply(lambda row: nltk.edit_distance(nltk.word_tokenize(row["sentence1"]), nltk.word_tokenize(row["sentence2"])), axis=1)
    return distances

if __name__ == "__main__":
    # Load the data
    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")
    text = text.join(labels)

    # Preprocess text
    text["sentence1"] = preprocess(text["sentence1"])
    text["sentence2"] = preprocess(text["sentence2"])

    # Load sentence transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Calculate sentence embeddings
    embeddings1, embeddings2 = calculate_embeddings(text["sentence1"], text["sentence2"], model)
    
    # Calculate cosine similarity
    text["similarity"] = cosine_similarity(embeddings1, embeddings2)

    # Calculate Levenshtein distance
    text["levenshtein_distance"] = levenshtein_distance(text[["sentence1", "sentence2"]])

    # Combine features
    X = text[["similarity", "levenshtein_distance"]]
    y = text["label"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate Matthews correlation coefficient
    mcc = matthews_corrcoef(y_test, y_pred)
    print(f"Matthews Correlation Coefficient: {mcc}")

    # Find the best threshold
    thresholds = sorted(text["similarity"].unique())
    mccs = {}
    for threshold in thresholds:
        y_pred = (text["similarity"] >= threshold).astype(int)
        mcc = matthews_corrcoef(text["label"], y_pred)
        mccs[threshold] = mcc
    
    best_threshold = max(mccs, key=mccs.get)
    print(f"Best threshold: {best_threshold}")
