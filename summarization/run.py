from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string

nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    filtered_words = [w for w in words if w not in set(stopwords.words('english'))]
    return ' '.join(filtered_words)

def summarize_text(text):
    cleaned = clean_text(text)
    sents = sent_tokenize(text)
    if len(sents) <= 2:
        return " ".join(sents), list(range(len(sents)))
    
    processed_sents = sent_tokenize(cleaned)
    if len(processed_sents) != len(sents):
        return " ".join(sents[:2]), list(range(2))
    
    tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf_vec.fit_transform(processed_sents)
    sim_matrix = cosine_similarity(tfidf_matrix)
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank_numpy(nx_graph)
    
    ranked_sents = sorted(((scores[i], i) for i in range(len(sents))), key=lambda x: x[0], reverse=True)
    top_indices = sorted([i for _, i in ranked_sents[:2]])
    top_sents = [sents[i] for i in top_indices]
    
    summary = " ".join(top_sents)
    return summary, top_indices

if __name__ == "__main__":
    tira_client = Client()
    data = tira_client.pd.inputs("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training").set_index("id")
    print("Data size: ", data.size)
    
    data["summary"], data["indices"] = zip(*data["story"].apply(lambda x: summarize_text(x)))
    result_df = data.drop(columns=["story", "indices"]).reset_index()
    
    output_dir = get_output_directory(str(Path(__file__).parent))
    result_df.to_json(Path(output_dir) / "predictions.jsonl", orient="records", lines=True)
