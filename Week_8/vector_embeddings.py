
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    df_text = df.apply(lambda x: ", ".join([f"{col}: {val}" for col, val in x.items()]), axis=1)
    return df_text.tolist()

def create_and_save_embeddings(data, output_path, vectorizer_path):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(data)
    
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Embeddings saved to {output_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

if __name__ == "__main__":
    training_data = load_and_process_data("/home/ubuntu/upload/TrainingDataset.csv")
    test_data = load_and_process_data("/home/ubuntu/upload/TestDataset.csv")

    create_and_save_embeddings(training_data, "training_embeddings.pkl", "tfidf_vectorizer.pkl")
    # For the RAG system, we primarily need embeddings for the knowledge base (training data).
    # Test data embeddings might be used for evaluating retrieval, but not for the core RAG process.


