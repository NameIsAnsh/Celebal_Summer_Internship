
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    df_text = df.apply(lambda x: ", ".join([f"{col}: {val}" for col, val in x.items()]), axis=1)
    return df_text.tolist()

def load_artifacts(embeddings_path, vectorizer_path):
    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return embeddings, vectorizer

def retrieve_documents(query, vectorizer, document_embeddings, documents, top_n=3):
    query_embedding = vectorizer.transform([query])
    similarities = cosine_similarity(query_embedding, document_embeddings).flatten()
    
    # Get the indices of the top_n most similar documents
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    retrieved_docs = [documents[i] for i in top_indices]
    return retrieved_docs

if __name__ == "__main__":
    # Load pre-processed data and artifacts
    training_data_docs = load_data("/home/ubuntu/upload/TrainingDataset.csv")
    training_embeddings, tfidf_vectorizer = load_artifacts("training_embeddings.pkl", "tfidf_vectorizer.pkl")

    # Example usage
    query = "What is the loan status for male applicants with high income?"
    retrieved_documents = retrieve_documents(query, tfidf_vectorizer, training_embeddings, training_data_docs)

    print(f"Query: {query}")
    print("\nRetrieved Documents:")
    for doc in retrieved_documents:
        print(f"- {doc}")


