import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from flask import Blueprint, request, jsonify
from flask_cors import cross_origin

# Load environment variables
load_dotenv()

chatbot_bp = Blueprint('chatbot', __name__)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Load pre-processed data and artifacts
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

# Initialize data and artifacts
try:
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    
    training_data_docs = load_data(os.path.join(parent_dir, "upload", "TrainingDataset.csv"))
    training_embeddings, tfidf_vectorizer = load_artifacts(
        os.path.join(parent_dir, "training_embeddings.pkl"), 
        os.path.join(parent_dir, "tfidf_vectorizer.pkl")
    )
except Exception as e:
    print(f"Error loading data: {e}")
    training_data_docs = []
    training_embeddings = None
    tfidf_vectorizer = None

# Retrieval function
def retrieve_documents(query, vectorizer, document_embeddings, documents, top_n=3):
    if not vectorizer or document_embeddings is None:
        return []
    
    query_embedding = vectorizer.transform([query])
    similarities = cosine_similarity(query_embedding, document_embeddings).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    retrieved_docs = [documents[i] for i in top_indices]
    return retrieved_docs

# Gemini response generation
def generate_response(query, retrieved_documents):
    if not GEMINI_API_KEY:
        return "Error: Gemini API key not configured. Please set GEMINI_API_KEY in .env file."
    
    try:
        model = genai.GenerativeModel("gemini-pro")
        context = "\n".join(retrieved_documents)
        prompt = f"Based on the following loan application data, answer the question:\n\nData:\n{context}\n\nQuestion: {query}\nAnswer:"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

@chatbot_bp.route('/chat', methods=['POST'])
@cross_origin()
def chat():
    try:
        data = request.get_json()
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Retrieve relevant documents
        retrieved_docs = retrieve_documents(
            user_query, 
            tfidf_vectorizer, 
            training_embeddings, 
            training_data_docs
        )
        
        # Generate response
        answer = generate_response(user_query, retrieved_docs)
        
        return jsonify({
            'query': user_query,
            'answer': answer,
            'retrieved_docs': retrieved_docs[:2]  # Return first 2 for reference
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chatbot_bp.route('/health', methods=['GET'])
@cross_origin()
def health():
    return jsonify({
        'status': 'healthy',
        'gemini_configured': bool(GEMINI_API_KEY),
        'data_loaded': bool(training_data_docs and training_embeddings is not None)
    })

