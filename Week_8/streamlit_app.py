import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="RAG Q&A Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 RAG Q&A Chatbot")
st.markdown("Ask questions about loan application data and get AI-powered answers!")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key input
    api_key = st.text_input(
        "Gemini API Key", 
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Enter your Gemini API key"
    )
    
    if api_key:
        genai.configure(api_key=api_key)
        st.success("✅ API Key configured!")
    else:
        st.warning("⚠️ Please enter your Gemini API key")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This chatbot uses RAG (Retrieval-Augmented Generation) to answer questions about loan application data.")

# Load data and artifacts
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df_text = df.apply(lambda x: ", ".join([f"{col}: {val}" for col, val in x.items()]), axis=1)
    return df_text.tolist()

@st.cache_data
def load_artifacts(embeddings_path, vectorizer_path):
    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return embeddings, vectorizer

# Initialize data
try:
    training_data_docs = load_data("/home/ubuntu/upload/TrainingDataset.csv")
    training_embeddings, tfidf_vectorizer = load_artifacts("training_embeddings.pkl", "tfidf_vectorizer.pkl")
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

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
    if not api_key:
        return "Error: Please configure your Gemini API key in the sidebar."
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        context = "\n".join(retrieved_documents)
        prompt = f"""Based on the following loan application data, answer the question accurately and concisely:

Data:
{context}

Question: {query}

Please provide a clear, informative answer based on the data provided. If the data doesn't contain enough information to answer the question, please say so.

Answer:"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main chat interface
if data_loaded:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "retrieved_docs" in message:
                with st.expander("📄 Retrieved Documents"):
                    for i, doc in enumerate(message["retrieved_docs"], 1):
                        st.text(f"{i}. {doc}")

    # Accept user input
    if prompt := st.chat_input("Ask a question about the loan data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve relevant documents
                retrieved_docs = retrieve_documents(
                    prompt, 
                    tfidf_vectorizer, 
                    training_embeddings, 
                    training_data_docs,
                    top_n=5
                )
                
                # Generate response
                response = generate_response(prompt, retrieved_docs)
                
                # Display response
                st.markdown(response)
                
                # Show retrieved documents
                with st.expander("📄 Retrieved Documents"):
                    for i, doc in enumerate(retrieved_docs[:3], 1):
                        st.text(f"{i}. {doc}")
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "retrieved_docs": retrieved_docs[:3]
        })

else:
    st.error("❌ Data not loaded. Please check if the required files are available.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Google Gemini")


