# RAG Q&A Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about loan application data using Google Gemini AI and document retrieval.

## 🚀 Features

- **RAG Architecture**: Combines document retrieval with generative AI for accurate responses
- **Interactive UI**: Clean Streamlit interface with chat functionality
- **Document Retrieval**: Uses TF-IDF vectorization and cosine similarity for relevant document retrieval
- **Gemini Integration**: Powered by Google's Gemini Pro model for intelligent response generation
- **Secure API Management**: Environment-based API key configuration with .gitignore protection

## 📁 Project Structure

```
├── streamlit_app.py          # Main Streamlit application
├── data_processing.py        # Data preprocessing utilities
├── vector_embeddings.py      # TF-IDF vectorization and embedding creation
├── retrieval_system.py       # Document retrieval system
├── main.py                   # Command-line version of the chatbot
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (API keys)
├── .gitignore               # Git ignore file (includes .env)
├── training_embeddings.pkl   # Pre-computed TF-IDF embeddings
├── tfidf_vectorizer.pkl     # Trained TF-IDF vectorizer
└── upload/                  # Data directory
    ├── TrainingDataset.csv
    ├── TestDataset.csv
    └── Sample_Submission.csv
```

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

1. Get a Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Replace `YOUR_GEMINI_API_KEY_HERE` in the `.env` file with your actual API key:

```bash
GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Generate Embeddings (if needed)

If you need to regenerate the embeddings:

```bash
python vector_embeddings.py
```

### 4. Run the Application

#### Streamlit Web Interface (Recommended)
```bash
streamlit run streamlit_app.py
```

#### Command Line Interface
```bash
python main.py
```

## 🌐 Deployment

The Streamlit app is configured to run on `0.0.0.0:8501` for easy deployment. 

## 💡 Usage

### Web Interface
1. Open the Streamlit app in your browser
2. Enter your Gemini API key in the sidebar
3. Ask questions about the loan data in the chat interface
4. View retrieved documents that informed the AI's response

### Sample Questions
- "What factors affect loan approval?"
- "What is the average income of approved applicants?"
- "How does education level impact loan approval rates?"
- "What is the distribution of loan amounts?"

## 🔧 Technical Details

### RAG Pipeline
1. **Document Processing**: CSV data is converted to text descriptions
2. **Vectorization**: TF-IDF vectorizer creates embeddings for all documents
3. **Retrieval**: User queries are vectorized and matched against documents using cosine similarity
4. **Generation**: Top relevant documents are provided as context to Gemini for response generation

### Data Format
The system processes loan application data with the following fields:
- Loan_ID, Gender, Married, Dependents, Education
- Self_Employed, ApplicantIncome, CoapplicantIncome
- LoanAmount, Loan_Amount_Term, Credit_History
- Property_Area, Loan_Status (for training data)

