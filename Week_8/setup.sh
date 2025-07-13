#!/bin/bash

echo "🤖 RAG Q&A Chatbot Setup"
echo "========================"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if embeddings exist
if [ ! -f "training_embeddings.pkl" ] || [ ! -f "tfidf_vectorizer.pkl" ]; then
    echo "🔄 Generating embeddings..."
    python vector_embeddings.py
else
    echo "✅ Embeddings already exist"
fi

# Check API key
if grep -q "YOUR_GEMINI_API_KEY_HERE" .env; then
    echo "⚠️  Please update your Gemini API key in .env file"
    echo "   Replace 'YOUR_GEMINI_API_KEY_HERE' with your actual API key"
else
    echo "✅ API key configured"
fi

echo ""
echo "🚀 Setup complete! Run the chatbot with:"
echo "   streamlit run streamlit_app.py"
echo ""
echo "🌐 Or access the live demo at:"
echo "   https://8501-i43qnl01ogt8pouypdoo9-dd62a052.manusvm.computer"

