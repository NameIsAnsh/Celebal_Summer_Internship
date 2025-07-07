import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from model.model import load_model

# Set page configuration
st.set_page_config(
    page_title="ML Model Deployment",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🤖 Machine Learning Model Deployment")
st.markdown("""
This web application demonstrates how to deploy a trained machine learning model using Streamlit.
You can input data, receive predictions, and understand model outputs through visualizations.
""")

# Load the trained model
@st.cache_resource
def load_trained_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'logistic_regression_model.pkl')
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        st.error("Model file not found. Please train the model first.")
        return None

model = load_trained_model()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Prediction", "Model Information", "Data Exploration"])

if page == "Prediction":
    st.header("🔮 Make Predictions")
    
    if model is not None:
        # Create input form
        st.subheader("Input Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature1 = st.number_input(
                "Feature 1", 
                min_value=0.0, 
                max_value=20.0, 
                value=5.0, 
                step=0.1,
                help="Enter a value for Feature 1"
            )
        
        with col2:
            feature2 = st.number_input(
                "Feature 2", 
                min_value=0.0, 
                max_value=20.0, 
                value=5.0, 
                step=0.1,
                help="Enter a value for Feature 2"
            )
        
        # Make prediction
        if st.button("🚀 Make Prediction", type="primary"):
            # Prepare input data
            input_data = np.array([[feature1, feature2]])
            
            # Get prediction and probability
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Display results
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", f"Class {prediction}")
            
            with col2:
                st.metric("Confidence (Class 0)", f"{prediction_proba[0]:.2%}")
            
            with col3:
                st.metric("Confidence (Class 1)", f"{prediction_proba[1]:.2%}")
            
            # Confidence visualization
            st.subheader("🎯 Prediction Confidence")
            confidence_df = pd.DataFrame({
                'Class': ['Class 0', 'Class 1'],
                'Probability': prediction_proba
            })
            st.bar_chart(confidence_df.set_index('Class'))
    else:
        st.error("Model not loaded. Please check if the model file exists.")

elif page == "Model Information":
    st.header("ℹ️ Model Information")
    
    st.subheader("Model Details")
    st.write("""
    - **Model Type**: Logistic Regression
    - **Features**: 2 numerical features (Feature 1, Feature 2)
    - **Target**: Binary classification (Class 0 or Class 1)
    - **Training Data**: Simple synthetic dataset
    """)
    
    if model is not None:
        st.subheader("Model Parameters")
        st.write(f"**Coefficients**: {model.coef_[0]}")
        st.write(f"**Intercept**: {model.intercept_[0]:.4f}")
        
        # Feature importance visualization
        st.subheader("📈 Feature Importance")
        feature_importance = abs(model.coef_[0])
        importance_df = pd.DataFrame({
            'Feature': ['Feature 1', 'Feature 2'],
            'Importance': feature_importance
        })
        st.bar_chart(importance_df.set_index('Feature'))

elif page == "Data Exploration":
    st.header("🔍 Data Exploration")
    
    st.subheader("Training Data Overview")
    st.write("""
    The model was trained on a simple synthetic dataset with the following characteristics:
    """)
    
    # Create sample data for visualization
    sample_data = pd.DataFrame({
        'Feature 1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Feature 2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'Target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    })
    
    st.subheader("📋 Sample Data")
    st.dataframe(sample_data)
    
    st.subheader("📊 Data Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Feature 1 Distribution**")
        st.bar_chart(sample_data['Feature 1'])
    
    with col2:
        st.write("**Feature 2 Distribution**")
        st.bar_chart(sample_data['Feature 2'])
    
    st.subheader("🎯 Target Distribution")
    target_counts = sample_data['Target'].value_counts()
    st.bar_chart(target_counts)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")

