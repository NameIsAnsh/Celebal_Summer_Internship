import joblib

def load_model(model_path):
    """Loads the trained machine learning model."""
    return joblib.load(model_path)


