import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pickle
import joblib

# Load models
FEATURE_EXTRACTOR_PATH = "C:\\Users\\Admin\\OneDrive\\Documents\\Schoolwork\\Third year\\First Sem\\Machine Learning\\Project\\Faces dataset\\AgePredictionDemo\\Models\\feature_extractor.keras"
ENSEMBLE_MODEL_PATH = "C:\\Users\\Admin\\OneDrive\\Documents\\Schoolwork\\Third year\\First Sem\\Machine Learning\\Project\\Faces dataset\\AgePredictionDemo\\Models\\ensemble_age_model.pkl"
feature_extractor = load_model(FEATURE_EXTRACTOR_PATH)
ensemble_model = joblib.load(ENSEMBLE_MODEL_PATH)

# 💡 Image Preprocessing
def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """Resizes and normalizes the input image for CNN extraction."""
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)

# 🧠 Feature Extraction
def extract_features(image_array: np.ndarray) -> np.ndarray:
    """Runs the preprocessed image through the CNN feature extractor."""
    features = feature_extractor.predict(image_array)
    return features.reshape(1, -1)

# 🎯 Age Prediction
def predict_age(features: np.ndarray) -> float:
    """Uses the ensemble model to predict age from CNN features."""
    age = ensemble_model.predict(features)[0]
    return round(age, 1)

