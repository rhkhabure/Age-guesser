from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import joblib
import sys
import os
import streamlit as st
import numpy as np
from PIL import Image

# Define the path to your ensemble model
ENSEMBLE_MODEL_PATH = "ensemble_model.joblib"  # Update if needed

# Add the directory containing 'Helpers.py' to sys.path
helpers_path = os.path.abspath(
    os.path.join(
        os.getcwd(),
        r"C:\Users\Admin\OneDrive\Documents\Schoolwork\Third year\First Sem\Machine Learning\Project\Faces dataset\AgePredictionDemo\Utils\Helpers.py"
    )
)
if helpers_path not in sys.path:
    sys.path.append(helpers_path)

try:
    from Helpers import preprocess_image, extract_features, predict_age
except ImportError as e:
    st.error(f"Could not import Helpers.py: {e}")
    st.stop()

# Load MobileNetV2 without top classification layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
feature_extractor = Model(inputs=base_model.input, outputs=x)

extractor = feature_extractor  # Use the in-memory model directly

# Load ensemble model
try:
    model = joblib.load(ENSEMBLE_MODEL_PATH)
except Exception as e:
    st.error(f"Could not load ensemble model: {e}")
    st.stop()

st.title("🎉 Live Age Group Prediction")
st.markdown("Upload a photo and let the ensemble model guess your age group!")

# Upload section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Classmate Photo", use_column_width=True)

        # Preprocess image using your helper (if it does more than resizing)
        image_np = preprocess_image(image)  # Should return shape (1, 224, 224, 3) and normalized

        # Feature extraction
        features = extractor.predict(image_np)

        # Predict
        prediction = model.predict(features)
        # If prediction is array-like, get the first value
        if hasattr(prediction, "__len__") and not isinstance(prediction, str):
            pred_value = prediction[0]
        else:
            pred_value = prediction

        # If you have a mapping from label to age group, use it here
        # Example: age_groups = {0: "Child", 1: "Teen", 2: "Adult", 3: "Senior"}
        # st.success(f"🧠 Predicted Age Group: **{age_groups.get(pred_value, pred_value)}**")
        st.success(f"🧠 Predicted Age Group: **{pred_value}**")
    except Exception as e:
        st.error(f"Error processing image: {e}")