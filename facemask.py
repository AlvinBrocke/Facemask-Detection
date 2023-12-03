
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load the model architecture
with open(r'model_architecture.json', 'r') as f:
    model_json = f.read()

# Load the model weights
model = model_from_json(model_json)
model.load_weights('model_weights.h5')

# Function to preprocess image
def preprocess_image(image):
    # Resize the image to match the input size expected by the model
    resized_image = cv2.resize(image, (224, 224))
    
    # Normalize pixel values to be between 0 and 1
    normalized_image = resized_image / 255.0
    
    # Add any other preprocessing steps as needed
    # ...

    return normalized_image

# Function to make predictions
def predict_mask(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make predictions
    # Example assuming binary classification (mask/no mask)
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))[0][0]

    return prediction

# Streamlit app
def main():
    st.title("Face Mask Detection")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Display the original image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make predictions
        prediction = predict_mask(image)

        # Display the prediction
        st.write(f"Prediction: {'With Mask' if prediction > 0.5 else 'Without Mask'}")

# Run the app
if __name__ == "__main__":
    main()