import os
import streamlit as st
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json

# Load model
with open("model_architecture.json", "r") as f:
    loaded_model = model_from_json(f.read())

file_path = "model_weights.h5"
if os.path.exists(file_path):
    loaded_model.load_weights("model_weights.h5")
else:
    print(f"The file {file_path} does not exist.")


# Load weights


# Function to predict if the user is wearing a mask or not


def predict_mask(image):
    input_image = cv2.imread(image)

    plt.imshow(input_image)

    input_image_resized = cv2.resize(input_image, (128, 128))

    input_image_scaled = input_image_resized / 255

    input_image_reshaped = np.reshape(input_image_scaled, [1, 128, 128, 3])

    input_prediction = loaded_model.predict(input_image_reshaped)

    print(input_prediction)

    input_pred_label = np.argmax(input_prediction)

    print(input_pred_label)

    if input_pred_label == 1:
        print("The person in the image is wearing a mask")

    else:
        print("The person in the image is not wearing a mask")


def main():
    st.title("Mask Detection App")

    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(
            np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR
        )

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess and predict
        prediction = predict_mask(image)

        # Display the prediction result
        st.write(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
