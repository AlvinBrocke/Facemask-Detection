import streamlit as st
from PIL import Image
import numpy as np
from tensorflow import keras
from keras import models
from keras.models import load_model


print("This application predicts whether an upload ")

# Function to preprocess the image for prediction
def preprocess_image(image):
    # Resize and preprocess the image
    image = image.resize((128, 128))
    image = image.convert('RGB')
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# Function to make predictions using the loaded model
def predict_mask(model, image):
    # Make predictions
    prediction = model.predict(image)[0][0]
    return prediction



def main():
    st.title("Face Mask Detection")
    st.subheader("This app predicts whether an image has a face mask on or not.")


    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")


    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)


        # Preprocess the image for prediction
        preprocessed_image = preprocess_image(image)


        # Load and preprocess the model
        model = model = load_model(r'C:\Users\labij\Downloads\Facemask folder\model.h5')
 


        # Make predictions
        prediction = predict_mask(model, preprocessed_image)
        # Display the raw prediction values
        st.write(f"Raw Prediction: {prediction}")
        # Display the prediction
        st.write(f"Prediction: {'With Mask' if prediction < 0.5 else 'Without Mask'}")


        # Display the prediction
        #st.write(f"Prediction: {'With Mask' if prediction > 0.5 else 'Without Mask'}")
        


# Run the app
if __name__ == "__main__":
    main()



