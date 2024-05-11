import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array

# Load your trained model
model = tf.keras.models.load_model('model.h5')

# Define function to make predictions
def predict(image):
    img = image.resize((224, 224))  # Resize the image using PIL
    img_array = img_to_array(img) / 255.0  # Convert the image to an array and normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    prediction = model.predict(img_array)  # Make prediction
    return prediction

# Streamlit app
def main():
    st.title('Image Segmentation App')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)  # Use PIL to open image
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Make prediction when 'Predict' button is clicked
        if st.button('Predict'):
            with st.spinner('Predicting...'):
                prediction = predict(image)
                # Assuming prediction is a PIL image; adjust if it's an array
                st.image(prediction, caption='Predicted Mask', use_column_width=True)

if __name__ == '__main__':
    main()
