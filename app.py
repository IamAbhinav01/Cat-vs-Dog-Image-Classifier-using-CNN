import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('cnn_model.h5')

# Preprocessing and prediction function
def predict_image(img):
    img = img.resize((64, 64))  # Resize to the input size your model expects
    img = image.img_to_array(img)
    img = img / 255.0  # Normalize (important if your model was trained with rescaled images)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    if result[0][0] >= 0.5:
        prediction = 'Dog 🐶'
    else:
        prediction = 'Cat 🐱'
    return prediction

# Streamlit UI
st.title("🐱🐶 Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    prediction = predict_image(img)
    st.subheader(f"Prediction: **{prediction}**")
