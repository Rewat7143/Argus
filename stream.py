import streamlit as st
from PIL import Image
import numpy as np

from mesonet import Meso4
# Load your model
meso = Meso4()
meso.load('weights/Meso4_DF.h5')

st.title('MesoNet Image Classifier')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    st.write("Classifying...")
    image = np.array(image.resize((256, 256))) / 255.0  # Resize and normalize the image
    image = np.expand_dims(image, axis=0)  # Expand dimensions for model prediction
    prediction = meso.predict(image)[0][0]  # Predict with your model

    if prediction > 0.5:
        st.write("The image is classified as real with a confidence of ", prediction)
    else:
        st.write("The image is classified as a deepfake with a confidence of ", 1 - prediction)