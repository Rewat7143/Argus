import streamlit as st
import streamlit_option_menu as option_menu
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = tf.keras.models.load_model("weights/Meso4_F2F.h5")

def predict_deepfake(image_file):
    # Load and preprocess the image
    try:
        img = image.load_img(image_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Perform inference
        prediction = model.predict(img_array)
        
        # Return prediction
        return prediction
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

def get_file_type(file):
    allowed_image_extensions = ['jpg', 'jpeg', 'png', 'gif']
    allowed_video_extensions = ['mp4', 'avi', 'mkv']

    file_extension = file.name.split('.')[-1].lower()

    if file_extension in allowed_image_extensions:
        return 'image'
    elif file_extension in allowed_video_extensions:
        return 'video'
    else:
        st.error(f"Unsupported file type: {file.name} ({file_extension})")
        return 'unknown'

# Main content
selected = option_menu.option_menu(
    menu_title=None,
    options=['About', 'Projects', 'Credit'],
    icons=['person', 'code-slash', 'credit-card-2-front', 'chat-left-text-fill'],
    orientation="horizontal"
)

if selected == 'Projects':
    st.title("DEEPFAKE DETECTION APPLICATION")
    st.markdown("## Welcome :clap:")

    # Deepfake model selection with informative descriptions
    selected_options = st.multiselect(
        'Select A Deep Fake Model:',
        [1, 2, 3],
        key='model_selection',
        help="Select one or more deepfake detection models to use for analysis.")

    # File upload with clear guidance and validation
    uploaded_files = st.file_uploader(
        "Choose files to upload:",
        type=['jpg', 'png', 'mp4'],
        accept_multiple_files=True,
        help="Upload images or videos to analyze for potential deepfakes. Supported file types: JPG, PNG, MP4.")

    if uploaded_files:
        # Process uploaded files
        for file in uploaded_files:
            file_type = get_file_type(file)

            if file_type == 'image':
                st.image(file, caption=f"Uploaded Image - {file.name}", use_column_width=True)
                # Perform deepfake prediction
                prediction = predict_deepfake(file)
                if prediction is not None:
                    st.write("Prediction:", prediction)  # Display prediction result

            elif file_type == 'video':
                st.video(file, caption=f"Uploaded Video - {file.name}")

                # For video files, you need to extract frames and perform predictions on each frame
                # You can use libraries like OpenCV for this task

            # Add placeholder for deepfake detection results based on selected models
            st.empty()  # Create an empty container for future content
            st.markdown("_(Deepfake Detection Results will be displayed here)_")  # Add a placeholder text

    # Button with clear action and feedback
    if st.button("Hit Me!", key='analyze_button', help="Trigger deepfake analysis based on uploaded files and selected models."):
        if not uploaded_files:
            st.error("Please upload files to analyze.")
        else:
            # Perform deepfake analysis using selected models and display results
            st.success("Deepfake analysis in progress...")
            # Replace ... with actual logic for deepfake analysis and result presentation
