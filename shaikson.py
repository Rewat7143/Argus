import streamlit as st
 
# Set page title and favicon
st.set_page_config(
    page_title="Deepfake Detection App",
    page_icon=":detective:",
    layout="wide",
)
 
# Main title and welcome message
st.title("DEEPFAKE DETECTION APPLICATION")
st.markdown("## Welcome :wave:")
 
# Multiselect widget
selected_options = st.multiselect('Select Options', [1, 2, 3])
 
# Upload video section
st.header("Upload your Video")
uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi"])
 
# Upload image section
st.header("Upload your Image")
uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
 
# Button to trigger actions
if st.button("Hit Me"):
    # Add functionality here when the button is clicked
    st.success("Processing...")
 
# Text area for user input
review_text = st.text_area("Review please ")
 
# Submit button
if st.button("Submit"):
    # Add functionality here when the submit button is clicked
    st.success("Submitted successfully!")
 
# Optional: Add a footer or additional information
st.sidebar.markdown("---")
st.sidebar.markdown("For support, contact shaikson7@gmail.com")
st.sidebar.markdown("Mobile : wa.me/+916281783825")
st.markdown(
    """
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        h2 {
            color: #007BFF;
        }
        .my-button {
            background-color: #28A745;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
    </style>
    """,
    unsafe_allow_html=True
)
 
