import json
import streamlit as st
import pickle
import numpy as np
import cv2
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from PIL import Image
model=pickle.load(open('model.pkl','rb'))

st.set_page_config(layout='wide')

def classify_image(uploaded_files):
    # Convert BytesIO to Image
    img = Image.open(uploaded_files)

    # Convert Image to numpy array
    img = np.array(img)

    # Ensure img is a numpy array
    img = np.array(img) if not isinstance(img, np.ndarray) else img

    # Load your deepfake model
    meso = pickle.load(open('model.pkl','rb'))  # Initialize your model here

    # Preprocess the image
    img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = cv2.resize(img_array, (256, 256))
    img_array = img_array / 255.0  # Rescale to match the training data preprocessing

    # Flatten the image data to match the input shape that the model expects
    img_array = img_array.flatten()[:100]  # Take the first 100 elements
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match input shape

    # Perform prediction
    prediction = meso.predict(img_array)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)



def cont(im1,t1,t11,im2,t2,t22,im3,t3,t33,im4,t4,t44):
    """Displays an image and text side-by-side in a Streamlit app.

    Args:
        image_path (str): Path to the image file.
        text (str): Text to display beside the image.
    """
    col1, col2, col3, col4 = st.columns(4)  # Adjust the number of columns for different layouts

  # Load the image using st.image()
    with col1:
        st.image(im1, width=300)# Adjust width as needed
        st.markdown(t1)
        st.markdown(t11)
  # Display the text using st.write()
    with col2:
        st.image(im2, width=300)# Adjust width as needed
        st.markdown(t2)
        st.markdown(t22)
    with col3:
        st.image(im3, width=300)# Adjust width as needed
        st.markdown(t3)
        st.markdown(t33)
    with col4:
        st.image(im4, width=300)# Adjust width as needed
        st.markdown(t4)
        st.markdown(t44)

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

def load_lottiefile(filepath: str):
    """Loads a Lottie animation from a JSON file, handling potential encoding issues.

    Args:
        filepath (str): Path to the Lottie JSON file.

    Returns:
        dict: The loaded Lottie animation data.

    Raises:
        UnicodeDecodeError: If the encoding of the file cannot be determined.
    """

    try:
        # Attempt to open the file with UTF-8 encoding (most common)
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        # If UTF-8 fails, try common encodings for JSON files
        for encoding in ("latin-1", "iso-8859-1"):
            try:
                with open(filepath, "r", encoding=encoding) as f:
                    return json.load(f)
            except UnicodeDecodeError:
                pass

    # Raise an error if no encoding works
    raise UnicodeDecodeError("Could not determine encoding of Lottie JSON file")

try:
    gif1 = load_lottiefile("Argus/code.json")# Load the Lottie animation with error handling
    gif2 = load_lottiefile("Argus/code2.json")
    gif3 = load_lottiefile("Argus/new.json")
except UnicodeDecodeError as e:
    st.error(f"Error loading Lottie animation: {e}")
    st.write("Please ensure your 'code.json' file is in a compatible encoding (e.g., UTF-8).")
else:
    with st.container():
        st.write("##")
        gif_column, text_column = st.columns((1, 2))
        with gif_column:
            st_lottie(
                gif1,
                height=200
                )
        with text_column:
            st.markdown("## Assalamualaikum ")

st.markdown(
    """
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        h2 {
            color: #007BFF;
            font-size: 80px;
        }
    </style>
    """,
    unsafe_allow_html=True
)
    
    # Display skills in two columns

# Main content
with st.container():
    selected = option_menu(
        menu_title=None,
        options=['About', 'Projects', 'Credit'],
        icons=['person', 'code-slash', 'credit-card-2-front', 'chat-left-text-fill'],
        orientation="horizontal"
    )

if selected == 'About':
    with st.container():
        st.write("##")
        st.markdown("# DEEP FAKE DETECTION")
        st.write("##")
        gif_column, text_column = st.columns((1, 2))
        with gif_column:
            st_lottie(
                gif3,
                height=200
                )
        with text_column:
            st.markdown(""":small_blue_diamond:**Combating misinformation:** Deepfakes can be used to spread lies or propaganda, making it crucial to have tools to detect them.\n\n:small_blue_diamond:**Protecting reputations:** Deepfakes could be used to damage someone's image by putting them in compromising situations.Detection helps prevent this.\n\n:small_blue_diamond:**Building trust online:** By making it easier to spot deepfakes, deepfake detection can help maintain trust in the authenticity of online content.""")
        text_cl , gif_cl = st.columns((2,1))
        with text_cl:
            st.markdown(":small_blue_diamond:**Combats fraud:**Deepfake detection helps law enforcement investigate deepfake-related crimes like CEO fraud, where impersonation is used for financial gain.\n\n:small_blue_diamond:**Protecting businesses:**Businesses can leverage deepfake detection to shield themselves from malicious attacks. Deepfakes can be used by competitors to damage a brand's image or sabotage marketing efforts. By identifying such manipulated content, businesses can take action to mitigate the harm before it impacts sales.")  
        with gif_cl:
            st_lottie(
                gif2,
                height=200
            )
            
        
elif selected == "Projects":
    st.markdown("#### Welcome :clap:")
    # Deepfake model selection with informative descriptions

    # File upload with clear guidance and validation
    uploaded_files = st.file_uploader("Choose an image...", type=["jpg", "png"])

    # Button with clear action and feedback
    if st.button("Hit Me!", key='analyze_button', help="Trigger deepfake analysis based on uploaded files and selected models."):
        if not uploaded_files:
            st.error("Please upload files to analyze.")
        else:
            # Perform deepfake analysis using selected models and display results
            output=classify_image(uploaded_files)
            if output >= 0.5:
                st.success("The image is real.")
            else:
                st.warning("This could be a deepfake!")
            # Replace ... with actual analysis logic and result presentation

    # Review text area with optional input and validation

    
elif selected == 'Credit':
    st.markdown("# Team Members")
    im1="Argus/Vin.jpg"
    t1="#### **Vinod Polinati**"
    t11="##### Machine Learning "

    im2="Argus/Raw.jpg"
    t2="#### **Reddy Rewat**"
    t22="##### Machine Learning"
    
    im3="Argus/Sha.jpg"
    t3="#### **Shaik Shajid**"
    t33="##### UI&UX Designer"
    
    im4="Argus/rah.jpg"
    t4="#### **Miazur Rahaman**"
    t44="##### Streamlit Developer"
    
    cont(im1,t1,t11,im2,t2,t22,im3,t3,t33,im4,t4,t44)

st.markdown(
    """
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        h2 {
            color: #007BFF;
            font-size: 70px;
        }
        h1{
            text-align: center;
        }
        h3{
            color: #0078FF;
            font-size: 50px;
        }
        h4{
            font-size: 40px;
        }
    </style>
    """,
    unsafe_allow_html=True
)
