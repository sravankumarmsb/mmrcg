import streamlit as st
from PIL import Image
import os
import configparser
import time

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read("config.ini")

# Read base path from config file
base_path_8k_img = config.get("paths", "base_path_8k_img")
base_path_30k_img = config.get("paths", "base_path_30k_img")
project_path = config.get("paths", "project_path")

# Load custom CSS
st.markdown("""
    <style>
    
    .title {
        font-family: 'Century Gothic', monospace;
        font-size: 42px;
        color: #4CAF50;
        font-weight: bold;
        text-align: center;
    }
    .subtitle {
        font-family: 'Century Gothic', sans-serif;
        font-size: 24px;
        color: #4CAF50;
        text-align: left;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px;
    }
    .stSpinner {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 48px;
        font-weight: bold;
        color: red;
        background-color: white;
        padding: 10px 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }
    .stSelectbox>div {
        font-size: 16px;
    }
            .overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 24px;
        font-weight: bold;
        color: white;
        z-index: 9999;
    }
    .hidden { display: none; }    
    </style>
    """, unsafe_allow_html=True)
st.markdown(
    """
    <div id="loading-overlay" class="hidden overlay">🔄 Processing... Please wait.</div>
    <script>
    function showLoading() {
        document.getElementById("loading-overlay").classList.remove("hidden");
    }
    function hideLoading() {
        document.getElementById("loading-overlay").classList.add("hidden");
    }
    </script>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="title">Multimodal Media Retrieval and Captioning System</div>', unsafe_allow_html=True)
st.markdown("""<div style="text-align: center; color: #003366; font-size: 18px;">
    Group - 22 (CH V S Adithya, Harini Pradeep Kumar, Madhuri Gupta, Sravan Kumar)
    </div><br>""", unsafe_allow_html=True)

# Initialize session state
if "task_option" not in st.session_state:
    st.session_state.task_option = ""
if "model_option" not in st.session_state:
    st.session_state.model_option = ""
if "retrieval_option" not in st.session_state:
    st.session_state.retrieval_option = ""
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Function to reset fields when model changes
def reset_fields():
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

st.session_state.task_option = st.radio("Select Task:", ["Retrieval", "Caption Generation"], index=0)

task_models = {
    "Retrieval": ["", "CLIP - PRE - 8k", "CLIP - PRE - 30k"],
    "Caption Generation": ["", "ResNet50 - LSTM - 8k", "ResNet50 - LSTM - 30k", "CLIP - LSTM - 8k", "CLIP - LSTM - 30k", "CLIP - ATTENTION - 8k", "CLIP - ATTENTION - 30k","VIT - GPT2"]
}

# Model selection dropdown with reset callback
model_option = st.selectbox("Select Model:", task_models[st.session_state.task_option], key="model_option",on_change=reset_fields)

# Dynamically adjust retrieval options
if st.session_state.task_option == "Retrieval":
    retrieval_options = ["", "Image to Text", "Text to Image","Image to Image"]
    retrieval_option = st.selectbox(
        "Select Retrieval Type:",
        retrieval_options,
        key="retrieval_option"
    )
else:
    retrieval_option = "Image to Text"
uploaded_file = None
input_text = None

# Dynamically show input fields based on retrieval type
if retrieval_option == "Image to Text":
    # Create a grid with two columns
    col1, col2 = st.columns(2)
    picture = None
    uploaded_file = None
    # Add buttons to each column
    with col1:
        # if st.button("Camera"):
        st.write("📷 Capture Image")
        picture = st.camera_input("")
    with col2:
        # Option to upload an image file
        st.write("📂 Upload Image")
        #uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="uploaded_file")
    
    # Display the captured or uploaded image
    if picture:
        # st.write("Picture NOT None: " + picture)
        uploaded_file = None
        uploaded_file = picture

    if uploaded_file is not None:
        file_name = uploaded_file.name
        saved_path = os.path.join(project_path,"uploaded_files", file_name)
        with open(saved_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
elif retrieval_option == "Image to Image":
    # Create a grid with two columns
    col1, col2 = st.columns(2)
    picture = None
    uploaded_file = None
    # Add buttons to each column
    with col1:
        # if st.button("Camera"):
        st.write("📷 Capture Image")
        picture = st.camera_input("")
    with col2:
        # Option to upload an image file
        st.write("📂 Upload Image")
        #uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="uploaded_file")
    
    # Display the captured or uploaded image
    if picture:
        # st.write("Picture NOT None: " + picture)
        uploaded_file = None
        uploaded_file = picture

    if uploaded_file is not None:
        file_name = uploaded_file.name
        saved_path = os.path.join(project_path,"uploaded_files", file_name)
        with open(saved_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
elif retrieval_option == "Text to Image":
    input_text = st.text_input("Enter a caption to retrieve an image:", "", key="input_text")
    submit_button = st.button("Retrieve Image")

def CLIP_PRE_8k():
    import mmrcs_clippretrained_flickr8k
    if uploaded_file is not None and retrieval_option == "Image to Text":
        caption = mmrcs_clippretrained_flickr8k.imageToCaptions(uploaded_file)
        st.markdown(f'Retrieved Caption : <div class="subtitle">{caption}</div>', unsafe_allow_html=True)
    elif uploaded_file is not None and retrieval_option == "Image to Image":
        import mmrcs_clippretrained_flickr8k    
        retrieved_image = mmrcs_clippretrained_flickr8k.imageToImage(uploaded_file)
        image_full_path = base_path_8k_img + retrieved_image
        st.markdown("""<h3 style="text-align: center; color: #4CAF50;">Retrieved Image</h3>""", unsafe_allow_html=True)
        st.image(image_full_path, use_column_width=True)

def CLIP_PRE_30k():
    import mmrcs_clippretrained_flickr30k
    if uploaded_file is not None and retrieval_option == "Image to Text":        
        caption = mmrcs_clippretrained_flickr30k.imageToCaptions(uploaded_file)
        st.markdown(f'Retrieved Caption : <div class="subtitle">{caption}</div>', unsafe_allow_html=True)
    elif uploaded_file is not None and retrieval_option == "Image to Image":
        import mmrcs_clippretrained_flickr30k    
        retrieved_image = mmrcs_clippretrained_flickr30k.imageToImage(uploaded_file)
        image_full_path = base_path_30k_img + retrieved_image
        st.markdown("""<h3 style="text-align: center; color: #4CAF50;">Retrieved Image</h3>""", unsafe_allow_html=True)
        st.image(image_full_path, use_column_width=True)

# Define functions for each retrieval method
def RESNET50_LSTM_8k():
    import mmrcs_resnet50_lstm_flickr8k
    if uploaded_file is not None:
        caption = mmrcs_resnet50_lstm_flickr8k.imageToCaptions(uploaded_file)
        st.markdown(f'Retrieved Caption : <div class="subtitle">{caption}</div>', unsafe_allow_html=True)

def RESNET50_LSTM_30k():
    import mmrcs_resnet50_lstm_flickr30k
    if uploaded_file is not None:        
        caption = mmrcs_resnet50_lstm_flickr30k.imageToCaptions(uploaded_file)
        st.markdown(f'Retrieved Caption : <div class="subtitle">{caption}</div>', unsafe_allow_html=True)

# Define functions for each retrieval method
def CLIP_LSTM_8k():
    import mmrcs_clip_lstm_flickr8k
    if uploaded_file is not None:
        caption = mmrcs_clip_lstm_flickr8k.imageToCaptions(uploaded_file)
        st.markdown(f'Retrieved Caption : <div class="subtitle">{caption}</div>', unsafe_allow_html=True)

def CLIP_LSTM_30k():
    import mmrcs_clip_lstm_flickr30k
    if uploaded_file is not None:        
        caption = mmrcs_clip_lstm_flickr30k.imageToCaptions(uploaded_file)
        st.markdown(f'Retrieved Caption : <div class="subtitle">{caption}</div>', unsafe_allow_html=True)

# Define functions for each retrieval method
def CLIP_ATTENTION_8k():
    import mmrcs_clip_attention_flickr8k
    if uploaded_file is not None:
        caption = mmrcs_clip_attention_flickr8k.imageToCaptions(uploaded_file)
        st.markdown(f'Retrieved Caption : <div class="subtitle">{caption}</div>', unsafe_allow_html=True)

def CLIP_ATTENTION_30k():
    import mmrcs_clip_attention_flickr30k
    if uploaded_file is not None:        
        caption = mmrcs_clip_attention_flickr30k.imageToCaptions(uploaded_file)
        st.markdown(f'Retrieved Caption : <div class="subtitle">{caption}</div>', unsafe_allow_html=True)

def VITmodel():
    import mmrcs_vit
    if uploaded_file is not None:
        caption = mmrcs_vit.imageToCaptions(uploaded_file)
        st.markdown(f'Retrieved Caption : <div class="subtitle">{caption}</div>', unsafe_allow_html=True)

# Execute corresponding function based on selected model
if model_option == "CLIP - PRE - 8k":
    with st.spinner("⏳ Processing... Please wait..."):
        CLIP_PRE_8k()    
elif model_option == "CLIP - PRE - 30k":
    with st.spinner("⏳ Processing... Please wait..."):
        CLIP_PRE_30k()
elif model_option == "ResNet50 - LSTM - 8k":
    with st.spinner("⏳ Processing... Please wait..."):
        RESNET50_LSTM_8k()
elif model_option == "ResNet50 - LSTM - 30k":
    with st.spinner("⏳ Processing... Please wait..."):
        RESNET50_LSTM_30k()  
elif model_option == "CLIP - LSTM - 8k":
    with st.spinner("⏳ Processing... Please wait..."):
        CLIP_LSTM_8k()
elif model_option == "CLIP - LSTM - 30k":
    with st.spinner("⏳ Processing... Please wait..."):
        CLIP_LSTM_30k()
elif model_option == "CLIP - ATTENTION - 8k":
    with st.spinner("⏳ Processing... Please wait..."):
        CLIP_ATTENTION_8k()
elif model_option == "CLIP - ATTENTION - 30k":
    with st.spinner("⏳ Processing... Please wait..."):
        CLIP_ATTENTION_30k()    
elif model_option == "VIT - GPT2":
    with st.spinner("⏳ Processing... Please wait..."):
        with st.spinner("⏳ Processing... Please wait..."):VITmodel()

def captionToImage(caption):
    if model_option == "CLIP - PRE - 8k": 
        import mmrcs_clippretrained_flickr8k    
        retrieved_image = mmrcs_clippretrained_flickr8k.captionToImage(caption)
        image_full_path = base_path_8k_img + retrieved_image
        st.markdown("""<h3 style="text-align: center; color: #4CAF50;">Retrieved Image</h3>""", unsafe_allow_html=True)
        st.image(image_full_path, use_column_width=True)
    if model_option == "CLIP - PRE - 30k": 
        import mmrcs_clippretrained_flickr30k    
        retrieved_image = mmrcs_clippretrained_flickr30k.captionToImage(caption)
        image_full_path = base_path_30k_img + retrieved_image
        st.markdown("""<h3 style="text-align: center; color: #4CAF50;">Retrieved Image</h3>""", unsafe_allow_html=True)
        st.image(image_full_path, use_column_width=True)

# Call captionToImage when submit is clicked
if retrieval_option == "Text to Image" and st.session_state.input_text and submit_button:
    captionToImage(st.session_state.input_text)