import streamlit as st
from PIL import Image
import os
import configparser

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read("config.ini")

# Read base path from config file
base_path = config.get("paths", "base_path_8k")
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
    </style>
    """, unsafe_allow_html=True)

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

# Task Selection
st.session_state.task_option = st.radio("Select Task:", ["Retrieval", "Caption Generation"], index=0)

# Model selection based on task
task_models = {
    "Retrieval": ["", "CLIP - PRE - 8k", "CLIP - PRE - 30k", "VIT - GPT2"],
    "Caption Generation": ["", "ResNet50 - LSTM - 8k", "ResNet50 - LSTM - 30k", "CLIP - LSTM - 8k", "CLIP - LSTM - 30k", "CLIP - ATTENTION - 8k", "CLIP - ATTENTION - 30k"]
}

model_option = st.selectbox("Select Model:", task_models[st.session_state.task_option], key="model_option")

# Dynamically adjust retrieval options
retrieval_options = ["", "Image to Text", "Text to Image"]
if model_option in task_models["Caption Generation"]:
    retrieval_options = ["", "Image to Text"]

retrieval_option = st.selectbox("Select Retrieval Type:", retrieval_options, key="retrieval_option")

uploaded_file = None
input_text = None

# File upload or text input based on retrieval type
if retrieval_option == "Image to Text":
    col1, col2 = st.columns(2)
    with col1:
        picture = st.camera_input("Take a Picture")
    with col2:
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key="uploaded_file")
    if picture:
        uploaded_file = picture
    if uploaded_file:
        file_name = uploaded_file.name
        saved_path = os.path.join(project_path, "uploaded_files", file_name)
        with open(saved_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

elif retrieval_option == "Text to Image":
    input_text = st.text_input("Enter a caption to retrieve an image:", "", key="input_text")
    submit_button = st.button("Submit")

# Define functions for each model
def CLIP_PRE_8k():
    import mmrcs_clippretrained_flickr8k
    if uploaded_file:
        caption = mmrcs_clippretrained_flickr8k.imageToCaptions(uploaded_file)
        st.markdown(f'Retrieved Caption: <div class="subtitle">{caption}</div>', unsafe_allow_html=True)

# (Add similar functions for other models here...)

def captionToImage(caption):
    import mmrcs_clippretrained_flickr30k
    retrieved_image = mmrcs_clippretrained_flickr30k.captionToImage(caption)
    image_full_path = os.path.join("E:\\AI&ML\\U5_Capstone_Project\\flickr30k_dataset\\flickr30k_images\\", retrieved_image)
    st.markdown("""<h3 style="text-align: center; color: #4CAF50;">Retrieved Image</h3>""", unsafe_allow_html=True)
    st.image(image_full_path, use_column_width=True)

# Execute model function
if model_option == "CLIP - PRE - 8k":
    CLIP_PRE_8k()
# (Add similar conditionals for other models...)

# Call captionToImage when submit is clicked
if retrieval_option == "Text to Image" and input_text and submit_button:
    captionToImage(input_text)
