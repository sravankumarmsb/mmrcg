from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import numpy as np
import streamlit as st

# Load model, processor, tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Adjusting inference parameters
gen_kwargs = {"max_length": 30, "num_beams": 5}  # Increased length and beams

def preprocess_image(image):
    """Preprocess image for ViT model"""
    if isinstance(image, st.runtime.uploaded_file_manager.UploadedFile):
        image = Image.open(image).convert("RGB")  # Convert to RGB to avoid issues

    if not isinstance(image, Image.Image):
        raise ValueError("Invalid image format. Ensure it's a valid image file.")

    # Convert to pixel values required by the model
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    return pixel_values.to(device)

def imageToCaptions(image):
    """Generate image caption using ViT-GPT2"""
    pixel_values = preprocess_image(image)  # Preprocess the image

    # Generate caption
    output_ids = model.generate(pixel_values, **gen_kwargs)

    # Decode the output
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return preds[0].strip()