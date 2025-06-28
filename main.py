import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io
import pandas as pd
import requests
from dotenv import load_dotenv
import os

# Load Hugging Face API key from .env
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

# Initialize InferenceClient for age classification
age_client = InferenceClient(token=HF_TOKEN)

# Hugging Face API details for AI image detector
API_URL_DETECTOR = "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_age(image):
    # Convert PIL image to bytes (JPEG)
    image = image.convert('RGB')
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)
    # Pass bytes, not BytesIO object!
    result = age_client.image_classification(
        image=image_bytes.getvalue(),
        model="nateraw/vit-age-classifier"
    )
    return result

def query_detector(image_bytes):
    # Send a POST request to the API with image data
    response = requests.post(API_URL_DETECTOR, headers=headers, data=image_bytes)
    return response.json()

def age_classification():
    st.title("Age Classification")

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if not HF_TOKEN:
            st.error("Hugging Face API token not found! Please set HUGGINGFACE_API_KEY in your .env file.")
            return

        # Call the Hugging Face API
        with st.spinner('Classifying...'):
            result = query_age(image)

        # Display API response
        if result and isinstance(result, list) and len(result) > 0:
            df = pd.DataFrame(result)
            st.write("API Response:")
            st.table(df)

            # Determine the label with the highest score
            top_result = df.loc[df['score'].idxmax()]
            label = top_result['label']
            st.write(f"The person in the image is likely in the age group: **{label}** (score: {top_result['score']:.2f})")
        else:
            st.write("An error occurred while processing the image. Please try again.")

def ai_image_detector():
    st.title("AI Image Detector")

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        # Convert image to bytes
        image_bytes = uploaded_file.read()

        # Call the Hugging Face API
        with st.spinner('Analyzing...'):
            result = query_detector(image_bytes)

        # Check and display the result
        if result:
            # Convert result to DataFrame for table display
            df = pd.DataFrame(result)
            st.write("API Response:")
            st.table(df)

            # Determine the label with the highest score
            if not df.empty:
                top_result = df.loc[df['score'].idxmax()]
                label = top_result['label']
                st.write(f"The image is likely **{label}** with a score of {top_result['score']:.2f}.")
            else:
                st.write("No results to display.")
        else:
            st.write("Failed to get a valid response from the API.")

def is_artificial_detector():
    st.title("Is Image Artificial?")

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        # Convert image to bytes
        image_bytes = uploaded_file.read()

        # Call the Hugging Face API
        with st.spinner('Analyzing...'):
            result = query_detector(image_bytes)

        # Check and display the result
        if result:
            # Determine the likelihood based on the scores
            is_artificial = False
            for item in result:
                if item['label'] == 'artificial' and item['score'] > 0.20:
                    is_artificial = True
                    break

            if is_artificial:
                st.write("The image may be artificially generated.")
            else:
                st.write("The image is likely human.")
        else:
            st.write("Failed to get a valid response from the API.")

def main():
    st.set_page_config(page_title="AI Image Tools", page_icon=":robot:")

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Age Classification", "AI Image Detector", "Is Image Artificial?"])

    if selection == "Age Classification":
        age_classification()
    elif selection == "AI Image Detector":
        ai_image_detector()
    elif selection == "Is Image Artificial?":
        is_artificial_detector()

if __name__ == "__main__":
    main()
