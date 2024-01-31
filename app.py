from io import BytesIO
import streamlit as st
import requests
from PIL import Image


# Streamlit UI
st.title("Traffic Object Detection")

# Upload image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    url = "http://localhost:8000/detect"
    files = {"file": uploaded_image}
    response = requests.post(url, files=files)
    # Make a request to the FastAPI endpoint for object detection
    if response.status_code == 200:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        detected_image_bytes = BytesIO(response.content)
        st.text(body=detected_image_bytes)
        detected_image = Image.open(detected_image_bytes)
        st.image(detected_image, caption="Processed Image", use_column_width=True)
    else:
        st.error("Failed to perform detection. Please try again.")
