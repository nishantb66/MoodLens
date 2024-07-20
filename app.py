import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image

# Load the pre-trained CNN model
model = load_model("models/imageclassifier.h5")


# Function to preprocess the image
def preprocess_image(image):
    # Convert the image to RGB
    img = image.convert("RGB")
    # Resize the image to match the model's expected input size
    img = img.resize((256, 256))  # Adjust the size to match your model's input
    # Convert the image to array
    img = np.array(img)
    # Normalize the image
    img = img / 255.0
    # Expand dimensions to match the model's expected input
    img = np.expand_dims(img, axis=0)
    return img


# Set page config to wide mode, add a title and favicon
st.set_page_config(page_title="MoodLens", page_icon=":camera:", layout="wide")

# Add a header and an explanatory text
st.title("MoodLens: Happy or Sad ? 📸")
st.subheader("Discover the sentiment behind every smile and frown.")
st.markdown(
    """
    MoodLens is an AI-powered application that uses Convolutional Neural Networks (CNN) to detect if you are happy or sad from facial images.
    Simply upload an image and let MoodLens reveal the sentiment.
"""
)

# Divide the page into three columns
col1, col2, col3 = st.columns([1, 2, 1])

# File uploader in the second column
with col2:
    uploaded_file = st.file_uploader("Upload an image and then the result will be displayed", type=["png", "jpg", "jpeg"])

# Check if an image has been uploaded
if uploaded_file is not None:
    # Display the uploaded image in the second column
    with col2:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")

        # Preprocess the uploaded image
        processed_img = preprocess_image(image)

        # Predict the class
        prediction = model.predict(processed_img)

        # Determine the class label based on the prediction
        class_label = "Happy" if prediction[0][0] < 0.5 else "Sad"

        # Display the classification result with a larger font and bold text
        st.markdown(
            f"<h1 style='text-align: center; color: white; background-color: #76b900; padding: 10px;'>"
            f"The sentiment of the image is: {class_label}</h1>",
            unsafe_allow_html=True,
        )
else:
    # Prompt in the second column
    with col2:
        st.write("Please upload an image to classify.")

# Add a footer
st.markdown(
    """
    ---
    MoodLens is part of a Learning project by me, exploring the fields of Data Science and Machine Learning. The detection accuracy of my model may vary based on the complexity of the images. The reason behined this is that the model was trained on a limited dataset (approx 200 images of both class) and may not generalize well to all images but it works to an great extent and validation accuracy is 96% and model accuracy after training is 100%
    For more information, visit GitHub.
"""
)

# Run the Streamlit app by typing `streamlit run app.py` in the terminal
# code to run the app in terminal : streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
