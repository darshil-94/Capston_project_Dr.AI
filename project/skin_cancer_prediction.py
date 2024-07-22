import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model(r'C:\Users\darshil\AI shaksham\Project_AI\project\model_files\Skin_model.keras')


# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def show_skin_cancer_prediction():
    # Streamlit app
    st.title("Skin Cancer Detection")

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        # Make prediction
        prediction = model.predict(processed_image)

        # Fetch probabilities
        probabilities = prediction[0]

        # Get class names
        class_names = ['Benign', 'Malignant']

        # Create a dictionary of class names and their probabilities
        probability_dict = dict(zip(class_names, probabilities))

        # Get the predicted class and its probability
        predicted_class = max(probability_dict, key=probability_dict.get)
        confidence = probability_dict[predicted_class] * 100

        st.write(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")

        # Optionally display the probabilities for each class
        st.write("Class Probabilities:")
        for class_name, prob in probability_dict.items():
            st.write(f"{class_name}: {prob * 100:.2f}%")
