import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


# Load the pre-trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r"C:\Users\darshil\AI shaksham\Project_AI\project\model_files\lstm_model.h5")
    return model


model = load_model()


# Load the tokenizer
@st.cache_resource
def load_tokenizer():
    with open(r"C:\Users\darshil\AI shaksham\Project_AI\project\model_files\tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def show_disease_prediction():

    tokenizer = load_tokenizer()

    # Define diseases list
    diseases = ['Psoriasis', 'Varicose Veins', 'Typhoid', 'Chicken pox',
                'Impetigo', 'Dengue', 'Fungal infection', 'Common Cold',
                'Pneumonia', 'Dimorphic Hemorrhoids', 'Arthritis', 'Acne',
                'Bronchial Asthma', 'Hypertension', 'Migraine',
                'Cervical spondylosis', 'Jaundice', 'Malaria',
                'urinary tract infection', 'allergy',
                'gastroesophageal reflux disease', 'drug reaction',
                'peptic ulcer disease', 'diabetes']


    # Function to preprocess input text
    def preprocess_text(text, tokenizer, maxlen=100):
        sequences = tokenizer.texts_to_sequences([text])
        padded_sequences = pad_sequences(sequences, maxlen=maxlen)
        return padded_sequences


    # Streamlit app
    st.title("Disease Prediction from Symptoms")

    st.write("Enter the symptoms you are experiencing below:")

    user_input = st.text_area("Symptoms", key="disease_symptoms")

    if st.button("Predict Disease",key='Disease_prediction'):
        if user_input:
            # Preprocess the user input
            preprocessed_input = preprocess_text(user_input, tokenizer)

            # Make prediction
            prediction = model.predict(preprocessed_input)

            # Get the predicted disease
            predicted_disease = diseases[np.argmax(prediction)]

            st.write(f"The predicted disease is: {predicted_disease}")
        else:
            st.write("Please enter your symptoms.")
