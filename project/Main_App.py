import streamlit as st
import stroke_prediction
import sleep_disorder_prediction
import disease_prediction
import skin_cancer_prediction
import streamlit.components.v1 as components

# Define the path to your custom component HTML file

from PIL import Image

# Load the local image
image = Image.open(r"C:\Users\darshil\AI shaksham\Project_AI\project\Image\Bg_img.jpg")

# Display the image as the background
st.image(image, use_column_width=True)

# st.markdown(page_bg_img,unsafe_allow_html=True)
# Initialize session state for page selection if not already set
if 'selection' not in st.session_state:
    st.session_state.selection = "Home"

# Set up the main app navigation
st.title("Dr. AI: AI Diagnosis and Care Chat")

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Home", "Stroke Prediction", "Sleep Disorder Prediction", "Disease Prediction", "Skin Cancer Prediction"], index=["Home", "Stroke Prediction", "Sleep Disorder Prediction", "Disease Prediction", "Skin Cancer Prediction"].index(st.session_state.selection))

# Update session state with the selected page
st.session_state.selection = selection

if selection == "Home":

    st.header("Welcome to Dr. AI: AI Diagnosis and Care Chat")
    st.write("""
         This app provides predictions for various healthcare conditions using machine learning models. 
        Use the sidebar to navigate to the specific prediction models:
        - **Stroke Prediction**: Predicts the likelihood of having a stroke based on user inputs.
        - **Sleep Disorder Prediction**: Predicts if a user has a sleep disorder based on their symptoms.
        - **Disease Prediction**: Predicts the disease based on user-reported symptoms.
        - **Skin Cancer Prediction**: Predicts the likelihood of skin cancer based on user inputs.

        **Note:** The chatbot is available at the bottom of the page for any additional questions or support.
    """)
    components.html(open(r'C:\Users\darshil\AI shaksham\Project_AI\project\index.html').read(), height=600)

elif selection == "Stroke Prediction":
    stroke_prediction.show_stroke_prediction()
elif selection == "Sleep Disorder Prediction":
    sleep_disorder_prediction.show_sleep_disorder_prediction()
elif selection == "Disease Prediction":
    disease_prediction.show_disease_prediction()
elif selection == "Skin Cancer Prediction":
    skin_cancer_prediction.show_skin_cancer_prediction()



