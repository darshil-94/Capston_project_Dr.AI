import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the model and encoders
model_path = r"C:\Users\darshil\AI shaksham\Project_AI\project\model_files\custom_model.pkl"
encoder_path = r"C:\Users\darshil\AI shaksham\Project_AI\project\model_files\label_encoder.pkl"

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(encoder_path, 'rb') as encoder_file:
    encoders = pickle.load(encoder_file)


# Define a function to preprocess user input
def preprocess_input(data):
    df = pd.DataFrame([data])

    # One-hot encoding for categorical variables
    one_hot_columns = ['gender_Male', 'ever_married_Yes', 'work_type_Private', 'work_type_Self-employed',
                       'work_type_children', 'Residence_type_Urban',
                       'smoking_status_formerly smoked', 'smoking_status_never smoked',
                       'smoking_status_smokes']

    # Ensure all required columns are in the DataFrame
    for column in one_hot_columns:
        if column not in df.columns:
            df[column] = 0

    # Encoding categorical variables
    for column, encoder in encoders.items():
        if column in df.columns:
            df[column] = encoder.transform(df[column])

    # Reorder columns to match training data columns
    required_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi'] + one_hot_columns
    for column in required_columns:
        if column not in df.columns:
            df[column] = 0
    df = df[required_columns]

    print("Final DataFrame columns for prediction:", df.columns)
    return df

def show_stroke_prediction() :
    # Title
    st.title('Stroke Prediction')

    # User input
    gender = st.selectbox('Gender', ['Male', 'Female'],key="stroke_age_2")
    age = st.number_input('Age', min_value=0, max_value=120,key="stroke_age")
    hypertension = st.selectbox('Hypertension', [0, 1],key="stroke_hypertention")
    heart_disease = st.selectbox('Heart Disease', [0, 1],key="strok_heart_disease")
    ever_married = st.selectbox('Ever Married', ['No', 'Yes'],key="stroke_ever_merried")
    work_type = st.selectbox('Work Type', ['Govt_job', 'Private', 'Self-employed'],key="stroke_work_type")
    residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'],key="stroke_recidencetype")
    avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, max_value=300.0,key="stroke_avg_glucose")
    bmi = st.number_input('BMI', min_value=0.0, max_value=100.0,key="stroke_BMI")
    smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'],key="stroke_smoking")

    # Button for prediction
    if st.button('Predict Stroke',key="Stroke_Prediction"):
        user_data = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }

        # Preprocess the user input
        user_data_df = preprocess_input(user_data)

        # Make prediction
        prediction = model.predict(user_data_df)[0]

        # Output result
        if prediction == 1:
            st.write("The model predicts that the person is at risk of stroke.")
        else:
            st.write("The model predicts that the person is not at risk of stroke.")
