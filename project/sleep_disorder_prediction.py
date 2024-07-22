import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the sleep disorder prediction model
model_path = r"C:\Users\darshil\AI shaksham\Project_AI\project\model_files\Sleep_model_2.pkl"
encoder_path = r"C:\Users\darshil\AI shaksham\Project_AI\project\model_files\label_encoder_Sleep.pkl"

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(encoder_path, 'rb') as encoder_file:
    encoders = pickle.load(encoder_file)


# Preprocessing function
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Ensure all required columns are present
    one_hot_columns = ['Gender_Female', 'Gender_Male', 'Occupation_Accountant', 'Occupation_Doctor',
                       'Occupation_Engineer','Occupation_Lawyer','Occupation_Manager','Occupation_Nurse',
                       'Occupation_Sales Representative','Occupation_Salesperson','Occupation_Scientist',
                       'Occupation_Software Engineer','Occupation_Teacher','BMI Category_Normal',
                       'BMI Category_Normal Weight','BMI Category_Obese','BMI Category_Overweight'
                       ]

    for column in one_hot_columns:
        if column not in df.columns:
            df[column] = 0

    # Encoding categorical variables
    for column, encoder in encoders.items():
        if column in df.columns:
            df[column] = encoder.transform(df[column])

    # Reorder columns to match training data columns
    required_columns = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level','Heart Rate','Daily Steps','Systolic','Diastolic'] + one_hot_columns
    for column in required_columns:
        if column not in df.columns:
            df[column] = 0
    df = df[required_columns]

    print("Final DataFrame columns for prediction:", df.columns)
    return df

def show_sleep_disorder_prediction():
    # Streamlit app
    st.title('Sleep Disorder Prediction')

    # Sleep Disorder Prediction
    st.header('Sleep Disorder Prediction')
    age = st.number_input('Age', min_value=0, max_value=120,key="sleep_age")
    sleep_duration = st.number_input('Sleep Duration', min_value=0.0,key="sleep_sleep_duration")
    quality_of_sleep = st.number_input('Quality of Sleep', min_value=1, max_value=10,key="sleep_quality_sleep")
    physical_activity_level = st.number_input('Physical Activity Level', min_value=0,key="sleep_physical_activity")
    stress_level = st.number_input('Stress Level', min_value=0,key="stress_level")
    heart_rate = st.number_input('Heart Rate', min_value=0,key="heart_rate")
    daily_steps = st.number_input('Daily Steps', min_value=0,key="Daily_step")
    systolic = st.number_input('Systolic Blood Pressure', min_value=0,key="systolic")
    diastolic = st.number_input('Diastolic Blood Pressure', min_value=0,key="diastolic")
    gender = st.selectbox('Gender', ['Male', 'Female'],key="sleep_gender")
    occupation = st.selectbox('Occupation', ['Software Engineer', 'Doctor', 'Sales Representative', 'Teacher',
           'Nurse', 'Engineer', 'Accountant', 'Scientist', 'Lawyer',
           'Salesperson', 'Manager'],key="sleep_occupation")
    bmi_category = st.selectbox('BMI Category', ['Overweight', 'Normal', 'Obese', 'Normal Weight'],key="BMI_category")


    if st.button('Predict Sleep Disorder',key="Sleep_Disorder"):
        user_data = {
            'Age': age,
            'Sleep Duration': sleep_duration,
            'Quality of Sleep': quality_of_sleep,
            'Physical Activity Level': physical_activity_level,
            'Stress Level': stress_level,
            'Heart Rate': heart_rate,
            'Daily Steps': daily_steps,
            'Systolic': systolic,
            'Diastolic': diastolic,
            'Gender': gender,
            'Occupation': occupation,
            'BMI Category': bmi_category,
        }

        # Preprocess the user input
        user_data_df = preprocess_input(user_data)

        # Make prediction
        prediction = model.predict(user_data_df)[0]

        # Output result
        if prediction == 'Sleep Apnea':
            st.write("The model predicts that the person has Sleep Apnea.")
        elif prediction == 'Insomnia':
            st.write("The model predicts that the person has Insomnia.")
        else:
            st.write("The model predicts that the person does not have a sleep disorder.")
