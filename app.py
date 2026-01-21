import streamlit as st
import pandas as pd
import joblib

# 1. Load the pre-trained Logistic Regression model and LabelEncoder
model = joblib.load('logistic_regression_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# 2. Set the title of the Streamlit application
st.title('Crop Recommendation System')

# 3. Create input fields for each feature
st.sidebar.header('Input Crop Parameters')

N = st.sidebar.number_input('Nitrogen (N)', min_value=0, max_value=140, value=90)
P = st.sidebar.number_input('Phosphorus (P)', min_value=0, max_value=145, value=42)
K = st.sidebar.number_input('Potassium (K)', min_value=0, max_value=205, value=43)
temperature = st.sidebar.number_input('Temperature (°C)', min_value=0.0, max_value=45.0, value=20.88, format="%.2f")
humidity = st.sidebar.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=82.00, format="%.2f")
ph = st.sidebar.number_input('pH Value', min_value=0.0, max_value=14.0, value=6.50, format="%.2f")
rainfall = st.sidebar.number_input('Rainfall (mm)', min_value=0.0, max_value=300.0, value=202.93, format="%.2f")

# 4. Create a prediction button
if st.sidebar.button('Predict Crop'):
    # 5. Create a Pandas DataFrame from the user inputs
    input_data = pd.DataFrame([{
        'N': N,
        'P': P,
        'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }])

    st.write('Input Data:')
    st.write(input_data)

    # 6. Make a prediction using the loaded model
    prediction = model.predict(input_data)

    # 7. Decode the numerical prediction back into a human-readable crop name
    predicted_crop = label_encoder.inverse_transform(prediction)

    # 8. Display the predicted crop to the user
    st.success(f'The recommended crop is: {predicted_crop[0]}')
