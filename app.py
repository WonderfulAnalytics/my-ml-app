# app.py
import streamlit as st
import joblib
import pandas as pd

# Load the trained model and the columns
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Streamlit app
st.title('Insurance Charges Prediction App')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data uploaded successfully")
    st.write(data)
    
    # Ensure the data is preprocessed the same way as the training data
    data_processed = pd.get_dummies(data, drop_first=True)
    
    # Reindex the data to ensure all expected columns are present
    data_processed = data_processed.reindex(columns=model_columns, fill_value=0)
    
    # Make predictions
    predictions = model.predict(data_processed)
    
    # Add predictions to the original data
    data['Predicted Charges'] = predictions
    
    # Display the data with predictions
    st.write("Data with Predictions")
    st.write(data)
