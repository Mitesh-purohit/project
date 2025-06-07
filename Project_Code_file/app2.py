import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the dataset to get feature names
data = pd.read_csv('creditcard.csv')
feature_names = data.columns[:-1]  # Exclude the target column (assuming 'Class' is the target)

def predict(features):
    prediction = model.predict(features)
    if prediction == 1:
        return "Fraudulent"
    else:
        return "Legitimate"

def main():
    st.title("Credit Card Fraud Detection App")

    # Create input fields for features
    input_features = []
    for col in feature_names:
        input_features.append(st.number_input(col, min_value=data[col].min(), max_value=data[col].max(), step=0.1))

    # Create a DataFrame from user input
    user_input = pd.DataFrame([input_features], columns=feature_names)

    # Make prediction and display result
    if st.button("Predict"):
        prediction = predict(user_input)
        st.write(f"The transaction is predicted to be: {prediction}")

if __name__ == "__main__":
    main()