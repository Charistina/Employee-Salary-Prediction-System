import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the entire pipeline
pipeline = joblib.load("best_pipeline.pkl")

st.title("Employee Salary Prediction App")

st.sidebar.header("User Input Features")

# Feature mapping based on original dataset
EDUCATION_MAPPING = {
    'Bachelors': 'Bachelors',
    'HS-grad': 'HS-grad',
    'Masters': 'Masters',
    'Assoc-voc': 'Assoc-voc',
    'Assoc-acdm': 'Assoc-acdm',
    'Some-college': 'Some-college',
    'Doctorate': 'Doctorate',
    'Prof-school': 'Prof-school',
    '7th-8th': '7th-8th',
    '11th': '11th',
    '10th': '10th',
    '12th': '12th',
    '9th': '9th',
    '5th-6th': '5th-6th',
    '1st-4th': '1st-4th',
    'Preschool': 'Preschool'
}

OCCUPATION_MAPPING = {
    'Tech-support': 'Tech-support',
    'Craft-repair': 'Craft-repair',
    'Other-service': 'Other-service',
    'Exec-managerial': 'Exec-managerial',
    'Prof-specialty': 'Prof-specialty',
    'Sales': 'Sales',
    'Adm-clerical': 'Adm-clerical',
    'Machine-op-inspct': 'Machine-op-inspct',
    'Transport-moving': 'Transport-moving',
    'Handlers-cleaners': 'Handlers-cleaners',
    'Farming-fishing': 'Farming-fishing',
    'Protective-serv': 'Protective-serv',
    'Priv-house-serv': 'Priv-house-serv',
    'Armed-Forces': 'Armed-Forces'
}

def user_input():
    age = st.sidebar.slider("Age", 17, 90, 30)
    workclass = st.sidebar.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", 
                                                 "Federal-gov", "Local-gov", "State-gov", 
                                                 "Without-pay", "Never-worked"])
    education = st.sidebar.selectbox("Education", list(EDUCATION_MAPPING.keys()))
    marital_status = st.sidebar.selectbox("Marital Status", ["Never-married", "Married-civ-spouse", 
                                                           "Divorced", "Separated", "Widowed",
                                                           "Married-spouse-absent", "Married-AF-spouse"])
    occupation = st.sidebar.selectbox("Occupation", list(OCCUPATION_MAPPING.keys()))
    relationship = st.sidebar.selectbox("Relationship", ["Wife", "Own-child", "Husband", 
                                                       "Not-in-family", "Other-relative", "Unmarried"])
    race = st.sidebar.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", 
                                       "Amer-Indian-Eskimo", "Other"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    capital_gain = st.sidebar.slider("Capital Gain", 0, 100000, 0)
    capital_loss = st.sidebar.slider("Capital Loss", 0, 5000, 0)
    hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40)
    native_country = st.sidebar.selectbox("Native Country", ["United-States", "Mexico", "Philippines",
                                                           "Germany", "Canada", "Puerto-Rico",
                                                           "El-Salvador", "India", "Cuba", "England"])
    
    # Map to dataset values
    education_val = EDUCATION_MAPPING[education]
    occupation_val = OCCUPATION_MAPPING[occupation]
    
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [189000],  # Median value from dataset
        'education': [education_val],
        'educational-num': [13 if education_val == 'Bachelors' else 
                            9 if education_val == 'HS-grad' else 
                            14 if education_val == 'Masters' else 10],  # Approximate mapping
        'marital-status': [marital_status],
        'occupation': [occupation_val],
        'relationship': [relationship],
        'race': [race],
        'gender': [gender],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })
    return input_data

input_df = user_input()

# Display user input
st.subheader("User Input Features")
st.write(input_df)

# Predict using the pipeline
try:
    prediction = pipeline.predict(input_df)
    prediction_proba = pipeline.predict_proba(input_df)
    
    # Display prediction
    st.subheader("Prediction Result")
    result = "> $50K/year" if prediction[0] == 1 else "<= $50K/year"
    st.write(f"Income: **{result}**")
    
    # Show confidence
    confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
    st.write(f"Confidence: {confidence:.1%}")
    
    # Show feature importance if available
    if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
        st.subheader("Top Influencing Factors")
        feature_importances = pipeline.named_steps['classifier'].feature_importances_
        
        # Get feature names from preprocessing
        if hasattr(pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
            feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importances
            }).sort_values('Importance', ascending=False).head(5)
            
            st.bar_chart(importance_df.set_index('Feature'))
    
except Exception as e:
    st.error(f"Prediction error: {str(e)}")
    st.info("Make sure all input features match the training data format")

# Batch Prediction
st.subheader("Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV file for batch predictions", type=["csv"])

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Sample of uploaded data:")
        st.write(batch_data.head())
        
        # Add missing columns with default values
        required_columns = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 
                           'marital-status', 'occupation', 'relationship', 'race', 
                           'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 
                           'native-country']
        
        for col in required_columns:
            if col not in batch_data.columns:
                batch_data[col] = 0 if col == 'fnlwgt' else 'Unknown'
        
        # Ensure correct column order
        batch_data = batch_data[required_columns]
        
        # Make predictions
        batch_preds = pipeline.predict(batch_data)
        batch_data['Prediction'] = ['> $50K' if p == 1 else '<= $50K' for p in batch_preds]
        
        st.subheader("Prediction Results")
        st.write(batch_data)
        
        # Download results
        st.download_button(
            label="Download Predictions",
            data=batch_data.to_csv(index=False),
            file_name='salary_predictions.csv',
            mime='text/csv'
        )
        
    except Exception as e:
        st.error(f"Batch processing error: {str(e)}")