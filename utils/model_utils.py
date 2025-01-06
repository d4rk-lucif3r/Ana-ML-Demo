import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

@st.cache_data
def prepare_training_data(data):
    """Prepare data for model training"""
    try:
        # Merge train with health camp details
        df = data['Train'].merge(data['Health Camp Detail'], on='Health_Camp_ID', how='left')
        
        # Add patient profile information
        df = df.merge(data['Patient Profile'], on='Patient_ID', how='left')
        
        # Create target variable
        first_camp_patients = set(data['First Health Camp']['Patient_ID'])
        second_camp_patients = set(data['Second Health Camp']['Patient_ID'])
        third_camp_patients = set(data['Third Health Camp']['Patient_ID'])
        
        df['Outcome'] = df.apply(
            lambda x: 1 if (
                (x['Category1'] == 'First' and x['Patient_ID'] in first_camp_patients) or
                (x['Category1'] == 'Second' and x['Patient_ID'] in second_camp_patients) or
                (x['Category1'] == 'Third' and x['Patient_ID'] in third_camp_patients)
            ) else 0,
            axis=1
        )
        
        return df
    except Exception as e:
        st.error(f"Error preparing training data: {str(e)}")
        return None

def save_model(model, scaler, features, filename_prefix="healthcare_camp"):
    """Save trained model and associated objects"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model
        model_filename = f"models/{filename_prefix}_model.joblib"
        joblib.dump(model, model_filename)
        
        # Save scaler
        scaler_filename = f"models/{filename_prefix}_scaler.joblib"
        joblib.dump(scaler, scaler_filename)
        
        # Save feature names
        features_filename = f"models/{filename_prefix}_features.joblib"
        joblib.dump(features, features_filename)
        
        return True
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        return False

def load_model(filename_prefix="healthcare_camp"):
    """Load trained model and associated objects"""
    try:
        model_path = f"models/{filename_prefix}_model.joblib"
        scaler_path = f"models/{filename_prefix}_scaler.joblib"
        features_path = f"models/{filename_prefix}_features.joblib"
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
            return None, None, None
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        
        return model, scaler, features
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None
