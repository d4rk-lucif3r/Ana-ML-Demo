import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    """Load all datasets and return them in a dictionary"""
    try:
        health_camp_detail = pd.read_csv('Data/Health_Camp_Detail.csv')
        train = pd.read_csv('Data/Train.csv')
        patient_profile = pd.read_csv('Data/Patient_Profile.csv')
        first_health_camp = pd.read_csv('Data/First_Health_Camp_Attended.csv')
        second_health_camp = pd.read_csv('Data/Second_Health_Camp_Attended.csv')
        third_health_camp = pd.read_csv('Data/Third_Health_Camp_Attended.csv')
        test = pd.read_csv('Data/test.csv')
        
        return {
            'Health Camp Detail': health_camp_detail,
            'Train': train,
            'Patient Profile': patient_profile,
            'First Health Camp': first_health_camp,
            'Second Health Camp': second_health_camp,
            'Third Health Camp': third_health_camp,
            'Test': test
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_dataset_description():
    """Return descriptions for each dataset"""
    return {
        'Health Camp Detail': 'Contains information about 65 health camps including dates and categories',
        'Train': 'Contains 75,278 registration records with anonymized variables',
        'Patient Profile': 'Contains demographic and social media information for 37,633 patients',
        'First Health Camp': 'Contains donation and health scores for first format camps',
        'Second Health Camp': 'Contains health scores for second format camps',
        'Third Health Camp': 'Contains stall visit information for third format camps',
        'Test': 'Contains registration records for camps after April 2006'
    }
