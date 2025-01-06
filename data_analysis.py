import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read all CSV files
health_camp_detail = pd.read_csv('Data/Health_Camp_Detail.csv')
train = pd.read_csv('Data/Train.csv')
patient_profile = pd.read_csv('Data/Patient_Profile.csv')
first_health_camp = pd.read_csv('Data/First_Health_Camp_Attended.csv')
second_health_camp = pd.read_csv('Data/Second_Health_Camp_Attended.csv')
third_health_camp = pd.read_csv('Data/Third_Health_Camp_Attended.csv')
test = pd.read_csv('Data/test.csv')

# Read Data Dictionary
data_dict = pd.read_excel('Data/Data_Dictionary.xlsx')

# Print basic information about each dataset
datasets = {
    'Health Camp Detail': health_camp_detail,
    'Train': train,
    'Patient Profile': patient_profile,
    'First Health Camp': first_health_camp,
    'Second Health Camp': second_health_camp,
    'Third Health Camp': third_health_camp,
    'Test': test
}

print("Data Dictionary:")
print(data_dict.head())
print("\n" + "="*50 + "\n")

for name, df in datasets.items():
    print(f"\n{name} Dataset:")
    print("-" * 30)
    print(f"Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nData Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\n" + "="*50 + "\n")
