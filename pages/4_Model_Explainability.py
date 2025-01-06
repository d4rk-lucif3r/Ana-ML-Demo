import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from utils.data_loader import load_data
from utils.model_utils import prepare_training_data, load_model

st.set_page_config(page_title="Model Explainability", page_icon="🔍", layout="wide")

st.title("Model Explainability 🔍")

st.markdown("""
### Understanding Model Decisions 🤔

This section helps you understand:
- How the model makes its predictions
- Which features influence the predictions most
- Why specific predictions were made

This transparency helps:
- Build trust in the model
- Identify potential biases
- Make better decisions using the model
""")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = None

# Load data
data = load_data()

# Try to get model from session state first, then from file
model = st.session_state.model
scaler = st.session_state.scaler
features = st.session_state.selected_features

# If not in session state, try loading from file
if not all([model, scaler, features]):
    model, scaler, features = load_model()
    if all([model, scaler, features]):
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.selected_features = features

if data and model and scaler and features:
    # Prepare data for explanation
    df = prepare_training_data(data)
    
    if df is not None:
        # Prepare features using saved feature list and scaler
        X = df[features].fillna(df[features].mean())
        X_scaled = scaler.transform(X)
        
        # Calculate SHAP values based on model type
        st.subheader("1️⃣ Global Feature Importance")
        st.markdown("""
        This shows how each feature affects predictions overall:
        - Red = Feature increases the prediction
        - Blue = Feature decreases the prediction
        - Wider bars = More impact on predictions
        """)
        
        with st.spinner("Calculating SHAP values... This might take a moment."):
            # Select appropriate explainer based on model type
            if isinstance(model, (RandomForestClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier)):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_scaled)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            
            elif isinstance(model, LogisticRegression):
                explainer = shap.LinearExplainer(model, X_scaled)
                shap_values = explainer.shap_values(X_scaled)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                expected_value = explainer.expected_value
            
            else:
                # Fallback to KernelExplainer for unknown model types
                background = shap.kmeans(X_scaled, 50)  # Use subset of data as background
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(X_scaled)[1]
                expected_value = explainer.expected_value[1]
            
            # Plot SHAP summary
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            st.pyplot(fig)
            plt.clf()
            
        # Individual prediction explanation
        st.subheader("2️⃣ Individual Prediction Explanation")
        st.markdown("""
        Analyze specific predictions:
        - See exactly why a prediction was made
        - Understand which features contributed most
        - Learn how changing features would affect the outcome
        """)
        
        # Select a sample for explanation
        sample_idx = st.number_input(
            "Select sample index", 
            0, 
            len(X)-1, 
            0,
            help="Choose a specific case to analyze"
        )
        
        if st.button("Explain Prediction", help="Click to analyze this specific prediction"):
            # Get prediction
            sample = X.iloc[sample_idx:sample_idx+1]
            sample_scaled = scaler.transform(sample)
            prediction = model.predict_proba(sample_scaled)[0][1]
            
            st.write(f"Predicted probability of favorable outcome: {prediction:.1%}")
            
            # SHAP force plot
            st.markdown("### Feature Contributions")
            fig, ax = plt.subplots(figsize=(10, 3))
            shap.force_plot(
                expected_value,
                shap_values[sample_idx,:],
                sample,
                matplotlib=True,
                show=False
            )
            st.pyplot(fig)
            plt.clf()
            
            # Feature contributions table
            st.markdown("""
            ### Detailed Impact Analysis
            - Positive values push towards favorable outcome
            - Negative values push against favorable outcome
            - Larger absolute values = stronger impact
            """)
            
            contributions = pd.DataFrame({
                'Feature': features,
                'Value': sample.values[0],
                'Impact': shap_values[sample_idx]
            }).sort_values('Impact', key=abs, ascending=False)
            
            st.dataframe(
                contributions.style.background_gradient(
                    subset=['Impact'],
                    cmap='RdBu'
                ),
                use_container_width=True
            )
    else:
        st.error("Error preparing data for explanation.")
else:
    if not data:
        st.error("⚠️ Error loading data. Please check if all data files are present in the Data directory.")
    if not model:
        st.warning("⚠️ No trained model found. Please train a model in the Model Training section first.")
