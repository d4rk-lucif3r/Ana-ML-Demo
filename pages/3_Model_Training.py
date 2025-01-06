import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.data_loader import load_data
from utils.model_utils import prepare_training_data, save_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = None

# Load data
data = load_data()

st.title("Model Training & Evaluation 🤖")

st.markdown("""
### Understanding the Prediction Task 📋

We're trying to predict whether a registered person will:
- Get a health score (for Format 1 & 2 camps)
- Visit at least one stall (for Format 3 camps)

This helps MedCamp:
- Better plan their resources
- Improve attendance rates
- Enhance camp effectiveness
""")

if data:
    # Data Preparation
    st.subheader("1️⃣ Data Preparation")
    st.markdown("""
    First, we'll prepare the data by:
    - Combining information from different datasets
    - Creating the target variable (favorable outcome)
    - Handling missing values
    """)
    
    df = prepare_training_data(data)
    
    if df is not None:
        # Feature Selection
        st.subheader("2️⃣ Feature Selection")
        st.markdown("""
        Select features to use for prediction:
        - Choose numerical features that might influence attendance
        - More features aren't always better - focus on relevant ones
        """)
        
        # Select numerical features
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numerical_features = [col for col in numerical_features 
                            if col not in ['Patient_ID', 'Health_Camp_ID', 'Outcome']]
        
        selected_features = st.multiselect(
            "Select features for training",
            numerical_features,
            default=numerical_features[:5],
            help="Choose the features you think might help predict attendance"
        )
        
        if selected_features:
            # Model Selection and Configuration
            st.subheader("3️⃣ Model Selection & Configuration")
            st.markdown("""
            Select and configure your model:
            - Choose from different algorithms
            - Configure model-specific parameters
            - Set training parameters
            """)
            
            # Model Selection
            model_type = st.selectbox(
                "Select Model Type",
                ["Random Forest", "XGBoost", "LightGBM", "Logistic Regression", "CatBoost"],
                help="Choose the machine learning algorithm to use"
            )
            
            # Common Parameters
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider(
                    "Test Set Size (%)", 
                    10, 40, 20,
                    help="Percentage of data to use for testing"
                )
                random_state = st.number_input(
                    "Random State", 
                    0, 100, 42,
                    help="For reproducible results"
                )
            
            # Model Specific Parameters
            with col2:
                if model_type in ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]:
                    n_estimators = st.number_input(
                        "Number of Trees/Iterations", 
                        100, 1000, 100, 100,
                        help="More trees generally means better performance but slower training"
                    )
                    max_depth = st.number_input(
                        "Max Depth", 
                        3, 20, 10,
                        help="Maximum depth of each tree. Higher values might lead to overfitting"
                    )
                
                if model_type == "Logistic Regression":
                    C = st.number_input(
                        "Regularization Strength (C)", 
                        0.01, 10.0, 1.0, 0.1,
                        help="Lower values mean stronger regularization"
                    )
            
            # Prepare features
            X = df[selected_features].fillna(df[selected_features].mean())
            y = df['Outcome']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=random_state
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Create placeholder for results
            results_container = st.container()
            
            # Model Training
            train_button = st.button("Train Model", help="Click to start training the model")
            if train_button:
                with st.spinner("Training model... This might take a moment."):
                    # Initialize selected model
                    if model_type == "Random Forest":
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=random_state
                        )
                    elif model_type == "XGBoost":
                        model = XGBClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=random_state
                        )
                    elif model_type == "LightGBM":
                        model = LGBMClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=random_state
                        )
                    elif model_type == "Logistic Regression":
                        model = LogisticRegression(
                            C=C,
                            random_state=random_state,
                            max_iter=1000
                        )
                    else:  # CatBoost
                        model = CatBoostClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=random_state,
                            verbose=False
                        )
                    model.fit(X_train_scaled, y_train)
                    
                    # Store in session state
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.session_state.selected_features = selected_features
                    st.session_state.trained = True
                    
                    # Make predictions
                    y_pred = model.predict_proba(X_test_scaled)[:, 1]
                    
                    # Calculate ROC-AUC
                    roc_auc = roc_auc_score(y_test, y_pred)
                    st.session_state.roc_auc = roc_auc
                    
                    # Store feature importance
                    importance_df = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    st.session_state.importance_df = importance_df
            
            # Display results if model is trained
            if 'trained' in st.session_state and st.session_state.trained:
                with results_container:
                    # Display results
                    st.subheader("4️⃣ Model Evaluation")
                    
                    # Show ROC-AUC score with interpretation
                    st.success(f"ROC-AUC Score: {st.session_state.roc_auc:.4f}")
                    st.markdown("""
                    **Understanding the Score:**
                    - 0.5 = Random guessing
                    - 1.0 = Perfect predictions
                    - Above 0.7 is generally considered good
                    - Above 0.8 is considered excellent
                    """)
                    
                    # Feature importance
                    st.subheader("5️⃣ Feature Importance")
                    st.markdown("""
                    See which features had the biggest impact on predictions:
                    - Higher importance = more influential in predictions
                    - Can help focus on what matters most for attendance
                    """)
                    
                    fig = px.bar(
                        st.session_state.importance_df,
                        x='Feature',
                        y='Importance',
                        title="Feature Importance Plot"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save model option
                    st.markdown("### Save Model")
                    st.markdown("Save this model to use it in the Model Explainability section.")
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        save_button = st.button("💾 Save Model", help="Save this model for later use")
                    
                    if save_button:
                        if save_model(st.session_state.model, st.session_state.scaler, st.session_state.selected_features):
                            with col2:
                                st.success("✅ Model saved successfully! You can now use it in the Model Explainability section.")
                            st.balloons()
        else:
            st.warning("Please select at least one feature for training.")
    else:
        st.error("Error preparing training data. Please check the data files.")
else:
    st.error("⚠️ Error loading data. Please check if all data files are present in the Data directory.")
