import streamlit as st
from utils.data_loader import load_data, get_dataset_description
from utils.session_state import init_session_state

# Initialize session state
init_session_state()

# Page config
st.set_page_config(
    page_title="Healthcare Camp Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
data = load_data()

# Main page
st.title("Healthcare Camp Analysis Dashboard 🏥")

st.markdown("""
### Welcome to the Healthcare Camp Analysis Dashboard! 👋

This tool helps analyze data from health camps organized by MedCamp in various cities. Here's what you can do:

1. **Explore Data** 📊
   - View and understand different datasets
   - Analyze missing values and patterns
   - Visualize distributions and relationships

2. **Analyze Features** 🔍
   - Discover correlations between variables
   - Understand categorical distributions
   - Identify important patterns

3. **Train Models** 🤖
   - Build predictive models
   - Evaluate model performance
   - Save models for future use

4. **Understand Predictions** 🎯
   - Explore how models make decisions
   - Analyze feature importance
   - Get detailed explanations

#### About the Data
MedCamp has conducted 65 health camps over 4 years, collecting data from about 110,000 registrations. The camps aim to:
- Provide health checks
- Increase health awareness
- Help working professionals maintain work-life balance

Use the sidebar to navigate through different analyses! 👈
""")

# Display some key statistics if data is loaded
if data:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"📊 Total Health Camps: {len(data['Health Camp Detail'])}")
    
    with col2:
        st.info(f"👥 Total Registrations: {len(data['Train']):,}")
    
    with col3:
        st.info(f"🏥 Unique Patients: {len(data['Patient Profile']):,}")

else:
    st.error("⚠️ Error loading data. Please check if all data files are present in the Data directory.")
