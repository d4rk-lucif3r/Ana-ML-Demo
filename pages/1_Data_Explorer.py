import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_data, get_dataset_description

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

# Load data
data = load_data()
dataset_descriptions = get_dataset_description()

st.title("Data Explorer 📊")

st.markdown("""
This page helps you understand the data collected from health camps. You can:
- View basic information about each dataset
- Check for missing values
- Analyze numerical distributions
- Get quick statistical summaries
""")

if data:
    # Dataset selection
    selected_dataset = st.selectbox(
        "Select Dataset to Explore",
        list(dataset_descriptions.keys()),
        help="Choose a dataset to analyze"
    )
    
    st.info(f"💡 **About this dataset**: {dataset_descriptions[selected_dataset]}")
    
    # Basic information
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Dataset Overview")
        st.write(f"Number of rows: {data[selected_dataset].shape[0]:,}")
        st.write(f"Number of columns: {data[selected_dataset].shape[1]}")
    
    with col2:
        st.subheader("📊 Data Types")
        dtypes = data[selected_dataset].dtypes.value_counts()
        st.write("Number of columns by type:")
        st.write(dtypes)

    # Sample data with explanation
    st.subheader("📎 Sample Data")
    st.markdown("Here's a peek at the first few rows of the data:")
    st.dataframe(data[selected_dataset].head(), use_container_width=True)
    
    # Missing values analysis
    st.subheader("🔍 Missing Values Analysis")
    missing_data = data[selected_dataset].isnull().sum()
    missing_percent = (missing_data / len(data[selected_dataset])) * 100
    
    if missing_data.any():
        st.markdown("""
        Understanding missing values is crucial because:
        - They might indicate data collection issues
        - They need to be handled before analysis
        - They could affect model performance
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=missing_data.index,
                y=missing_data.values,
                title="Missing Values Count",
                labels={'x': 'Column', 'y': 'Missing Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=missing_data.index,
                y=missing_percent,
                title="Missing Values Percentage",
                labels={'x': 'Column', 'y': 'Missing %'}
            )
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        
        # Table view of missing values
        st.markdown("### Detailed Missing Values Information")
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': missing_percent.round(2)
        }).query('`Missing Count` > 0')
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("✅ Great news! This dataset has no missing values.")
    
    # Numerical analysis
    st.subheader("📈 Numerical Columns Analysis")
    num_cols = data[selected_dataset].select_dtypes(include=['int64', 'float64']).columns
    
    if len(num_cols) > 0:
        st.markdown("""
        Analyze the distribution of numerical columns:
        - The histogram shows how values are distributed
        - The box plot shows outliers and quartiles
        - The statistics provide key numerical summaries
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_num_col = st.selectbox("Select Column to Analyze", num_cols)
            fig = px.histogram(
                data[selected_dataset],
                x=selected_num_col,
                title=f"Distribution of {selected_num_col}",
                marginal="box"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Statistical Summary")
            stats_df = pd.DataFrame({
                'Statistic': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
                'Value': [
                    f"{data[selected_dataset][selected_num_col].count():,.0f}",
                    f"{data[selected_dataset][selected_num_col].mean():,.2f}",
                    f"{data[selected_dataset][selected_num_col].std():,.2f}",
                    f"{data[selected_dataset][selected_num_col].min():,.2f}",
                    f"{data[selected_dataset][selected_num_col].quantile(0.25):,.2f}",
                    f"{data[selected_dataset][selected_num_col].quantile(0.5):,.2f}",
                    f"{data[selected_dataset][selected_num_col].quantile(0.75):,.2f}",
                    f"{data[selected_dataset][selected_num_col].max():,.2f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
            
            st.markdown("""
            **Understanding these statistics:**
            - **Count**: Number of non-missing values
            - **Mean**: Average value
            - **Std**: How spread out the values are
            - **Min/Max**: Smallest/largest values
            - **25%/50%/75%**: Values at these percentiles
            """)
else:
    st.error("⚠️ Error loading data. Please check if all data files are present in the Data directory.")
