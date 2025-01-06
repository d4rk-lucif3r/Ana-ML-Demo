import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_data

st.set_page_config(page_title="Feature Analysis", page_icon="🔍", layout="wide")

# Load data
data = load_data()

st.title("Feature Analysis 🔍")

st.markdown("""
This page helps you understand relationships between different features in the data:
- Discover correlations between numerical variables
- Analyze categorical variable distributions
- Identify important patterns and insights
""")

if data:
    # Dataset selection
    selected_dataset = st.selectbox(
        "Choose Dataset to Analyze",
        list(data.keys()),
        help="Select a dataset to analyze its features"
    )
    
    st.info(f"Dataset Shape: {data[selected_dataset].shape[0]:,} rows × {data[selected_dataset].shape[1]} columns")
    
    # Correlation analysis
    st.subheader("📊 Correlation Analysis")
    st.markdown("""
    Correlation shows how strongly pairs of variables are related:
    - Values close to 1 indicate strong positive correlation
    - Values close to -1 indicate strong negative correlation
    - Values close to 0 indicate little to no correlation
    """)
    
    num_cols = data[selected_dataset].select_dtypes(include=['int64', 'float64']).columns
    
    if len(num_cols) > 0:
        # Allow user to select specific columns
        selected_cols = st.multiselect(
            "Select columns for correlation analysis",
            num_cols,
            default=list(num_cols)[:5],
            help="Choose columns to analyze their relationships"
        )
        
        if selected_cols:
            corr_matrix = data[selected_dataset][selected_cols].corr()
            
            # Correlation heatmap
            fig = px.imshow(
                corr_matrix,
                title="Correlation Matrix Heatmap",
                labels=dict(color="Correlation"),
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show strongest correlations
            correlations = []
            for i in range(len(selected_cols)):
                for j in range(i+1, len(selected_cols)):
                    correlations.append({
                        'Feature 1': selected_cols[i],
                        'Feature 2': selected_cols[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            
            if correlations:
                st.subheader("🔝 Top Feature Correlations")
                st.markdown("""
                These are the strongest relationships found between features:
                - Strong positive correlations suggest features increase together
                - Strong negative correlations suggest as one increases, the other decreases
                """)
                
                corr_df = pd.DataFrame(correlations)
                corr_df['Abs Correlation'] = abs(corr_df['Correlation'])
                corr_df = corr_df.sort_values('Abs Correlation', ascending=False)
                
                st.dataframe(
                    corr_df[['Feature 1', 'Feature 2', 'Correlation']]
                    .head(10)
                    .style.background_gradient(subset=['Correlation'], cmap='RdBu'),
                    use_container_width=True
                )
    
    # Categorical analysis
    st.subheader("📊 Categorical Features Analysis")
    cat_cols = data[selected_dataset].select_dtypes(include=['object']).columns
    
    if len(cat_cols) > 0:
        st.markdown("""
        Analyze the distribution of categorical variables:
        - See how many items fall into each category
        - Understand the balance of categories
        - Identify rare or dominant categories
        """)
        
        selected_cat_col = st.selectbox(
            "Select Categorical Column",
            cat_cols,
            help="Choose a categorical column to analyze its distribution"
        )
        
        # Value counts and visualization
        value_counts = data[selected_dataset][selected_cat_col].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Distribution of {selected_cat_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Count Plot of {selected_cat_col}",
                labels={'x': selected_cat_col, 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show value counts table
        st.markdown("### Detailed Category Information")
        value_counts_df = pd.DataFrame({
            'Category': value_counts.index,
            'Count': value_counts.values,
            'Percentage': (value_counts.values / len(data[selected_dataset]) * 100).round(2)
        })
        st.dataframe(value_counts_df, use_container_width=True)
        
else:
    st.error("⚠️ Error loading data. Please check if all data files are present in the Data directory.")
