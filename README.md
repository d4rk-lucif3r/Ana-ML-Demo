# Healthcare Camp Attendance Predictor 🏥

## 👋 Created by Ana

[Ana](https://openana.ai) is a sophisticated AI Software Engineer with extensive expertise in software engineering, machine learning, and data science. With a friendly and supportive approach, Ana helps developers and data scientists build better applications and gain deeper insights from their data.

Visit [openana.ai](https://openana.ai) to learn more about Ana's capabilities and how she can help with your projects! 🚀

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ana-ml-demo.streamlit.app/) [![Kaggle Dataset](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/datasets/shivan118/healthcare-analytics)

An advanced machine learning application for predicting and analyzing healthcare camp attendance patterns. This project helps MedCamp optimize their healthcare camp resources and improve attendance rates through sophisticated ML predictions and comprehensive insights.

## 🎯 Project Overview

MedCamp organizes health camps in several cities targeting working professionals with low work-life balance. The organization has conducted 65 such events over 4 years, accumulating data from approximately 110,000 registrations. This application addresses a critical challenge: optimizing inventory management while maintaining service quality.

### Business Problem

MedCamp faces two key challenges:
1. **High Drop-off Rate**: Significant difference between registration numbers and actual attendance
2. **Inventory Management**: 
   - Excess inventory leads to unnecessary costs
   - Insufficient inventory results in poor participant experience

### Solution Approach

Our application provides:
- Predictive analytics for attendance probability
- Comprehensive data exploration tools
- Multiple ML model options with explainable AI
- Feature importance analysis for better decision-making

## 📊 Dataset Details

Source: [Healthcare Analytics Dataset on Kaggle](https://www.kaggle.com/datasets/shivan118/healthcare-analytics)

### Files Structure

1. **Health_Camp_Detail.csv**
   - Health_Camp_ID
   - Camp_Start_Date
   - Camp_End_Date
   - Category details

2. **Train.csv**
   - Registration details
   - Patient_ID
   - Health_Camp_ID
   - Registration_Date
   - Anonymized variables

3. **Patient_Profile.csv**
   - Patient_ID
   - Online_Follower
   - Social media details
   - Income
   - Education
   - Age
   - First_Interaction_Date
   - City_Type
   - Employer_Category

4. **First_Health_Camp_Attended.csv**
   - Format 1 camp details
   - Donation amounts
   - Health_Score

5. **Second_Health_Camp_Attended.csv**
   - Format 2 camp details
   - Health_Score

6. **Third_Health_Camp_Attended.csv**
   - Format 3 camp details
   - Number_of_stall_visited
   - Last_Stall_Visited_Number

### Camp Formats

1. **Format 1**: Provides instantaneous health score + Donation based
2. **Format 2**: Provides instantaneous health score
3. **Format 3**: Awareness through various health stalls

### Target Variable Definition

A "favorable outcome" is defined as:
- **Format 1 & 2**: Patient receives a health score
- **Format 3**: Patient visits at least one stall

## 🔧 Technical Implementation

### Tech Stack

#### Frontend
- **Streamlit**: Interactive web interface
- **Plotly**: Dynamic visualizations
- **Matplotlib/Seaborn**: Statistical plots

#### Backend/ML
- **Core**: Python 3.8+
- **Data Processing**: 
  - pandas
  - numpy
  - scikit-learn
- **ML Models**:
  - Random Forest
  - XGBoost
  - LightGBM
  - Logistic Regression
  - CatBoost
- **Model Explainability**: SHAP (SHapley Additive exPlanations)

### Key Features

1. **Exploratory Data Analysis** 📊
   - Temporal attendance patterns
   - Demographic analysis
   - Registration vs. attendance correlation
   - Category-wise performance metrics

2. **Model Training Interface** 🤖
   - Multiple algorithm options
   - Hyperparameter tuning
   - Cross-validation
   - Performance metrics visualization

3. **Feature Analysis** 🎯
   - Importance rankings
   - Correlation studies
   - Interactive feature exploration

4. **Model Explainability** 🔍
   - SHAP value analysis
   - Feature contribution plots
   - Individual prediction explanations

5. **Model Management** 💾
   - Save/load functionality
   - Version tracking
   - Performance comparison

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip
git
```

### Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## 📱 Usage Guide

### 1. Data Exploration
- View comprehensive camp statistics
- Analyze temporal patterns
- Explore demographic distributions
- Investigate feature correlations

### 2. Model Training
- Select relevant features
- Choose ML algorithm
- Configure model parameters
- Train and evaluate performance

### 3. Model Interpretation
- Examine global feature importance
- Analyze individual predictions
- Understand model decisions
- Export insights

## 📈 Model Evaluation

### Metrics
- Primary: ROC-AUC Score
- Supporting metrics:
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix

### Validation Strategy
- Train/Test split based on temporal cutoff
- Camps before March 31st, 2006: Training
- Camps after April 1st, 2006: Testing

## 🌟 Live Demo

Experience the application: [Healthcare Camp Predictor](https://ana-ml-demo.streamlit.app/)

### Demo Features
- Interactive data exploration
- Real-time model training
- Dynamic visualizations
- Instant predictions

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.


## 🙏 Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/shivan118/healthcare-analytics)
- MedCamp for the valuable healthcare initiative
- Streamlit for the excellent web framework

---
Built with ❤️ by [Ana](https://openana.ai) | [Live Demo](https://ana-ml-demo.streamlit.app/)
